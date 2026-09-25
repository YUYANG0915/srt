#!/usr/bin/env python3
"""Build Task3 P5 cleaning / household state-change candidate pack from COCO val2017."""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import ssl
import time
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import certifi


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ZIP_PATH = DATA_DIR / "coco_annotations" / "annotations_trainval2017.zip"
IMAGE_DIR = DATA_DIR / "task3_cleaning_coco_images"
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


@dataclass
class Candidate:
    image_id: int
    image_path: str
    captions: list[str]
    categories: list[str]
    suggested_label: str
    suggested_boundary_family: str
    suggested_tool: str
    suggested_target: str
    suggested_state_cues: str
    suggestion_reason: str
    trigger_words: list[str]
    priority_score: int


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def load_coco_metadata() -> tuple[dict[int, list[str]], dict[int, list[str]], dict[int, str]]:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        caps = json.load(io.TextIOWrapper(zf.open("annotations/captions_val2017.json"), encoding="utf-8"))
        inst = json.load(io.TextIOWrapper(zf.open("annotations/instances_val2017.json"), encoding="utf-8"))

    cat_name = {row["id"]: row["name"] for row in inst["categories"]}
    caps_by_img: dict[int, list[str]] = defaultdict(list)
    cats_by_img: dict[int, list[str]] = defaultdict(list)
    file_by_img: dict[int, str] = {}
    for image in inst["images"]:
        file_by_img[image["id"]] = image["file_name"]
    for ann in caps["annotations"]:
        caps_by_img[ann["image_id"]].append(ann["caption"])
    for ann in inst["annotations"]:
        cats_by_img[ann["image_id"]].append(cat_name[ann["category_id"]])
    return caps_by_img, cats_by_img, file_by_img


def contains_any(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term in text]


def has_phrase(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms)


def infer_candidate(captions: list[str], categories: list[str]) -> tuple[str, str, str, str, str, str, list[str], int] | None:
    text = " ".join(captions).lower()
    cats = set(categories)
    trigger_terms = [
        "cleaning",
        "clean",
        "washing",
        "wash",
        "rinsing",
        "rinse",
        "wiping",
        "wipe",
        "scrubbing",
        "scrub",
        "brushing",
        "brush",
        "toothbrush",
        "sink",
        "soap",
        "hose",
        "shower",
        "bath",
        "car wash",
        "dishes",
        "dish",
        "toilet",
        "bathroom",
    ]
    triggers = contains_any(text, trigger_terms)
    if "toothbrush" in cats:
        triggers.append("toothbrush")
    if "sink" in cats:
        triggers.append("sink")
    if "toilet" in cats:
        triggers.append("toilet")

    if not triggers:
        return None
    if "moped" in text and not has_phrase(text, ["mop ", "mopping", "mopped"]):
        return None

    score = len(set(triggers))
    if "person" in cats:
        score += 2
    if has_phrase(text, ["cleaning", "washing", "wiping", "scrubbing", "brushing", "rinsing", "car wash"]):
        score += 6
    if {"sink", "toothbrush", "toilet"} & cats:
        score += 2

    if has_phrase(text, ["brushing teeth", "brushes her teeth", "brushes his teeth", "brushing his teeth", "brushing her teeth"]):
        return (
            "active_cleaning",
            "setup_vs_active",
            "toothbrush",
            "teeth/mouth",
            "toothbrush is in use on teeth",
            "caption directly describes active tooth cleaning",
            triggers,
            score + 8,
        )
    if "toothbrush" in cats:
        return (
            "pre_cleaning_setup",
            "setup_vs_active",
            "toothbrush",
            "teeth/mouth",
            "toothbrush present but active contact must be visually checked",
            "toothbrush presence may indicate setup rather than active cleaning",
            triggers,
            score + 3,
        )
    if has_phrase(text, ["washing", "wash ", "rinsing", "rinse", "car wash"]):
        target = "vehicle/object/body/dishes"
        if has_phrase(text, ["motorcycle", "motorcycles", "bike"]):
            target = "motorcycle"
        elif has_phrase(text, ["dishes", "dish", "plate", "plates"]):
            target = "dishes/tableware"
        elif has_phrase(text, ["hands", "face", "body", "bath"]):
            target = "person/body"
        return (
            "active_cleaning",
            "setup_vs_active",
            "water/hose/hand",
            target,
            "water/cleaning action is described",
            "washing/rinsing cue suggests an active cleaning state",
            triggers,
            score + 7,
        )
    if has_phrase(text, ["cleaning", "wiping", "scrubbing"]):
        return (
            "active_cleaning",
            "setup_vs_active",
            "cloth/brush/hand",
            "surface/object",
            "cleaning contact or wiping/scrubbing action is described",
            "caption suggests active contact with a target surface/object",
            triggers,
            score + 6,
        )
    if has_phrase(text, ["clean motorcycle", "clean car", "clean room", "clean bathroom", "clean kitchen", "clean sink", "clean toilet"]) and not has_phrase(
        text, ["cleaning", "washing", "wiping", "scrubbing", "brushing", "rinsing", "car wash"]
    ):
        target = "cleaned object/space"
        if "motorcycle" in text:
            target = "motorcycle"
        elif "car" in text:
            target = "car"
        elif has_phrase(text, ["bathroom", "sink", "toilet"]):
            target = "bathroom fixture/space"
        return (
            "cleaned_or_finished",
            "active_vs_result",
            "none_or_not_active",
            target,
            "clean/result state is described without active cleaning contact",
            "caption suggests a cleaned/result state rather than ongoing cleaning",
            triggers,
            score + 4,
        )
    if has_phrase(text, ["sink", "bathroom", "toilet"]) or {"sink", "toilet"} & cats:
        return (
            "unclear_or_non_process",
            "cleaning_vs_noncleaning_salience",
            "none_or_unclear",
            "bathroom/sink/toilet scene",
            "cleaning-related object is salient but active cleaning is not described",
            "bathroom fixtures can be salient without a cleaning process",
            triggers,
            score,
        )
    return None


def download_image(image_id: int, file_name: str) -> Path | None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = IMAGE_DIR / file_name
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    url = f"http://images.cocodataset.org/val2017/{file_name}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60, context=SSL_CONTEXT) as response:
                out_path.write_bytes(response.read())
            if out_path.exists() and out_path.stat().st_size > 0:
                return out_path
        except Exception:
            time.sleep(2**attempt)
    return None


def write_csv(path: Path, rows: list[Candidate]) -> None:
    fieldnames = [
        "id",
        "image_path",
        "keep",
        "question",
        "gold_label",
        "neighbor_label",
        "boundary_family",
        "why_not_neighbor",
        "visible_actor",
        "visible_tool",
        "visible_target",
        "visible_state_cues",
        "issue_type",
        "notes",
        "source_dataset",
        "source_image_id",
        "suggested_label",
        "suggested_boundary_family",
        "suggested_tool",
        "suggested_target",
        "suggested_state_cues",
        "suggestion_reason",
        "captions",
        "categories",
        "trigger_words",
        "priority_score",
        "scenario_family",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, cand in enumerate(rows, start=1):
            writer.writerow(
                {
                    "id": f"task3clean_{idx:04d}_{cand.image_id}",
                    "image_path": cand.image_path,
                    "keep": "",
                    "question": "What cleaning stage is shown in this scene?",
                    "gold_label": "",
                    "neighbor_label": "",
                    "boundary_family": "",
                    "why_not_neighbor": "",
                    "visible_actor": "",
                    "visible_tool": "",
                    "visible_target": "",
                    "visible_state_cues": "",
                    "issue_type": "",
                    "notes": "",
                    "source_dataset": "coco_val2017_downloaded",
                    "source_image_id": str(cand.image_id),
                    "suggested_label": cand.suggested_label,
                    "suggested_boundary_family": cand.suggested_boundary_family,
                    "suggested_tool": cand.suggested_tool,
                    "suggested_target": cand.suggested_target,
                    "suggested_state_cues": cand.suggested_state_cues,
                    "suggestion_reason": cand.suggestion_reason,
                    "captions": " || ".join(cand.captions),
                    "categories": " | ".join(cand.categories),
                    "trigger_words": " | ".join(sorted(set(cand.trigger_words))),
                    "priority_score": str(cand.priority_score),
                    "scenario_family": "cleaning_household_state_change",
                }
            )


def write_html(path: Path, rows: list[Candidate]) -> None:
    cards = []
    for idx, cand in enumerate(rows, start=1):
        caps = "".join(f"<li>{html.escape(cap)}</li>" for cap in cand.captions)
        cards.append(
            f"""
            <div class="card">
              <img src="file://{html.escape(cand.image_path)}" alt="{cand.image_id}">
              <div class="body">
                <h3>{idx}. image {cand.image_id}</h3>
                <p><b>suggested:</b> {html.escape(cand.suggested_label)} / {html.escape(cand.suggested_boundary_family)}</p>
                <p><b>tool-target:</b> {html.escape(cand.suggested_tool)} -> {html.escape(cand.suggested_target)}</p>
                <p><b>state cues:</b> {html.escape(cand.suggested_state_cues)}</p>
                <p><b>reason:</b> {html.escape(cand.suggestion_reason)}</p>
                <p><b>categories:</b> {html.escape(', '.join(cand.categories))}</p>
                <p><b>triggers:</b> {html.escape(', '.join(sorted(set(cand.trigger_words))))}</p>
                <p><b>score:</b> {cand.priority_score}</p>
                <ul>{caps}</ul>
                <p class="path">{html.escape(cand.image_path)}</p>
              </div>
            </div>
            """
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Task3 P5 Cleaning / Household Task Candidates</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 24px; color: #111; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 18px; }}
    .card {{ border: 1.5px solid #222; border-radius: 8px; overflow: hidden; background: #fff; }}
    img {{ width: 100%; height: 270px; object-fit: contain; display: block; background: #f4f4f4; }}
    .body {{ padding: 14px; }}
    h3 {{ margin: 0 0 8px; font-size: 21px; }}
    p, li {{ font-size: 15px; line-height: 1.35; }}
    ul {{ padding-left: 20px; }}
    .path {{ color: #555; font-size: 12px; word-break: break-all; }}
  </style>
</head>
<body>
  <h1>Task3 P5 Cleaning / Household Task Candidates</h1>
  <p>Total candidates: {len(rows)}</p>
  <p>Review goal: keep only images where a cleaning stage or cleaning-related non-process boundary is visually grounded.</p>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_eval_jsonl(path: Path, rows: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx, cand in enumerate(rows, start=1):
            row = {
                "id": f"task3clean_{idx:04d}_{cand.image_id}",
                "image_id": str(cand.image_id),
                "source": "task3_cleaning_bootstrap",
                "source_dataset": "coco_val2017_downloaded",
                "task": "task3_cleaning_household_state_change",
                "scenario_family": "cleaning_household_state_change",
                "underspecified_question": "What cleaning stage is shown in this scene?",
                "original_question": "What cleaning stage is shown in this scene?",
                "answer_space": [
                    "pre_cleaning_setup",
                    "active_cleaning",
                    "cleaned_or_finished",
                    "unclear_or_non_process",
                ],
                "gold_label": cand.suggested_label,
                "expected_answer": cand.suggested_label,
                "answer": cand.suggested_label,
                "label_source": "suggested_bootstrap",
                "boundary_family": cand.suggested_boundary_family,
                "boundary_family_hint": cand.suggested_boundary_family,
                "boundary_source": "suggested_bootstrap",
                "neighbor_label": "",
                "why_not_neighbor": cand.suggestion_reason,
                "visible_tool": cand.suggested_tool,
                "visible_target": cand.suggested_target,
                "visible_state_cues": cand.suggested_state_cues,
                "captions": " || ".join(cand.captions),
                "categories": " | ".join(cand.categories),
                "image_path": cand.image_path,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--download-limit", type=int, default=90)
    parser.add_argument("--output-csv", default="review/task3_cleaning_candidates_v1_review.csv")
    parser.add_argument("--output-html", default="review/task3_cleaning_candidates_v1_review.html")
    parser.add_argument("--output-jsonl", default="data/task3_cleaning_eval_v1_bootstrap.jsonl")
    parser.add_argument("--output-summary", default="runs/task3_cleaning_candidates_v1_summary_2026-06-30.json")
    args = parser.parse_args()

    caps_by_img, cats_by_img, file_by_img = load_coco_metadata()
    candidates: list[tuple[int, tuple[str, str, str, str, str, str, list[str], int]]] = []
    for image_id, captions in caps_by_img.items():
        inferred = infer_candidate(captions[:5], sorted(set(cats_by_img.get(image_id, []))))
        if inferred:
            candidates.append((image_id, inferred))
    candidates.sort(key=lambda row: (-row[1][-1], row[0]))

    selected: list[Candidate] = []
    for image_id, inferred in candidates:
        if len(selected) >= args.limit:
            break
        if len(selected) >= args.download_limit:
            break
        file_name = file_by_img.get(image_id, f"{image_id:012d}.jpg")
        image_path = download_image(image_id, file_name)
        if not image_path:
            continue
        label, family, tool, target, state_cues, reason, triggers, score = inferred
        selected.append(
            Candidate(
                image_id=image_id,
                image_path=str(image_path.resolve()),
                captions=caps_by_img.get(image_id, [])[:5],
                categories=sorted(set(cats_by_img.get(image_id, []))),
                suggested_label=label,
                suggested_boundary_family=family,
                suggested_tool=tool,
                suggested_target=target,
                suggested_state_cues=state_cues,
                suggestion_reason=reason,
                trigger_words=triggers,
                priority_score=score,
            )
        )

    write_csv(ROOT / args.output_csv, selected)
    write_html(ROOT / args.output_html, selected)
    write_eval_jsonl(ROOT / args.output_jsonl, selected)

    summary = {
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "image_dir": str(IMAGE_DIR),
        "output_csv": str(ROOT / args.output_csv),
        "output_html": str(ROOT / args.output_html),
        "output_jsonl": str(ROOT / args.output_jsonl),
        "label_distribution": {},
        "family_distribution": {},
    }
    for row in selected:
        summary["label_distribution"][row.suggested_label] = summary["label_distribution"].get(row.suggested_label, 0) + 1
        summary["family_distribution"][row.suggested_boundary_family] = summary["family_distribution"].get(row.suggested_boundary_family, 0) + 1
    out_summary = ROOT / args.output_summary
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
