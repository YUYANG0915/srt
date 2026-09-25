#!/usr/bin/env python3
"""Build Task3 tool-use / object-operation candidate review pack from local COCO subsets."""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ZIP_PATH = DATA_DIR / "coco_annotations" / "annotations_trainval2017.zip"
IMAGE_DIRS = [
    DATA_DIR / "coco_val2017_task1_raw500_images",
    DATA_DIR / "coco_val2017_task2_raw500_images",
    DATA_DIR / "coco_val2017_pilot_images",
    DATA_DIR / "coco_val2017_unique_strict_images",
    DATA_DIR / "coco_val2017_unique_images",
]


@dataclass
class Candidate:
    image_id: int
    image_path: str
    captions: list[str]
    categories: list[str]
    suggested_label: str
    suggested_boundary_family: str
    suggested_tool: str
    suggested_object: str
    suggestion_reason: str
    trigger_words: list[str]
    priority_score: int


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def load_coco_metadata() -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        caps = json.load(io.TextIOWrapper(zf.open("annotations/captions_val2017.json"), encoding="utf-8"))
        inst = json.load(io.TextIOWrapper(zf.open("annotations/instances_val2017.json"), encoding="utf-8"))

    cat_name = {row["id"]: row["name"] for row in inst["categories"]}
    caps_by_img: dict[int, list[str]] = defaultdict(list)
    cats_by_img: dict[int, list[str]] = defaultdict(list)
    for ann in caps["annotations"]:
        caps_by_img[ann["image_id"]].append(ann["caption"])
    for ann in inst["annotations"]:
        cats_by_img[ann["image_id"]].append(cat_name[ann["category_id"]])
    return caps_by_img, cats_by_img


def find_local_images() -> dict[int, str]:
    image_map: dict[int, str] = {}
    for image_dir in IMAGE_DIRS:
        for path in sorted(image_dir.glob("*.jpg")):
            image_map.setdefault(int(path.stem), str(path.resolve()))
    return image_map


def contains_any(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if term in text]


def infer_candidate(captions: list[str], categories: list[str]) -> tuple[str, str, str, str, str, list[str], int] | None:
    text = " ".join(captions).lower()
    cats = set(categories)
    score = 0

    # Strong tool-use signals with visible human-tool interaction in COCO-style captions.
    action_terms = contains_any(
        text,
        [
            "using",
            "uses",
            "holding",
            "holds",
            "brushing",
            "brush",
            "cutting",
            "cut",
            "slicing",
            "typing",
            "working on",
            "talking on",
            "taking a picture",
            "taking photo",
            "looking at",
            "playing",
            "drying",
            "operating",
            "remote",
        ],
    )
    if "person" in cats:
        score += 3
    score += min(len(action_terms), 4)

    if "toothbrush" in cats or "tooth brush" in text or "brushing" in text:
        triggers = action_terms + ["toothbrush"]
        if "brushing" in text or "brush" in text:
            return (
                "active_tool_use",
                "setup_vs_active",
                "toothbrush",
                "teeth/mouth",
                "toothbrush plus brushing cue suggests active tool-object contact",
                triggers,
                score + 10,
            )
        return (
            "preparing_tool",
            "setup_vs_active",
            "toothbrush",
            "teeth/mouth",
            "toothbrush is visible but active brushing may need visual confirmation",
            triggers,
            score + 5,
        )

    if "scissors" in cats or "scissors" in text:
        triggers = action_terms + ["scissors"]
        if any(term in text for term in ["cutting", "cut ", "slicing"]):
            return (
                "active_tool_use",
                "setup_vs_active",
                "scissors",
                "target material/object",
                "scissors plus cutting cue suggests active operation",
                triggers,
                score + 9,
            )
        return (
            "preparing_tool",
            "setup_vs_active",
            "scissors",
            "target material/object",
            "scissors are visible; keep only if image shows setup or contact clearly",
            triggers,
            score + 5,
        )

    if "knife" in cats or "knife" in text:
        triggers = action_terms + ["knife"]
        if any(term in text for term in ["cutting", "cut ", "slicing", "slice"]):
            return (
                "active_tool_use",
                "setup_vs_active",
                "knife",
                "food/object",
                "knife plus cutting/slicing cue suggests active operation",
                triggers,
                score + 8,
            )
        return (
            "preparing_tool",
            "setup_vs_active",
            "knife",
            "food/object",
            "knife is present but active cutting must be checked visually",
            triggers,
            score + 4,
        )

    if "hair drier" in cats or "hair dryer" in text or "hair drier" in text:
        triggers = action_terms + ["hair drier"]
        if any(term in text for term in ["drying", "using", "holding"]):
            return (
                "active_tool_use",
                "setup_vs_active",
                "hair drier",
                "hair/person",
                "hair drier plus holding/using cue suggests active drying",
                triggers,
                score + 8,
            )
        return (
            "preparing_tool",
            "setup_vs_active",
            "hair drier",
            "hair/person",
            "hair drier is visible but active operation needs confirmation",
            triggers,
            score + 4,
        )

    if {"keyboard", "mouse", "laptop"} & cats:
        triggers = action_terms + sorted({"keyboard", "mouse", "laptop"} & cats)
        if any(term in text for term in ["typing", "working on", "using", "at a computer", "computer"]):
            return (
                "active_tool_use",
                "tool_vs_object_salience",
                "keyboard/mouse/laptop",
                "computer interface",
                "person-computer interaction cue suggests active tool use",
                triggers,
                score + 7,
            )
        return (
            "preparing_tool",
            "tool_vs_object_salience",
            "keyboard/mouse/laptop",
            "computer interface",
            "computer tool is present but active use may be unclear",
            triggers,
            score + 3,
        )

    if "cell phone" in cats or "phone" in text:
        triggers = action_terms + ["cell phone"]
        if any(term in text for term in ["talking on", "using", "holding", "taking a picture", "taking photo", "looking at"]):
            return (
                "active_tool_use",
                "tool_vs_object_salience",
                "cell phone",
                "communication/photo target",
                "phone plus hand/attention cue suggests active use",
                triggers,
                score + 7,
            )
        return (
            "preparing_tool",
            "tool_vs_object_salience",
            "cell phone",
            "communication/photo target",
            "phone is visible but active use needs visual confirmation",
            triggers,
            score + 3,
        )

    if "remote" in cats:
        triggers = action_terms + ["remote"]
        return (
            "preparing_tool" if not action_terms else "active_tool_use",
            "tool_vs_object_salience",
            "remote",
            "device/TV",
            "remote-control tool is visible; active operation depends on hand/target relation",
            triggers,
            score + 4,
        )

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
        "visible_object",
        "visible_state_cues",
        "issue_type",
        "notes",
        "source_dataset",
        "source_image_id",
        "suggested_label",
        "suggested_boundary_family",
        "suggested_tool",
        "suggested_object",
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
                    "id": f"task3tool_{idx:04d}_{cand.image_id}",
                    "image_path": cand.image_path,
                    "keep": "",
                    "question": "What operation stage is shown in this scene?",
                    "gold_label": "",
                    "neighbor_label": "",
                    "boundary_family": "",
                    "why_not_neighbor": "",
                    "visible_actor": "",
                    "visible_tool": "",
                    "visible_object": "",
                    "visible_state_cues": "",
                    "issue_type": "",
                    "notes": "",
                    "source_dataset": "coco_val2017_local_subset",
                    "source_image_id": str(cand.image_id),
                    "suggested_label": cand.suggested_label,
                    "suggested_boundary_family": cand.suggested_boundary_family,
                    "suggested_tool": cand.suggested_tool,
                    "suggested_object": cand.suggested_object,
                    "suggestion_reason": cand.suggestion_reason,
                    "captions": " || ".join(cand.captions),
                    "categories": " | ".join(cand.categories),
                    "trigger_words": " | ".join(cand.trigger_words),
                    "priority_score": str(cand.priority_score),
                    "scenario_family": "tool_use_object_operation",
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
                <p><b>tool-object:</b> {html.escape(cand.suggested_tool)} -> {html.escape(cand.suggested_object)}</p>
                <p><b>reason:</b> {html.escape(cand.suggestion_reason)}</p>
                <p><b>categories:</b> {html.escape(', '.join(cand.categories))}</p>
                <p><b>triggers:</b> {html.escape(', '.join(cand.trigger_words))}</p>
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
  <title>Task3 Tool-use / Object-operation Candidates</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 24px; }}
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
  <h1>Task3 Tool-use / Object-operation Candidates</h1>
  <p>Total candidates: {len(rows)}</p>
  <p>Review goal: keep only images where a human/tool/object operation stage is visually grounded.</p>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--max-per-tool", type=int, default=16)
    parser.add_argument("--exclude-jsonl", action="append", default=[])
    parser.add_argument("--output-csv", default="review/task3_tooluse_candidates_v1_review.csv")
    parser.add_argument("--output-html", default="review/task3_tooluse_candidates_v1_review.html")
    parser.add_argument("--output-summary", default="runs/task3_tooluse_candidates_v1_summary_2026-06-29.json")
    args = parser.parse_args()

    excluded: set[str] = set()
    for rel_path in args.exclude_jsonl:
        excluded.update(str(row.get("image_id")) for row in load_jsonl(ROOT / rel_path))

    image_map = find_local_images()
    caps_by_img, cats_by_img = load_coco_metadata()

    candidates: list[Candidate] = []
    for image_id, image_path in image_map.items():
        if str(image_id) in excluded:
            continue
        captions = caps_by_img.get(image_id, [])[:5]
        categories = sorted(set(cats_by_img.get(image_id, [])))
        inferred = infer_candidate(captions, categories)
        if not inferred:
            continue
        label, family, tool, obj, reason, triggers, score = inferred
        candidates.append(
            Candidate(
                image_id=image_id,
                image_path=image_path,
                captions=captions,
                categories=categories,
                suggested_label=label,
                suggested_boundary_family=family,
                suggested_tool=tool,
                suggested_object=obj,
                suggestion_reason=reason,
                trigger_words=triggers,
                priority_score=score,
            )
        )

    candidates.sort(key=lambda row: (-row.priority_score, row.suggested_boundary_family, row.image_id))
    selected: list[Candidate] = []
    per_tool_count: dict[str, int] = {}
    for cand in candidates:
        if len(selected) >= args.limit:
            break
        count = per_tool_count.get(cand.suggested_tool, 0)
        if count >= args.max_per_tool:
            continue
        selected.append(cand)
        per_tool_count[cand.suggested_tool] = count + 1
    write_csv(ROOT / args.output_csv, selected)
    write_html(ROOT / args.output_html, selected)

    summary = {
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "excluded_eval_images": len(excluded),
        "output_csv": str(ROOT / args.output_csv),
        "output_html": str(ROOT / args.output_html),
        "label_distribution": {},
        "family_distribution": {},
        "tool_distribution": {},
    }
    for row in selected:
        summary["label_distribution"][row.suggested_label] = summary["label_distribution"].get(row.suggested_label, 0) + 1
        summary["family_distribution"][row.suggested_boundary_family] = summary["family_distribution"].get(row.suggested_boundary_family, 0) + 1
        summary["tool_distribution"][row.suggested_tool] = summary["tool_distribution"].get(row.suggested_tool, 0) + 1
    out_summary = ROOT / args.output_summary
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
