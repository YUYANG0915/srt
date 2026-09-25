#!/usr/bin/env python3
"""Build a broader procedural-cooking candidate pack from local COCO subsets."""

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
REVIEW_DIR = ROOT / "review"
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
    suggested_step_label: str
    suggested_boundary_family: str
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


def infer_candidate(captions: list[str], categories: list[str]) -> tuple[str, str, str, list[str], int] | None:
    text = " ".join(captions).lower()
    cats = set(categories)
    food_cats = {"banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"}
    if not cats & food_cats:
        return None

    triggers: list[str] = []
    score = 1 + len(cats & food_cats)

    def has(*terms: str) -> bool:
        return any(term in text for term in terms)

    if has("oven", "grill", "grilling", "stove", "cooking", "cook", "baking", "kitchen"):
        for term in ["oven", "grill", "stove", "cooking", "baking", "kitchen"]:
            if term in text:
                triggers.append(term)
        return (
            "heating_cooking",
            "active_heat_vs_ready_result",
            "caption/category cues suggest active heat or kitchen cooking context",
            triggers,
            score + 8,
        )

    if has("cut", "cutting", "chopping", "knife", "slicing", "slice being"):
        for term in ["cut", "cutting", "chopping", "knife", "slicing", "slice"]:
            if term in text:
                triggers.append(term)
        return (
            "cutting",
            "explicit_action",
            "caption cues suggest a visible cutting/slicing action or boundary",
            triggers,
            score + 6,
        )

    if has("bowl", "mix", "mixing", "stir", "stirring", "blend", "blender"):
        for term in ["bowl", "mix", "stir", "blend"]:
            if term in text:
                triggers.append(term)
        return (
            "mixing",
            "explicit_action",
            "caption/category cues suggest mixing or ingredient combination",
            triggers,
            score + 5,
        )

    if has(
        "plate",
        "platter",
        "table",
        "served",
        "serving",
        "display",
        "counter",
        "holding",
        "eating",
        "ready",
        "topped",
        "topping",
        "frosted",
        "birthday",
        "bakery",
        "slice of",
    ):
        for term in ["plate", "table", "served", "display", "counter", "holding", "eating", "topped", "bakery", "slice"]:
            if term in text:
                triggers.append(term)
        return (
            "plating_serving",
            "assembly_vs_serving",
            "caption/category cues suggest food assembly, presentation, serving, or ready-to-eat state",
            triggers,
            score + 4,
        )

    if cats & {"fork", "knife", "spoon", "bowl", "dining table", "cup", "wine glass"}:
        triggers.append("food+tableware")
        return (
            "plating_serving",
            "assembly_vs_serving",
            "food appears with tableware/dining context; likely presentation or serving boundary",
            triggers,
            score + 2,
        )

    return (
        "plating_serving",
        "assembly_vs_serving",
        "food object is visible but process boundary is weak; keep only if image evidence is clear",
        ["food_object"],
        score,
    )


def write_csv(path: Path, rows: list[Candidate]) -> None:
    fieldnames = [
        "id",
        "image_path",
        "keep",
        "boundary_family",
        "question",
        "gold_step_label",
        "neighbor_label",
        "why_not_neighbor_label",
        "visible_actor",
        "visible_tool",
        "visible_object",
        "visible_state_cues",
        "issue_type",
        "notes",
        "source_dataset",
        "source_image_id",
        "suggested_step_label",
        "suggested_boundary_family",
        "suggestion_reason",
        "captions",
        "categories",
        "trigger_words",
        "priority_score",
        "scenario_domain",
        "activity_name",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, cand in enumerate(rows, start=1):
            writer.writerow(
                {
                    "id": f"task3cookv2_{idx:04d}_{cand.image_id}",
                    "image_path": cand.image_path,
                    "keep": "",
                    "boundary_family": "",
                    "question": "What step of the process is happening in this scene?",
                    "gold_step_label": "",
                    "neighbor_label": "",
                    "why_not_neighbor_label": "",
                    "visible_actor": "",
                    "visible_tool": "",
                    "visible_object": "",
                    "visible_state_cues": "",
                    "issue_type": "",
                    "notes": "",
                    "source_dataset": "coco_val2017_local_subset",
                    "source_image_id": str(cand.image_id),
                    "suggested_step_label": cand.suggested_step_label,
                    "suggested_boundary_family": cand.suggested_boundary_family,
                    "suggestion_reason": cand.suggestion_reason,
                    "captions": " || ".join(cand.captions),
                    "categories": " | ".join(cand.categories),
                    "trigger_words": " | ".join(cand.trigger_words),
                    "priority_score": str(cand.priority_score),
                    "scenario_domain": "cooking",
                    "activity_name": "cooking_or_food_preparation",
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
                <p><b>suggested:</b> {html.escape(cand.suggested_step_label)} / {html.escape(cand.suggested_boundary_family)}</p>
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
  <title>Task3 Procedural Cooking Candidates V2</title>
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
  <h1>Task3 Procedural Cooking Candidates V2</h1>
  <p>Total candidates: {len(rows)}</p>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--exclude-jsonl", default="data/task3_procedural_eval_v2.jsonl")
    parser.add_argument("--output-csv", default="review/task3_procedural_cooking_candidates_v2_review.csv")
    parser.add_argument("--output-html", default="review/task3_procedural_cooking_candidates_v2_review.html")
    parser.add_argument("--output-summary", default="runs/task3_procedural_cooking_candidates_v2_summary_2026-06-28.json")
    args = parser.parse_args()

    excluded = {str(row.get("image_id")) for row in load_jsonl(ROOT / args.exclude_jsonl)}
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
        label, family, reason, triggers, score = inferred
        candidates.append(
            Candidate(
                image_id=image_id,
                image_path=image_path,
                captions=captions,
                categories=categories,
                suggested_step_label=label,
                suggested_boundary_family=family,
                suggestion_reason=reason,
                trigger_words=triggers,
                priority_score=score,
            )
        )

    candidates.sort(key=lambda row: (-row.priority_score, row.suggested_step_label, row.image_id))
    selected = candidates[: args.limit]
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
    }
    for row in selected:
        summary["label_distribution"][row.suggested_step_label] = summary["label_distribution"].get(row.suggested_step_label, 0) + 1
        summary["family_distribution"][row.suggested_boundary_family] = summary["family_distribution"].get(row.suggested_boundary_family, 0) + 1
    out_summary = ROOT / args.output_summary
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
