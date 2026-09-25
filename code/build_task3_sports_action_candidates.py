#!/usr/bin/env python3
"""Build a review pack for Phase B S3 sports-action process priors."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ANN_PATH = ROOT / "data" / "coco_annotations" / "instances_val2017.json"
OUT_CSV = ROOT / "review" / "task3_sports_action_candidates_v1_review.csv"
OUT_HTML = ROOT / "review" / "task3_sports_action_candidates_v1_review.html"
OUT_SUMMARY = ROOT / "runs" / "task3_sports_action_candidates_v1_summary_2026-06-26.json"

IMAGE_DIRS = [
    ROOT / "data" / "coco_val2017_task1_raw500_images",
    ROOT / "data" / "coco_val2017_task2_raw500_images",
    ROOT / "data" / "coco_val2017_pilot_images",
    ROOT / "data" / "coco_val2017_unique_strict_images",
    ROOT / "data" / "coco_val2017_unique_images",
]

SPORT_EQUIPMENT = {
    "sports ball",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "skis",
    "snowboard",
    "tennis racket",
    "frisbee",
    "kite",
    "bicycle",
}

SPORT_HINTS = {
    "sports ball": "ball_sport",
    "baseball bat": "baseball",
    "baseball glove": "baseball",
    "skateboard": "skateboarding",
    "surfboard": "surfing",
    "skis": "skiing",
    "snowboard": "snowboarding",
    "tennis racket": "tennis",
    "frisbee": "frisbee",
    "kite": "kite_activity",
    "bicycle": "cycling",
}


def load_coco() -> dict[str, Any]:
    return json.loads(ANN_PATH.read_text(encoding="utf-8"))


def image_path_for(image_id: int) -> Path | None:
    filename = f"{image_id:012d}.jpg"
    for directory in IMAGE_DIRS:
        path = directory / filename
        if path.exists():
            return path.resolve()
    return None


def bbox_area(box: list[float]) -> float:
    return max(0.0, float(box[2])) * max(0.0, float(box[3]))


def dominant_equipment(equipment: set[str]) -> str:
    priority = [
        "tennis racket",
        "baseball bat",
        "baseball glove",
        "sports ball",
        "skateboard",
        "surfboard",
        "skis",
        "snowboard",
        "frisbee",
        "kite",
        "bicycle",
    ]
    for name in priority:
        if name in equipment:
            return name
    return sorted(equipment)[0]


def build_candidates(limit_per_sport: int = 6, max_total: int = 45) -> list[dict[str, str]]:
    coco = load_coco()
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    images = {image["id"]: image for image in coco["images"]}

    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image_id, anns in anns_by_image.items():
        image_path = image_path_for(image_id)
        if image_path is None:
            continue

        cat_names = [categories[ann["category_id"]] for ann in anns]
        cat_set = set(cat_names)
        equipment = cat_set & SPORT_EQUIPMENT
        if "person" not in cat_set or not equipment:
            continue

        person_area = sum(bbox_area(ann["bbox"]) for ann in anns if categories[ann["category_id"]] == "person")
        equipment_area = sum(bbox_area(ann["bbox"]) for ann in anns if categories[ann["category_id"]] in equipment)
        image_info = images[image_id]
        image_area = float(image_info["width"] * image_info["height"])
        score = (person_area + equipment_area) / max(1.0, image_area)
        main_equipment = dominant_equipment(equipment)
        groups[main_equipment].append(
            {
                "image_id": image_id,
                "image_path": str(image_path),
                "categories": sorted(cat_set),
                "sports_equipment": sorted(equipment),
                "main_equipment": main_equipment,
                "score": score,
            }
        )

    for rows in groups.values():
        rows.sort(key=lambda row: row["score"], reverse=True)

    selected: list[dict[str, Any]] = []
    # Round-robin gives us better scenario diversity than taking the top overall images.
    equipment_order = [
        "sports ball",
        "tennis racket",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "skis",
        "snowboard",
        "frisbee",
        "kite",
        "bicycle",
    ]
    for _ in range(limit_per_sport):
        for equipment in equipment_order:
            rows = groups.get(equipment, [])
            if rows:
                selected.append(rows.pop(0))
                if len(selected) >= max_total:
                    break
        if len(selected) >= max_total:
            break

    output_rows: list[dict[str, str]] = []
    for idx, row in enumerate(selected, start=1):
        sport_hint = SPORT_HINTS.get(row["main_equipment"], row["main_equipment"].replace(" ", "_"))
        output_rows.append(
            {
                "id": f"sports_v1_{idx:04d}_{row['image_id']}",
                "image_path": row["image_path"],
                "keep": "",
                "sport_hint": sport_hint,
                "question": "What phase of the sports action is happening in this scene?",
                "gold_action_phase": "",
                "phase_boundary_family": "",
                "neighbor_phase": "",
                "why_not_neighbor_phase": "",
                "visible_actor_pose": "",
                "visible_equipment": ", ".join(row["sports_equipment"]),
                "visible_ball_or_target": "",
                "visible_motion_cues": "",
                "issue_type": "",
                "notes": "",
                "source_dataset": "coco_val2017_local_subset",
                "source_image_id": str(row["image_id"]),
                "coco_sports_equipment": ", ".join(row["sports_equipment"]),
                "coco_categories": ", ".join(row["categories"]),
                "selection_score": f"{row['score']:.4f}",
            }
        )
    return output_rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "image_path",
        "keep",
        "sport_hint",
        "question",
        "gold_action_phase",
        "phase_boundary_family",
        "neighbor_phase",
        "why_not_neighbor_phase",
        "visible_actor_pose",
        "visible_equipment",
        "visible_ball_or_target",
        "visible_motion_cues",
        "issue_type",
        "notes",
        "source_dataset",
        "source_image_id",
        "coco_sports_equipment",
        "coco_categories",
        "selection_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_html(rows: list[dict[str, str]], path: Path) -> None:
    cards = []
    for row in rows:
        cards.append(
            f"""
            <div class="card">
              <img src="file://{html.escape(row['image_path'])}" alt="{html.escape(row['id'])}">
              <div class="meta">
                <p><b>id:</b> {html.escape(row['id'])}</p>
                <p><b>sport_hint:</b> {html.escape(row['sport_hint'])}</p>
                <p><b>equipment:</b> {html.escape(row['coco_sports_equipment'])}</p>
                <p><b>categories:</b> {html.escape(row['coco_categories'])}</p>
                <p><b>score:</b> {html.escape(row['selection_score'])}</p>
                <p class="path">{html.escape(row['image_path'])}</p>
              </div>
            </div>
            """
        )
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Task3 Sports Action Candidates V1</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 18px; }}
    .card {{ border: 1px solid #bbb; border-radius: 8px; padding: 12px; }}
    img {{ width: 100%; height: 270px; object-fit: contain; background: #f7f7f7; }}
    .meta {{ font-size: 14px; line-height: 1.35; }}
    .path {{ color: #666; font-size: 12px; word-break: break-all; }}
  </style>
</head>
<body>
  <h1>Task3 Sports Action Candidates V1</h1>
  <p>Total candidates: {len(rows)}</p>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-sport", type=int, default=6)
    parser.add_argument("--max-total", type=int, default=45)
    parser.add_argument("--output-csv", default=str(OUT_CSV))
    parser.add_argument("--output-html", default=str(OUT_HTML))
    parser.add_argument("--output-summary", default=str(OUT_SUMMARY))
    args = parser.parse_args()

    out_csv = Path(args.output_csv)
    out_html = Path(args.output_html)
    out_summary = Path(args.output_summary)

    rows = build_candidates(limit_per_sport=args.limit_per_sport, max_total=args.max_total)
    write_csv(rows, out_csv)
    write_html(rows, out_html)
    summary = {
        "candidate_count": len(rows),
        "review_csv": str(out_csv),
        "review_html": str(out_html),
        "limit_per_sport": args.limit_per_sport,
        "max_total": args.max_total,
        "sports_distribution": {},
    }
    for row in rows:
        summary["sports_distribution"][row["sport_hint"]] = summary["sports_distribution"].get(row["sport_hint"], 0) + 1
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_html}")
    print(f"Wrote {out_summary}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
