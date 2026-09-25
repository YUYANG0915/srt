#!/usr/bin/env python3
"""Prepare a review sheet for real egocentric Task3 frames.

This script does not download licensed datasets. Put extracted real frames from
Ego4D, EPIC-KITCHENS, GoPro footage, or another allowed source into an image
folder, then use this script to create a raw jsonl plus CSV/HTML review files.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def find_images(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def make_id(prefix: str, index: int, path: Path) -> str:
    stem = "".join(ch.lower() if ch.isalnum() else "_" for ch in path.stem).strip("_")
    stem = "_".join(part for part in stem.split("_") if part)
    return f"{prefix}_{index:04d}_{stem[:40]}"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "image_path",
        "source_dataset",
        "source_video_id",
        "timestamp_sec",
        "keep",
        "split",
        "activity",
        "question",
        "visible_participant_count",
        "implied_camera_holder_count",
        "expected_total_participants",
        "expected_reasoning",
        "first_person_cues",
        "visible_participant_boxes",
        "viewpoint_cue_boxes",
        "issue_type",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_html(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for idx, row in enumerate(rows, start=1):
        image_path = Path(str(row["image_path"])).resolve()
        cards.append(
            f"""
<section class="card">
  <h2>{idx}. {h(row["id"])}</h2>
  <img src="{h(image_path)}" alt="{h(row["id"])}" />
  <table>
    <tr><th>keep</th><td>yes / no</td></tr>
    <tr><th>split</th><td>first_person_prior / third_person_control</td></tr>
    <tr><th>activity</th><td>skiing / cycling / cooking / ...</td></tr>
    <tr><th>question</th><td>How many people are [activity] in this scene?</td></tr>
    <tr><th>visible_participant_count</th><td>integer count of directly visible participants</td></tr>
    <tr><th>implied_camera_holder_count</th><td>1 if first-person cue implies wearer participates, else 0</td></tr>
    <tr><th>expected_total_participants</th><td>visible + implied if justified</td></tr>
    <tr><th>first_person_cues</th><td>hands / skis / handlebars / cooking hands / camera wearer viewpoint</td></tr>
    <tr><th>visible_participant_boxes</th><td>optional: x1,y1,x2,y2; x1,y1,x2,y2</td></tr>
    <tr><th>viewpoint_cue_boxes</th><td>optional: x1,y1,x2,y2; x1,y1,x2,y2</td></tr>
  </table>
</section>
"""
        )
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{h(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f7f7f4; color: #1f2937; }}
    h1 {{ font-size: 24px; margin-bottom: 16px; }}
    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 18px; }}
    h2 {{ margin: 0 0 10px; font-size: 17px; }}
    img {{ max-width: 760px; max-height: 460px; border: 1px solid #ccc; display: block; margin: 10px 0 14px; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 960px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ width: 240px; background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>{h(title)}</h1>
  {''.join(cards)}
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Folder containing real egocentric frames.")
    parser.add_argument("--output-image-dir", default="data/task3_real_egocentric_raw_images")
    parser.add_argument("--raw-output", default="data/task3_real_egocentric_raw_items.jsonl")
    parser.add_argument("--review-csv", default="review/task3_real_egocentric_review.csv")
    parser.add_argument("--review-html", default="review/task3_real_egocentric_review.html")
    parser.add_argument("--source-dataset", default="ego4d_or_epic_or_user")
    parser.add_argument("--id-prefix", default="task3real")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    images = find_images(input_dir)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise FileNotFoundError(f"No images found under {input_dir}")

    output_image_dir = Path(args.output_image_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, src in enumerate(images, start=1):
        item_id = make_id(args.id_prefix, idx, src)
        if args.copy_images:
            dst = output_image_dir / f"{item_id}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            image_path = dst.resolve()
        else:
            image_path = src.resolve()
        rows.append(
            {
                "id": item_id,
                "image_id": item_id,
                "image_path": str(image_path),
                "source_dataset": args.source_dataset,
                "source_video_id": "",
                "timestamp_sec": "",
                "keep": "",
                "split": "",
                "activity": "",
                "question": "",
                "visible_participant_count": "",
                "implied_camera_holder_count": "",
                "expected_total_participants": "",
                "expected_reasoning": "",
                "first_person_cues": "",
                "visible_participant_boxes": "",
                "viewpoint_cue_boxes": "",
                "issue_type": "",
                "notes": "",
            }
        )

    write_jsonl(Path(args.raw_output), rows)
    write_csv(Path(args.review_csv), rows)
    write_html(Path(args.review_html), rows, "Task3 Real Egocentric Review")
    print(f"Wrote {len(rows)} raw items to {args.raw_output}")
    print(f"Wrote review CSV to {args.review_csv}")
    print(f"Wrote review HTML to {args.review_html}")


if __name__ == "__main__":
    main()
