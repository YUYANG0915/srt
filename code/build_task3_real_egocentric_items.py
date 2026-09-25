#!/usr/bin/env python3
"""Build Task3 real egocentric clean/marked items from a reviewed CSV."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "visible_participant": (230, 57, 70),
    "viewpoint_cue": (29, 112, 184),
}


def truthy(value: str) -> bool:
    return value.strip().lower() in {"yes", "y", "true", "1", "keep"}


def safe_int(value: str, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def safe_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def parse_boxes(text: str) -> list[tuple[float, float, float, float]]:
    """Parse semicolon-separated xyxy boxes: x1,y1,x2,y2; x1,y1,x2,y2."""
    boxes = []
    for chunk in str(text or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [part.strip() for part in chunk.replace("|", ",").split(",")]
        if len(parts) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(part) for part in parts]
        except ValueError:
            continue
        boxes.append((x1, y1, x2, y2))
    return boxes


def normalize_question(row: dict[str, str]) -> str:
    question = row.get("question", "").strip()
    if question:
        return question
    activity = row.get("activity", "").strip() or "doing this activity"
    return f"How many people are {activity} in this scene?"


def read_review(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def draw_marked_image(row: dict[str, str], output_path: Path) -> list[dict[str, Any]]:
    src = Path(row["image_path"]).expanduser()
    image = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = safe_font(max(16, round(min(image.size) * 0.03)), bold=True)
    small = safe_font(max(13, round(min(image.size) * 0.022)), bold=True)

    marked: list[dict[str, Any]] = []
    specs = [
        ("visible_participant", "visible_participant_boxes"),
        ("viewpoint_cue", "viewpoint_cue_boxes"),
    ]
    for kind, column in specs:
        boxes = parse_boxes(row.get(column, ""))
        color = COLORS[kind]
        for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
            mark_id = f"{kind}_{idx}"
            draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=max(4, round(min(image.size) * 0.006)))
            label_bbox = draw.textbbox((0, 0), mark_id, font=font)
            label_w = label_bbox[2] - label_bbox[0]
            label_h = label_bbox[3] - label_bbox[1]
            label_y = max(0, y1 - label_h - 10)
            draw.rectangle([x1, label_y, x1 + label_w + 12, label_y + label_h + 8], fill=color + (230,))
            draw.text((x1 + 6, label_y + 4), mark_id, fill=(255, 255, 255, 255), font=small)
            marked.append(
                {
                    "mark_id": mark_id,
                    "kind": kind,
                    "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                }
            )

    footer = "Sketchpad visual action: marked visible participants and first-person viewpoint cues."
    footer_bbox = draw.textbbox((0, 0), footer, font=small)
    footer_h = footer_bbox[3] - footer_bbox[1]
    draw.rectangle([0, image.size[1] - footer_h - 12, image.size[0], image.size[1]], fill=(0, 0, 0, 150))
    draw.text((8, image.size[1] - footer_h - 7), footer, fill=(255, 255, 255, 255), font=small)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return marked


def build_row(row: dict[str, str], *, marked: bool, marked_path: Path | None, marked_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    split = row.get("split", "").strip()
    hidden_expected = split == "first_person_prior" or safe_int(row.get("implied_camera_holder_count", "0")) > 0
    visible_count = safe_int(row.get("visible_participant_count", "0"))
    implied_count = safe_int(row.get("implied_camera_holder_count", "0"))
    expected_total = safe_int(row.get("expected_total_participants", ""), visible_count + implied_count)
    question = normalize_question(row)
    image_path = Path(row["image_path"]).expanduser().resolve()

    item = {
        "id": row["id"],
        "image_id": row.get("image_id", row["id"]),
        "source": "real_egocentric_reviewed",
        "source_dataset": row.get("source_dataset", ""),
        "source_video_id": row.get("source_video_id", ""),
        "timestamp_sec": row.get("timestamp_sec", ""),
        "task": "task3_real_egocentric_prior",
        "split": split,
        "underspecified_question": question,
        "original_question": question,
        "activity": row.get("activity", "").strip(),
        "visible_participant_count": visible_count,
        "implied_camera_holder_count": implied_count,
        "expected_total_participants": expected_total,
        "expected_policy": "answer",
        "expected_reasoning": row.get("expected_reasoning", "").strip()
        or ("include_hidden_camera_holder" if hidden_expected else "count_visible_only"),
        "hidden_camera_holder_expected": hidden_expected,
        "first_person_cues": row.get("first_person_cues", "").strip(),
        "main_error": "visible_only_count" if hidden_expected else "over_infer_hidden_camera_holder",
        "answer": str(expected_total),
        "manual_status": "reviewed",
        "issue_type": row.get("issue_type", "").strip(),
        "notes": row.get("notes", "").strip(),
        "image_path": str(image_path),
    }
    if marked:
        item["original_image_path"] = str(image_path)
        item["image_path"] = str(marked_path.resolve()) if marked_path else str(image_path)
        item["sketchpad_action"] = "mark_visible_participants_and_viewpoint_cues"
        item["marked_candidate_count"] = len(marked_candidates)
        item["marked_candidates"] = marked_candidates
    return item


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output-clean", default="data/task3_real_egocentric_clean_items.jsonl")
    parser.add_argument("--output-marked", default="data/task3_real_egocentric_marked_items.jsonl")
    parser.add_argument("--marked-image-dir", default="data/task3_real_egocentric_marked_images")
    parser.add_argument("--copy-unboxed-marked", action="store_true", help="Copy image with footer even if no boxes are provided.")
    args = parser.parse_args()

    rows = [row for row in read_review(Path(args.review_csv)) if truthy(row.get("keep", ""))]
    if not rows:
        raise ValueError("No kept rows found. Fill keep=yes in the review CSV first.")

    marked_dir = Path(args.marked_image_dir)
    clean_items = []
    marked_items = []
    for row in rows:
        clean_items.append(build_row(row, marked=False, marked_path=None, marked_candidates=[]))
        marked_path = marked_dir / f"{row['id']}_marked.jpg"
        has_boxes = bool(parse_boxes(row.get("visible_participant_boxes", "")) or parse_boxes(row.get("viewpoint_cue_boxes", "")))
        if has_boxes or args.copy_unboxed_marked:
            marked_candidates = draw_marked_image(row, marked_path)
        else:
            marked_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(row["image_path"]).expanduser(), marked_path)
            marked_candidates = []
        marked_items.append(build_row(row, marked=True, marked_path=marked_path, marked_candidates=marked_candidates))

    write_jsonl(Path(args.output_clean), clean_items)
    write_jsonl(Path(args.output_marked), marked_items)
    print(f"Wrote {len(clean_items)} clean items to {args.output_clean}")
    print(f"Wrote {len(marked_items)} marked items to {args.output_marked}")
    print(f"Wrote marked images to {marked_dir}")


if __name__ == "__main__":
    main()
