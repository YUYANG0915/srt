#!/usr/bin/env python3
"""Build procedural cooking clean/marked items from a reviewed CSV."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "actor": (230, 57, 70),
    "tool": (29, 112, 184),
    "object": (46, 160, 67),
    "state": (132, 94, 194),
}


def truthy(value: str) -> bool:
    return value.strip().lower() in {"yes", "y", "true", "1", "keep"}


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
        ("actor", "actor_box"),
        ("tool", "tool_box"),
        ("object", "object_box"),
        ("state", "state_box"),
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

    footer = "Sketchpad visual action: marked actor, tool, object, and state cues."
    footer_bbox = draw.textbbox((0, 0), footer, font=small)
    footer_h = footer_bbox[3] - footer_bbox[1]
    draw.rectangle([0, image.size[1] - footer_h - 12, image.size[0], image.size[1]], fill=(0, 0, 0, 150))
    draw.text((8, image.size[1] - footer_h - 7), footer, fill=(255, 255, 255, 255), font=small)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return marked


def build_row(row: dict[str, str], *, marked: bool, marked_path: Path | None, marked_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    question = row.get("question", "").strip() or "What step of the process is happening in this scene?"
    gold_step = row.get("gold_step_label", "").strip()
    image_path = Path(row["image_path"]).expanduser().resolve()
    item = {
        "id": row["id"],
        "image_id": row.get("source_image_id", row["id"]),
        "source": "procedural_cooking_reviewed",
        "source_dataset": row.get("source_dataset", "manual_bootstrap"),
        "task": "task3_procedural_cooking",
        "scenario_domain": row.get("scenario_domain", "cooking").strip() or "cooking",
        "activity_name": row.get("activity_name", "").strip(),
        "underspecified_question": question,
        "original_question": question,
        "question": question,
        "gold_step_label": gold_step,
        "expected_answer": gold_step,
        "answer": gold_step,
        "visible_actor": row.get("visible_actor", "").strip(),
        "visible_tool": row.get("visible_tool", "").strip(),
        "visible_object": row.get("visible_object", "").strip(),
        "visible_state_cues": row.get("visible_state_cues", "").strip(),
        "expected_policy": "answer",
        "issue_type": row.get("issue_type", "").strip(),
        "notes": row.get("notes", "").strip(),
        "image_path": str(image_path),
    }
    if marked:
        item["original_image_path"] = str(image_path)
        item["image_path"] = str(marked_path.resolve()) if marked_path else str(image_path)
        item["sketchpad_action"] = "mark_actor_tool_object_state"
        item["marked_candidate_count"] = len(marked_candidates)
        item["marked_candidates"] = marked_candidates
    return item


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output-clean", default="data/task3_procedural_cooking_clean_items.jsonl")
    parser.add_argument("--output-marked", default="data/task3_procedural_cooking_marked_items.jsonl")
    parser.add_argument("--marked-image-dir", default="data/task3_procedural_cooking_marked_images")
    parser.add_argument("--copy-unboxed-marked", action="store_true")
    args = parser.parse_args()

    rows = [
        row
        for row in read_review(Path(args.review_csv))
        if truthy(row.get("keep", "")) and row.get("gold_step_label", "").strip()
    ]
    if not rows:
        raise ValueError("No kept rows with gold_step_label found. Fill keep=yes and gold_step_label first.")

    marked_dir = Path(args.marked_image_dir)
    clean_items = []
    marked_items = []
    for row in rows:
        clean_items.append(build_row(row, marked=False, marked_path=None, marked_candidates=[]))
        marked_path = marked_dir / f"{row['id']}_marked.jpg"
        has_boxes = any(parse_boxes(row.get(col, "")) for col in ["actor_box", "tool_box", "object_box", "state_box"])
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
