#!/usr/bin/env python3
"""Create Sketchpad-style marked-image items from GQA scene-graph candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


COLORS = [
    (230, 57, 70),
    (29, 112, 184),
    (42, 157, 143),
    (245, 166, 35),
    (131, 56, 236),
    (255, 99, 146),
    (0, 150, 136),
    (251, 86, 7),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def safe_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def image_path_for(item: dict[str, Any], image_dir: Path) -> Path:
    explicit = str(item.get("image_path") or "")
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return path
    image_id = str(item["image_id"])
    for suffix in (".jpg", ".jpeg", ".png"):
        path = image_dir / f"{image_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No image found for GQA image_id={image_id} under {image_dir}")


def draw_marked_image(item: dict[str, Any], image_dir: Path, output_path: Path) -> list[dict[str, Any]]:
    image = Image.open(image_path_for(item, image_dir)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = safe_font(max(18, round(min(image.size) * 0.035)))
    label_font = safe_font(max(14, round(min(image.size) * 0.026)))

    target_object = str(item.get("target_object", "object"))
    candidates = []
    for idx, obj in enumerate(item.get("candidate_objects_scene_graph", []), start=1):
        x = float(obj.get("x", 0))
        y = float(obj.get("y", 0))
        w = float(obj.get("w", 0))
        h = float(obj.get("h", 0))
        if w <= 0 or h <= 0:
            continue
        label = f"{target_object}_{idx}"
        color = COLORS[(idx - 1) % len(COLORS)]
        width = max(3, round(min(image.size) * 0.006))
        for offset in range(width):
            draw.rectangle([x - offset, y - offset, x + w + offset, y + h + offset], outline=color + (255,))
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        pad = max(4, width)
        label_x = max(0, min(x, image.size[0] - text_w - 2 * pad))
        label_y = max(0, y - text_h - 2 * pad)
        if label_y < 2:
            label_y = min(image.size[1] - text_h - 2 * pad, y + 2)
        draw.rectangle(
            [label_x, label_y, label_x + text_w + 2 * pad, label_y + text_h + 2 * pad],
            fill=color + (230,),
        )
        draw.text((label_x + pad, label_y + pad), label, fill=(255, 255, 255, 255), font=font)
        candidates.append(
            {
                "mark_id": label,
                "object_id": obj.get("object_id"),
                "name": obj.get("name"),
                "attributes": obj.get("attributes", []),
                "bbox_xywh": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
            }
        )

    footer = f"Sketchpad visual action: marked all GQA '{target_object}' candidates ({len(candidates)})."
    footer_bbox = draw.textbbox((0, 0), footer, font=label_font)
    footer_h = footer_bbox[3] - footer_bbox[1]
    draw.rectangle([0, image.size[1] - footer_h - 12, image.size[0], image.size[1]], fill=(0, 0, 0, 145))
    draw.text((8, image.size[1] - footer_h - 7), footer, fill=(255, 255, 255, 255), font=label_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-items", required=True)
    parser.add_argument("--output-image-dir", required=True)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.items))
    image_dir = Path(args.image_dir)
    output_image_dir = Path(args.output_image_dir)
    output_rows = []
    for item in rows:
        marked_path = output_image_dir / f"{item['id']}_marked.jpg"
        marked_candidates = draw_marked_image(item, image_dir, marked_path)
        row = dict(item)
        row["original_image_path"] = item.get("image_path", "")
        row["image_path"] = str(marked_path.resolve())
        row["sketchpad_action"] = f"mark_all_gqa_candidate_referents:{item.get('target_object', '')}"
        row["marked_candidate_count"] = len(marked_candidates)
        row["marked_candidates"] = marked_candidates
        output_rows.append(row)

    output_items = Path(args.output_items)
    output_items.parent.mkdir(parents=True, exist_ok=True)
    output_items.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    print(f"Wrote {len(output_rows)} marked items to {output_items}")
    print(f"Wrote marked images to {output_image_dir}")


if __name__ == "__main__":
    main()
