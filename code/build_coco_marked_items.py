#!/usr/bin/env python3
"""Create Sketchpad-style marked-image items from COCO annotations."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


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


def draw_marked_image(
    *,
    image_path: Path,
    annotations: list[dict[str, Any]],
    category_name: str,
    output_path: Path,
    min_area: float,
) -> list[dict[str, Any]]:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = safe_font(max(18, round(min(image.size) * 0.035)))
    label_font = safe_font(max(14, round(min(image.size) * 0.026)))

    candidates = []
    kept = [ann for ann in annotations if float(ann.get("area", 0.0)) >= min_area and not ann.get("iscrowd")]
    kept.sort(key=lambda ann: (ann["bbox"][1], ann["bbox"][0]))
    for idx, ann in enumerate(kept, start=1):
        x, y, w, h = [float(v) for v in ann["bbox"]]
        label = f"{category_name}_{idx}"
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
                "bbox_xywh": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "area": float(ann.get("area", 0.0)),
            }
        )

    footer = f"Sketchpad visual action: marked all visible '{category_name}' candidates ({len(candidates)})."
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
    parser.add_argument("--instances", required=True)
    parser.add_argument("--output-items", required=True)
    parser.add_argument("--output-image-dir", required=True)
    parser.add_argument("--min-area", type=float, default=1400.0)
    args = parser.parse_args()

    items = read_jsonl(Path(args.items))
    coco = load_json(Path(args.instances))
    category_by_id = {row["id"]: row["name"] for row in coco["categories"]}
    annotations_by_image_and_name: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        name = category_by_id.get(ann["category_id"])
        if name:
            annotations_by_image_and_name[(int(ann["image_id"]), name)].append(ann)

    output_image_dir = Path(args.output_image_dir)
    output_rows = []
    for item in items:
        image_id_int = int(str(item["image_id"]))
        target_object = str(item["target_object"])
        image_path = Path(item["image_path"])
        annotations = annotations_by_image_and_name[(image_id_int, target_object)]
        marked_path = output_image_dir / f"{item['id']}_marked.jpg"
        marked_candidates = draw_marked_image(
            image_path=image_path,
            annotations=annotations,
            category_name=target_object,
            output_path=marked_path,
            min_area=args.min_area,
        )
        row = dict(item)
        row["original_image_path"] = item["image_path"]
        row["image_path"] = str(marked_path.resolve())
        row["sketchpad_action"] = f"mark_all_candidate_referents:{target_object}"
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
