#!/usr/bin/env python3
"""Build a small real-image unique-referent pilot from COCO val2017."""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TEMPLATES = {
    "person": {
        "question": "What is the person doing?",
        "original": "What is the only visible person doing?",
        "target": "person",
    },
    "car": {
        "question": "What color is the car?",
        "original": "What color is the only visible car?",
        "target": "car",
    },
    "motorcycle": {
        "question": "What color is the motorcycle?",
        "original": "What color is the only visible motorcycle?",
        "target": "motorcycle",
    },
    "bicycle": {
        "question": "What color is the bicycle?",
        "original": "What color is the only visible bicycle?",
        "target": "bicycle",
    },
    "bus": {
        "question": "What color is the bus?",
        "original": "What color is the only visible bus?",
        "target": "bus",
    },
}

TARGETS = ["person", "car", "motorcycle", "bicycle", "bus"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def download_image(image_id: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{image_id:012d}.jpg"
    path = out_dir / file_name
    if path.exists():
        return path
    urllib.request.urlretrieve(f"http://images.cocodataset.org/val2017/{file_name}", path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-category", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--min-area", type=float, default=2500.0)
    parser.add_argument("--all-min-area", type=float, default=300.0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data = load_json(Path(args.instances))
    categories = {row["id"]: row["name"] for row in data["categories"]}

    by_image = defaultdict(list)
    all_by_image = defaultdict(list)
    for ann in data["annotations"]:
        name = categories.get(ann["category_id"])
        if name not in TARGETS:
            continue
        if ann.get("iscrowd"):
            continue
        if float(ann.get("area", 0.0)) >= args.all_min_area:
            all_by_image[ann["image_id"]].append(name)
        if float(ann.get("area", 0.0)) < args.min_area:
            continue
        by_image[ann["image_id"]].append(name)

    buckets = defaultdict(list)
    for image_id, names in by_image.items():
        counts = Counter(names)
        all_counts = Counter(all_by_image.get(image_id, []))
        for target in TARGETS:
            if counts[target] == 1 and all_counts[target] == 1:
                buckets[target].append((image_id, len(names)))

    image_dir = Path(args.image_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    used_images = set()
    for target in TARGETS:
        candidates = [row for row in buckets[target] if row[0] not in used_images]
        rng.shuffle(candidates)
        for image_id, total in candidates[: args.per_category]:
            used_images.add(image_id)
            template = TEMPLATES[target]
            path = download_image(image_id, image_dir).resolve()
            rows.append(
                {
                    "id": f"coco_unique_{target}_{image_id}",
                    "image_id": f"{image_id:012d}",
                    "image_path": str(path),
                    "source": "coco_val2017_unique_auto",
                    "original_question": template["original"],
                    "underspecified_question": template["question"],
                    "answer": "needs_manual_review",
                    "target_object": template["target"],
                    "relation_phrase": "unique_visible_referent",
                    "candidate_count_scene_graph": 1,
                    "total_preferred_objects_in_image": total,
                    "expected_policy": "answer",
                    "manual_status": "auto_needs_review",
                }
            )

    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} items to {output}")


if __name__ == "__main__":
    main()
