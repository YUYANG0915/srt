#!/usr/bin/env python3
"""Build a small real-image ambiguous-object pilot from COCO val2017."""

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
        "original": "What is the most prominent person doing?",
        "answer": "needs_manual_review",
        "target": "person",
    },
    "car": {
        "question": "What color is the car?",
        "original": "What color is the most central car?",
        "answer": "needs_manual_review",
        "target": "car",
    },
    "motorcycle": {
        "question": "What color is the motorcycle?",
        "original": "What color is the most prominent motorcycle?",
        "answer": "needs_manual_review",
        "target": "motorcycle",
    },
    "bicycle": {
        "question": "What color is the bicycle?",
        "original": "What color is the most prominent bicycle?",
        "answer": "needs_manual_review",
        "target": "bicycle",
    },
    "bus": {
        "question": "What color is the bus?",
        "original": "What color is the most central bus?",
        "answer": "needs_manual_review",
        "target": "bus",
    },
    "truck": {
        "question": "What color is the truck?",
        "original": "What color is the most prominent truck?",
        "answer": "needs_manual_review",
        "target": "truck",
    },
    "bird": {
        "question": "What color is the bird?",
        "original": "What color is the bird on the right?",
        "answer": "needs_manual_review",
        "target": "bird",
    },
}

PREFERRED = ["person", "car", "motorcycle", "bicycle", "bus", "truck", "bird"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def download_image(image_id: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{image_id:012d}.jpg"
    path = out_dir / file_name
    if path.exists():
        return path
    url = f"http://images.cocodataset.org/val2017/{file_name}"
    urllib.request.urlretrieve(url, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-area", type=float, default=1400.0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data = load_json(Path(args.instances))
    categories = {row["id"]: row["name"] for row in data["categories"]}
    images = {row["id"]: row for row in data["images"]}

    by_image = defaultdict(list)
    for ann in data["annotations"]:
        name = categories.get(ann["category_id"])
        if name not in TEMPLATES:
            continue
        if ann.get("iscrowd"):
            continue
        if float(ann.get("area", 0.0)) < args.min_area:
            continue
        by_image[ann["image_id"]].append(name)

    buckets = defaultdict(list)
    for image_id, names in by_image.items():
        counts = Counter(names)
        for name in PREFERRED:
            if counts[name] >= 2:
                buckets[name].append((image_id, counts[name], len(names)))
                break

    selected = []
    per_category_cap = max(3, args.limit // 5)
    for name in PREFERRED:
        candidates = buckets[name]
        rng.shuffle(candidates)
        for image_id, count, total in candidates[:per_category_cap]:
            if len(selected) >= args.limit:
                break
            selected.append((name, image_id, count, total))
        if len(selected) >= args.limit:
            break

    if len(selected) < args.limit:
        leftovers = []
        used = {image_id for _, image_id, _, _ in selected}
        for name in PREFERRED:
            leftovers.extend((name, *row) for row in buckets[name] if row[0] not in used)
        rng.shuffle(leftovers)
        selected.extend(leftovers[: args.limit - len(selected)])

    image_dir = Path(args.image_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, image_id, count, total in selected[: args.limit]:
        path = download_image(image_id, image_dir).resolve()
        template = TEMPLATES[name]
        rows.append(
            {
                "id": f"coco_{name}_{image_id}",
                "image_id": f"{image_id:012d}",
                "image_path": str(path),
                "source": "coco_val2017_auto",
                "original_question": template["original"],
                "underspecified_question": template["question"],
                "answer": template["answer"],
                "target_object": template["target"],
                "relation_phrase": "removed_or_unspecified",
                "candidate_count_scene_graph": count,
                "total_preferred_objects_in_image": total,
                "expected_policy": "clarify",
                "manual_status": "auto_needs_review",
            }
        )

    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} items to {output}")


if __name__ == "__main__":
    main()

