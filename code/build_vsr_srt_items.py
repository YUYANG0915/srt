#!/usr/bin/env python3
"""Build SRT spatial-layout items from VSR."""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset


FRAME_BY_RELATION = {
    "left of": "image_plane",
    "right of": "image_plane",
    "at the left side of": "image_plane",
    "at the right side of": "image_plane",
    "at the side of": "image_plane",
    "above": "image_plane",
    "below": "image_plane",
    "over": "image_plane",
    "across from": "image_plane",
    "opposite to": "image_plane",
    "in front of": "depth",
    "behind": "depth",
    "at the back of": "depth",
    "far away from": "depth",
    "away from": "depth",
    "facing": "orientation",
    "facing away from": "orientation",
    "parallel to": "orientation",
    "perpendicular to": "orientation",
    "inside": "containment",
    "contains": "containment",
    "in": "containment",
    "within": "containment",
    "into": "containment",
    "enclosed by": "containment",
    "surrounding": "containment",
    "consists of": "containment",
    "part of": "containment",
    "has as a part": "containment",
    "on": "contact_support",
    "on top of": "contact_support",
    "under": "contact_support",
    "beneath": "contact_support",
    "touching": "contact_support",
    "attached to": "contact_support",
    "covering": "contact_support",
    "beside": "contact_support",
    "next to": "contact_support",
    "adjacent to": "contact_support",
    "alongside": "contact_support",
    "by": "contact_support",
    "at the edge of": "contact_support",
}


def download(url: str, path: Path, timeout: int = 120, retries: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "vlm-underspec-pilot/1.0"})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                path.write_bytes(response.read())
                return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"Failed to download {url} after {retries} retries: {last_error}")


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cambridgeltl/vsr_zeroshot")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="data/vsr_srt_spatial_items.jsonl")
    parser.add_argument("--image-dir", default="data/vsr_images")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-relation-label", type=int, default=12)
    parser.add_argument("--download-images", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    ds = load_dataset(args.dataset, data_files={args.split: f"{args.split}.jsonl"}, split=args.split)
    rows = [dict(row) for row in ds]
    rng.shuffle(rows)

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        relation = clean_text(row.get("relation"))
        label = int(row.get("label"))
        if not relation:
            continue
        grouped[(relation, label)].append(row)

    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        rng.shuffle(bucket)
        selected.extend(bucket[: args.max_per_relation_label])
    rng.shuffle(selected)
    selected = selected[: args.limit]

    image_dir = Path(args.image_dir)
    out_rows = []
    for idx, row in enumerate(selected, start=1):
        image_name = clean_text(row["image"])
        image_path = image_dir / image_name
        if args.download_images:
            download(clean_text(row["image_link"]), image_path)
        expected = "true" if int(row["label"]) == 1 else "false"
        relation = clean_text(row.get("relation"))
        item = {
            "id": f"vsr_{args.split}_{idx:04d}_{Path(image_name).stem}",
            "image_id": Path(image_name).stem,
            "image_path": str(image_path.resolve()),
            "source": args.dataset,
            "task": "srt_spatial_layout",
            "underspecified_question": clean_text(row["caption"]),
            "statement": clean_text(row["caption"]),
            "caption": clean_text(row["caption"]),
            "label": int(row["label"]),
            "expected_answer": expected,
            "expected_policy": expected,
            "relation": relation,
            "expected_spatial_frame": FRAME_BY_RELATION.get(relation, "uncertain"),
            "subj": clean_text(row.get("subj")),
            "obj": clean_text(row.get("obj")),
            "image_link": clean_text(row.get("image_link")),
            "annotator_id": row.get("annotator_id"),
            "vote_true_validator_id": row.get("vote_true_validator_id"),
            "vote_false_validator_id": row.get("vote_false_validator_id"),
        }
        out_rows.append(item)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in out_rows), encoding="utf-8")
    print(f"Wrote {len(out_rows)} VSR SRT items to {output}")
    print(f"Image directory: {image_dir}")
    if args.download_images:
        print("Downloaded images for selected items.")
    else:
        print("Images were not downloaded; rerun with --download-images when ready.")


if __name__ == "__main__":
    main()
