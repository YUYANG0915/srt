#!/usr/bin/env python3
"""Download only the Visual Genome images needed by GQA item files."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


URL_TEMPLATES = [
    "https://cs.stanford.edu/people/rak248/VG_100K/{image_id}.jpg",
    "https://cs.stanford.edu/people/rak248/VG_100K_2/{image_id}.jpg",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def download_one(image_id: str, output_dir: Path, timeout: int) -> Path:
    output = output_dir / f"{image_id}.jpg"
    if output.exists() and output.stat().st_size > 0:
        return output
    last_error: Exception | None = None
    for template in URL_TEMPLATES:
        url = template.format(image_id=image_id)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "vlm-underspec-pilot/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
            if data:
                output.write_bytes(data)
                return output
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 404:
                continue
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Could not download image_id={image_id}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", nargs="+", required=True)
    parser.add_argument("--output-items", nargs="+", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    if len(args.items) != len(args.output_items):
        raise ValueError("--items and --output-items must have the same length")

    output_dir = Path(args.image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows_by_path = [(Path(path), read_jsonl(Path(path))) for path in args.items]
    image_ids = sorted({str(row["image_id"]) for _, rows in all_rows_by_path for row in rows})
    downloaded: dict[str, Path] = {}
    for idx, image_id in enumerate(image_ids, start=1):
        path = download_one(image_id, output_dir, args.timeout)
        downloaded[image_id] = path.resolve()
        print(f"[{idx}/{len(image_ids)}] {image_id} -> {path}")

    for output_path, (_, rows) in zip(args.output_items, all_rows_by_path, strict=True):
        updated = []
        for row in rows:
            new_row = dict(row)
            new_row["image_path"] = str(downloaded[str(row["image_id"])])
            updated.append(new_row)
        write_jsonl(Path(output_path), updated)
        print(f"Wrote {len(updated)} rows to {output_path}")


if __name__ == "__main__":
    main()
