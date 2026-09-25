#!/usr/bin/env python3
"""Build clean/oracle JSONL files from the reviewed procedural boundary microset CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1", "keep"}


def read_review(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_item(row: dict[str, str]) -> dict[str, object]:
    question = row.get("question", "").strip() or "What step of the process is happening in this scene?"
    gold_step = row.get("gold_step_label", "").strip()
    image_path = str(Path(row["image_path"]).expanduser().resolve())
    boundary_family = row.get("boundary_family", "").strip() or "other"
    return {
        "id": row["id"],
        "image_id": row.get("source_image_id", "").strip() or row["id"],
        "source": "procedural_boundary_microset_reviewed",
        "source_dataset": row.get("source_dataset", "").strip() or "manual_boundary_microset",
        "task": "task3_procedural_boundary_microset",
        "scenario_domain": "cooking",
        "activity_name": "cooking_or_food_preparation",
        "underspecified_question": question,
        "original_question": question,
        "question": question,
        "gold_step_label": gold_step,
        "expected_answer": gold_step,
        "answer": gold_step,
        "expected_policy": "answer",
        "boundary_family": boundary_family,
        "boundary_family_hint": boundary_family,
        "neighbor_label": row.get("neighbor_label", "").strip(),
        "why_not_neighbor_label": row.get("why_not_neighbor_label", "").strip(),
        "visible_actor": row.get("visible_actor", "").strip(),
        "visible_tool": row.get("visible_tool", "").strip(),
        "visible_object": row.get("visible_object", "").strip(),
        "visible_state_cues": row.get("visible_state_cues", "").strip(),
        "issue_type": row.get("issue_type", "").strip(),
        "notes": row.get("notes", "").strip(),
        "image_path": image_path,
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review-csv",
        default=str(ROOT / "review" / "task3_procedural_boundary_microset_review.csv"),
    )
    parser.add_argument(
        "--output-clean",
        default=str(ROOT / "data" / "task3_procedural_boundary_microset_clean_items.jsonl"),
    )
    parser.add_argument(
        "--output-oracle",
        default=str(ROOT / "data" / "task3_procedural_boundary_microset_clean_items_oracle_family.jsonl"),
    )
    args = parser.parse_args()

    rows = [
        build_item(row)
        for row in read_review(Path(args.review_csv))
        if truthy(row.get("keep", "")) and row.get("gold_step_label", "").strip()
    ]
    if not rows:
        raise ValueError("No kept rows with gold_step_label found in reviewed microset CSV.")

    write_jsonl(Path(args.output_clean), rows)
    write_jsonl(Path(args.output_oracle), rows)
    print(f"Wrote {len(rows)} rows to {args.output_clean}")
    print(f"Wrote {len(rows)} rows to {args.output_oracle}")


if __name__ == "__main__":
    main()
