#!/usr/bin/env python3
"""Build Task3 procedural eval v2 by merging eval v1 with reviewed expansion items."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REVIEW_DIR = ROOT / "review"
RUNS_DIR = ROOT / "runs"


def truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1", "keep"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_item(row: dict[str, str]) -> dict:
    question = row.get("question", "").strip() or "What step of the process is happening in this scene?"
    gold = row.get("gold_step_label", "").strip()
    family = row.get("boundary_family", "").strip() or "other"
    return {
        "id": row.get("id", ""),
        "image_id": row.get("source_image_id", "").strip() or row.get("id", ""),
        "source": "procedural_expansion_pack_v1_reviewed",
        "source_dataset": row.get("source_dataset", "").strip() or "coco_val2017_local_subset",
        "task": "task3_procedural_eval_v2",
        "scenario_domain": row.get("scenario_domain", "").strip() or "cooking",
        "activity_name": row.get("activity_name", "").strip() or "cooking_or_food_preparation",
        "underspecified_question": question,
        "original_question": question,
        "question": question,
        "gold_step_label": gold,
        "expected_answer": gold,
        "answer": gold,
        "expected_policy": "answer",
        "boundary_family": family,
        "boundary_family_hint": family,
        "neighbor_label": row.get("neighbor_label", "").strip(),
        "why_not_neighbor_label": row.get("why_not_neighbor_label", "").strip(),
        "visible_actor": row.get("visible_actor", "").strip(),
        "visible_tool": row.get("visible_tool", "").strip(),
        "visible_object": row.get("visible_object", "").strip(),
        "visible_state_cues": row.get("visible_state_cues", "").strip(),
        "issue_type": row.get("issue_type", "").strip(),
        "notes": row.get("notes", "").strip(),
        "image_path": str(Path(row["image_path"]).expanduser().resolve()),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    base_rows = load_jsonl(DATA_DIR / "task3_procedural_eval_v1.jsonl")
    reviewed_rows = read_csv(REVIEW_DIR / "task3_procedural_expansion_pack_v1_review.csv")

    valid_expansion: list[dict] = []
    incomplete_rows: list[dict[str, str]] = []
    rejected_rows: list[dict[str, str]] = []

    for row in reviewed_rows:
        keep = truthy(row.get("keep", ""))
        gold = row.get("gold_step_label", "").strip()
        family = row.get("boundary_family", "").strip()
        if keep and gold and family:
            valid_expansion.append(build_item(row))
        elif keep:
            incomplete_rows.append(row)
        else:
            rejected_rows.append(row)

    seen_image_ids = {str(row.get("image_id", "")) for row in base_rows}
    merged_rows = list(base_rows)
    added_rows = []
    for row in valid_expansion:
        if str(row.get("image_id", "")) in seen_image_ids:
            continue
        merged_rows.append(row)
        added_rows.append(row)
        seen_image_ids.add(str(row.get("image_id", "")))

    output_jsonl = DATA_DIR / "task3_procedural_eval_v2.jsonl"
    write_jsonl(output_jsonl, merged_rows)

    summary = {
        "base_eval_v1_size": len(base_rows),
        "valid_expansion_size": len(valid_expansion),
        "added_to_eval_v2_size": len(added_rows),
        "eval_v2_size": len(merged_rows),
        "added_image_ids": [row.get("image_id", "") for row in added_rows],
        "incomplete_keep_yes_rows": [
            {
                "id": row.get("id", ""),
                "source_image_id": row.get("source_image_id", ""),
                "boundary_family": row.get("boundary_family", ""),
                "gold_step_label": row.get("gold_step_label", ""),
            }
            for row in incomplete_rows
        ],
        "rejected_rows": [
            {
                "id": row.get("id", ""),
                "source_image_id": row.get("source_image_id", ""),
                "keep": row.get("keep", ""),
            }
            for row in rejected_rows
        ],
    }
    summary_path = RUNS_DIR / "task3_procedural_eval_v2_build_summary_2026-06-26.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {output_jsonl}")
    print(f"Wrote {summary_path}")
    print(f"eval_v2_size={len(merged_rows)}")
    print(f"added_size={len(added_rows)}")
    print(f"incomplete_keep_yes={len(incomplete_rows)}")


if __name__ == "__main__":
    main()
