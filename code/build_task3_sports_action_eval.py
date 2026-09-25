#!/usr/bin/env python3
"""Build sports-action eval JSONL from the reviewed candidate CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW_CSV = ROOT / "review" / "task3_sports_action_candidates_v1_review.csv"
OUT_JSONL = ROOT / "data" / "task3_sports_action_eval_v1.jsonl"
OUT_SUMMARY = ROOT / "runs" / "task3_sports_action_eval_v1_build_summary_2026-06-26.json"


def truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1", "keep"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def build_item(row: dict[str, str]) -> dict:
    question = row.get("question", "").strip() or "What phase of the sports action is happening in this scene?"
    gold = row.get("gold_action_phase", "").strip()
    family = row.get("phase_boundary_family", "").strip() or "other"
    return {
        "id": row["id"],
        "image_id": row.get("source_image_id", "").strip() or row["id"],
        "source": "sports_action_candidates_v1_reviewed",
        "source_dataset": row.get("source_dataset", "").strip() or "coco_val2017_local_subset",
        "task": "task3_sports_action_phase",
        "scenario_domain": "sports_action",
        "sport_hint": row.get("sport_hint", "").strip(),
        "underspecified_question": question,
        "original_question": question,
        "question": question,
        "gold_action_phase": gold,
        "expected_answer": gold,
        "answer": gold,
        "expected_policy": "answer",
        "phase_boundary_family": family,
        "boundary_family": family,
        "neighbor_phase": row.get("neighbor_phase", "").strip(),
        "why_not_neighbor_phase": row.get("why_not_neighbor_phase", "").strip(),
        "visible_actor_pose": row.get("visible_actor_pose", "").strip(),
        "visible_equipment": row.get("visible_equipment", "").strip(),
        "visible_ball_or_target": row.get("visible_ball_or_target", "").strip(),
        "visible_motion_cues": row.get("visible_motion_cues", "").strip(),
        "issue_type": row.get("issue_type", "").strip(),
        "notes": row.get("notes", "").strip(),
        "image_path": str(Path(row["image_path"]).expanduser().resolve()),
    }


def main() -> None:
    rows = read_csv(REVIEW_CSV)
    valid = []
    incomplete = []
    rejected = []
    for row in rows:
        if truthy(row.get("keep", "")) and row.get("gold_action_phase", "").strip():
            valid.append(build_item(row))
        elif truthy(row.get("keep", "")):
            incomplete.append(row)
        else:
            rejected.append(row)

    write_jsonl(OUT_JSONL, valid)
    summary = {
        "valid_size": len(valid),
        "incomplete_keep_yes_size": len(incomplete),
        "rejected_or_unreviewed_size": len(rejected),
        "output_jsonl": str(OUT_JSONL),
        "incomplete_ids": [row.get("id", "") for row in incomplete],
    }
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSONL}")
    print(f"Wrote {OUT_SUMMARY}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
