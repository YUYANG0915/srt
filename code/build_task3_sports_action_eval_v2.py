#!/usr/bin/env python3
"""Build sports-action eval v2 from seed annotations plus reviewed shortlist suggestions."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEED_JSONL = ROOT / "data" / "task3_sports_action_eval_v1.jsonl"
SHORTLIST_CSV = ROOT / "review" / "task3_sports_action_shortlist_v1_review.csv"
OUT_JSONL = ROOT / "data" / "task3_sports_action_eval_v2.jsonl"
OUT_SUMMARY = ROOT / "runs" / "task3_sports_action_eval_v2_build_summary_2026-06-26.json"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def accepted(value: str) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "accept", "accepted", "true", "1", "keep"}


def build_item(row: dict[str, str]) -> dict:
    question = "What phase of the sports action is happening in this scene?"
    gold = row.get("review_gold_action_phase", "").strip()
    family = row.get("review_phase_boundary_family", "").strip() or "other"
    image_id = row["id"].split("_")[-1]
    return {
        "id": row["id"],
        "image_id": image_id,
        "source": "sports_action_shortlist_v1_reviewed",
        "source_dataset": "coco_val2017_local_subset",
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
        "neighbor_phase": row.get("review_neighbor_phase", "").strip(),
        "why_not_neighbor_phase": row.get("review_why_not_neighbor_phase", "").strip(),
        "visible_actor_pose": row.get("review_visible_actor_pose", "").strip(),
        "visible_equipment": "",
        "visible_ball_or_target": row.get("review_visible_ball_or_target", "").strip(),
        "visible_motion_cues": row.get("review_visible_motion_cues", "").strip(),
        "issue_type": "",
        "notes": row.get("suggested_notes", "").strip(),
        "human_decision": row.get("human_decision", "").strip(),
        "human_correction_notes": row.get("human_correction_notes", "").strip(),
        "image_path": str(Path(row["image_path"]).expanduser().resolve()),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    seed_rows = load_jsonl(SEED_JSONL)
    shortlist_rows = read_csv(SHORTLIST_CSV)
    accepted_rows = [
        build_item(row)
        for row in shortlist_rows
        if accepted(row.get("human_decision", "")) and row.get("review_gold_action_phase", "").strip()
    ]

    merged = list(seed_rows)
    seen = {row["id"] for row in merged}
    added = []
    for row in accepted_rows:
        if row["id"] in seen:
            continue
        merged.append(row)
        added.append(row)
        seen.add(row["id"])

    write_jsonl(OUT_JSONL, merged)
    summary = {
        "seed_size": len(seed_rows),
        "accepted_shortlist_size": len(accepted_rows),
        "added_size": len(added),
        "eval_v2_size": len(merged),
        "output_jsonl": str(OUT_JSONL),
        "added_ids": [row["id"] for row in added],
    }
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_JSONL}")
    print(f"Wrote {OUT_SUMMARY}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
