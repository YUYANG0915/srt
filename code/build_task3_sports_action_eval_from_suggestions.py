#!/usr/bin/env python3
"""Build a sports-action eval JSONL from OpenAI-assisted suggestion CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def image_id_from_row(row: dict[str, str]) -> str:
    if row.get("source_image_id"):
        return row["source_image_id"]
    parts = row["id"].split("_")
    return parts[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suggestions-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.9)
    parser.add_argument("--min-index", type=int, default=1)
    parser.add_argument("--max-index", type=int, default=10_000)
    args = parser.parse_args()

    rows = read_csv(Path(args.suggestions_csv))
    output_rows: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for row in rows:
        try:
            idx = int(row["id"].split("_")[2])
        except (IndexError, ValueError):
            idx = 0
        try:
            confidence = float(row.get("suggested_confidence") or 0.0)
        except ValueError:
            confidence = 0.0
        keep = row.get("suggested_keep", "").strip().lower()
        phase = row.get("suggested_gold_action_phase", "").strip()
        family = row.get("suggested_phase_boundary_family", "").strip() or "other"
        if not (args.min_index <= idx <= args.max_index and keep == "yes" and phase and confidence >= args.min_confidence):
            skipped.append(row)
            continue
        item = {
            "id": row["id"],
            "image_id": image_id_from_row(row),
            "source": "sports_action_openai_suggestions",
            "source_dataset": "coco_val2017_local_subset",
            "task": "task3_sports_action_phase",
            "scenario_domain": "sports_action",
            "sport_hint": row.get("sport_hint", ""),
            "underspecified_question": "What phase of the sports action is happening in this scene?",
            "original_question": "What phase of the sports action is happening in this scene?",
            "question": "What phase of the sports action is happening in this scene?",
            "gold_action_phase": phase,
            "expected_answer": phase,
            "answer": phase,
            "expected_policy": "answer",
            "phase_boundary_family": family,
            "boundary_family": family,
            "neighbor_phase": row.get("suggested_neighbor_phase", ""),
            "why_not_neighbor_phase": row.get("suggested_why_not_neighbor_phase", ""),
            "visible_actor_pose": row.get("suggested_visible_actor_pose", ""),
            "visible_equipment": "",
            "visible_ball_or_target": row.get("suggested_visible_ball_or_target", ""),
            "visible_motion_cues": row.get("suggested_visible_motion_cues", ""),
            "suggested_confidence": str(confidence),
            "issue_type": "openai_assisted_high_confidence",
            "notes": row.get("suggested_notes", ""),
            "image_path": row.get("image_path", ""),
        }
        output_rows.append(item)

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for item in output_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "suggestions_csv": args.suggestions_csv,
        "output_jsonl": args.output_jsonl,
        "min_confidence": args.min_confidence,
        "min_index": args.min_index,
        "max_index": args.max_index,
        "kept": len(output_rows),
        "skipped": len(skipped),
        "phase_distribution": {},
        "family_distribution": {},
        "sport_distribution": {},
    }
    for item in output_rows:
        summary["phase_distribution"][item["expected_answer"]] = summary["phase_distribution"].get(item["expected_answer"], 0) + 1
        summary["family_distribution"][item["phase_boundary_family"]] = summary["family_distribution"].get(item["phase_boundary_family"], 0) + 1
        summary["sport_distribution"][item["sport_hint"]] = summary["sport_distribution"].get(item["sport_hint"], 0) + 1
    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_jsonl}")
    print(f"Wrote {output_summary}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
