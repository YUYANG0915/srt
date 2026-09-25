#!/usr/bin/env python3
"""Apply human spot-check annotations to Task3 procedural auto46 eval."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"yes", "y", "true", "1", "ok"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", default="data/task3_procedural_eval_v3_auto46.jsonl")
    parser.add_argument("--spotcheck-csv", default="review/task3_procedural_eval_v3_auto46_spotcheck.csv")
    parser.add_argument("--output-jsonl", default="data/task3_procedural_eval_v3_auto46_humanchecked.jsonl")
    parser.add_argument("--output-summary", default="runs/task3_procedural_eval_v3_auto46_humancheck_summary_2026-06-29.json")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input_jsonl))
    review_by_id = {row["id"]: row for row in read_csv(Path(args.spotcheck_csv))}

    changed_label = 0
    changed_family = 0
    checked = 0
    rejected: list[str] = []
    output_rows: list[dict] = []

    for row in rows:
        item = dict(row)
        review = review_by_id.get(str(row.get("id")))
        if review:
            checked += 1
            if not truthy(review.get("auto_label_ok", "")):
                rejected.append(str(row.get("id")))
                item["human_spotcheck_status"] = "flagged"
            else:
                item["human_spotcheck_status"] = "checked_ok"
            new_label = review.get("gold_step_label", "").strip()
            new_family = review.get("boundary_family", "").strip()
            if new_label and new_label != item.get("gold_step_label"):
                changed_label += 1
                item["gold_step_label"] = new_label
                item["expected_answer"] = new_label
                item["answer"] = new_label
            if new_family and new_family != item.get("boundary_family"):
                changed_family += 1
                item["boundary_family"] = new_family
                item["boundary_family_hint"] = new_family
            item["human_notes"] = review.get("human_notes", "").strip()
        else:
            item["human_spotcheck_status"] = item.get("human_spotcheck_status", "base_human_reviewed")
        output_rows.append(item)

    write_jsonl(Path(args.output_jsonl), output_rows)
    summary = {
        "input_jsonl": args.input_jsonl,
        "spotcheck_csv": args.spotcheck_csv,
        "output_jsonl": args.output_jsonl,
        "total_rows": len(output_rows),
        "spotchecked_rows": checked,
        "checked_ok_rows": sum(1 for row in output_rows if row.get("human_spotcheck_status") == "checked_ok"),
        "flagged_rows": rejected,
        "changed_label": changed_label,
        "changed_family": changed_family,
    }
    out_summary = Path(args.output_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
