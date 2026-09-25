#!/usr/bin/env python3
"""Learn a relation-family router policy from VSR dev predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import summarize_vsr_srt_runs as summary
from evaluate_vsr_srt_router import relation_family


def load_best_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in summary.load_jsonl(path):
        record_id = str(record.get("id"))
        previous = records.get(record_id)
        if previous is None or previous.get("error") or not record.get("error"):
            records[record_id] = record
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--run", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--output-dir", default="runs/vsr_srt_learned_router")
    args = parser.parse_args()

    items = {row["id"]: row for row in summary.load_jsonl(Path(args.items))}
    runs = {name: load_best_records(Path(path)) for name, path in args.run}

    per_condition = []
    by_family_condition: dict[tuple[str, str], list[bool]] = {}
    for item_id, item in items.items():
        family = relation_family(str(item.get("relation", "")))
        for condition, records in runs.items():
            record = records.get(item_id)
            if record is None:
                continue
            row = summary.classify(item, record)
            row["condition"] = condition
            row["relation_family"] = family
            per_condition.append(row)
            by_family_condition.setdefault((family, condition), []).append(bool(row["final_correct"]))

    families = sorted({family for family, _ in by_family_condition})
    conditions = sorted(runs)
    policy: dict[str, str] = {}
    family_rows = []
    for family in families:
        best_condition = None
        best_acc = -1.0
        for condition in conditions:
            values = by_family_condition.get((family, condition), [])
            if not values:
                continue
            acc = sum(values) / len(values)
            family_rows.append(
                {
                    "relation_family": family,
                    "condition": condition,
                    "n": len(values),
                    "accuracy": acc,
                }
            )
            if acc > best_acc or (acc == best_acc and condition == "default"):
                best_acc = acc
                best_condition = condition
        if best_condition is not None:
            policy[family] = best_condition

    routed = []
    for item_id, item in items.items():
        family = relation_family(str(item.get("relation", "")))
        condition = policy.get(family, "default")
        record = runs[condition].get(item_id)
        if record is None:
            continue
        row = summary.classify(item, record)
        row["router_mode"] = "learned_relation_family_router"
        row["relation_family"] = family
        row["selected_condition"] = condition
        routed.append(row)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "dev_condition_by_family.csv", family_rows)
    write_csv(out_dir / "dev_router_per_item.csv", routed)
    result = {
        "policy": policy,
        "dev_n": len(routed),
        "dev_accuracy": sum(row["final_correct"] for row in routed) / len(routed) if routed else 0.0,
        "conditions": conditions,
    }
    (out_dir / "router_policy.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# Learned VSR SRT Router Policy", ""]
    md.append(f"Dev accuracy: **{result['dev_accuracy'] * 100:.1f}%** on {result['dev_n']} items.")
    md.extend(["", "## Policy", "", "| relation_family | selected_condition |", "|---|---|"])
    for family, condition in sorted(policy.items()):
        md.append(f"| {family} | {condition} |")
    md.extend(["", "## Dev Accuracy By Family And Condition", "", "| relation_family | condition | n | acc. |", "|---|---|---:|---:|"])
    for row in sorted(family_rows, key=lambda r: (r["relation_family"], r["condition"])):
        md.append(f"| {row['relation_family']} | {row['condition']} | {row['n']} | {row['accuracy'] * 100:.1f}% |")
    (out_dir / "router_policy.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote learned policy to {out_dir / 'router_policy.md'}")


if __name__ == "__main__":
    main()
