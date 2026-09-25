#!/usr/bin/env python3
"""Learn a hierarchical VSR SRT router from dev predictions."""

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


def pick_best_condition(scores: dict[str, list[bool]], default_name: str) -> str:
    best_condition = default_name
    best_acc = -1.0
    for condition, values in scores.items():
        if not values:
            continue
        acc = sum(values) / len(values)
        tie_break = 1 if condition == default_name else 0
        if acc > best_acc or (acc == best_acc and tie_break):
            best_acc = acc
            best_condition = condition
    return best_condition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--run", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--min-relation-support", type=int, default=3)
    parser.add_argument("--default-condition", default="default")
    parser.add_argument("--output-dir", default="runs/vsr_srt_hierarchical_router")
    args = parser.parse_args()

    items = {row["id"]: row for row in summary.load_jsonl(Path(args.items))}
    runs = {name: load_best_records(Path(path)) for name, path in args.run}

    by_family_condition: dict[tuple[str, str], list[bool]] = {}
    by_relation_condition: dict[tuple[str, str], list[bool]] = {}
    relation_counts: dict[str, int] = {}
    detail_rows = []

    for item_id, item in items.items():
        relation = str(item.get("relation", ""))
        family = relation_family(relation)
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
        for condition, records in runs.items():
            record = records.get(item_id)
            if record is None:
                continue
            row = summary.classify(item, record)
            correct = bool(row["final_correct"])
            by_family_condition.setdefault((family, condition), []).append(correct)
            by_relation_condition.setdefault((relation, condition), []).append(correct)
            detail_rows.append(
                {
                    "id": item_id,
                    "relation_family": family,
                    "relation": relation,
                    "condition": condition,
                    "final_correct": correct,
                }
            )

    family_policy: dict[str, str] = {}
    for family in sorted({family for family, _ in by_family_condition}):
        scores = {
            condition: by_family_condition.get((family, condition), [])
            for condition in runs
        }
        family_policy[family] = pick_best_condition(scores, args.default_condition)

    relation_policy: dict[str, str] = {}
    relation_rows = []
    for relation in sorted(relation_counts):
        scores = {
            condition: by_relation_condition.get((relation, condition), [])
            for condition in runs
        }
        for condition, values in scores.items():
            if values:
                relation_rows.append(
                    {
                        "relation": relation,
                        "condition": condition,
                        "n": len(values),
                        "accuracy": sum(values) / len(values),
                    }
                )
        if relation_counts[relation] >= args.min_relation_support:
            relation_policy[relation] = pick_best_condition(scores, args.default_condition)

    routed = []
    for item_id, item in items.items():
        relation = str(item.get("relation", ""))
        family = relation_family(relation)
        selected_condition = relation_policy.get(relation, family_policy.get(family, args.default_condition))
        record = runs[selected_condition].get(item_id)
        if record is None:
            continue
        row = summary.classify(item, record)
        row["router_mode"] = "hierarchical_relation_router"
        row["relation_family"] = family
        row["relation"] = relation
        row["selected_condition"] = selected_condition
        row["selected_level"] = "relation" if relation in relation_policy else "family"
        routed.append(row)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "dev_relation_condition_scores.csv", relation_rows)
    write_csv(out_dir / "dev_router_per_item.csv", routed)
    result = {
        "router_type": "hierarchical_relation_router",
        "min_relation_support": args.min_relation_support,
        "default_condition": args.default_condition,
        "family_policy": family_policy,
        "relation_policy": relation_policy,
        "dev_n": len(routed),
        "dev_accuracy": sum(row["final_correct"] for row in routed) / len(routed) if routed else 0.0,
        "conditions": sorted(runs),
    }
    (out_dir / "router_policy.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# Learned Hierarchical VSR SRT Router", ""]
    md.append(f"Dev accuracy: **{result['dev_accuracy'] * 100:.1f}%** on {result['dev_n']} items.")
    md.append(f"Min relation support for relation-level override: **{args.min_relation_support}**.")
    md.extend(["", "## Family Fallback Policy", "", "| relation_family | selected_condition |", "|---|---|"])
    for family, condition in sorted(family_policy.items()):
        md.append(f"| {family} | {condition} |")
    md.extend(["", "## Relation Overrides", "", "| relation | n | selected_condition |", "|---|---:|---|"])
    for relation in sorted(relation_policy):
        md.append(f"| {relation} | {relation_counts[relation]} | {relation_policy[relation]} |")
    (out_dir / "router_policy.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote hierarchical policy to {out_dir / 'router_policy.md'}")


if __name__ == "__main__":
    main()
