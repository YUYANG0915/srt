#!/usr/bin/env python3
"""Evaluate a rule-based relation-family SRT router on VSR runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import summarize_vsr_srt_runs as summary


RELATION_TO_FAMILY = {
    "left of": "image_position",
    "right of": "image_position",
    "at the left side of": "image_position",
    "at the right side of": "image_position",
    "above": "image_position",
    "below": "image_position",
    "over": "image_position",
    "between": "image_position",
    "in the middle of": "image_position",
    "next to": "proximity",
    "beside": "proximity",
    "by": "proximity",
    "adjacent to": "proximity",
    "alongside": "proximity",
    "at the side of": "proximity",
    "at the edge of": "proximity",
    "close to": "proximity",
    "near": "proximity",
    "on": "contact_support",
    "on top of": "contact_support",
    "under": "contact_support",
    "beneath": "contact_support",
    "touching": "contact_support",
    "attached to": "contact_support",
    "covering": "contact_support",
    "detached from": "contact_support",
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
    "around": "containment",
    "in front of": "depth",
    "behind": "depth",
    "at the back of": "depth",
    "far away from": "depth",
    "away from": "depth",
    "far from": "depth",
    "ahead of": "depth",
    "facing": "orientation",
    "facing away from": "orientation",
    "parallel to": "orientation",
    "perpendicular to": "orientation",
    "across from": "opposition",
    "opposite to": "opposition",
}


def relation_family(relation: str) -> str:
    rel = relation.lower().strip()
    if rel in RELATION_TO_FAMILY:
        return RELATION_TO_FAMILY[rel]
    return "other"


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
    parser.add_argument("--items", default="data/vsr_srt_spatial_items_120.jsonl")
    parser.add_argument("--default-run", required=True)
    parser.add_argument("--evidence-run", required=True)
    parser.add_argument("--verify-run", required=True)
    parser.add_argument("--run", action="append", nargs=2, metavar=("NAME", "PATH"))
    parser.add_argument("--policy-json")
    parser.add_argument("--output-dir", default="runs/vsr_srt_router_summary")
    args = parser.parse_args()

    items = {row["id"]: row for row in summary.load_jsonl(Path(args.items))}
    runs = {
        "default": load_best_records(Path(args.default_run)),
        "evidence": load_best_records(Path(args.evidence_run)),
        "verify": load_best_records(Path(args.verify_run)),
    }
    for name, path in args.run or []:
        runs[name] = load_best_records(Path(path))
    policy = {
        "contact_support": "verify",
        "containment": "default",
        "depth": "evidence",
        "image_position": "default",
        "opposition": "evidence",
        "orientation": "default",
        "other": "default",
        "proximity": "default",
    }
    relation_policy: dict[str, str] = {}
    if args.policy_json:
        loaded = json.loads(Path(args.policy_json).read_text(encoding="utf-8"))
        relation_policy = loaded.get("relation_policy", {})
        policy = loaded.get("family_policy", loaded.get("policy", loaded))

    per_item = []
    for item_id, item in items.items():
        family = relation_family(str(item.get("relation", "")))
        relation = str(item.get("relation", ""))
        selected_condition = relation_policy.get(relation, policy.get(family, "default"))
        record = runs[selected_condition].get(item_id)
        if record is None:
            continue
        row = summary.classify(item, record)
        row["router_mode"] = "srt_router_relation_family"
        row["relation_family"] = family
        row["relation"] = relation
        row["selected_condition"] = selected_condition
        row["selected_level"] = "relation" if relation in relation_policy else "family"
        per_item.append(row)

    n = len(per_item)
    final_acc = sum(row["final_correct"] for row in per_item) / n if n else 0.0
    violation = sum(row["failure_types"] != "ok" for row in per_item) / n if n else 0.0

    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in per_item:
        by_family.setdefault(str(row["relation_family"]), []).append(row)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "vsr_srt_router_per_item.csv", per_item)
    (out_dir / "vsr_srt_router_per_item.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in per_item),
        encoding="utf-8",
    )
    result = {
        "mode": "srt_router_relation_family",
        "n": n,
        "final_accuracy": final_acc,
        "srt_violation_rate": violation,
        "policy": policy,
        "relation_policy": relation_policy,
    }
    (out_dir / "vsr_srt_router_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# VSR SRT Router Summary", ""]
    md.append("| mode | n | final acc. | SRT violation |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| srt_router_relation_family | {n} | {final_acc * 100:.1f}% | {violation * 100:.1f}% |")
    md.extend(["", "## Router Policy", "", "| relation_family | selected_condition |", "|---|---|"])
    for family, selected in sorted(policy.items()):
        md.append(f"| {family} | {selected} |")
    if relation_policy:
        md.extend(["", "## Relation Overrides", "", "| relation | selected_condition |", "|---|---|"])
        for relation, selected in sorted(relation_policy.items()):
            md.append(f"| {relation} | {selected} |")
    md.extend(["", "## By Relation Family", "", "| relation_family | n | acc. |", "|---|---:|---:|"])
    for family, rows in sorted(by_family.items()):
        acc = sum(row["final_correct"] for row in rows) / len(rows)
        md.append(f"| {family} | {len(rows)} | {acc * 100:.1f}% |")
    (out_dir / "vsr_srt_router_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote router summary to {out_dir / 'vsr_srt_router_summary.md'}")


if __name__ == "__main__":
    main()
