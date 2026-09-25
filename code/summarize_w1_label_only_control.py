#!/usr/bin/env python3
"""Summarize the W1 label-only oracle control for Task3 process families."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from summarize_task3_process_runs import (
    dedupe_retry_rows,
    extract_pred,
    labels_from_items,
    load_jsonl,
    parse_jsonish,
)


ROOT = Path(__file__).resolve().parent.parent

FAMILIES = {
    "physical": {
        "name": "Physical state transition",
        "items": "task3_physical_eval_v2_refined.jsonl",
        "tag": "task3_physical_v2refined",
        "summary": "task3_physical_v2refined_5model_4cond_summary_2026-07-03/task3_process_summary.json",
    },
    "assembly": {
        "name": "Assembly / construction",
        "items": "task3_assembly_eval_v1_reviewed.jsonl",
        "tag": "task3_assembly_v1reviewed",
        "summary": "task3_assembly_v1reviewed_5model_4cond_summary_2026-07-03/task3_process_summary.json",
    },
    "traffic": {
        "name": "Navigation / traffic",
        "items": "task3_traffic_eval_v1_reviewed.jsonl",
        "tag": "task3_traffic_v1reviewed",
        "summary": "task3_traffic_v1reviewed_5model_4cond_summary_2026-07-03/task3_process_summary.json",
    },
    "affordance": {
        "name": "Object-use affordance",
        "items": "task3_affordance_eval_v1_reviewed.jsonl",
        "tag": "task3_affordance_v1reviewed",
        "summary": "task3_affordance_v1reviewed_5model_4cond_summary_2026-07-03/task3_process_summary.json",
    },
    "tooluse": {
        "name": "Tool-use",
        "items": "task3_tooluse_eval_v2_focused_corrected52.jsonl",
        "tag": "task3_tooluse_v2focused52",
        "summary": "task3_tooluse_v2focused52_5model_4cond_summary_2026-06-30/task3_tooluse_summary.json",
    },
    "cleaning": {
        "name": "Cleaning",
        "items": "task3_cleaning_eval_v1_reviewed.jsonl",
        "tag": "task3_cleaning_v1reviewed56",
        "summary": "task3_cleaning_v1reviewed56_5model_router_summary_2026-07-01/task3_cleaning_summary.json",
    },
    "craft": {
        "name": "Craft",
        "items": "task3_craft_eval_v1_corrected.jsonl",
        "tag": "task3_craft_v1corrected77",
        "summary": "task3_craft_v1corrected77_5model_5cond_summary_2026-07-02/task3_craft_summary.json",
    },
    "mobility": {
        "name": "Mobility",
        "items": "task3_mobility_eval_v1_reviewed.jsonl",
        "tag": "task3_mobility_v1reviewed106",
        "summary": "task3_mobility_v1reviewed106_5model_5cond_summary_2026-07-02/task3_mobility_summary.json",
    },
}

MODELS = {
    "gpt54mini": "GPT-5.4-mini",
    "qwen25vl72b": "Qwen2.5-VL-72B",
    "glm46v": "GLM-4.6V",
    "gemini31pro": "Gemini-3.1-Pro",
    "mistralsmall32": "Mistral-Small-3.2",
}


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}"


def load_existing_summary(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(str(row["model"]), str(row["condition"])): row for row in rows}


def compute_label_only(family_key: str, meta: dict[str, str], model_key: str) -> dict[str, Any]:
    item_rows = load_jsonl(ROOT / "data" / meta["items"])
    items = {str(row["id"]): row for row in item_rows}
    labels = labels_from_items(items)
    run_path = ROOT / "results" / "raw" / f"{meta['tag']}_{model_key}_process_boundary_label_only.jsonl"
    rows = dedupe_retry_rows(load_jsonl(run_path))
    correct = 0
    total = 0
    missing = 0
    for row in rows:
        item = items.get(str(row.get("id")))
        if item is None:
            continue
        raw_text = str(row.get("raw_text", ""))
        parsed = parse_jsonish(raw_text)
        pred = extract_pred(parsed, raw_text, labels)
        gold = str(item.get("gold_label") or item.get("expected_answer") or "")
        if pred is None:
            missing += 1
        total += 1
        correct += int(pred == gold)
    expected = len(items)
    return {
        "family": family_key,
        "model": MODELS[model_key],
        "condition": "label_only",
        "correct": correct,
        "total": total,
        "expected_total": expected,
        "missing_or_unparsed": missing,
        "accuracy": correct / total if total else None,
    }


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def micro_accuracy(rows: list[dict[str, Any]], condition: str) -> float | None:
    correct_key = f"{condition}_correct"
    correct = 0
    total = 0
    for row in rows:
        if row.get(condition) is None or row.get(correct_key) is None:
            continue
        correct += int(row[correct_key])
        total += int(row["n"])
    return correct / total if total else None


def main() -> None:
    out_dir = ROOT / "results" / "summaries" / "w1_label_only_oracle_control_2026-07-13"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for family_key, meta in FAMILIES.items():
        existing = load_existing_summary(ROOT / "results" / "summaries" / meta["summary"])
        for model_key, model_name in MODELS.items():
            label_only = compute_label_only(family_key, meta, model_key)
            record: dict[str, Any] = {
                "family_key": family_key,
                "family": meta["name"],
                "model": model_name,
                "n": label_only["total"],
                "expected_n": label_only["expected_total"],
                "label_only_correct": label_only["correct"],
                "label_only": label_only["accuracy"],
                "label_only_missing_or_unparsed": label_only["missing_or_unparsed"],
            }
            for condition in ("default", "generic_srt", "scenario_srt", "scenario_srt_v2", "boundary_router"):
                summary_row = existing.get((model_name, condition))
                record[condition] = summary_row["accuracy"] if summary_row else None
                record[f"{condition}_correct"] = summary_row["correct"] if summary_row else None
            if record["boundary_router"] is not None and record["label_only"] is not None:
                record["oracle_minus_label_only_pp"] = (record["boundary_router"] - record["label_only"]) * 100
            else:
                record["oracle_minus_label_only_pp"] = None
            rows.append(record)

    with (out_dir / "w1_label_only_control_by_model.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "family_key",
            "family",
            "model",
            "n",
            "expected_n",
            "default",
            "generic_srt",
            "scenario_srt",
            "scenario_srt_v2",
            "label_only",
            "boundary_router",
            "oracle_minus_label_only_pp",
            "label_only_correct",
            "boundary_router_correct",
            "label_only_missing_or_unparsed",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)

    family_rows: list[dict[str, Any]] = []
    for family_key, meta in FAMILIES.items():
        subset = [row for row in rows if row["family_key"] == family_key]
        family_rows.append(
            {
                "family_key": family_key,
                "family": meta["name"],
                "models": len(subset),
                "n_per_model": subset[0]["n"] if subset else 0,
                "default": mean([row["default"] for row in subset if row["default"] is not None]),
                "generic_srt": mean([row["generic_srt"] for row in subset if row["generic_srt"] is not None]),
                "label_only": mean([row["label_only"] for row in subset if row["label_only"] is not None]),
                "boundary_router": mean([row["boundary_router"] for row in subset if row["boundary_router"] is not None]),
                "oracle_minus_label_only_pp": mean(
                    [row["oracle_minus_label_only_pp"] for row in subset if row["oracle_minus_label_only_pp"] is not None]
                ),
            }
        )

    with (out_dir / "w1_label_only_control_by_family.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "family_key",
            "family",
            "models",
            "n_per_model",
            "default",
            "generic_srt",
            "label_only",
            "boundary_router",
            "oracle_minus_label_only_pp",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(family_rows)

    oracle_rows = [row for row in rows if row["boundary_router"] is not None]
    aggregate = {
        "all_rows": len(rows),
        "families": len(FAMILIES),
        "models": len(MODELS),
        "with_boundary_router_rows": len(oracle_rows),
        "label_only_mean_all_rows": mean([row["label_only"] for row in rows if row["label_only"] is not None]),
        "default_mean_all_rows": mean([row["default"] for row in rows if row["default"] is not None]),
        "generic_srt_mean_all_rows": mean([row["generic_srt"] for row in rows if row["generic_srt"] is not None]),
        "label_only_mean_oracle_rows": mean([row["label_only"] for row in oracle_rows if row["label_only"] is not None]),
        "boundary_router_mean_oracle_rows": mean([row["boundary_router"] for row in oracle_rows if row["boundary_router"] is not None]),
        "oracle_minus_label_only_pp_mean": mean(
            [row["oracle_minus_label_only_pp"] for row in oracle_rows if row["oracle_minus_label_only_pp"] is not None]
        ),
        "label_only_micro_all_rows": micro_accuracy(rows, "label_only"),
        "default_micro_all_rows": micro_accuracy(rows, "default"),
        "generic_srt_micro_all_rows": micro_accuracy(rows, "generic_srt"),
        "label_only_micro_oracle_rows": micro_accuracy(oracle_rows, "label_only"),
        "boundary_router_micro_oracle_rows": micro_accuracy(oracle_rows, "boundary_router"),
    }
    if aggregate["label_only_micro_oracle_rows"] is not None and aggregate["boundary_router_micro_oracle_rows"] is not None:
        aggregate["oracle_minus_label_only_micro_pp"] = (
            aggregate["boundary_router_micro_oracle_rows"] - aggregate["label_only_micro_oracle_rows"]
        ) * 100
    else:
        aggregate["oracle_minus_label_only_micro_pp"] = None
    (out_dir / "w1_label_only_control_summary.json").write_text(
        json.dumps({"aggregate": aggregate, "by_family": family_rows, "by_model": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# W1 Label-only Oracle Control",
        "",
        "Label-only gives the model the oracle boundary and the two candidate labels, but no SRT process description.",
        "",
        "## Aggregate",
        "",
        f"- Rows: {aggregate['all_rows']} family-model pairs; oracle comparison rows: {aggregate['with_boundary_router_rows']}.",
        f"- Mean label-only accuracy over all rows: {pct(aggregate['label_only_mean_all_rows'])}%.",
        f"- Mean label-only accuracy where boundary-router exists: {pct(aggregate['label_only_mean_oracle_rows'])}%.",
        f"- Mean boundary-router accuracy on the same rows: {pct(aggregate['boundary_router_mean_oracle_rows'])}%.",
        f"- Mean oracle minus label-only gap: {aggregate['oracle_minus_label_only_pp_mean']:+.1f} pp.",
        f"- Micro label-only accuracy over all rows: {pct(aggregate['label_only_micro_all_rows'])}%.",
        f"- Micro label-only accuracy where boundary-router exists: {pct(aggregate['label_only_micro_oracle_rows'])}%.",
        f"- Micro boundary-router accuracy on the same rows: {pct(aggregate['boundary_router_micro_oracle_rows'])}%.",
        f"- Micro oracle minus label-only gap: {aggregate['oracle_minus_label_only_micro_pp']:+.1f} pp.",
        "",
        "## By Family",
        "",
        "| Family | n/model | Default | Generic SRT | Label-only | Boundary router | Oracle - label-only |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in family_rows:
        gap = row["oracle_minus_label_only_pp"]
        gap_str = "-" if gap is None else f"{gap:+.1f} pp"
        lines.append(
            f"| {row['family']} | {row['n_per_model']} | {pct(row['default'])}% | {pct(row['generic_srt'])}% | "
            f"{pct(row['label_only'])}% | {pct(row['boundary_router'])}% | {gap_str} |"
        )
    lines.extend(
        [
            "",
            "## By Family and Model",
            "",
            "| Family | Model | n | Default | Generic SRT | Label-only | Boundary router | Oracle - label-only |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        gap = row["oracle_minus_label_only_pp"]
        gap_str = "-" if gap is None else f"{gap:+.1f} pp"
        lines.append(
            f"| {row['family']} | {row['model']} | {row['n']} | {pct(row['default'])}% | {pct(row['generic_srt'])}% | "
            f"{pct(row['label_only'])}% | {pct(row['boundary_router'])}% | {gap_str} |"
        )
    (out_dir / "w1_label_only_control_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "aggregate": aggregate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
