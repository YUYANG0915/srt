#!/usr/bin/env python3
"""Summarize full Task3 CoT baselines against existing SRT conditions."""

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

COT_MODES = {
    "plain_cot": "process_plain_cot",
    "multimodal_cot": "process_multimodal_cot",
    "self_verification": "process_self_verification",
    "llava_cot_style": "process_llava_cot_style",
    "grounded_cot": "process_grounded_cot",
}

BASE_CONDITIONS = [
    "default",
    "generic_srt",
    "scenario_srt",
    "scenario_srt_v2",
    "label_only",
    "boundary_router",
]


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}"


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def load_existing_summary(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(str(row["model"]), str(row["condition"])): row for row in rows}


def compute_run(meta: dict[str, str], model_key: str, mode: str) -> dict[str, Any]:
    item_rows = load_jsonl(ROOT / "data" / meta["items"])
    items = {str(row["id"]): row for row in item_rows}
    labels = labels_from_items(items)
    run_path = ROOT / "results" / "raw" / f"{meta['tag']}_{model_key}_{mode}.jsonl"
    rows = dedupe_retry_rows(load_jsonl(run_path))
    correct = 0
    attempted = 0
    total = 0
    missing = 0
    api_errors = 0
    for row in rows:
        item = items.get(str(row.get("id")))
        if item is None:
            continue
        raw_text = str(row.get("raw_text", ""))
        attempted += 1
        if row.get("error"):
            api_errors += 1
            continue
        if not raw_text.strip():
            missing += 1
            continue
        parsed = parse_jsonish(raw_text)
        pred = extract_pred(parsed, raw_text, labels)
        gold = str(item.get("gold_label") or item.get("expected_answer") or "")
        if pred is None:
            missing += 1
        total += 1
        correct += int(pred == gold)
    return {
        "correct": correct,
        "total": total,
        "attempted": attempted,
        "expected_total": len(items),
        "missing_or_unparsed": missing,
        "api_errors": api_errors,
        "accuracy": correct / total if total else None,
    }


def micro_accuracy(rows: list[dict[str, Any]], condition: str) -> float | None:
    correct = 0
    total = 0
    for row in rows:
        if row.get(condition) is None:
            continue
        correct_value = row.get(f"{condition}_correct")
        total_value = row.get(f"{condition}_total", row.get("n"))
        if correct_value is None or total_value is None:
            continue
        correct += int(correct_value)
        total += int(total_value)
    return correct / total if total else None


def best_condition(row: dict[str, Any], conditions: list[str]) -> tuple[str | None, float | None]:
    best_name: str | None = None
    best_value: float | None = None
    for condition in conditions:
        value = row.get(condition)
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_name = condition
            best_value = value
    return best_name, best_value


def main() -> None:
    out_dir = ROOT / "results" / "summaries" / "task3_cot_baseline_control_2026-07-15"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for family_key, meta in FAMILIES.items():
        existing = load_existing_summary(ROOT / "results" / "summaries" / meta["summary"])
        for model_key, model_name in MODELS.items():
            record: dict[str, Any] = {
                "family_key": family_key,
                "family": meta["name"],
                "model": model_name,
            }
            for condition in BASE_CONDITIONS:
                summary_row = existing.get((model_name, condition))
                record[condition] = summary_row["accuracy"] if summary_row else None
                record[f"{condition}_correct"] = summary_row["correct"] if summary_row else None
                record[f"{condition}_total"] = summary_row["total"] if summary_row else None
            for cot_condition, mode in COT_MODES.items():
                stats = compute_run(meta, model_key, mode)
                record[cot_condition] = stats["accuracy"]
                record[f"{cot_condition}_correct"] = stats["correct"]
                record[f"{cot_condition}_total"] = stats["total"]
                record[f"{cot_condition}_attempted"] = stats["attempted"]
                record[f"{cot_condition}_missing_or_unparsed"] = stats["missing_or_unparsed"]
                record[f"{cot_condition}_api_errors"] = stats["api_errors"]
                record["n"] = stats["total"]
                record["expected_n"] = stats["expected_total"]
            cot_best_name, cot_best = best_condition(record, list(COT_MODES))
            srt_best_name, srt_best = best_condition(record, ["generic_srt", "scenario_srt", "scenario_srt_v2", "boundary_router"])
            deployable_srt_name, deployable_srt = best_condition(record, ["generic_srt", "scenario_srt", "scenario_srt_v2"])
            record["best_cot_condition"] = cot_best_name
            record["best_cot"] = cot_best
            record["best_srt_condition"] = srt_best_name
            record["best_srt"] = srt_best
            record["best_deployable_srt_condition"] = deployable_srt_name
            record["best_deployable_srt"] = deployable_srt
            if cot_best is not None and record.get("default") is not None:
                record["best_cot_minus_default_pp"] = (cot_best - record["default"]) * 100
            if cot_best is not None and deployable_srt is not None:
                record["deployable_srt_minus_best_cot_pp"] = (deployable_srt - cot_best) * 100
            if cot_best is not None and srt_best is not None:
                record["best_srt_minus_best_cot_pp"] = (srt_best - cot_best) * 100
            rows.append(record)

    fields = [
        "family_key",
        "family",
        "model",
        "n",
        "expected_n",
        "default",
        "plain_cot",
        "multimodal_cot",
        "self_verification",
        "llava_cot_style",
        "grounded_cot",
        "best_cot_condition",
        "best_cot",
        "generic_srt",
        "scenario_srt",
        "scenario_srt_v2",
        "label_only",
        "boundary_router",
        "best_deployable_srt_condition",
        "best_deployable_srt",
        "best_srt_condition",
        "best_srt",
        "best_cot_minus_default_pp",
        "deployable_srt_minus_best_cot_pp",
        "best_srt_minus_best_cot_pp",
    ]
    with (out_dir / "task3_cot_baseline_by_model.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)

    family_rows: list[dict[str, Any]] = []
    for family_key, meta in FAMILIES.items():
        subset = [row for row in rows if row["family_key"] == family_key]
        family_rows.append(
            {
                "family_key": family_key,
                "family": meta["name"],
                "models": len(subset),
                "n_per_model": subset[0]["n"] if subset else 0,
                **{
                    condition: mean([row[condition] for row in subset if row.get(condition) is not None])
                    for condition in [
                        "default",
                        "plain_cot",
                        "multimodal_cot",
                        "self_verification",
                        "llava_cot_style",
                        "grounded_cot",
                        "best_cot",
                        "generic_srt",
                        "scenario_srt",
                        "scenario_srt_v2",
                        "best_deployable_srt",
                        "label_only",
                        "boundary_router",
                        "best_srt",
                    ]
                },
            }
        )
    with (out_dir / "task3_cot_baseline_by_family.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(family_rows[0].keys()))
        writer.writeheader()
        writer.writerows(family_rows)

    model_rows: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    for model, subset in by_model.items():
        model_rows.append(
            {
                "model": model,
                "families": len(subset),
                **{
                    condition: mean([row[condition] for row in subset if row.get(condition) is not None])
                    for condition in [
                        "default",
                        "plain_cot",
                        "multimodal_cot",
                        "self_verification",
                        "llava_cot_style",
                        "grounded_cot",
                        "best_cot",
                        "generic_srt",
                        "scenario_srt",
                        "scenario_srt_v2",
                        "best_deployable_srt",
                        "label_only",
                        "boundary_router",
                        "best_srt",
                    ]
                },
            }
        )
    with (out_dir / "task3_cot_baseline_by_model_macro.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(model_rows[0].keys()))
        writer.writeheader()
        writer.writerows(model_rows)

    aggregate = {
        "family_model_rows": len(rows),
        "families": len(FAMILIES),
        "models": len(MODELS),
        "macro": {
            condition: mean([row[condition] for row in rows if row.get(condition) is not None])
            for condition in [
                "default",
                "plain_cot",
                "multimodal_cot",
                "self_verification",
                "llava_cot_style",
                "grounded_cot",
                "best_cot",
                "generic_srt",
                "scenario_srt",
                "scenario_srt_v2",
                "best_deployable_srt",
                "label_only",
                "boundary_router",
                "best_srt",
            ]
        },
        "micro": {
            condition: micro_accuracy(rows, condition)
            for condition in [
                "default",
                "plain_cot",
                "multimodal_cot",
                "self_verification",
                "llava_cot_style",
                "grounded_cot",
                "generic_srt",
                "scenario_srt",
                "scenario_srt_v2",
                "label_only",
                "boundary_router",
            ]
        },
        "counts": {
            "cot_best_beats_default": sum(
                1 for row in rows if row.get("best_cot") is not None and row.get("default") is not None and row["best_cot"] > row["default"]
            ),
            "cot_best_beats_deployable_srt": sum(
                1
                for row in rows
                if row.get("best_cot") is not None
                and row.get("best_deployable_srt") is not None
                and row["best_cot"] > row["best_deployable_srt"]
            ),
            "deployable_srt_beats_cot_best": sum(
                1
                for row in rows
                if row.get("best_cot") is not None
                and row.get("best_deployable_srt") is not None
                and row["best_deployable_srt"] > row["best_cot"]
            ),
            "ties_deployable_srt_best_cot": sum(
                1
                for row in rows
                if row.get("best_cot") is not None
                and row.get("best_deployable_srt") is not None
                and abs(row["best_deployable_srt"] - row["best_cot"]) < 1e-12
            ),
        },
    }

    (out_dir / "task3_cot_baseline_summary.json").write_text(
        json.dumps({"aggregate": aggregate, "by_family": family_rows, "by_model_macro": model_rows, "by_family_model": rows}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Task3 CoT Baseline Control",
        "",
        "Full comparison over 8 process families, 5 VLMs, and five CoT-style baselines.",
        "",
        "## Aggregate",
        "",
        f"- Family-model rows: {aggregate['family_model_rows']} ({aggregate['families']} families x {aggregate['models']} models).",
        f"- Macro default: {pct(aggregate['macro']['default'])}%.",
        f"- Macro plain CoT: {pct(aggregate['macro']['plain_cot'])}%.",
        f"- Macro multimodal CoT: {pct(aggregate['macro']['multimodal_cot'])}%.",
        f"- Macro self-verification: {pct(aggregate['macro']['self_verification'])}%.",
        f"- Macro LLaVA-CoT-style: {pct(aggregate['macro']['llava_cot_style'])}%.",
        f"- Macro grounded CoT: {pct(aggregate['macro']['grounded_cot'])}%.",
        f"- Macro best CoT-style baseline: {pct(aggregate['macro']['best_cot'])}%.",
        f"- Macro generic SRT: {pct(aggregate['macro']['generic_srt'])}%.",
        f"- Macro best deployable SRT (generic/scenario): {pct(aggregate['macro']['best_deployable_srt'])}%.",
        f"- Macro label-only oracle control: {pct(aggregate['macro']['label_only'])}%.",
        f"- Macro best SRT/oracle condition: {pct(aggregate['macro']['best_srt'])}%.",
        "",
        "## By Family",
        "",
        "| Family | n/model | Default | Plain CoT | MM-CoT | Self-Verify | LLaVA-CoT | Grounded-CoT | Best CoT | Generic SRT | Best deployable SRT | Label-only | Boundary router | Best SRT/oracle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in family_rows:
        lines.append(
            f"| {row['family']} | {row['n_per_model']} | {pct(row['default'])}% | {pct(row['plain_cot'])}% | "
            f"{pct(row['multimodal_cot'])}% | {pct(row['self_verification'])}% | {pct(row['llava_cot_style'])}% | "
            f"{pct(row['grounded_cot'])}% | {pct(row['best_cot'])}% | "
            f"{pct(row['generic_srt'])}% | {pct(row['best_deployable_srt'])}% | {pct(row['label_only'])}% | "
            f"{pct(row['boundary_router'])}% | {pct(row['best_srt'])}% |"
        )
    lines.extend(
        [
            "",
            "## By Model Macro",
            "",
            "| Model | Default | Plain CoT | MM-CoT | Self-Verify | LLaVA-CoT | Grounded-CoT | Best CoT | Generic SRT | Best deployable SRT | Label-only | Boundary router | Best SRT/oracle |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(model_rows, key=lambda r: r["model"]):
        lines.append(
            f"| {row['model']} | {pct(row['default'])}% | {pct(row['plain_cot'])}% | {pct(row['multimodal_cot'])}% | "
            f"{pct(row['self_verification'])}% | {pct(row['llava_cot_style'])}% | {pct(row['grounded_cot'])}% | "
            f"{pct(row['best_cot'])}% | {pct(row['generic_srt'])}% | "
            f"{pct(row['best_deployable_srt'])}% | {pct(row['label_only'])}% | {pct(row['boundary_router'])}% | {pct(row['best_srt'])}% |"
        )
    lines.extend(
        [
            "",
            "## By Family and Model",
            "",
            "| Family | Model | Default | Best CoT | Best CoT cond. | Best deployable SRT | Best deployable cond. | Label-only | Boundary router |",
            "|---|---|---:|---:|---|---:|---|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['family']} | {row['model']} | {pct(row['default'])}% | {pct(row['best_cot'])}% | "
            f"`{row['best_cot_condition']}` | {pct(row['best_deployable_srt'])}% | `{row['best_deployable_srt_condition']}` | "
            f"{pct(row['label_only'])}% | {pct(row['boundary_router'])}% |"
        )
    (out_dir / "task3_cot_baseline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"output_dir": str(out_dir), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
