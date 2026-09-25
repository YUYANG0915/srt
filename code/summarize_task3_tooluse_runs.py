#!/usr/bin/env python3
"""Summarize Task3 P4 tool-use / object-operation runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def parse_jsonish(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
    return {}


def infer_model_label(path: Path, rows: list[dict[str, Any]]) -> str:
    stem = path.stem
    if "qwen25vl72b" in stem:
        return "Qwen2.5-VL-72B"
    if "glm46v" in stem:
        return "GLM-4.6V"
    if "mistralsmall32" in stem:
        return "Mistral-Small-3.2"
    if "gpt54mini" in stem:
        return "GPT-5.4-mini"
    if "gemini31pro" in stem:
        return "Gemini-3.1-Pro"
    if rows:
        return str(rows[0].get("model", stem))
    return stem


def infer_condition(path: Path, rows: list[dict[str, Any]]) -> str:
    mode = str(rows[0].get("mode", "")) if rows else path.stem
    if mode == "tooluse_default" or "tooluse_default" in path.stem:
        return "default"
    if mode == "tooluse_srt_generic" or "tooluse_srt_generic" in path.stem:
        return "generic_srt"
    if mode == "tooluse_srt_scenario" or "tooluse_srt_scenario" in path.stem:
        if mode == "tooluse_srt_scenario_v2" or "tooluse_srt_scenario_v2" in path.stem:
            return "scenario_srt_v2"
        return "scenario_srt"
    return mode


def extract_pred(parsed: dict[str, Any], raw_text: str) -> str | None:
    for key in ("operation_stage", "stage", "answer", "label", "process_step"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    text = raw_text.lower()
    labels = ["preparing_tool", "active_tool_use", "result_or_finished", "unclear_or_non_process"]
    for label in labels:
        if label in text:
            return label
    return None


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="runs/task3_tooluse_summary")
    args = parser.parse_args()

    items = {str(row["id"]): row for row in load_jsonl(Path(args.items))}
    case_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for run_str in args.runs:
        run_path = Path(run_str)
        rows = load_jsonl(run_path)
        model = infer_model_label(run_path, rows)
        condition = infer_condition(run_path, rows)
        label_total: Counter[str] = Counter()
        label_correct: Counter[str] = Counter()
        boundary_total: Counter[str] = Counter()
        boundary_correct: Counter[str] = Counter()
        correct = 0

        for row in rows:
            item = items.get(str(row.get("id")), {})
            parsed = parse_jsonish(row.get("raw_text", ""))
            pred = extract_pred(parsed, str(row.get("raw_text", "")))
            gold = str(item.get("gold_label") or row.get("expected_answer") or row.get("answer") or "").strip()
            ok = pred == gold
            correct += int(ok)
            label_total[gold] += 1
            label_correct[gold] += int(ok)
            boundary = str(item.get("boundary_family") or item.get("boundary_family_hint") or "").strip()
            boundary_total[boundary] += 1
            boundary_correct[boundary] += int(ok)
            case_rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "id": row.get("id", ""),
                    "image_id": row.get("image_id", ""),
                    "gold": gold,
                    "pred": pred or "",
                    "ok": ok,
                    "boundary_family": boundary,
                    "label_source": item.get("label_source", ""),
                    "visible_tool": item.get("visible_tool", ""),
                    "visible_object": item.get("visible_object", ""),
                    "raw_text": row.get("raw_text", ""),
                }
            )

        total = len(rows)
        summary_rows.append(
            {
                "model": model,
                "condition": condition,
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
                "by_gold_label": {
                    label: {
                        "correct": label_correct[label],
                        "total": label_total[label],
                        "accuracy": label_correct[label] / label_total[label],
                    }
                    for label in sorted(label_total)
                },
                "by_boundary": {
                    boundary: {
                        "correct": boundary_correct[boundary],
                        "total": boundary_total[boundary],
                        "accuracy": boundary_correct[boundary] / boundary_total[boundary],
                    }
                    for boundary in sorted(boundary_total)
                },
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "task3_tooluse_summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "task3_tooluse_case_table.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "model",
            "condition",
            "id",
            "image_id",
            "gold",
            "pred",
            "ok",
            "boundary_family",
            "label_source",
            "visible_tool",
            "visible_object",
            "raw_text",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_rows)

    model_to_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for row in summary_rows:
        model_to_rows.setdefault(row["model"], {})[row["condition"]] = row

    lines = ["# Task3 P4 Tool-use / Object-operation Summary", ""]
    lines.append("Note: this v1 set uses suggested fallback labels where human gold labels were not filled.")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("| Model | Default | Generic SRT | Scenario SRT | Scenario SRT v2 | Best Delta |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model in sorted(model_to_rows):
        default = model_to_rows[model].get("default")
        generic = model_to_rows[model].get("generic_srt")
        scenario = model_to_rows[model].get("scenario_srt")
        scenario_v2 = model_to_rows[model].get("scenario_srt_v2")
        default_cell = f"{default['correct']} / {default['total']} ({pct(default['accuracy'])})" if default else "-"
        generic_cell = f"{generic['correct']} / {generic['total']} ({pct(generic['accuracy'])})" if generic else "-"
        scenario_cell = f"{scenario['correct']} / {scenario['total']} ({pct(scenario['accuracy'])})" if scenario else "-"
        scenario_v2_cell = f"{scenario_v2['correct']} / {scenario_v2['total']} ({pct(scenario_v2['accuracy'])})" if scenario_v2 else "-"
        candidates = [row for row in [generic, scenario, scenario_v2] if row]
        if default and candidates:
            best = max(row["accuracy"] for row in candidates)
            delta_cell = f"{(best - default['accuracy']) * 100:+.1f} pp"
        else:
            delta_cell = "-"
        lines.append(f"| {model} | {default_cell} | {generic_cell} | {scenario_cell} | {scenario_v2_cell} | {delta_cell} |")

    lines.append("")
    lines.append("## By Boundary")
    for row in sorted(summary_rows, key=lambda x: (x["model"], x["condition"])):
        lines.append("")
        lines.append(f"### {row['model']} / `{row['condition']}`")
        lines.append("")
        lines.append("| Boundary | Correct | Accuracy |")
        lines.append("|---|---:|---:|")
        for boundary, stats in row["by_boundary"].items():
            lines.append(f"| `{boundary or '(blank)'}` | {stats['correct']} / {stats['total']} | {pct(stats['accuracy'])} |")

    lines.append("")
    lines.append("## By Gold Label")
    for row in sorted(summary_rows, key=lambda x: (x["model"], x["condition"])):
        lines.append("")
        lines.append(f"### {row['model']} / `{row['condition']}`")
        lines.append("")
        lines.append("| Gold label | Correct | Accuracy |")
        lines.append("|---|---:|---:|")
        for label, stats in row["by_gold_label"].items():
            lines.append(f"| `{label}` | {stats['correct']} / {stats['total']} | {pct(stats['accuracy'])} |")

    (out_dir / "task3_tooluse_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
