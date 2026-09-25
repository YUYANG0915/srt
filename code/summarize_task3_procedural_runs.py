#!/usr/bin/env python3
"""Summarize Task3 procedural/cooking model runs."""

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
    if "gpt54mini" in stem:
        return "GPT-5.4-mini"
    if "gemini31pro" in stem:
        return "Gemini-3.1-Pro"
    if "qwen25vl72b" in stem:
        return "Qwen2.5-VL-72B"
    if "glm46v" in stem:
        return "GLM-4.6V"
    if "mistralsmall32" in stem:
        return "Mistral-Small-3.2"
    if rows:
        return str(rows[0].get("model", stem))
    return stem


def infer_condition(path: Path, rows: list[dict[str, Any]]) -> str:
    if rows and rows[0].get("mode"):
        mode = str(rows[0]["mode"])
    else:
        mode = path.stem
    if mode.endswith("procedural_default") or mode == "procedural_default" or "default" in path.stem:
        return "default"
    if "srt_targeted_process" in mode or "srt_targeted_process" in path.stem:
        return "targeted_process"
    if "srt_action_priority" in mode or "srt_action_priority" in path.stem:
        return "action_priority"
    return mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--items", default=None, help="Optional source item JSONL for metadata merge.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    item_by_id: dict[str, dict[str, Any]] = {}
    if args.items:
        item_by_id = {str(row.get("id")): row for row in load_jsonl(Path(args.items))}

    case_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for run_path_str in args.runs:
        run_path = Path(run_path_str)
        rows = load_jsonl(run_path)
        model = infer_model_label(run_path, rows)
        condition = infer_condition(run_path, rows)
        correct = 0
        family_total: Counter[str] = Counter()
        family_correct: Counter[str] = Counter()
        label_total: Counter[str] = Counter()
        label_correct: Counter[str] = Counter()

        for row in rows:
            item = item_by_id.get(str(row.get("id")), {})
            parsed = parse_jsonish(row.get("raw_text", ""))
            pred = (
                parsed.get("step_label")
                or parsed.get("process_step")
                or parsed.get("answer")
                or parsed.get("label")
            )
            gold = (
                row.get("expected_answer")
                or row.get("gold_step_label")
                or item.get("expected_answer")
                or item.get("gold_step_label")
                or row.get("answer")
            )
            ok = pred == gold
            correct += int(ok)
            family = (
                row.get("boundary_family")
                or row.get("boundary_family_hint")
                or item.get("boundary_family")
                or item.get("boundary_family_hint")
                or ""
            )
            family_total[family] += 1
            family_correct[family] += int(ok)
            label_total[str(gold)] += 1
            label_correct[str(gold)] += int(ok)
            case_rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "id": row.get("id", ""),
                    "image_id": row.get("image_id", ""),
                    "gold": gold,
                    "pred": pred,
                    "ok": ok,
                    "boundary_family": family,
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
                "by_family": {
                    family: {
                        "correct": family_correct[family],
                        "total": family_total[family],
                        "accuracy": family_correct[family] / family_total[family],
                    }
                    for family in sorted(family_total)
                },
                "by_gold_label": {
                    label: {
                        "correct": label_correct[label],
                        "total": label_total[label],
                        "accuracy": label_correct[label] / label_total[label],
                    }
                    for label in sorted(label_total)
                },
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "model",
            "condition",
            "id",
            "image_id",
            "gold",
            "pred",
            "ok",
            "boundary_family",
            "raw_text",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_rows)

    ordered = sorted(summary_rows, key=lambda row: (row["model"], row["condition"]))
    model_to_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for row in ordered:
        model_to_rows.setdefault(row["model"], {})[row["condition"]] = row

    lines = ["# Task3 Procedural Cooking Run Summary", ""]
    lines.append("## Overall")
    lines.append("")
    lines.append("| Model | Default | Targeted-process SRT | Delta |")
    lines.append("|---|---:|---:|---:|")
    for model in sorted(model_to_rows):
        default = model_to_rows[model].get("default")
        targeted = model_to_rows[model].get("targeted_process")
        default_acc = default["accuracy"] if default else None
        targeted_acc = targeted["accuracy"] if targeted else None
        delta = None
        if default_acc is not None and targeted_acc is not None:
            delta = targeted_acc - default_acc
        default_cell = f"{default['correct']} / {default['total']} ({default_acc * 100:.1f}%)" if default else "-"
        targeted_cell = f"{targeted['correct']} / {targeted['total']} ({targeted_acc * 100:.1f}%)" if targeted else "-"
        delta_cell = f"{delta * 100:+.1f} pp" if delta is not None else "-"
        lines.append(f"| {model} | {default_cell} | {targeted_cell} | {delta_cell} |")

    lines.append("")
    lines.append("## By Boundary Family")
    for row in ordered:
        lines.append("")
        lines.append(f"### {row['model']} / `{row['condition']}`")
        lines.append("")
        lines.append("| Boundary family | Correct | Accuracy |")
        lines.append("|---|---:|---:|")
        for family, stats in row["by_family"].items():
            label = family or "(blank)"
            lines.append(f"| `{label}` | {stats['correct']} / {stats['total']} | {stats['accuracy'] * 100:.1f}% |")

    lines.append("")
    lines.append("## By Gold Step Label")
    for row in ordered:
        lines.append("")
        lines.append(f"### {row['model']} / `{row['condition']}`")
        lines.append("")
        lines.append("| Gold label | Correct | Accuracy |")
        lines.append("|---|---:|---:|")
        for label, stats in row["by_gold_label"].items():
            lines.append(f"| `{label}` | {stats['correct']} / {stats['total']} | {stats['accuracy'] * 100:.1f}% |")

    output_md = Path(args.output_md)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_md}")


if __name__ == "__main__":
    main()
