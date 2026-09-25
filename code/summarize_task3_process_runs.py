#!/usr/bin/env python3
"""Summarize generic Task3 SRT process-family runs."""

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


def dedupe_retry_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("id", ""))
        if not item_id:
            continue
        current = by_id.get(item_id)
        row_usable = not row.get("error") and bool(str(row.get("raw_text", "")).strip())
        current_usable = bool(current) and not current.get("error") and bool(str(current.get("raw_text", "")).strip())
        if current is None or row_usable or not current_usable:
            by_id[item_id] = row
    return list(by_id.values())


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
    stem = path.stem
    if mode == "process_default" or "process_default" in stem:
        return "default"
    if mode == "process_visual_evidence_only" or "process_visual_evidence_only" in stem:
        return "visual_evidence_only"
    if mode == "process_plain_cot" or "process_plain_cot" in stem:
        return "plain_cot"
    if mode == "process_multimodal_cot" or "process_multimodal_cot" in stem:
        return "multimodal_cot"
    if mode == "process_self_verification" or "process_self_verification" in stem:
        return "self_verification"
    if mode == "process_srt_policy_only" or "process_srt_policy_only" in stem:
        return "policy_only_T"
    if mode == "process_srt_s_t" or "process_srt_s_t" in stem:
        return "partial_S_T"
    if mode == "process_srt_r_t" or "process_srt_r_t" in stem:
        return "partial_R_T"
    if mode == "process_srt_s_r_no_t" or "process_srt_s_r_no_t" in stem:
        return "partial_S_R_no_T"
    if mode == "process_srt_generic" or "process_srt_generic" in stem:
        return "generic_srt"
    if mode == "process_srt_scenario" or "process_srt_scenario" in stem:
        return "scenario_srt"
    if mode == "process_srt_wrong_boundary_router" or "process_srt_wrong_boundary_router" in stem:
        return "wrong_boundary_router"
    if mode == "process_srt_self_router" or "process_srt_self_router" in stem:
        return "self_router"
    if mode == "process_srt_boundary_router_oracle" or "process_srt_boundary_router_oracle" in stem:
        return "boundary_router"
    if mode == "process_boundary_label_only" or "process_boundary_label_only" in stem:
        return "label_only"
    return mode


def labels_from_items(items: dict[str, dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for item in items.values():
        for label in item.get("answer_space") or []:
            label_str = str(label)
            if label_str and label_str not in labels:
                labels.append(label_str)
    return labels


def extract_pred(parsed: dict[str, Any], raw_text: str, labels: list[str]) -> str | None:
    for key in (
        "process_step",
        "stage",
        "answer",
        "label",
        "assembly_stage",
        "transition_stage",
        "event_stage",
        "navigation_stage",
        "use_stage",
    ):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            candidate = value.strip()
            if candidate in labels:
                return candidate
            for label in labels:
                if label in candidate:
                    return label
            return candidate
    text = raw_text.lower()
    for label in labels:
        if label.lower() in text:
            return label
    return None


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Task3 Generic Process Summary")
    args = parser.parse_args()

    items = {str(row["id"]): row for row in load_jsonl(Path(args.items))}
    labels = labels_from_items(items)
    summary_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []

    for run_str in args.runs:
        run_path = Path(run_str)
        rows = dedupe_retry_rows(load_jsonl(run_path))
        model = infer_model_label(run_path, rows)
        condition = infer_condition(run_path, rows)
        label_total: Counter[str] = Counter()
        label_correct: Counter[str] = Counter()
        boundary_total: Counter[str] = Counter()
        boundary_correct: Counter[str] = Counter()
        correct = 0
        total = 0
        for row in rows:
            item = items.get(str(row.get("id")))
            if item is None:
                continue
            raw_text = str(row.get("raw_text", ""))
            parsed = parse_jsonish(raw_text)
            pred = extract_pred(parsed, raw_text, labels)
            gold = str(item.get("gold_label") or item.get("expected_answer") or "")
            ok = pred == gold
            total += 1
            correct += int(ok)
            label_total[gold] += 1
            label_correct[gold] += int(ok)
            boundary = str(item.get("boundary_family") or "unknown")
            boundary_total[boundary] += 1
            boundary_correct[boundary] += int(ok)
            case_rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "id": row.get("id"),
                    "image_id": item.get("image_id"),
                    "scenario_family": item.get("scenario_family"),
                    "subfamily": item.get("subfamily"),
                    "gold": gold,
                    "pred": pred or "",
                    "ok": ok,
                    "boundary_family": boundary,
                    "label_source": item.get("label_source"),
                    "visible_actor": item.get("visible_actor"),
                    "visible_object": item.get("visible_object"),
                    "visible_relation": item.get("visible_relation"),
                    "visible_state_cues": item.get("visible_state_cues"),
                    "raw_text": raw_text,
                }
            )
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
                        "accuracy": label_correct[label] / label_total[label] if label_total[label] else 0.0,
                    }
                    for label in sorted(label_total)
                },
                "by_boundary": {
                    boundary: {
                        "correct": boundary_correct[boundary],
                        "total": boundary_total[boundary],
                        "accuracy": boundary_correct[boundary] / boundary_total[boundary] if boundary_total[boundary] else 0.0,
                    }
                    for boundary in sorted(boundary_total)
                },
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "task3_process_summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "task3_process_case_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0].keys()) if case_rows else [])
        if case_rows:
            writer.writeheader()
            writer.writerows(case_rows)

    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for row in summary_rows:
        by_model.setdefault(row["model"], {})[row["condition"]] = row
    condition_order = [
        "default",
        "visual_evidence_only",
        "plain_cot",
        "multimodal_cot",
        "self_verification",
        "generic_srt",
        "scenario_srt",
        "label_only",
        "wrong_boundary_router",
        "self_router",
        "boundary_router",
    ]
    lines = [
        f"# {args.title}",
        "",
        "Note: this summary is evaluated on the supplied reviewed item file.",
        "",
        "## Overall",
        "",
        "| Model | Default | Visual Evidence | Plain CoT | Multimodal CoT | Self-Verify | Generic SRT | Scenario SRT | Label-only | Wrong Boundary | Self-router | Boundary Router | Best Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in sorted(by_model):
        default_acc = by_model[model].get("default", {}).get("accuracy", 0.0)
        best_acc = default_acc
        cells = []
        for condition in condition_order:
            row = by_model[model].get(condition)
            if row:
                best_acc = max(best_acc, row["accuracy"])
                cells.append(f"{row['correct']} / {row['total']} ({pct(row['accuracy'])})")
            else:
                cells.append("-")
        delta = f"{(best_acc - default_acc) * 100:+.1f} pp" if by_model[model].get("default") else "-"
        lines.append(f"| {model} | {' | '.join(cells)} | {delta} |")

    lines.append("")
    lines.append("## By Boundary")
    for row in sorted(summary_rows, key=lambda x: (x["model"], x["condition"])):
        lines.extend(["", f"### {row['model']} / `{row['condition']}`", "", "| Boundary | Correct | Accuracy |", "|---|---:|---:|"])
        for boundary, stats in row["by_boundary"].items():
            lines.append(f"| `{boundary}` | {stats['correct']} / {stats['total']} | {pct(stats['accuracy'])} |")

    (out_dir / "task3_process_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "runs": len(summary_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
