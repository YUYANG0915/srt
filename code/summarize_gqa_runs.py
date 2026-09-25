#!/usr/bin/env python3
"""Summarize GQA underspecified referent runs across models."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parents[1]
RUNS = BASE / "runs"
OUT_DIR = RUNS / "gqa_multimodel_summary"

MODELS = {
    "openai_gpt54mini": "GPT-5.4-mini",
    "gemini31pro": "Gemini-3.1-Pro",
    "glm46v": "GLM-4.6V",
    "mistralsmall32": "Mistral-Small-3.2-Vision",
    "qwen25vl72b": "Qwen2.5-VL-72B",
}

TASKS = {
    "gqa_task1": {
        "label": "GQA Task1 ambiguous referent",
        "prefix": "gqa_task1_clean124",
        "expected_n": 100,
        "correct": {"clarify", "enumerate"},
        "main_error_label": "over-answer",
        "main_error": {"answer"},
    },
    "gqa_task2": {
        "label": "GQA Task2 unique referent",
        "prefix": "gqa_task2_clean96",
        "expected_n": 96,
        "correct": {"answer"},
        "main_error_label": "unnecessary clarification",
        "main_error": {"clarify", "enumerate"},
    },
}

MODES = ["default", "evidence", "marked_no_prior", "visual_sketchpad"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_jsonish(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
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


def pct(num: int, den: int) -> str:
    return "0.0%" if den == 0 else f"{num / den:.1%}"


def run_path(task: str, model_key: str, mode: str) -> Path:
    prefix = TASKS[task]["prefix"]
    return RUNS / f"{prefix}_{model_key}_{mode}_limit100.jsonl"


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("id"))
        previous = by_id.get(item_id)
        if previous is None:
            by_id[item_id] = row
            continue
        previous_valid = not previous.get("error") and bool(parse_jsonish(str(previous.get("raw_text", ""))).get("policy"))
        current_valid = not row.get("error") and bool(parse_jsonish(str(row.get("raw_text", ""))).get("policy"))
        if current_valid or not previous_valid:
            by_id[item_id] = row
    return list(by_id.values())


def summarize_file(task: str, model_key: str, mode: str, path: Path) -> dict[str, Any]:
    rows = dedupe_rows(read_jsonl(path))
    policies: Counter[str] = Counter()
    parse_fail = 0
    errors = 0

    for row in rows:
        if row.get("error"):
            errors += 1
        parsed = parse_jsonish(str(row.get("raw_text", "")))
        if not parsed:
            parse_fail += 1
        policy = str(parsed.get("policy", "")).lower() or "<blank>"
        policies[policy] += 1

    task_cfg = TASKS[task]
    n = len(rows)
    correct = sum(policies[p] for p in task_cfg["correct"])
    main_error = sum(policies[p] for p in task_cfg["main_error"])
    expected_n = task_cfg["expected_n"]
    return {
        "task": task,
        "task_label": task_cfg["label"],
        "model_key": model_key,
        "model": MODELS[model_key],
        "mode": mode,
        "path": str(path),
        "n": n,
        "expected_n": expected_n,
        "complete": n == expected_n and errors == 0,
        "policy_accuracy_n": correct,
        "policy_accuracy": correct / n if n else 0.0,
        "main_error_label": task_cfg["main_error_label"],
        "main_error_n": main_error,
        "main_error": main_error / n if n else 0.0,
        "answer": policies["answer"],
        "clarify": policies["clarify"],
        "enumerate": policies["enumerate"],
        "uncertain": policies["uncertain"],
        "blank": policies["<blank>"],
        "parse_fail": parse_fail,
        "errors": errors,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for model_key in MODELS:
        for task in TASKS:
            for mode in MODES:
                path = run_path(task, model_key, mode)
                if path.exists():
                    summaries.append(summarize_file(task, model_key, mode, path))

    with (OUT_DIR / "condition_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    with (OUT_DIR / "condition_summary.csv").open("w", encoding="utf-8", newline="") as f:
        if summaries:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)

    md = ["# GQA Multimodel Summary", ""]
    md.append("Complete rows have the expected item count and zero API errors.")
    for task, cfg in TASKS.items():
        md += [
            "",
            f"## {cfg['label']}",
            "",
            f"| model | mode | complete | n | policy accuracy | {cfg['main_error_label']} | answer | clarify | enumerate | uncertain | parse fail | errors |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in [r for r in summaries if r["task"] == task]:
            md.append(
                "| {model} | {mode} | {complete} | {n}/{expected_n} | {acc} | {err} | {answer} | {clarify} | {enumerate} | {uncertain} | {parse_fail} | {errors} |".format(
                    model=row["model"],
                    mode=row["mode"],
                    complete="yes" if row["complete"] else "no",
                    n=row["n"],
                    expected_n=row["expected_n"],
                    acc=pct(row["policy_accuracy_n"], row["n"]),
                    err=pct(row["main_error_n"], row["n"]),
                    answer=row["answer"],
                    clarify=row["clarify"],
                    enumerate=row["enumerate"],
                    uncertain=row["uncertain"],
                    parse_fail=row["parse_fail"],
                    errors=row["errors"],
                )
            )

    md += [
        "",
        "Reading guide:",
        "- Task1 expects `clarify` or `enumerate`; `answer` is over-answering.",
        "- Task2 expects `answer`; `clarify` or `enumerate` is unnecessary clarification.",
        "- `marked_no_prior` isolates marked visual evidence; `visual_sketchpad` adds the explicit referent-uniqueness process prior.",
    ]
    (OUT_DIR / "condition_summary.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
