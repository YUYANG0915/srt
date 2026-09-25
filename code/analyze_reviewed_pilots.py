#!/usr/bin/env python3
"""Analyze manually reviewed pilot spreadsheets."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_jsonish(text: str) -> dict[str, Any]:
    stripped = text.strip()
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


def norm(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip().lower()


def load_runs(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    runs = {}
    for path in paths:
        for row in load_jsonl(path):
            runs[(str(row["id"]), str(row["mode"]))] = row
    return runs


def run_policy(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    parsed = parse_jsonish(str(row.get("raw_text", "")))
    return norm(parsed.get("policy"))


def run_candidate_count(row: dict[str, Any] | None) -> int | None:
    if not row:
        return None
    parsed = parse_jsonish(str(row.get("raw_text", "")))
    candidates = parsed.get("relevant_candidates")
    if isinstance(candidates, list):
        return len(candidates)
    return None


def is_acceptable_policy(pred: str, expected: str) -> bool:
    pred = norm(pred)
    expected = norm(expected)
    if expected in {"clarify/enumerate", "clarify or enumerate"}:
        return pred in {"clarify", "enumerate"}
    if expected == "clarify":
        return pred == "clarify"
    if expected == "enumerate":
        return pred in {"enumerate", "clarify"}
    if expected == "uncertain":
        return pred in {"uncertain", "clarify"}
    if expected == "answer":
        return pred == "answer"
    return pred == expected


def analyze(review_xlsx: Path, run_paths: list[Path], task_name: str) -> tuple[list[dict[str, Any]], str]:
    df = pd.read_excel(review_xlsx)
    df.columns = [str(c).strip() for c in df.columns]
    runs = load_runs(run_paths)
    kept = df[df["keep"].map(norm) == "yes"].copy()

    lines = []
    lines.append(f"# {task_name} Reviewed Pilot Summary")
    lines.append("")
    lines.append(f"Review file: `{review_xlsx}`")
    lines.append(f"Total reviewed: {len(df)}")
    lines.append(f"Kept: {len(kept)}")
    lines.append(f"Rejected: {len(df) - len(kept)}")
    lines.append("")
    lines.append("## Human Labels")
    lines.append("")
    lines.append(f"- keep: {dict(Counter(df['keep'].map(norm)))}")
    lines.append(f"- human_expected_policy: {dict(Counter(df['human_expected_policy'].map(norm)))}")
    lines.append(f"- issue_type: {dict(Counter(df['issue_type'].map(norm)))}")
    lines.append("")

    cleaned_rows = []
    for _, review in kept.iterrows():
        item = {k: (None if pd.isna(v) else v) for k, v in review.to_dict().items()}
        item["human_expected_policy"] = norm(review.get("human_expected_policy"))
        item["human_candidate_count"] = norm(review.get("human_candidate_count"))
        item["issue_type"] = norm(review.get("issue_type"))
        cleaned_rows.append(item)

    lines.append("## Model Policy on Kept Items")
    lines.append("")
    lines.append("| Condition | n | policy counts | acceptable policy | invalid answer | process-answer inconsistency |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for mode in ["default", "evidence"]:
        policies = Counter()
        acceptable = 0
        invalid_answer = 0
        inconsistency = 0
        n = 0
        for _, review in kept.iterrows():
            item_id = str(review["id"])
            expected = norm(review.get("human_expected_policy"))
            row = runs.get((item_id, mode))
            pred = run_policy(row)
            policies[pred or "missing"] += 1
            n += 1
            if is_acceptable_policy(pred, expected):
                acceptable += 1
            if pred == "answer" and expected != "answer":
                invalid_answer += 1
            if mode == "evidence":
                count = run_candidate_count(row)
                if count is not None and count >= 2 and pred == "answer" and expected != "answer":
                    inconsistency += 1
        lines.append(
            f"| {mode} | {n} | {dict(policies)} | {acceptable}/{n} | {invalid_answer}/{n} | "
            f"{inconsistency}/{n} |"
        )
    lines.append("")

    lines.append("## Kept Items")
    lines.append("")
    lines.append("| id | target | human_expected_policy | human_candidate_count | issue_type |")
    lines.append("|---|---|---|---:|---|")
    for item in cleaned_rows:
        lines.append(
            f"| {item.get('id')} | {item.get('target_object')} | {item.get('human_expected_policy')} | "
            f"{item.get('human_candidate_count') or ''} | {item.get('issue_type') or ''} |"
        )
    lines.append("")
    return cleaned_rows, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task1-review", required=True)
    parser.add_argument("--task1-runs", nargs="+", required=True)
    parser.add_argument("--task2-review", required=True)
    parser.add_argument("--task2-runs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task1_rows, task1_md = analyze(
        Path(args.task1_review),
        [Path(p) for p in args.task1_runs],
        "Task 1 Ambiguous Referent",
    )
    task2_rows, task2_md = analyze(
        Path(args.task2_review),
        [Path(p) for p in args.task2_runs],
        "Task 2 Unique Referent",
    )

    (output_dir / "task1_reviewed_summary.md").write_text(task1_md, encoding="utf-8")
    (output_dir / "task2_reviewed_summary.md").write_text(task2_md, encoding="utf-8")
    with (output_dir / "task1_clean_items.jsonl").open("w", encoding="utf-8") as f:
        for row in task1_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "task2_clean_items.jsonl").open("w", encoding="utf-8") as f:
        for row in task2_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote reviewed summaries and clean items to {output_dir}")


if __name__ == "__main__":
    main()

