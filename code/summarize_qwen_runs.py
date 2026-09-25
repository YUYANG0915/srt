#!/usr/bin/env python3
"""Summarize Qwen OpenRouter runs for visual-sketchpad ablations."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parents[1]
RUNS = BASE / "runs"
OUT_DIR = RUNS / "qwen_visual_sketchpad_analysis"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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


def load_run(path: Path, *retry_paths: Path) -> dict[str, dict[str, Any]]:
    rows = {str(row["id"]): row for row in read_jsonl(path)}
    for retry_path in retry_paths:
        rows.update({str(row["id"]): row for row in read_jsonl(retry_path)})
    return rows


def policy(parsed: dict[str, Any]) -> str:
    return str(parsed.get("policy", "")).lower() or "<blank>"


def candidate_counts(parsed: dict[str, Any]) -> tuple[int, int, int]:
    candidates = parsed.get("relevant_candidates")
    if isinstance(candidates, list):
        possible = 0
        target = 0
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            status = str(candidate.get("target_status", "")).lower()
            if status in {"possible", "target"}:
                possible += 1
            if status == "target":
                target += 1
        return len(candidates), possible, target

    observations = parsed.get("observation_0")
    if isinstance(observations, list):
        plausible = sum(
            1
            for observation in observations
            if isinstance(observation, dict) and bool(observation.get("is_plausible_referent"))
        )
        uniqueness = str(parsed.get("referent_uniqueness", "")).lower()
        target = 1 if uniqueness == "unique" and plausible == 1 else 0
        return len(observations), plausible, target

    return 0, 0, 0


def pct(num: int, den: int) -> str:
    return "0.0%" if den == 0 else f"{num / den:.1%}"


def summarize(task: str, mode: str, rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    policies: Counter[str] = Counter()
    parse_fail = 0
    bad_or_error = 0
    process_answer_inconsistency = 0
    over_correction = 0

    for row in rows.values():
        if row.get("error"):
            bad_or_error += 1
        parsed = parse_jsonish(str(row.get("raw_text", "")))
        if not parsed:
            parse_fail += 1
        pol = policy(parsed)
        policies[pol] += 1
        _, possible, target = candidate_counts(parsed)
        if task == "task1" and possible >= 2 and pol == "answer":
            process_answer_inconsistency += 1
        if task == "task2" and possible <= 1 and target <= 1 and pol in {"clarify", "enumerate"}:
            over_correction += 1

    n = len(rows)
    if task == "task1":
        correct = policies["clarify"] + policies["enumerate"]
        main_error = policies["answer"]
    else:
        correct = policies["answer"]
        main_error = policies["clarify"] + policies["enumerate"]

    return {
        "task": task,
        "mode": mode,
        "n": n,
        "policy_accuracy": correct,
        "main_error": main_error,
        "answer": policies["answer"],
        "clarify": policies["clarify"],
        "enumerate": policies["enumerate"],
        "uncertain": policies["uncertain"],
        "blank": policies["<blank>"],
        "parse_fail": parse_fail,
        "bad_or_error": bad_or_error,
        "process_answer_inconsistency": process_answer_inconsistency,
        "over_correction": over_correction,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = {
        ("task1", "default"): load_run(RUNS / "task1_auto_clean457_qwen25vl72b_default.jsonl"),
        ("task1", "evidence"): load_run(
            RUNS / "task1_auto_clean457_qwen25vl72b_evidence.jsonl",
            RUNS / "task1_auto_clean457_qwen25vl72b_evidence_retry.jsonl",
        ),
        ("task1", "marked_no_prior"): load_run(
            RUNS / "task1_auto_clean457_qwen25vl72b_marked_no_prior.jsonl",
        ),
        ("task1", "visual_sketchpad"): load_run(
            RUNS / "task1_auto_clean457_qwen25vl72b_visual_sketchpad.jsonl",
        ),
        ("task2", "default"): load_run(
            RUNS / "task2_auto_clean330_qwen25vl72b_default.jsonl",
            RUNS / "task2_auto_clean330_qwen25vl72b_default_retry.jsonl",
        ),
        ("task2", "evidence"): load_run(RUNS / "task2_auto_clean330_qwen25vl72b_evidence.jsonl"),
        ("task2", "marked_no_prior"): load_run(
            RUNS / "task2_auto_clean330_qwen25vl72b_marked_no_prior.jsonl",
        ),
        ("task2", "visual_sketchpad"): load_run(
            RUNS / "task2_auto_clean330_qwen25vl72b_visual_sketchpad.jsonl",
        ),
    }

    summaries = [summarize(task, mode, rows) for (task, mode), rows in runs.items()]

    md = ["# Qwen2.5-VL-72B Visual Sketchpad Summary", ""]
    md += [
        "## Task 1: Ambiguous Referent",
        "",
        "| mode | n | policy accuracy | over-answer | answer | clarify | enumerate | uncertain | P-A inconsistency | parse fail | errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [r for r in summaries if r["task"] == "task1"]:
        md.append(
            "| {mode} | {n} | {acc} | {err} | {answer} | {clarify} | {enumerate} | {uncertain} | {pai} | {parse_fail} | {errors} |".format(
                mode=row["mode"],
                n=row["n"],
                acc=pct(row["policy_accuracy"], row["n"]),
                err=pct(row["main_error"], row["n"]),
                answer=row["answer"],
                clarify=row["clarify"],
                enumerate=row["enumerate"],
                uncertain=row["uncertain"],
                pai=pct(row["process_answer_inconsistency"], row["n"]),
                parse_fail=row["parse_fail"],
                errors=row["bad_or_error"],
            )
        )

    md += [
        "",
        "## Task 2: Unique Referent",
        "",
        "| mode | n | policy accuracy | unnecessary clarification | answer | clarify | enumerate | uncertain | over-correction | parse fail | errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [r for r in summaries if r["task"] == "task2"]:
        md.append(
            "| {mode} | {n} | {acc} | {err} | {answer} | {clarify} | {enumerate} | {uncertain} | {over} | {parse_fail} | {errors} |".format(
                mode=row["mode"],
                n=row["n"],
                acc=pct(row["policy_accuracy"], row["n"]),
                err=pct(row["main_error"], row["n"]),
                answer=row["answer"],
                clarify=row["clarify"],
                enumerate=row["enumerate"],
                uncertain=row["uncertain"],
                over=pct(row["over_correction"], row["n"]),
                parse_fail=row["parse_fail"],
                errors=row["bad_or_error"],
            )
        )

    md += [
        "",
        "Interpretation:",
        "- `marked_no_prior` tests whether marked visual artifacts alone change behavior.",
        "- `visual_sketchpad` tests the marked-artifact plus explicit process/prior condition.",
        "- Task 1 should reduce over-answering; Task 2 should remain mostly answer-oriented.",
    ]

    (OUT_DIR / "condition_summary.md").write_text("\n".join(md), encoding="utf-8")
    with (OUT_DIR / "condition_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
