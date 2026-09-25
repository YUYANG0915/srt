#!/usr/bin/env python3
"""Lightweight summary for pilot outputs.

This intentionally uses simple parsing plus manual-review fields. The pilot is
about finding motivating cases, not producing final automatic labels.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


POLICY_RE = re.compile(r'"?policy"?\s*[:=]\s*"?([a-zA-Z_-]+)"?', re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_jsonish(text: str) -> dict[str, Any] | None:
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
                return None
    return None


def infer_policy(row: dict[str, Any]) -> str:
    parsed = parse_jsonish(row.get("raw_text", ""))
    if parsed and isinstance(parsed.get("policy"), str):
        return parsed["policy"].lower()
    match = POLICY_RE.search(row.get("raw_text", ""))
    if match:
        return match.group(1).lower()
    text = row.get("raw_text", "").lower()
    if "clarify" in text or "which" in text and "?" in text:
        return "clarify?"
    return "unknown"


def candidate_count(row: dict[str, Any]) -> int | None:
    parsed = parse_jsonish(row.get("raw_text", ""))
    if not parsed:
        return None
    candidates = parsed.get("relevant_candidates")
    if isinstance(candidates, list):
        return len(candidates)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()

    grouped = defaultdict(list)
    for run in args.runs:
        path = Path(run)
        for row in load_jsonl(path):
            row["_run"] = str(path)
            row["_policy"] = infer_policy(row)
            row["_candidate_count_model"] = candidate_count(row)
            grouped[(row.get("model"), row.get("mode"))].append(row)

    for key, rows in sorted(grouped.items()):
        model, mode = key
        policies = Counter(row["_policy"] for row in rows)
        expected_clarify = [row for row in rows if row.get("expected_policy") == "clarify"]
        over_answer = [
            row
            for row in expected_clarify
            if row["_policy"] in {"answer", "direct_answer"} or row["_policy"].startswith("answer")
        ]
        evidence_rows = [row for row in rows if row.get("mode") == "evidence"]
        evidence_multi_but_answer = [
            row
            for row in evidence_rows
            if (row.get("_candidate_count_model") or 0) >= 2
            and (row["_policy"] in {"answer", "direct_answer"} or row["_policy"].startswith("answer"))
        ]
        print(f"\n{model} / {mode}")
        print(f"  n={len(rows)}")
        print(f"  policies={dict(policies)}")
        if expected_clarify:
            print(f"  expected_clarify_answer_rate={len(over_answer)}/{len(expected_clarify)}")
        if evidence_rows:
            print(f"  process_answer_inconsistent={len(evidence_multi_but_answer)}/{len(evidence_rows)}")
            for row in evidence_multi_but_answer[:5]:
                print(f"    case {row.get('id')}: model listed {row.get('_candidate_count_model')} candidates but policy={row['_policy']}")


if __name__ == "__main__":
    main()

