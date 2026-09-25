#!/usr/bin/env python3
"""Summarize Task3 sports-action model runs."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="Run JSONL files to summarize.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    run_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for run_path_str in args.runs:
        run_path = Path(run_path_str)
        rows = load_jsonl(run_path)
        correct = 0
        family_total: Counter[str] = Counter()
        family_correct: Counter[str] = Counter()
        mode = rows[0].get("mode", run_path.stem) if rows else run_path.stem
        for row in rows:
            parsed = parse_jsonish(row.get("raw_text", ""))
            pred = parsed.get("action_phase") or parsed.get("phase") or parsed.get("answer")
            gold = row.get("expected_answer", "")
            ok = pred == gold
            correct += int(ok)
            family = row.get("phase_boundary_family") or row.get("boundary_family") or ""
            family_total[family] += 1
            family_correct[family] += int(ok)
            run_rows.append(
                {
                    "mode": mode,
                    "id": row.get("id", ""),
                    "image_id": row.get("image_id", ""),
                    "gold": gold,
                    "pred": pred,
                    "ok": ok,
                    "family": family,
                    "raw_text": row.get("raw_text", ""),
                }
            )
        total = len(rows)
        summary_rows.append(
            {
                "mode": mode,
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
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["mode", "id", "image_id", "gold", "pred", "ok", "family", "raw_text"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_rows)

    lines = ["# Task3 Sports Action Run Summary", ""]
    lines.append("| Mode | Correct | Accuracy |")
    lines.append("|---|---:|---:|")
    for row in summary_rows:
        lines.append(f"| `{row['mode']}` | {row['correct']} / {row['total']} | {row['accuracy'] * 100:.1f}% |")
    lines.append("")
    lines.append("## By Family")
    for row in summary_rows:
        lines.append("")
        lines.append(f"### `{row['mode']}`")
        lines.append("")
        lines.append("| Family | Correct | Accuracy |")
        lines.append("|---|---:|---:|")
        for family, stats in row["by_family"].items():
            label = family or "(blank)"
            lines.append(f"| `{label}` | {stats['correct']} / {stats['total']} | {stats['accuracy'] * 100:.1f}% |")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
