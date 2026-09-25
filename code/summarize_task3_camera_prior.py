#!/usr/bin/env python3
"""Summarize Task3 camera-holder prior runs."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


def extract_count(parsed: dict[str, Any], raw_text: str) -> int | None:
    for key in ("estimated_total_participants", "total_participants", "count"):
        value = parsed.get(key)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    answer = str(parsed.get("answer", "") or raw_text or "").lower()
    if re.search(r"\bat least\s+2\b|\btwo or more\b|\b2\+\b", answer):
        return 2
    for word, number in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", answer):
            return number
    match = re.search(r"\b([0-6])\b", answer)
    if match:
        return int(match.group(1))
    return None


def hidden_inferred(parsed: dict[str, Any], raw_text: str, predicted_count: int | None, visible_count: int) -> bool:
    value = parsed.get("hidden_camera_holder_likely")
    if isinstance(value, bool):
        return value
    text = json.dumps(parsed, ensure_ascii=False).lower() + "\n" + (raw_text or "").lower()
    cue_terms = [
        "camera wearer",
        "camera holder",
        "person filming",
        "person holding the camera",
        "photographer",
        "first-person",
        "implied participant",
        "hidden participant",
    ]
    if any(term in text for term in cue_terms):
        return True
    return predicted_count is not None and predicted_count > visible_count


def classify(item: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    raw_text = str(record.get("raw_text", ""))
    parsed = parse_jsonish(raw_text)
    predicted_count = extract_count(parsed, raw_text)
    visible_count = int(item.get("visible_participant_count", 1))
    expected_total = int(item.get("expected_total_participants", visible_count))
    hidden_expected = bool(item.get("hidden_camera_holder_expected"))
    inferred_hidden = hidden_inferred(parsed, raw_text, predicted_count, visible_count)

    error_type = "ok"
    is_correct = predicted_count == expected_total
    if record.get("error"):
        error_type = "api_or_runtime_error"
        is_correct = False
    elif predicted_count is None:
        error_type = "count_not_extractable"
        is_correct = False
    elif hidden_expected and predicted_count <= visible_count:
        error_type = "visible_only_count"
    elif not hidden_expected and predicted_count > visible_count:
        error_type = "over_infer_hidden_camera_holder"
    elif predicted_count != expected_total:
        error_type = "wrong_count_other"

    return {
        "id": item["id"],
        "model": record.get("model"),
        "mode": record.get("mode"),
        "activity": item.get("activity"),
        "hidden_camera_holder_expected": hidden_expected,
        "visible_count": visible_count,
        "expected_total": expected_total,
        "predicted_count": predicted_count,
        "hidden_inferred": inferred_hidden,
        "is_correct": is_correct,
        "error_type": error_type,
        "policy": parsed.get("policy"),
        "answer": parsed.get("answer", raw_text[:240]),
    }


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="runs/task3_camera_prior_summary")
    args = parser.parse_args()

    items = {row["id"]: row for row in load_jsonl(Path(args.items))}
    rows: list[dict[str, Any]] = []
    for run_path in args.runs:
        for record in load_jsonl(Path(run_path)):
            item = items.get(str(record.get("id")))
            if item:
                rows.append(classify(item, record))

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["model"]),
            str(row["mode"]),
            "first_person" if row["hidden_camera_holder_expected"] else "third_person_control",
        )
        grouped[key].append(row)

    summary = []
    for (model, mode, split), group in sorted(grouped.items()):
        n = len(group)
        correct = sum(1 for row in group if row["is_correct"])
        visible_only = sum(1 for row in group if row["error_type"] == "visible_only_count")
        over_infer = sum(1 for row in group if row["error_type"] == "over_infer_hidden_camera_holder")
        hidden_inferred_count = sum(1 for row in group if row["hidden_inferred"])
        summary.append(
            {
                "model": model,
                "mode": mode,
                "split": split,
                "n": n,
                "accuracy": correct / n if n else 0.0,
                "hidden_inference_rate": hidden_inferred_count / n if n else 0.0,
                "visible_only_error_rate": visible_only / n if n else 0.0,
                "over_infer_hidden_rate": over_infer / n if n else 0.0,
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "task3_per_item_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (out_dir / "task3_condition_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    headers = [
        "model",
        "mode",
        "split",
        "n",
        "accuracy",
        "hidden_inference_rate",
        "visible_only_error_rate",
        "over_infer_hidden_rate",
    ]
    csv_lines = [",".join(headers)]
    for row in summary:
        csv_lines.append(",".join(str(row[h]) for h in headers))
    (out_dir / "task3_condition_summary.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    md = ["# Task3 Camera-Holder Prior Summary", ""]
    md.append("| model | mode | split | n | accuracy | hidden inference | visible-only error | over-infer hidden |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|")
    for row in summary:
        md.append(
            "| {model} | `{mode}` | {split} | {n} | {acc} | {hir} | {vo} | {oi} |".format(
                model=row["model"],
                mode=row["mode"],
                split=row["split"],
                n=row["n"],
                acc=pct(row["accuracy"]),
                hir=pct(row["hidden_inference_rate"]),
                vo=pct(row["visible_only_error_rate"]),
                oi=pct(row["over_infer_hidden_rate"]),
            )
        )
    md.extend(
        [
            "",
            "Interpretation:",
            "- On first-person items, `visible_only_error_rate` is the main failure: the model ignores the implied camera-holder participant.",
            "- On third-person controls, `over_infer_hidden_rate` is the main failure: the model invents a hidden participant without first-person evidence.",
        ]
    )
    (out_dir / "task3_condition_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote {len(rows)} per-item rows and {len(summary)} summary rows to {out_dir}")


if __name__ == "__main__":
    main()
