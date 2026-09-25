#!/usr/bin/env python3
"""Summarize SRT spatial VSR runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


FRAME_BY_RELATION = {
    "left of": "image_plane",
    "right of": "image_plane",
    "at the left side of": "image_plane",
    "at the right side of": "image_plane",
    "at the side of": "image_plane",
    "above": "image_plane",
    "below": "image_plane",
    "over": "image_plane",
    "across from": "image_plane",
    "opposite to": "image_plane",
    "in front of": "depth",
    "behind": "depth",
    "at the back of": "depth",
    "far away from": "depth",
    "away from": "depth",
    "facing": "orientation",
    "facing away from": "orientation",
    "parallel to": "orientation",
    "perpendicular to": "orientation",
    "inside": "containment",
    "contains": "containment",
    "in": "containment",
    "within": "containment",
    "into": "containment",
    "enclosed by": "containment",
    "surrounding": "containment",
    "consists of": "containment",
    "part of": "containment",
    "has as a part": "containment",
    "on": "contact_support",
    "on top of": "contact_support",
    "under": "contact_support",
    "beneath": "contact_support",
    "touching": "contact_support",
    "attached to": "contact_support",
    "covering": "contact_support",
    "beside": "contact_support",
    "next to": "contact_support",
    "adjacent to": "contact_support",
    "alongside": "contact_support",
    "by": "contact_support",
    "at the edge of": "contact_support",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def norm_answer(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "correct"}:
        return "true"
    if text in {"false", "no", "incorrect"}:
        return "false"
    if text in {"uncertain", "unknown", "cannot determine", "can't determine"}:
        return "uncertain"
    if re.search(r"\b(true|yes|correct)\b", text):
        return "true"
    if re.search(r"\b(false|no|incorrect)\b", text):
        return "false"
    if re.search(r"\b(uncertain|unknown|cannot determine|can't determine)\b", text):
        return "uncertain"
    return ""


def nested_get(data: dict[str, Any], path: list[str]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def classify(item: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_jsonish(str(record.get("raw_text", "")))
    expected = str(item.get("expected_answer") or ("true" if int(item.get("label", 0)) == 1 else "false"))
    answer = norm_answer(parsed.get("answer"))
    if not answer:
        answer = norm_answer(record.get("raw_text"))
    relation = str(item.get("relation", ""))
    expected_frame = str(item.get("expected_spatial_frame") or FRAME_BY_RELATION.get(relation, "uncertain"))
    predicted_frame = str(nested_get(parsed, ["sequence", "spatial_frame"]) or parsed.get("spatial_frame") or "").strip()
    relation_holds = nested_get(parsed, ["relation", "relation_holds"])
    if relation_holds is None:
        relation_holds = nested_get(parsed, ["relation_evidence", "supports"])
    if isinstance(relation_holds, str):
        relation_holds_text = relation_holds.strip().lower()
        if relation_holds_text in {"true", "yes", "1"}:
            relation_holds_norm: bool | None = True
        elif relation_holds_text in {"false", "no", "0"}:
            relation_holds_norm = False
        else:
            relation_holds_norm = None
    elif isinstance(relation_holds, bool):
        relation_holds_norm = relation_holds
    else:
        relation_holds_norm = None
    timeline_consistent = nested_get(parsed, ["timeline_check", "is_consistent"])
    if isinstance(timeline_consistent, str):
        timeline_consistent_norm: bool | None = timeline_consistent.strip().lower() in {"true", "yes", "1"}
    elif isinstance(timeline_consistent, bool):
        timeline_consistent_norm = timeline_consistent
    else:
        timeline_consistent_norm = None

    final_correct = answer == expected
    frame_correct = predicted_frame == expected_frame if predicted_frame else None
    if record.get("mode") == "srt_spatial_v2" and predicted_frame in {"proximity", "mixed"}:
        frame_correct = None
    relation_correct = relation_holds_norm == (expected == "true") if relation_holds_norm is not None else None
    process_answer_consistent = None
    if relation_holds_norm is not None and answer in {"true", "false"}:
        process_answer_consistent = answer == ("true" if relation_holds_norm else "false")

    failures = []
    if record.get("error"):
        failures.append("api_or_runtime_error")
    if not answer:
        failures.append("answer_parse_failure")
    if predicted_frame and predicted_frame != expected_frame and frame_correct is not None:
        failures.append("spatial_frame_failure")
    if relation_holds_norm is not None and relation_holds_norm != (expected == "true"):
        failures.append("visual_relation_failure")
    if process_answer_consistent is False:
        failures.append("process_answer_inconsistent")
    if timeline_consistent_norm is False:
        failures.append("timeline_check_inconsistent")
    if not final_correct and not failures:
        failures.append("wrong_final_answer_other")

    return {
        "id": item["id"],
        "model": record.get("model"),
        "mode": record.get("mode"),
        "relation": relation,
        "expected_frame": expected_frame,
        "predicted_frame": predicted_frame,
        "expected_answer": expected,
        "answer": answer,
        "final_correct": final_correct,
        "frame_correct": frame_correct,
        "relation_correct": relation_correct,
        "process_answer_consistent": process_answer_consistent,
        "timeline_consistent": timeline_consistent_norm,
        "failure_types": ";".join(failures) if failures else "ok",
    }


def pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.1f}%"


def mean_bool(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), bool)]
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="runs/vsr_srt_summary")
    args = parser.parse_args()

    items = {row["id"]: row for row in load_jsonl(Path(args.items))}
    per_item = []
    for run in args.runs:
        records_by_id: dict[str, dict[str, Any]] = {}
        for record in load_jsonl(Path(run)):
            record_id = str(record.get("id"))
            previous = records_by_id.get(record_id)
            if previous is None or previous.get("error") or not record.get("error"):
                records_by_id[record_id] = record
        for record in records_by_id.values():
            item = items.get(str(record.get("id")))
            if item:
                per_item.append(classify(item, record))

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_item:
        grouped[(str(row["model"]), str(row["mode"]))].append(row)

    summary = []
    for (model, mode), rows in sorted(grouped.items()):
        n = len(rows)
        final_acc = mean_bool(rows, "final_correct")
        frame_acc = mean_bool(rows, "frame_correct")
        relation_acc = mean_bool(rows, "relation_correct")
        consistency = mean_bool(rows, "process_answer_consistent")
        timeline = mean_bool(rows, "timeline_consistent")
        srt_violation = sum(1 for row in rows if row["failure_types"] != "ok") / n if n else 0.0
        summary.append(
            {
                "model": model,
                "mode": mode,
                "n": n,
                "final_accuracy": final_acc,
                "frame_accuracy": frame_acc,
                "relation_accuracy": relation_acc,
                "process_answer_consistency": consistency,
                "timeline_consistency": timeline,
                "srt_violation_rate": srt_violation,
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "vsr_srt_per_item.csv", per_item)
    write_csv(out_dir / "vsr_srt_condition_summary.csv", summary)
    (out_dir / "vsr_srt_per_item.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in per_item),
        encoding="utf-8",
    )
    (out_dir / "vsr_srt_condition_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# VSR SRT Summary", ""]
    md.append("| model | mode | n | final acc. | frame acc. | relation acc. | process-answer consistency | timeline consistency | SRT violation |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary:
        md.append(
            "| {model} | `{mode}` | {n} | {fa} | {fra} | {ra} | {pac} | {tc} | {sv} |".format(
                model=row["model"],
                mode=row["mode"],
                n=row["n"],
                fa=pct(row["final_accuracy"]),
                fra=pct(row["frame_accuracy"]),
                ra=pct(row["relation_accuracy"]),
                pac=pct(row["process_answer_consistency"]),
                tc=pct(row["timeline_consistency"]),
                sv=pct(row["srt_violation_rate"]),
            )
        )
    (out_dir / "vsr_srt_condition_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {len(per_item)} per-item rows and {len(summary)} summary rows to {out_dir}")


if __name__ == "__main__":
    main()
