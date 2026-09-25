#!/usr/bin/env python3
"""Evaluate sports-action boundary routers against existing answer runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_FAMILIES = {
    "setup_vs_execution",
    "execution_vs_followthrough",
    "action_vs_nonaction",
    "equipment_context",
    "other",
}


POLICIES = {
    "hand": {
        "setup_vs_execution": "default",
        "execution_vs_followthrough": "boundary_verify",
        "action_vs_nonaction": "default",
        "equipment_context": "phase_prior",
        "other": "default",
        # Legacy/noisy labels from early seed rows.
        "pre_action_setup": "visible_boundary_srt",
        "": "default",
    },
    "dev_best": {
        "setup_vs_execution": "default",
        "execution_vs_followthrough": "generic_srt",
        "action_vs_nonaction": "phase_prior",
        "equipment_context": "visible_boundary_srt",
        "other": "default",
        # Legacy/noisy labels from early seed rows.
        "pre_action_setup": "visible_boundary_srt",
        "": "default",
    },
}

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


def normalize_family(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    if text in VALID_FAMILIES:
        return text
    for family in VALID_FAMILIES:
        if family in text:
            return family
    return "other"


def parse_phase(text: str) -> str:
    parsed = parse_jsonish(text)
    phase = parsed.get("action_phase") or parsed.get("phase") or parsed.get("answer")
    if phase:
        return str(phase)
    match = re.search(
        r"(pre_action_setup|active_execution|follow_through_or_result|resting_or_non_action)",
        str(text or ""),
    )
    return match.group(1) if match else ""


def accuracy(rows: list[dict[str, Any]]) -> tuple[int, int, float]:
    correct = sum(1 for row in rows if row["ok"])
    total = len(rows)
    return correct, total, correct / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--router-run", required=True)
    parser.add_argument("--default-run", required=True)
    parser.add_argument("--generic-run", required=True)
    parser.add_argument("--phase-prior-run", required=True)
    parser.add_argument("--visible-boundary-run", required=True)
    parser.add_argument("--boundary-verify-run", required=True)
    parser.add_argument("--policy", choices=sorted(POLICIES), default="hand")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    policy = POLICIES[args.policy]

    items = {row["id"]: row for row in load_jsonl(Path(args.items))}
    router_rows = {row["id"]: row for row in load_jsonl(Path(args.router_run))}
    answer_runs = {
        "default": load_jsonl(Path(args.default_run)),
        "generic_srt": load_jsonl(Path(args.generic_run)),
        "phase_prior": load_jsonl(Path(args.phase_prior_run)),
        "visible_boundary_srt": load_jsonl(Path(args.visible_boundary_run)),
        "boundary_verify": load_jsonl(Path(args.boundary_verify_run)),
    }
    answer_by_mode: dict[str, dict[str, dict[str, Any]]] = {
        mode: {row["id"]: row for row in rows} for mode, rows in answer_runs.items()
    }

    rows: list[dict[str, Any]] = []
    for item_id, item in items.items():
        router_text = str(router_rows.get(item_id, {}).get("raw_text", ""))
        router_json = parse_jsonish(router_text)
        predicted_family = normalize_family(router_json.get("boundary_family"))
        gold_family = str(item.get("phase_boundary_family") or item.get("boundary_family") or "")
        selected_mode = policy.get(predicted_family, "default")
        selected_answer = answer_by_mode[selected_mode][item_id]
        pred_phase = parse_phase(str(selected_answer.get("raw_text", "")))
        gold_phase = str(item.get("expected_answer", ""))
        rows.append(
            {
                "id": item_id,
                "image_id": item.get("image_id", ""),
                "sport_hint": item.get("sport_hint", ""),
                "gold_family": gold_family,
                "predicted_family": predicted_family,
                "router_family_ok": normalize_family(gold_family) == predicted_family,
                "selected_mode": selected_mode,
                "gold_phase": gold_phase,
                "pred_phase": pred_phase,
                "ok": pred_phase == gold_phase,
                "router_rationale": router_json.get("rationale", ""),
            }
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    family_correct = sum(1 for row in rows if row["router_family_ok"])
    correct, total, acc = accuracy(rows)
    selected_counter = Counter(row["selected_mode"] for row in rows)
    pred_family_counter = Counter(row["predicted_family"] for row in rows)

    by_gold_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_gold_family[row["gold_family"]].append(row)

    lines = ["# Task3 Sports Automatic Router Summary", ""]
    lines.append(f"- Routing policy: `{args.policy}`")
    lines.append(f"- Router family accuracy: {family_correct} / {total} = {family_correct / total * 100:.1f}%")
    lines.append(f"- Routed action accuracy: {correct} / {total} = {acc * 100:.1f}%")
    lines.append("")
    lines.append("## Selected Modes")
    lines.append("")
    for mode, count in selected_counter.most_common():
        lines.append(f"- `{mode}`: {count}")
    lines.append("")
    lines.append("## Predicted Families")
    lines.append("")
    for family, count in pred_family_counter.most_common():
        lines.append(f"- `{family}`: {count}")
    lines.append("")
    lines.append("## By Gold Family")
    lines.append("")
    lines.append("| Gold family | Correct | Accuracy |")
    lines.append("|---|---:|---:|")
    for family, family_rows in sorted(by_gold_family.items()):
        fam_correct, fam_total, fam_acc = accuracy(family_rows)
        label = family or "(blank)"
        lines.append(f"| `{label}` | {fam_correct} / {fam_total} | {fam_acc * 100:.1f}% |")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
