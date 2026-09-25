#!/usr/bin/env python3
"""Check valid/error coverage for Task3 CoT baseline reruns."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

FAMILIES = {
    "physical": ("task3_physical_v2refined", 60),
    "assembly": ("task3_assembly_v1reviewed", 60),
    "traffic": ("task3_traffic_v1reviewed", 60),
    "affordance": ("task3_affordance_v1reviewed", 60),
    "tooluse": ("task3_tooluse_v2focused52", 52),
    "cleaning": ("task3_cleaning_v1reviewed56", 56),
    "craft": ("task3_craft_v1corrected77", 77),
    "mobility": ("task3_mobility_v1reviewed106", 106),
}

MODELS = ["gpt54mini", "qwen25vl72b", "glm46v", "gemini31pro", "mistralsmall32"]
MODES = [
    "process_plain_cot",
    "process_multimodal_cot",
    "process_self_verification",
    "process_llava_cot_style",
    "process_grounded_cot",
]


def best_by_id(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        item_id = str(row.get("id"))
        current = rows.get(item_id)
        row_valid = bool(str(row.get("raw_text", "")).strip()) and not row.get("error")
        current_valid = (
            current is not None
            and bool(str(current.get("raw_text", "")).strip())
            and not current.get("error")
        )
        if current is None or row_valid or not current_valid:
            rows[item_id] = row
    return rows


def main() -> None:
    out_dir = ROOT / "results" / "summaries" / "task3_cot_baseline_control_2026-07-15"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for family, (tag, expected) in FAMILIES.items():
        for model in MODELS:
            for mode in MODES:
                path = ROOT / "results" / "raw" / f"{tag}_{model}_{mode}.jsonl"
                latest = best_by_id(path)
                valid = 0
                errors = 0
                empty = 0
                for row in latest.values():
                    raw = str(row.get("raw_text", "")).strip()
                    if row.get("error"):
                        errors += 1
                    if not raw:
                        empty += 1
                    if raw and not row.get("error"):
                        valid += 1
                need = max(expected - valid, 0)
                rows.append(
                    {
                        "family": family,
                        "model": model,
                        "mode": mode,
                        "expected": expected,
                        "latest_rows": len(latest),
                        "valid": valid,
                        "errors": errors,
                        "empty": empty,
                        "need_rerun": need,
                        "status": "complete" if need == 0 else "needs_rerun",
                        "output": str(path),
                    }
                )

    manifest = out_dir / "cot_validity_status.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    failed = [row for row in rows if row["need_rerun"]]
    print(
        json.dumps(
            {
                "jobs": len(rows),
                "complete_jobs": len(rows) - len(failed),
                "needs_rerun_jobs": len(failed),
                "missing_valid_samples": sum(int(row["need_rerun"]) for row in failed),
                "status_csv": str(manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
