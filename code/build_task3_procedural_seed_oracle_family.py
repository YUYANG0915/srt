#!/usr/bin/env python3
"""Attach manual boundary-family hints to the procedural seed items."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FAMILY_MAP = {
    "task3cook_seed_0001_258883": "active_heat_vs_ready_result",
    "task3cook_seed_0002_309678": "assembly_vs_serving",
    "task3cook_seed_0003_330818": "active_heat_vs_ready_result",
    "task3cook_seed_0004_346703": "assembly_vs_serving",
    "task3cook_seed_0005_402433": "active_heat_vs_ready_result",
    "task3cook_seed_0006_470773": "assembly_vs_serving",
    "task3cook_seed_0007_279541": "assembly_vs_serving",
    "task3cook_seed_0008_226903": "assembly_vs_serving",
    "task3cook_seed_0009_63965": "assembly_vs_serving",
    "task3cook_seed_0010_353027": "assembly_vs_serving",
}


def attach(src: Path, dst: Path) -> None:
    rows = []
    for line in src.open("r", encoding="utf-8"):
        obj = json.loads(line)
        obj["boundary_family_hint"] = FAMILY_MAP.get(obj["id"], "other")
        rows.append(obj)
    dst.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    attach(
        ROOT / "data" / "task3_procedural_cooking_seed_clean_items.jsonl",
        ROOT / "data" / "task3_procedural_cooking_seed_clean_items_oracle_family.jsonl",
    )
    attach(
        ROOT / "data" / "task3_procedural_cooking_seed_marked_items.jsonl",
        ROOT / "data" / "task3_procedural_cooking_seed_marked_items_oracle_family.jsonl",
    )
    print("Wrote oracle-family seed item files.")


if __name__ == "__main__":
    main()
