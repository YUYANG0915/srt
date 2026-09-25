#!/usr/bin/env python3
"""Summarize VSR-500 S2 runs across models, splits, and learned routers."""

from __future__ import annotations

import csv
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs"
OUT_DIR = RUNS / "vsr500_multimodel_summary"

MODELS = {
    "openai_gpt54mini": "GPT-5.4-mini",
    "qwen25vl72b": "Qwen2.5-VL-72B",
    "gemini31pro": "Gemini-3.1-Pro",
    "glm46v": "GLM-4.6V",
    "mistralsmall32": "Mistral-Small-3.2-Vision",
}

SPLITS = {"dev100": 100, "test400": 400}
GLOBAL_MODES = ["default", "evidence", "verify"]
ROUTER_VARIANTS = ["learned_router_fixed", "hierarchical_router_r3_fixed"]
SPLIT_ALIASES = {"dev100": ["dev100", "dev"], "test400": ["test400", "test"]}
ALIASES = {
    "qwen25vl72b": ["qwen25vl72b", "qwen"],
    "openai_gpt54mini": ["openai_gpt54mini", "gpt54mini", "gpt"],
    "gemini31pro": ["gemini31pro", "gemini"],
    "glm46v": ["glm46v", "glm"],
    "mistralsmall32": ["mistralsmall32", "mistral"],
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_read_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    return read_json(path)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def global_summary_path(model_key: str, split: str) -> Path | None:
    aliases = ALIASES.get(model_key, [model_key])
    split_aliases = SPLIT_ALIASES.get(split, [split])
    candidates = [
        RUNS / f"vsr500_{split_alias}_{alias}_condition_summary" / "vsr_srt_condition_summary.json"
        for split_alias in split_aliases
        for alias in aliases
    ]
    if model_key == "qwen25vl72b":
        for split_alias in split_aliases:
            candidates.append(RUNS / f"vsr500_{split_alias}_qwen_condition_summary" / "vsr_srt_condition_summary.json")
    return first_existing(candidates)


def router_summary_path(model_key: str, split: str, variant: str) -> Path | None:
    aliases = ALIASES.get(model_key, [model_key])
    split_aliases = SPLIT_ALIASES.get(split, [split])
    candidates = [
        RUNS / f"vsr500_{split_alias}_{alias}_{variant}" / "vsr_srt_router_summary.json"
        for split_alias in split_aliases
        for alias in aliases
    ]
    if model_key == "qwen25vl72b":
        for split_alias in split_aliases:
            candidates.append(RUNS / f"vsr500_{split_alias}_qwen_{variant}" / "vsr_srt_router_summary.json")
    return first_existing(candidates)


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for model_key, model_name in MODELS.items():
        for split, expected_n in SPLITS.items():
            summary_path = global_summary_path(model_key, split)
            global_summary = maybe_read_summary(summary_path) if summary_path else None
            if global_summary:
                summary_items = global_summary if isinstance(global_summary, list) else [global_summary]
                for item in summary_items:
                    mode = str(item["mode"]).replace("srt_spatial_", "")
                    rows.append(
                        {
                            "model_key": model_key,
                            "model": model_name,
                            "split": split,
                            "method": mode,
                            "n": item.get("n", 0),
                            "expected_n": expected_n,
                            "complete": item.get("n", 0) == expected_n,
                            "final_accuracy": item.get("final_accuracy"),
                            "srt_violation_rate": item.get("srt_violation_rate"),
                        }
                    )
            for variant in ROUTER_VARIANTS:
                router_path = router_summary_path(model_key, split, variant)
                router_summary = maybe_read_summary(router_path) if router_path else None
                if router_summary:
                    rows.append(
                        {
                            "model_key": model_key,
                            "model": model_name,
                            "split": split,
                            "method": variant,
                            "n": router_summary.get("n", 0),
                            "expected_n": expected_n,
                            "complete": router_summary.get("n", 0) == expected_n,
                            "final_accuracy": router_summary.get("final_accuracy"),
                            "srt_violation_rate": router_summary.get("srt_violation_rate"),
                        }
                    )
    return rows


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    with (OUT_DIR / "condition_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    if rows:
        with (OUT_DIR / "condition_summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    md = ["# VSR-500 Multimodel Summary", ""]
    md.append("This is the main S2 summary table across models, splits, global SRT-base conditions, and learned routing variants.")
    for split, expected_n in SPLITS.items():
        md.extend(
            [
                "",
                f"## {split}",
                "",
                "| model | method | complete | n | final acc. | SRT violation |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        split_rows = [row for row in rows if row["split"] == split]
        order = {name: idx for idx, name in enumerate(GLOBAL_MODES + ROUTER_VARIANTS)}
        split_rows.sort(key=lambda row: (row["model"], order.get(row["method"], 99)))
        for row in split_rows:
            md.append(
                "| {model} | {method} | {complete} | {n}/{expected_n} | {acc} | {viol} |".format(
                    model=row["model"],
                    method=row["method"],
                    complete="yes" if row["complete"] else "no",
                    n=row["n"],
                    expected_n=row["expected_n"],
                    acc=pct(row["final_accuracy"]),
                    viol=pct(row["srt_violation_rate"]),
                )
            )

    md.extend(
        [
            "",
            "Reading guide:",
            "- `default`, `evidence`, and `verify` are the three global SRT-base conditions.",
            "- `learned_router_fixed` is the dev-learned family router.",
            "- `hierarchical_router_r3_fixed` is the family router plus relation-level overrides with min support 3.",
            "- The paper claim for S2 should compare learned routing against the best single global condition, not only against `default`.",
        ]
    )
    (OUT_DIR / "condition_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
