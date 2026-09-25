#!/usr/bin/env python3
"""Run full Task3 CoT-style baselines across process families/models."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent

FAMILIES = {
    "physical": ("task3_physical_eval_v2_refined.jsonl", "task3_physical_v2refined"),
    "assembly": ("task3_assembly_eval_v1_reviewed.jsonl", "task3_assembly_v1reviewed"),
    "traffic": ("task3_traffic_eval_v1_reviewed.jsonl", "task3_traffic_v1reviewed"),
    "affordance": ("task3_affordance_eval_v1_reviewed.jsonl", "task3_affordance_v1reviewed"),
    "tooluse": ("task3_tooluse_eval_v2_focused_corrected52.jsonl", "task3_tooluse_v2focused52"),
    "cleaning": ("task3_cleaning_eval_v1_reviewed.jsonl", "task3_cleaning_v1reviewed56"),
    "craft": ("task3_craft_eval_v1_corrected.jsonl", "task3_craft_v1corrected77"),
    "mobility": ("task3_mobility_eval_v1_reviewed.jsonl", "task3_mobility_v1reviewed106"),
}

MODELS = {
    "gpt54mini": "openai/gpt-5.4-mini",
    "qwen25vl72b": "qwen/qwen2.5-vl-72b-instruct",
    "glm46v": "z-ai/glm-4.6v",
    "gemini31pro": "google/gemini-3.1-pro-preview",
    "mistralsmall32": "mistralai/mistral-small-3.2-24b-instruct",
}

MODES = [
    "process_plain_cot",
    "process_multimodal_cot",
    "process_self_verification",
    "process_llava_cot_style",
    "process_grounded_cot",
]


def run_one(family: str, model_key: str, mode: str, args: argparse.Namespace) -> None:
    item_file, tag = FAMILIES[family]
    output = PROJECT / "results" / "raw" / f"{tag}_{model_key}_{mode}.jsonl"
    cmd = [
        sys.executable,
        str(ROOT / "run_openrouter_vlm.py"),
        "--items",
        str(PROJECT / "data" / item_file),
        "--image-dir",
        str(PROJECT / "data"),
        "--model",
        MODELS[model_key],
        "--mode",
        mode,
        "--output",
        str(output),
        "--temperature",
        "0",
        "--max-tokens",
        str(args.max_tokens),
        "--timeout",
        str(args.timeout),
        "--retries",
        str(args.retries),
        "--request-delay",
        str(args.request_delay),
        "--continue-on-error",
    ]
    print(f"\n=== {family} / {model_key} / {mode} ===", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=os.environ.copy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", choices=sorted(FAMILIES), default=sorted(FAMILIES))
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=sorted(MODELS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-delay", type=float, default=0.2)
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    for family in args.families:
        for model_key in args.models:
            for mode in args.modes:
                run_one(family, model_key, mode, args)


if __name__ == "__main__":
    main()
