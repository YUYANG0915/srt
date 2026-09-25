#!/usr/bin/env python3
"""Run the Task3 label-only oracle control across process families/models."""

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

OPENROUTER_MODELS = {
    "gpt54mini": "openai/gpt-5.4-mini",
    "qwen25vl72b": "qwen/qwen2.5-vl-72b-instruct",
    "glm46v": "z-ai/glm-4.6v",
    "gemini31pro": "google/gemini-3.1-pro-preview",
    "mistralsmall32": "mistralai/mistral-small-3.2-24b-instruct",
}


def run_openrouter(family: str, model_key: str, model: str, args: argparse.Namespace) -> None:
    item_file, tag = FAMILIES[family]
    output = PROJECT / "results" / "raw" / f"{tag}_{model_key}_process_boundary_label_only.jsonl"
    cmd = [
        sys.executable,
        str(ROOT / "run_openrouter_vlm.py"),
        "--items",
        str(PROJECT / "data" / item_file),
        "--image-dir",
        str(PROJECT / "data"),
        "--model",
        model,
        "--mode",
        "process_boundary_label_only",
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
    print(f"\n=== {family} / {model_key} / label-only ===", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=os.environ.copy())


def run_openai(family: str, args: argparse.Namespace) -> None:
    item_file, tag = FAMILIES[family]
    output = PROJECT / "results" / "raw" / f"{tag}_gpt54mini_process_boundary_label_only.jsonl"
    cmd = [
        sys.executable,
        str(ROOT / "run_openai_vlm.py"),
        "--items",
        str(PROJECT / "data" / item_file),
        "--image-dir",
        str(PROJECT / "data"),
        "--model",
        args.openai_model,
        "--mode",
        "process_boundary_label_only",
        "--output",
        str(output),
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--timeout",
        str(args.timeout),
        "--retries",
        str(args.retries),
        "--continue-on-error",
    ]
    print(f"\n=== {family} / gpt54mini / label-only ===", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=os.environ.copy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", choices=sorted(FAMILIES), default=sorted(FAMILIES))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(OPENROUTER_MODELS),
        default=sorted(OPENROUTER_MODELS),
    )
    parser.add_argument("--openai-model", default="gpt-5.4-mini")
    parser.add_argument("--max-output-tokens", type=int, default=350)
    parser.add_argument("--max-tokens", type=int, default=350)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-delay", type=float, default=0.0)
    parser.add_argument("--gpt-via-openrouter", action="store_true")
    args = parser.parse_args()

    for family in args.families:
        if "gpt54mini" in args.models and not args.gpt_via_openrouter:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set.")
            run_openai(family, args)
        if any(model in OPENROUTER_MODELS for model in args.models if model != "gpt54mini" or args.gpt_via_openrouter):
            if not os.environ.get("OPENROUTER_API_KEY"):
                raise RuntimeError("OPENROUTER_API_KEY is not set.")
            for model_key in args.models:
                if model_key == "gpt54mini" and not args.gpt_via_openrouter:
                    continue
                if model_key in OPENROUTER_MODELS:
                    run_openrouter(family, model_key, OPENROUTER_MODELS[model_key], args)


if __name__ == "__main__":
    main()
