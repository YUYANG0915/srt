#!/usr/bin/env python3
"""Run generic SRT process-family batches through OpenRouter."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

MODELS = {
    "qwen25vl72b": "qwen/qwen2.5-vl-72b-instruct",
    "glm46v": "z-ai/glm-4.6v",
    "gemini31pro": "google/gemini-3.1-pro-preview",
    "mistralsmall32": "mistralai/mistral-small-3.2-24b-instruct",
}

MODES = [
    "process_default",
    "process_visual_evidence_only",
    "process_plain_cot",
    "process_multimodal_cot",
    "process_self_verification",
    "process_srt_policy_only",
    "process_srt_s_t",
    "process_srt_r_t",
    "process_srt_s_r_no_t",
    "process_srt_generic",
    "process_srt_scenario",
    "process_srt_wrong_boundary_router",
    "process_srt_self_router",
    "process_srt_boundary_router_oracle",
    "process_boundary_label_only",
]


def run_one(family: str, model_key: str, mode: str, args: argparse.Namespace) -> None:
    model = MODELS[model_key]
    items = ROOT / "data" / args.items_template.format(family=family)
    image_dir = ROOT / "data" / f"task3_{family}_coco_images"
    output = ROOT / "runs" / f"task3_{family}_{args.output_tag}_{model_key}_{mode}.jsonl"
    cmd = [
        sys.executable,
        str(ROOT / "run_openrouter_vlm.py"),
        "--items",
        str(items),
        "--image-dir",
        str(image_dir),
        "--model",
        model,
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
        "--continue-on-error",
    ]
    print(f"\n=== {family} / {model_key} / {mode} ===", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=os.environ.copy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", default=["assembly", "traffic", "affordance"])
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), required=True)
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--items-template", default="task3_{family}_eval_v1_reviewed.jsonl")
    parser.add_argument("--output-tag", default="v1reviewed")
    parser.add_argument("--max-tokens", type=int, default=650)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    for family in args.families:
        for model_key in args.models:
            for mode in args.modes:
                run_one(family, model_key, mode, args)


if __name__ == "__main__":
    main()
