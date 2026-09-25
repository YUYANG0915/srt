#!/usr/bin/env python3
"""Run generic SRT process-family batches through OpenAI Responses API."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
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


def run_one(family: str, mode: str, args: argparse.Namespace) -> None:
    items = ROOT / "data" / args.items_template.format(family=family)
    image_dir = ROOT / "data" / f"task3_{family}_coco_images"
    output = ROOT / "runs" / f"task3_{family}_{args.output_tag}_gpt54mini_{mode}.jsonl"
    cmd = [
        sys.executable,
        str(ROOT / "run_openai_vlm.py"),
        "--items",
        str(items),
        "--image-dir",
        str(image_dir),
        "--model",
        args.model,
        "--mode",
        mode,
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
    print(f"\n=== {family} / gpt54mini / {mode} ===", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=os.environ.copy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", default=["assembly", "traffic", "affordance"])
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--items-template", default="task3_{family}_eval_v1_reviewed.jsonl")
    parser.add_argument("--output-tag", default="v1reviewed")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--max-output-tokens", type=int, default=650)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    for family in args.families:
        for mode in args.modes:
            run_one(family, mode, args)


if __name__ == "__main__":
    main()
