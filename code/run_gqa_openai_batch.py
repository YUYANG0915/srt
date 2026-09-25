#!/usr/bin/env python3
"""Run GQA referent-disambiguation batches through the OpenAI Responses API."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
MODES = ["default", "evidence", "marked_no_prior", "visual_sketchpad"]

TASKS = {
    "gqa_task1": {
        "n": 124,
        "base_items": BASE / "data/gqa_task1_ambiguous_clean_items.jsonl",
        "marked_items": BASE / "data/gqa_task1_ambiguous_marked_items.jsonl",
        "image_dir": BASE / "data/gqa_images",
        "marked_image_dir": BASE / "data/gqa_task1_marked_images",
    },
    "gqa_task2": {
        "n": 96,
        "base_items": BASE / "data/gqa_task2_unique_clean_items.jsonl",
        "marked_items": BASE / "data/gqa_task2_unique_marked_items.jsonl",
        "image_dir": BASE / "data/gqa_images",
        "marked_image_dir": BASE / "data/gqa_task2_marked_images",
    },
}

MAX_TOKENS = {
    "default": 700,
    "evidence": 1200,
    "marked_no_prior": 1000,
    "visual_sketchpad": 2200,
}


def items_for(task: str, mode: str) -> Path:
    if mode in {"marked_no_prior", "visual_sketchpad"}:
        return TASKS[task]["marked_items"]
    return TASKS[task]["base_items"]


def image_dir_for(task: str, mode: str) -> Path:
    if mode in {"marked_no_prior", "visual_sketchpad"}:
        return TASKS[task]["marked_image_dir"]
    return TASKS[task]["image_dir"]


def output_path(task: str, mode: str, limit: int | None) -> Path:
    n = TASKS[task]["n"]
    suffix = f"limit{limit}" if limit else "full"
    return BASE / "runs" / f"{task}_clean{n}_openai_gpt54mini_{mode}_{suffix}.jsonl"


def run_one(task: str, mode: str, limit: int | None, model: str, retries: int) -> None:
    output = output_path(task, mode, limit)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_openai_vlm.py"),
        "--items",
        str(items_for(task, mode)),
        "--image-dir",
        str(image_dir_for(task, mode)),
        "--model",
        model,
        "--mode",
        mode,
        "--output",
        str(output),
        "--max-output-tokens",
        str(MAX_TOKENS[mode]),
        "--retries",
        str(retries),
        "--timeout",
        "180",
        "--continue-on-error",
        "--error-log",
        str(output.with_name(output.stem + "_errors.jsonl")),
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
    print("RUN", task, "openai_gpt54mini", mode, "->", output, flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["gqa_task1", "gqa_task2"])
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    for task in args.tasks:
        for mode in args.modes:
            run_one(task, mode, args.limit, args.model, args.retries)


if __name__ == "__main__":
    main()
