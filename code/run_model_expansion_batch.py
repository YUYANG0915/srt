#!/usr/bin/env python3
"""Run the next model-expansion batch through OpenRouter."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]

MODELS = {
    "glm46v": {
        "id": "z-ai/glm-4.6v",
        "max_tokens": {"marked_no_prior": 1000, "visual_sketchpad": 2200, "default": 500, "evidence": 1200},
        "reasoning_effort": "none",
    },
    "mistralsmall32": {
        "id": "mistralai/mistral-small-3.2-24b-instruct",
        "max_tokens": {"marked_no_prior": 900, "visual_sketchpad": 1800, "default": 700, "evidence": 1100},
        "reasoning_effort": "none",
    },
    "gemini31pro": {
        "id": "google/gemini-3.1-pro-preview",
        "max_tokens": {"marked_no_prior": 900, "visual_sketchpad": 1600, "default": 700, "evidence": 1100},
        "reasoning_effort": "low",
    },
}

TASKS = {
    "task1": {
        "n": 457,
        "items": {
            "default": BASE / "data/task1_ambiguous_auto_clean_items.jsonl",
            "evidence": BASE / "data/task1_ambiguous_auto_clean_items.jsonl",
            "marked_no_prior": BASE / "data/task1_ambiguous_auto_clean_marked_items.jsonl",
            "visual_sketchpad": BASE / "data/task1_ambiguous_auto_clean_marked_items.jsonl",
        },
        "image_dir": {
            "default": BASE / "data/coco_val2017_task1_raw500_images",
            "evidence": BASE / "data/coco_val2017_task1_raw500_images",
            "marked_no_prior": BASE / "data/coco_val2017_task1_marked_images",
            "visual_sketchpad": BASE / "data/coco_val2017_task1_marked_images",
        },
    },
    "task2": {
        "n": 330,
        "items": {
            "default": BASE / "data/task2_unique_auto_clean_items.jsonl",
            "evidence": BASE / "data/task2_unique_auto_clean_items.jsonl",
            "marked_no_prior": BASE / "data/task2_unique_auto_clean_marked_items.jsonl",
            "visual_sketchpad": BASE / "data/task2_unique_auto_clean_marked_items.jsonl",
        },
        "image_dir": {
            "default": BASE / "data/coco_val2017_task2_raw500_images",
            "evidence": BASE / "data/coco_val2017_task2_raw500_images",
            "marked_no_prior": BASE / "data/coco_val2017_task2_marked_images",
            "visual_sketchpad": BASE / "data/coco_val2017_task2_marked_images",
        },
    },
}


def output_path(task: str, model_key: str, mode: str, limit: int | None) -> Path:
    n = TASKS[task]["n"]
    suffix = f"limit{limit}" if limit else "full"
    return BASE / "runs" / f"{task}_auto_clean{n}_{model_key}_{mode}_{suffix}.jsonl"


def run_one(task: str, model_key: str, mode: str, limit: int | None) -> None:
    model = MODELS[model_key]
    output = output_path(task, model_key, mode, limit)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_openrouter_vlm.py"),
        "--items",
        str(TASKS[task]["items"][mode]),
        "--image-dir",
        str(TASKS[task]["image_dir"][mode]),
        "--model",
        model["id"],
        "--mode",
        mode,
        "--output",
        str(output),
        "--max-tokens",
        str(model["max_tokens"][mode]),
        "--reasoning-effort",
        model["reasoning_effort"],
        "--retries",
        "5",
        "--timeout",
        "180",
        "--continue-on-error",
        "--error-log",
        str(output.with_name(output.stem + "_errors.jsonl")),
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
    print("RUN", task, model_key, mode, "->", output, flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--tasks", nargs="+", default=["task1", "task2"])
    parser.add_argument("--modes", nargs="+", default=["marked_no_prior", "visual_sketchpad"])
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    for model_key in args.models:
        for task in args.tasks:
            for mode in args.modes:
                run_one(task, model_key, mode, args.limit)


if __name__ == "__main__":
    main()
