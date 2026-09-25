#!/usr/bin/env python3
"""Run GQA referent-disambiguation batches through OpenRouter."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]

MODELS = {
    "qwen25vl72b": {
        "id": "qwen/qwen2.5-vl-72b-instruct",
        "max_tokens": {"default": 700, "evidence": 1200, "marked_no_prior": 1000, "visual_sketchpad": 2200},
        "reasoning_effort": "none",
    },
    "glm46v": {
        "id": "z-ai/glm-4.6v",
        "max_tokens": {"default": 500, "evidence": 1200, "marked_no_prior": 1000, "visual_sketchpad": 2200},
        "reasoning_effort": "none",
    },
    "mistralsmall32": {
        "id": "mistralai/mistral-small-3.2-24b-instruct",
        "max_tokens": {"default": 700, "evidence": 1100, "marked_no_prior": 900, "visual_sketchpad": 1800},
        "reasoning_effort": "none",
    },
    "gemini31pro": {
        "id": "google/gemini-3.1-pro-preview",
        "max_tokens": {"default": 700, "evidence": 1100, "marked_no_prior": 900, "visual_sketchpad": 1600},
        "reasoning_effort": "low",
    },
}

TASKS = {
    "gqa_task1": {
        "base_items": BASE / "data/gqa_task1_ambiguous_clean_items.jsonl",
        "marked_items": BASE / "data/gqa_task1_ambiguous_marked_items.jsonl",
        "image_dir_env": "GQA_IMAGE_DIR",
        "marked_image_dir": BASE / "data/gqa_task1_marked_images",
    },
    "gqa_task2": {
        "base_items": BASE / "data/gqa_task2_unique_clean_items.jsonl",
        "marked_items": BASE / "data/gqa_task2_unique_marked_items.jsonl",
        "image_dir_env": "GQA_IMAGE_DIR",
        "marked_image_dir": BASE / "data/gqa_task2_marked_images",
    },
}

MODES = ["default", "evidence", "marked_no_prior", "visual_sketchpad"]


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def output_path(task: str, model_key: str, mode: str, limit: int | None) -> Path:
    n = count_jsonl(TASKS[task]["base_items"])
    suffix = f"limit{limit}" if limit else "full"
    return BASE / "runs" / f"{task}_clean{n}_{model_key}_{mode}_{suffix}.jsonl"


def items_for(task: str, mode: str) -> Path:
    if mode in {"marked_no_prior", "visual_sketchpad"}:
        return TASKS[task]["marked_items"]
    return TASKS[task]["base_items"]


def image_dir_for(task: str, mode: str, gqa_image_dir: str) -> Path:
    if mode in {"marked_no_prior", "visual_sketchpad"}:
        return TASKS[task]["marked_image_dir"]
    return Path(gqa_image_dir)


def run_one(task: str, model_key: str, mode: str, limit: int | None, gqa_image_dir: str) -> None:
    model = MODELS[model_key]
    items = items_for(task, mode)
    image_dir = image_dir_for(task, mode, gqa_image_dir)
    output = output_path(task, model_key, mode, limit)
    if not items.exists():
        raise FileNotFoundError(f"Missing items file for {task}/{mode}: {items}")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_openrouter_vlm.py"),
        "--items",
        str(items),
        "--image-dir",
        str(image_dir),
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
    parser.add_argument("--models", nargs="+", default=["gemini31pro"])
    parser.add_argument("--tasks", nargs="+", default=["gqa_task1", "gqa_task2"])
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    gqa_image_dir = os.environ.get("GQA_IMAGE_DIR")
    if not gqa_image_dir:
        raise RuntimeError("GQA_IMAGE_DIR is not set.")
    for model_key in args.models:
        for task in args.tasks:
            for mode in args.modes:
                run_one(task, model_key, mode, args.limit, gqa_image_dir)


if __name__ == "__main__":
    main()
