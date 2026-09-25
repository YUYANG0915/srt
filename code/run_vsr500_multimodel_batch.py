#!/usr/bin/env python3
"""Run VSR-500 S2 batches across models and splits."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parent

OPENROUTER_MODELS = {
    "qwen25vl72b": {
        "id": "qwen/qwen2.5-vl-72b-instruct",
        "max_tokens": {"default": 700, "evidence": 700, "verify": 700},
        "reasoning_effort": "none",
        "request_delay": "5",
    },
    "gemini31pro": {
        "id": "google/gemini-3.1-pro-preview",
        "max_tokens": {"default": 700, "evidence": 700, "verify": 700},
        "reasoning_effort": "low",
        "request_delay": "2",
    },
    "glm46v": {
        "id": "z-ai/glm-4.6v",
        "max_tokens": {"default": 700, "evidence": 700, "verify": 700},
        "reasoning_effort": "none",
        "request_delay": "2",
    },
    "mistralsmall32": {
        "id": "mistralai/mistral-small-3.2-24b-instruct",
        "max_tokens": {"default": 700, "evidence": 700, "verify": 700},
        "reasoning_effort": "none",
        "request_delay": "2",
    },
}

OPENAI_MODELS = {
    "openai_gpt54mini": {
        "id": "gpt-5.4-mini",
        "max_output_tokens": {"default": 900, "evidence": 900, "verify": 900},
    },
}

SPLITS = {
    "dev100": BASE / "data/vsr_srt_spatial_items_500_dev100.jsonl",
    "test400": BASE / "data/vsr_srt_spatial_items_500_test400.jsonl",
}

MODES = {
    "default": "srt_spatial_default",
    "evidence": "srt_spatial_evidence",
    "verify": "srt_spatial_verify",
}

IMAGE_DIR = BASE / "data/vsr_images"


def provider_for(model_key: str) -> str:
    if model_key in OPENAI_MODELS:
        return "openai"
    if model_key in OPENROUTER_MODELS:
        return "openrouter"
    raise KeyError(f"Unknown model key: {model_key}")


def output_path(model_key: str, split: str, mode: str) -> Path:
    return BASE / "runs" / f"vsr500_{split}_{model_key}_{MODES[mode]}.jsonl"


def run_openrouter(model_key: str, split: str, mode: str) -> None:
    model = OPENROUTER_MODELS[model_key]
    output = output_path(model_key, split, mode)
    cmd = [
        sys.executable,
        str(BASE / "run_openrouter_vlm.py"),
        "--items",
        str(SPLITS[split]),
        "--image-dir",
        str(IMAGE_DIR),
        "--model",
        model["id"],
        "--mode",
        MODES[mode],
        "--output",
        str(output),
        "--max-tokens",
        str(model["max_tokens"][mode]),
        "--reasoning-effort",
        model["reasoning_effort"],
        "--request-delay",
        model["request_delay"],
        "--retries",
        "5",
        "--timeout",
        "180",
        "--continue-on-error",
        "--error-log",
        str(output.with_name(output.stem + "_errors.jsonl")),
    ]
    print("RUN", split, model_key, mode, "->", output, flush=True)
    subprocess.run(cmd, check=True)


def run_openai(model_key: str, split: str, mode: str) -> None:
    model = OPENAI_MODELS[model_key]
    output = output_path(model_key, split, mode)
    cmd = [
        sys.executable,
        str(BASE / "run_openai_vlm.py"),
        "--items",
        str(SPLITS[split]),
        "--image-dir",
        str(IMAGE_DIR),
        "--model",
        model["id"],
        "--mode",
        MODES[mode],
        "--output",
        str(output),
        "--max-output-tokens",
        str(model["max_output_tokens"][mode]),
        "--retries",
        "5",
        "--timeout",
        "180",
        "--continue-on-error",
        "--error-log",
        str(output.with_name(output.stem + "_errors.jsonl")),
    ]
    print("RUN", split, model_key, mode, "->", output, flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["openai_gpt54mini", "qwen25vl72b", "gemini31pro", "glm46v", "mistralsmall32"])
    parser.add_argument("--splits", nargs="+", default=["dev100", "test400"])
    parser.add_argument("--modes", nargs="+", default=["default", "evidence", "verify"])
    args = parser.parse_args()

    if any(provider_for(model_key) == "openai" for model_key in args.models) and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    if any(provider_for(model_key) == "openrouter" for model_key in args.models) and not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    for model_key in args.models:
        for split in args.splits:
            for mode in args.modes:
                if provider_for(model_key) == "openai":
                    run_openai(model_key, split, mode)
                else:
                    run_openrouter(model_key, split, mode)


if __name__ == "__main__":
    main()
