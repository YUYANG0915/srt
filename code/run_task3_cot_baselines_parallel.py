#!/usr/bin/env python3
"""Run Task3 CoT baselines with one subprocess per family/model/mode job."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_task3_cot_baselines_all import FAMILIES, MODELS, MODES


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def make_cmd(family: str, model_key: str, mode: str, args: argparse.Namespace) -> list[str]:
    item_file, tag = FAMILIES[family]
    output = PROJECT / "results" / "raw" / f"{tag}_{model_key}_{mode}.jsonl"
    return [
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


def run_job(job: tuple[str, str, str], args: argparse.Namespace) -> tuple[tuple[str, str, str], int, str]:
    family, model_key, mode = job
    cmd = make_cmd(family, model_key, mode, args)
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    wrote = sum(1 for line in proc.stdout.splitlines() if " wrote " in line)
    tail = "\n".join(proc.stdout.splitlines()[-6:])
    return job, proc.returncode, f"wrote={wrote}\n{tail}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+", choices=sorted(FAMILIES), default=sorted(FAMILIES))
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=sorted(MODELS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-delay", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=500)
    args = parser.parse_args()

    jobs = [(family, model, mode) for family in args.families for model in args.models for mode in args.modes]
    print(f"Running {len(jobs)} jobs with workers={args.workers}", flush=True)
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, job, args): job for job in jobs}
        for future in as_completed(futures):
            job, returncode, summary = future.result()
            family, model_key, mode = job
            status = "OK" if returncode == 0 else f"FAIL({returncode})"
            if returncode != 0:
                failures += 1
            print(f"\n[{status}] {family}/{model_key}/{mode}\n{summary}", flush=True)
    if failures:
        raise SystemExit(f"{failures} jobs failed")


if __name__ == "__main__":
    main()
