#!/usr/bin/env python3
"""Evaluate the frozen SRT bank for a given model and split."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs"
DATA = BASE / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--default-run", required=True)
    parser.add_argument("--evidence-run", required=True)
    parser.add_argument("--verify-run", required=True)
    parser.add_argument("--vsr-rules-run", required=True)
    parser.add_argument("--proximity-strict-run", required=True)
    parser.add_argument("--enclosure-open-run", required=True)
    parser.add_argument("--opposition-shared-space-run", required=True)
    parser.add_argument("--facing-away-bearing-run", required=True)
    parser.add_argument("--policy-json", default=str(RUNS / "vsr500_dev_qwen_hierarchical_router_targeted_policy" / "router_policy_with_surrounding_opposite_facingaway_beneath.json"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(BASE / "evaluate_vsr_srt_router.py"),
        "--items",
        args.items,
        "--default-run",
        args.default_run,
        "--evidence-run",
        args.evidence_run,
        "--verify-run",
        args.verify_run,
        "--run",
        "vsr_rules",
        args.vsr_rules_run,
        "--run",
        "proximity_strict",
        args.proximity_strict_run,
        "--run",
        "enclosure_open",
        args.enclosure_open_run,
        "--run",
        "opposition_shared_space",
        args.opposition_shared_space_run,
        "--run",
        "facing_away_bearing",
        args.facing_away_bearing_run,
        "--policy-json",
        args.policy_json,
        "--output-dir",
        args.output_dir,
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
