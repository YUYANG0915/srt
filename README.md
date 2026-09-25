# Anonymous Code Release: VPAC-Bench and SRT

This repository contains the code used to construct and evaluate the experiments reported in the accompanying anonymous submission. It covers the three experimental stages: underspecified visual question answering (S1), spatial-relation transfer (S2), and process-grounded visual reasoning with VPAC-Bench (S3).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

API-backed experiments read credentials only from environment variables:

```bash
export OPENAI_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

## Code Map

- `code/prompts.py`: all SRT, boundary-aligned SRT, label-only, self-router, and chain-of-thought baseline prompts.
- `code/run_openai_vlm.py` and `code/run_openrouter_vlm.py`: provider-specific model runners.
- `code/run_gqa_*` and `code/build_coco_*`: S1 construction and evaluation.
- `code/build_vsr_*`, `code/run_vsr*`, `code/evaluate_vsr_*`, and `code/learn_vsr_*`: S2 construction, evaluation, and routing.
- `code/build_task3_*`: VPAC-Bench candidate construction and review-processing utilities.
- `code/run_task3_process_*`: S3 default, generic SRT, scenario SRT, and boundary-aligned SRT runs.
- `code/run_task3_cot_*`: chain-of-thought baseline runs.
- `code/run_task3_label_only_all.py`: label-only controls.
- `code/summarize_*` and `code/analyze_*`: aggregation, confidence intervals, paired comparisons, ablations, and figure/table inputs.