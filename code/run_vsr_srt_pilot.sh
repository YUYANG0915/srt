#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL="${MODEL:-gpt-5.4-mini}"
PROVIDER="${PROVIDER:-openai}"
LIMIT="${LIMIT:-120}"
PYTHON="${PYTHON:-python3}"
ITEMS="${ITEMS:-data/vsr_srt_spatial_items_120.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-data/vsr_images}"
RUN_PREFIX="${RUN_PREFIX:-vsr_srt_${MODEL//[^A-Za-z0-9]/_}}"

if [[ "$PROVIDER" == "openai" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not set." >&2
    exit 1
  fi
  RUNNER="run_openai_vlm.py"
  TOKEN_ARG=(--max-output-tokens 1200)
elif [[ "$PROVIDER" == "openrouter" ]]; then
  if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "OPENROUTER_API_KEY is not set." >&2
    exit 1
  fi
  RUNNER="run_openrouter_vlm.py"
  REASONING_EFFORT="${REASONING_EFFORT:-none}"
  TOKEN_ARG=(--max-tokens 1200 --reasoning-effort "$REASONING_EFFORT")
else
  echo "Unknown PROVIDER=$PROVIDER. Use openai or openrouter." >&2
  exit 1
fi

run_condition() {
  local mode="$1"
  local suffix="$2"
  echo "Running $MODEL / $mode"
  "$PYTHON" "$RUNNER" \
    --items "$ITEMS" \
    --image-dir "$IMAGE_DIR" \
    --model "$MODEL" \
    --mode "$mode" \
    --output "runs/${RUN_PREFIX}_${suffix}.jsonl" \
    --limit "$LIMIT" \
    "${TOKEN_ARG[@]}" \
    --continue-on-error \
    --error-log "runs/${RUN_PREFIX}_${suffix}_errors.jsonl"
}

run_condition srt_spatial_default default
run_condition srt_spatial_evidence evidence
run_condition srt_spatial srt_spatial

"$PYTHON" summarize_vsr_srt_runs.py \
  --items "$ITEMS" \
  --runs "runs/${RUN_PREFIX}_default.jsonl" \
         "runs/${RUN_PREFIX}_evidence.jsonl" \
         "runs/${RUN_PREFIX}_srt_spatial.jsonl" \
  --output-dir "runs/${RUN_PREFIX}_summary"

echo "Done. Summary: runs/${RUN_PREFIX}_summary/vsr_srt_condition_summary.md"
