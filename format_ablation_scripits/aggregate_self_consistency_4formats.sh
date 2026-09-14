#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/data/zhw/sheet_router"

CANDIDATE_ROOT="${CANDIDATE_ROOT:-$REPO_DIR/sc_outs/candidates}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/format_ablation_outs}"
RESUME="${RESUME:-0}"
IDS="${IDS:-}"
LIMIT="${LIMIT:-0}"
TIE_BREAK_LOGPROB="${TIE_BREAK_LOGPROB:-mean}"

MODELS=(
  "gemma-3-12b-it"
  "gemma-4-12B-it"
  "Qwen3.5-9B"
  "Qwen3-VL-30B-A3B-Instruct"
)
FORMATS=(latex markdown json_rows json_cells)

cd "$REPO_DIR"

EXTRA_ARGS=()
if [[ "$RESUME" == "1" ]]; then EXTRA_ARGS+=(--resume); fi
if [[ -n "$IDS" ]]; then EXTRA_ARGS+=(--ids "$IDS"); fi
if [[ "$LIMIT" -gt 0 ]]; then EXTRA_ARGS+=(--limit "$LIMIT"); fi

for MODEL in "${MODELS[@]}"; do
  for FORMAT in "${FORMATS[@]}"; do
    for BENCHMARK in realhit spreadsheet; do
      echo "[4-format Self-Consistency] model=$MODEL format=$FORMAT benchmark=$BENCHMARK seeds=42,43,44,45"
      python format_ablation.py self_consistency "$BENCHMARK" \
        --model "$MODEL" \
        --format "$FORMAT" \
        --candidate_root "$CANDIDATE_ROOT" \
        --output_root "$OUTPUT_ROOT" \
        --tie_break_logprob "$TIE_BREAK_LOGPROB" \
        "${EXTRA_ARGS[@]}"
    done
  done
done
