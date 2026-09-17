#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/data/zhw/sheet_router"

SOURCE_ROOT="${SOURCE_ROOT:-$REPO_DIR/lp_outs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/format_ablation_outs}"
RESUME="${RESUME:-0}"
IDS="${IDS:-}"
LIMIT="${LIMIT:-0}"
TIE_BREAK_LOGPROB="${TIE_BREAK_LOGPROB:-mean}"
TIE_BREAK_ORDER="${TIE_BREAK_ORDER:-recommend}"
VOTE_METHOD="${VOTE_METHOD:-vote_mean}"

MODELS=(
  "gemma-3-12b-it"
  "gemma-4-12B-it"
  "Qwen3.5-9B"
  "Qwen3-VL-30B-A3B-Instruct"
)

cd "$REPO_DIR"

EXTRA_ARGS=()
if [[ "$RESUME" == "1" ]]; then EXTRA_ARGS+=(--resume); fi
if [[ -n "$IDS" ]]; then EXTRA_ARGS+=(--ids "$IDS"); fi
if [[ "$LIMIT" -gt 0 ]]; then EXTRA_ARGS+=(--limit "$LIMIT"); fi

for MODEL in "${MODELS[@]}"; do
  for BENCHMARK in realhit spreadsheet; do
    echo "[4-format SheetFlex-vote] model=$MODEL benchmark=$BENCHMARK tie_logprob=$TIE_BREAK_LOGPROB order=$TIE_BREAK_ORDER"
    python format_ablation.py "$VOTE_METHOD" "$BENCHMARK" \
      --model "$MODEL" \
      --source_root "$SOURCE_ROOT" \
      --output_root "$OUTPUT_ROOT" \
      --tie_break_logprob "$TIE_BREAK_LOGPROB" \
      --tie_break_order "$TIE_BREAK_ORDER" \
      "${EXTRA_ARGS[@]}"
  done
done
