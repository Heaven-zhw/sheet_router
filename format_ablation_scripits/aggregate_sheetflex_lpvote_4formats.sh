#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/mnt/data/zhw/sheet_router"

SOURCE_ROOT="${SOURCE_ROOT:-$REPO_DIR/lp_outs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/format_ablation_outs}"
LP_WEIGHT_STRENGTH="${LP_WEIGHT_STRENGTH:-1.0}"
MISSING_LOGPROB_POLICY="${MISSING_LOGPROB_POLICY:-error}"
TIE_BREAK_ORDER="${TIE_BREAK_ORDER:-recommend}"
RESUME="${RESUME:-0}"
IDS="${IDS:-}"
LIMIT="${LIMIT:-0}"

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
    echo "[4-format SheetFlex-LPVote] model=$MODEL benchmark=$BENCHMARK alpha=$LP_WEIGHT_STRENGTH"
    python format_ablation.py lpvote "$BENCHMARK" \
      --model "$MODEL" \
      --source_root "$SOURCE_ROOT" \
      --output_root "$OUTPUT_ROOT" \
      --lp_weight_strength "$LP_WEIGHT_STRENGTH" \
      --missing_logprob_policy "$MISSING_LOGPROB_POLICY" \
      --tie_break_order "$TIE_BREAK_ORDER" \
      "${EXTRA_ARGS[@]}"
  done
done
