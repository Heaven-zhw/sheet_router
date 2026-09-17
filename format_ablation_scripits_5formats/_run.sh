#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-$REPO_DIR/sc_outs/candidates}"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_DIR/lp_outs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/format_ablation_outs_5formats}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_DIR/dataset/spreadsheetbench/spreadsheetbench_verified_400}"
RESUME="${RESUME:-1}"
IDS="${IDS:-}"
LIMIT="${LIMIT:-0}"
TIE_BREAK_LOGPROB="${TIE_BREAK_LOGPROB:-mean}"
TIE_BREAK_ORDER="${TIE_BREAK_ORDER:-recommend}"
LP_WEIGHT_STRENGTH="${LP_WEIGHT_STRENGTH:-1.0}"
CONFIDENCE_GATE_STRENGTH="${CONFIDENCE_GATE_STRENGTH:-1.0}"
MISSING_LOGPROB_POLICY="${MISSING_LOGPROB_POLICY:-error}"
SELF_CONSISTENCY_SEEDS="${SELF_CONSISTENCY_SEEDS:-42,43,44,45,46}"
METHOD="${1:?method is required}"

MODEL_NAMES="${MODEL_NAMES:-gemma-3-12b-it gemma-4-12B-it Qwen3.5-9B Qwen3-VL-30B-A3B-Instruct}"
BENCHMARKS="${BENCHMARKS:-realhit spreadsheet}"
SC_FORMATS="${SC_FORMATS:-latex markdown json_rows json_cells image excel_1_image}"
read -r -a models <<< "$MODEL_NAMES"
read -r -a benchmarks <<< "$BENCHMARKS"
read -r -a formats <<< "$SC_FORMATS"
case "$METHOD" in self_consistency|vote_mean|vote_fixed|lpvote|cgvote) ;; *) echo "Unknown method: $METHOD" >&2; exit 2;; esac
for benchmark in "${benchmarks[@]}"; do
  case "$benchmark" in realhit|spreadsheet) ;; *) echo "Unknown benchmark: $benchmark" >&2; exit 2;; esac
done

cd "$REPO_DIR"

EXTRA_ARGS=()
if [[ "$RESUME" == "1" ]]; then EXTRA_ARGS+=(--resume); fi
if [[ "${SKIP_COMPLETED:-1}" == "1" ]]; then EXTRA_ARGS+=(--skip_completed); fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then EXTRA_ARGS+=(--dry_run); fi
if [[ -n "$IDS" ]]; then EXTRA_ARGS+=(--ids "$IDS"); fi
if [[ "$LIMIT" -gt 0 ]]; then EXTRA_ARGS+=(--limit "$LIMIT"); fi

run_sc() {
  local model="$1" benchmark="$2" format="$3"
  echo "[5-format Self-Consistency] model=$model benchmark=$benchmark format=$format seeds=$SELF_CONSISTENCY_SEEDS"
  python format_ablation.py self_consistency "$benchmark" \
    --model "$model" \
    --format "$format" \
    --format_set five \
    --candidate_root "$CANDIDATE_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --dataset_root "$DATASET_ROOT" \
    --tie_break_logprob "$TIE_BREAK_LOGPROB" \
    --self_consistency_seeds "$SELF_CONSISTENCY_SEEDS" \
    "${EXTRA_ARGS[@]}"
}

run_cross() {
  local model="$1" benchmark="$2"
  echo "[5-format $METHOD] model=$model benchmark=$benchmark order=$TIE_BREAK_ORDER"
  python format_ablation.py "$METHOD" "$benchmark" \
    --model "$model" \
    --format_set five \
    --source_root "$SOURCE_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --tie_break_logprob "$TIE_BREAK_LOGPROB" \
    --tie_break_order "$TIE_BREAK_ORDER" \
    --lp_weight_strength "$LP_WEIGHT_STRENGTH" \
    --confidence_gate_strength "$CONFIDENCE_GATE_STRENGTH" \
    --missing_logprob_policy "$MISSING_LOGPROB_POLICY" \
    --dataset_root "$DATASET_ROOT" \
    "${EXTRA_ARGS[@]}"
}

for model in "${models[@]}"; do
  for benchmark in "${benchmarks[@]}"; do
    if [[ "$METHOD" == "self_consistency" ]]; then
      for format in "${formats[@]}"; do
        run_sc "$model" "$benchmark" "$format"
      done
    else
      run_cross "$model" "$benchmark"
    fi
  done
done
