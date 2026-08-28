#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

DATASET="${DATASET:-spreadsheet}"
MODEL_NAME="${MODEL_NAME:-gemma-4-12B-it}"
URL="${URL:-10.26.33.169:33374}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/sc_outs/candidates}"
WORKERS="${WORKERS:-8}"

# RESUME="${RESUME:-0}"
# SAMPLE_INDEX="${SAMPLE_INDEX:-}"
# IDS="${IDS:-}"

FORMATS=(
  latex 
  markdown 
  json_rows 
  json_cells 
  image
  excel_1_image 
)

cd "$REPO_DIR"

EXTRA_ARGS=()

# if [[ "$RESUME" == "1" ]]; then
#   EXTRA_ARGS+=(--resume)
# fi
# if [[ -n "$SAMPLE_INDEX" ]]; then
#   EXTRA_ARGS+=(--sample_index "$SAMPLE_INDEX")
# fi
# if [[ -n "$IDS" ]]; then
#   EXTRA_ARGS+=(--ids "$IDS")
# fi

if [[ "$DATASET" == "realhit" ]]; then
  MAX_TEXT_TOKENS=100000
elif [[ "$DATASET" == "spreadsheet" ]]; then
  MAX_TEXT_TOKENS=40000
  EXTRA_ARGS+=(
    --code_exec_url localhost:8081
    --render_formulas_before_eval
  )
else
  echo "DATASET must be realhit or spreadsheet" >&2
  exit 1
fi

for FORMAT in "${FORMATS[@]}"; do
  echo "============================================================"
  echo "[Self-Consistency candidates] dataset=$DATASET model=$MODEL_NAME format=$FORMAT"
  echo "============================================================"

  python self_consistency_generate.py \
    --dataset "$DATASET" \
    --model_name "$MODEL_NAME" \
    --url "$URL" \
    --table_format "$FORMAT" \
    --num_samples 6 \
    --temperature 0.1 \
    --top_p 1.0 \
    --base_seed 42 \
    --workers "$WORKERS" \
    --max_retries 3 \
    --max_text_tokens "$MAX_TEXT_TOKENS" \
    --output_root "$OUTPUT_ROOT" \
    "${EXTRA_ARGS[@]}"
done
