#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

# RUN_TAG="${RUN_TAG:-rerun}"
WORKERS="${WORKERS:-8}"
OUTPUT_ROOT="lp_outs"

MODEL="Qwen3.5-9B"
URL="10.26.35.171:33370"
MAX_TEXT_TOKENS=100000
# FORMATS=(
#   official_latex
#   latex 
#   markdown 
#   html 
#   csv 
#   dataframe 
#   json_rows 
#   json_cells 
#   image 
#   excel_1_image 
#   default_image
# )
FORMATS=(
  latex 
  markdown 
  json_rows 
  json_cells 
  image 
  excel_1_image 
)

cd "$REPO_DIR"

for FORMAT in "${FORMATS[@]}"; do
  echo "============================================================"
  echo "[RealHiTBench] model=$MODEL format=$FORMAT"
  echo "============================================================"

  python realhit_cot.py \
    --url "$URL" \
    --model_name "$MODEL" \
    --table_format "$FORMAT" \
    --temperature 0 \
    --top_p 1.0 \
    --max_retries 3 \
    --workers "$WORKERS" \
    --max_text_tokens "$MAX_TEXT_TOKENS" \
    --save_logprobs \
    -s 100ktoken \
    --output_root "$OUTPUT_ROOT" \
    --report_every 100 \
    --save_every 100
done
