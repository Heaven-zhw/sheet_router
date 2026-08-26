#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

# RUN_TAG="${RUN_TAG:-rerun}"
WORKERS="${WORKERS:-8}"

MODEL="gemma-4-12B-it"
URL="10.26.33.169:33374"
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
    -s 100ktoken \
    --report_every 100 \
    --save_every 100
done
