#!/usr/bin/env bash
set -euo pipefail

# Start code executor first:
# cd code_exec_docker
# bash start_jupyter_server.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

# RUN_TAG="${RUN_TAG:-rerun}"
WORKERS="${WORKERS:-8}"
CODE_EXEC_URL="${CODE_EXEC_URL:-localhost:8081}"

MODEL="Qwen3.5-9B"
URL="10.26.35.171:33370"
MAX_TEXT_TOKENS=40000

# FORMATS=(
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
  echo "[SpreadsheetBench verified_400] model=$MODEL format=$FORMAT"
  echo "============================================================"

  python spreadsheet_pot.py \
    --url "$URL" \
    --model_name "$MODEL" \
    --table_format "$FORMAT" \
    --data_split verified_400 \
    --code_exec_url "$CODE_EXEC_URL" \
    --temperature 0 \
    --top_p 1.0 \
    --max_retries 3 \
    --workers "$WORKERS" \
    --report_every 50 \
    --save_every 50 \
    --max_text_tokens "$MAX_TEXT_TOKENS" \
    --save_logprobs \
    -s 40ktoken \
    --render_formulas_before_eval
done
