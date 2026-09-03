cd /mnt/data/zhw/sheet_router

REPO_DIR="$(pwd)"
RUN_MAP_ROOT="$REPO_DIR/configs/sheetflex/logprobs_generated_maps"
VOTE_ROOT="$REPO_DIR/lp_outs/sheetflex_vote_logprobs"

mkdir -p "$RUN_MAP_ROOT" "$VOTE_ROOT"

MODELS=(
  "gemma-3-12b-it"
  "gemma-4-12B-it"
  #"gemma-4-26B-A4B-it"
  "Qwen3.5-9B"
  #"Qwen3-VL-30B-A3B-Instruct"
)

for MODEL in "${MODELS[@]}"; do
  echo "============================================================"
  echo "Building run maps for $MODEL"
  echo "============================================================"

  REALHIT_MAP="$RUN_MAP_ROOT/${MODEL}_realhit.json"
  SPREADSHEET_MAP="$RUN_MAP_ROOT/${MODEL}_spreadsheet.json"

  cat > "$REALHIT_MAP" <<EOF
{
  "latex": "$REPO_DIR/lp_outs/realhitbench/$MODEL/cot_latex_logprobs_100ktoken",
  "markdown": "$REPO_DIR/lp_outs/realhitbench/$MODEL/cot_markdown_logprobs_100ktoken",
  "json_cells": "$REPO_DIR/lp_outs/realhitbench/$MODEL/cot_json_cells_logprobs_100ktoken",
  "json_rows": "$REPO_DIR/lp_outs/realhitbench/$MODEL/cot_json_rows_logprobs_100ktoken",
  "image": "$REPO_DIR/lp_outs/realhitbench/$MODEL/cot_image_logprobs_100ktoken",
  "excel_1_image": "$REPO_DIR/lp_outs/realhitbench/$MODEL/cot_excel_1_image_logprobs_100ktoken"
}
EOF

  cat > "$SPREADSHEET_MAP" <<EOF
{
  "latex": "$REPO_DIR/lp_outs/spreadsheetbench_verified_400/$MODEL/pot_latex_logprobs_40ktoken",
  "markdown": "$REPO_DIR/lp_outs/spreadsheetbench_verified_400/$MODEL/pot_markdown_logprobs_40ktoken",
  "json_cells": "$REPO_DIR/lp_outs/spreadsheetbench_verified_400/$MODEL/pot_json_cells_logprobs_40ktoken",
  "json_rows": "$REPO_DIR/lp_outs/spreadsheetbench_verified_400/$MODEL/pot_json_rows_logprobs_40ktoken",
  "image": "$REPO_DIR/lp_outs/spreadsheetbench_verified_400/$MODEL/pot_image_logprobs_40ktoken",
  "excel_1_image": "$REPO_DIR/lp_outs/spreadsheetbench_verified_400/$MODEL/pot_excel_1_image_logprobs_40ktoken"
}
EOF

  echo "[RealHiTBench] SheetFlex-vote: $MODEL"
  python sheetflex_vote.py realhit \
    --run_map "$REALHIT_MAP" \
    --output_dir "$VOTE_ROOT/$MODEL/realhit"

  echo "[SpreadsheetBench verified_400] SheetFlex-vote: $MODEL"
  python sheetflex_vote.py spreadsheet \
    --run_map "$SPREADSHEET_MAP" \
    --output_dir "$VOTE_ROOT/$MODEL/spreadsheet"
done