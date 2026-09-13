#!/usr/bin/env bash
# 加上 --exclude_unchanged_target_values参数；修改LPVOTE_ROOT；注释RealHiTBench的相关代码
cd /mnt/data/zhw/sheet_router

REPO_DIR="$(pwd)"
RUN_MAP_ROOT="$REPO_DIR/configs/sheetflex/logprobs_generated_maps"
LP_WEIGHT_STRENGTH="${LP_WEIGHT_STRENGTH:-1.0}"
MISSING_LOGPROB_POLICY="${MISSING_LOGPROB_POLICY:-error}"
TIE_BREAK_ORDER="${TIE_BREAK_ORDER:-recommend}"
STRENGTH_TAG="${LP_WEIGHT_STRENGTH//./p}"
LPVOTE_ROOT="${LPVOTE_ROOT:-$REPO_DIR/lp_outs/sheetflex_lp_vote_noop/alpha_${STRENGTH_TAG}}"

mkdir -p "$RUN_MAP_ROOT" "$LPVOTE_ROOT"

MODELS=(
  "gemma-3-12b-it"
  "gemma-4-12B-it"
  # "gemma-4-26B-A4B-it"
  "Qwen3.5-9B"
  "Qwen3-VL-30B-A3B-Instruct"
)

for MODEL in "${MODELS[@]}"; do
  echo "============================================================"
  echo "SheetFlex-LPVote: $MODEL (alpha=$LP_WEIGHT_STRENGTH)"
  echo "============================================================"

  REALHIT_MAP="$RUN_MAP_ROOT/${MODEL}_realhit.json"
  SPREADSHEET_MAP="$RUN_MAP_ROOT/${MODEL}_spreadsheet.json"

#   cat > "$REALHIT_MAP" <<EOF
# {
#   "latex": "$REPO_DIR/lp_outs/realhitbench/$MODEL/cot_latex_logprobs_100ktoken",
#   "markdown": "$REPO_DIR/lp_outs/realhitbench/$MODEL/cot_markdown_logprobs_100ktoken",
#   "json_cells": "$REPO_DIR/lp_outs/realhitbench/$MODEL/cot_json_cells_logprobs_100ktoken",
#   "json_rows": "$REPO_DIR/lp_outs/realhitbench/$MODEL/cot_json_rows_logprobs_100ktoken",
#   "image": "$REPO_DIR/lp_outs/realhitbench/$MODEL/cot_image_logprobs_100ktoken",
#   "excel_1_image": "$REPO_DIR/lp_outs/realhitbench/$MODEL/cot_excel_1_image_logprobs_100ktoken"
# }
# EOF

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

  # echo "[RealHiTBench] SheetFlex-LPVote: $MODEL"
  # python sheetflex_lp_vote.py realhit \
  #   --run_map "$REALHIT_MAP" \
  #   --output_dir "$LPVOTE_ROOT/$MODEL/realhit" \
  #   --lp_weight_strength "$LP_WEIGHT_STRENGTH" \
  #   --missing_logprob_policy "$MISSING_LOGPROB_POLICY" \
  #   --tie_break_order "$TIE_BREAK_ORDER"

  echo "[SpreadsheetBench verified_400] SheetFlex-LPVote: $MODEL"
  python sheetflex_lp_vote.py spreadsheet \
    --run_map "$SPREADSHEET_MAP" \
    --output_dir "$LPVOTE_ROOT/$MODEL/spreadsheet" \
    --lp_weight_strength "$LP_WEIGHT_STRENGTH" \
    --missing_logprob_policy "$MISSING_LOGPROB_POLICY" \
    --tie_break_order "$TIE_BREAK_ORDER" \
    --exclude_unchanged_target_values
done
