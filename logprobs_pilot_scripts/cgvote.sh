#!/usr/bin/env bash

cd /mnt/data/zhw/sheet_router

REPO_DIR="$(pwd)"
RUN_MAP_ROOT="$REPO_DIR/configs/sheetflex/logprobs_generated_maps"
LP_WEIGHT_STRENGTH="${LP_WEIGHT_STRENGTH:-1.0}"
CONFIDENCE_GATE_STRENGTH="${CONFIDENCE_GATE_STRENGTH:-1.0}"
MISSING_LOGPROB_POLICY="${MISSING_LOGPROB_POLICY:-error}"
TIE_BREAK_ORDER="${TIE_BREAK_ORDER:-recommend}"
ALPHA_TAG="${LP_WEIGHT_STRENGTH//./p}"
BETA_TAG="${CONFIDENCE_GATE_STRENGTH//./p}"
CGVOTE_ROOT="${CGVOTE_ROOT:-$REPO_DIR/lp_outs/sheetflex_cg_vote/alpha_${ALPHA_TAG}_beta_${BETA_TAG}}"

mkdir -p "$RUN_MAP_ROOT" "$CGVOTE_ROOT"

MODELS=(
  "gemma-3-12b-it"
  "gemma-4-12B-it"
  # "gemma-4-26B-A4B-it"
  "Qwen3.5-9B"
  # "Qwen3-VL-30B-A3B-Instruct"
)

for MODEL in "${MODELS[@]}"; do
  echo "============================================================"
  echo "SheetFlex-CGVote: $MODEL (alpha=$LP_WEIGHT_STRENGTH, beta=$CONFIDENCE_GATE_STRENGTH)"
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

  echo "[RealHiTBench] SheetFlex-CGVote: $MODEL"
  python sheetflex_cg_vote.py realhit \
    --run_map "$REALHIT_MAP" \
    --output_dir "$CGVOTE_ROOT/$MODEL/realhit" \
    --lp_weight_strength "$LP_WEIGHT_STRENGTH" \
    --confidence_gate_strength "$CONFIDENCE_GATE_STRENGTH" \
    --missing_logprob_policy "$MISSING_LOGPROB_POLICY" \
    --tie_break_order "$TIE_BREAK_ORDER"

  echo "[SpreadsheetBench verified_400] SheetFlex-CGVote: $MODEL"
  python sheetflex_cg_vote.py spreadsheet \
    --run_map "$SPREADSHEET_MAP" \
    --output_dir "$CGVOTE_ROOT/$MODEL/spreadsheet" \
    --lp_weight_strength "$LP_WEIGHT_STRENGTH" \
    --confidence_gate_strength "$CONFIDENCE_GATE_STRENGTH" \
    --missing_logprob_policy "$MISSING_LOGPROB_POLICY" \
    --tie_break_order "$TIE_BREAK_ORDER"
done
