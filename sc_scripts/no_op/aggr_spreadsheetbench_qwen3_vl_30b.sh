#!/usr/bin/env bash
# 修改RESULT_ROOT，修改REPO_DIR，加参数exclude_unchanged_target_values
REPO_DIR="/mnt/data/zhw/sheet_router"

DATASET_ROOT="${DATASET_ROOT:-$REPO_DIR/dataset/spreadsheetbench/spreadsheetbench_verified_400}"
MODEL_NAME="${MODEL_NAME:-Qwen3-VL-30B-A3B-Instruct}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-$REPO_DIR/sc_outs/candidates}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_DIR/sc_outs/aggregated_noop}"
RESUME="${RESUME:-0}"
IDS="${IDS:-}"
LIMIT="${LIMIT:-0}"

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
if [[ "$RESUME" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
fi
if [[ -n "$IDS" ]]; then
  EXTRA_ARGS+=(--ids "$IDS")
fi
if [[ "$LIMIT" -gt 0 ]]; then
  EXTRA_ARGS+=(--limit "$LIMIT")
fi

for FORMAT in "${FORMATS[@]}"; do
  MANIFEST="$CANDIDATE_ROOT/self_consistency_manifests/spreadsheet/$MODEL_NAME/$FORMAT/manifest.json"
  OUTPUT_DIR="$RESULT_ROOT/spreadsheetbench_verified_400/$MODEL_NAME/$FORMAT"

  if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing manifest: $MANIFEST" >&2
    exit 1
  fi

  echo "============================================================"
  echo "[Self-Consistency aggregation] dataset=spreadsheetbench_verified_400 model=$MODEL_NAME format=$FORMAT"
  echo "manifest=$MANIFEST"
  echo "output=$OUTPUT_DIR"
  echo "============================================================"

  python self_consistency.py spreadsheet \
    --manifest "$MANIFEST" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_root "$DATASET_ROOT" \
    --exclude_unchanged_target_values \
    "${EXTRA_ARGS[@]}"
done
