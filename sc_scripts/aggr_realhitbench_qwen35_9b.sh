#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

MODEL_NAME="${MODEL_NAME:-Qwen3.5-9B}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-$REPO_DIR/sc_outs/candidates}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_DIR/sc_outs/aggregated}"
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
  MANIFEST="$CANDIDATE_ROOT/self_consistency_manifests/realhit/$MODEL_NAME/$FORMAT/manifest.json"
  OUTPUT_DIR="$RESULT_ROOT/realhitbench/$MODEL_NAME/$FORMAT"

  if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing manifest: $MANIFEST" >&2
    exit 1
  fi

  echo "============================================================"
  echo "[Self-Consistency aggregation] dataset=realhit model=$MODEL_NAME format=$FORMAT"
  echo "manifest=$MANIFEST"
  echo "output=$OUTPUT_DIR"
  echo "============================================================"

  python self_consistency.py realhit \
    --manifest "$MANIFEST" \
    --output_dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"
done
