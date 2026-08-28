#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

DATASET="${DATASET:?Set DATASET to realhit or spreadsheet}"
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME}"
URL="${URL:?Set URL to the OpenAI-compatible service}"
TABLE_FORMAT="${TABLE_FORMAT:?Set one explicit table format}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_DIR/sc_outs/candidates}"

cd "$REPO_DIR"

python self_consistency_generate.py \
  --dataset "$DATASET" \
  --model_name "$MODEL_NAME" \
  --url "$URL" \
  --table_format "$TABLE_FORMAT" \
  --num_samples 6 \
  --temperature 0.1 \
  --top_p 1.0 \
  --base_seed 42 \
  --output_root "$OUTPUT_ROOT"
