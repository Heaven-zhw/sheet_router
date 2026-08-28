#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"

DATASET="${DATASET:?Set DATASET to realhit or spreadsheet}"
MANIFEST="${MANIFEST:?Set MANIFEST to the stage-1 manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:?Set a new Self-Consistency output directory}"
PYTHON="${PYTHON:-python}"

cd "$REPO_DIR"
"$PYTHON" self_consistency.py "$DATASET" \
  --manifest "$MANIFEST" \
  --output_dir "$OUTPUT_DIR"
