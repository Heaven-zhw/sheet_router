#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# The already-run combination is 42,43,44,45,46. This default list runs the
# other five combinations from choosing five seeds out of 42..47.
SEED_COMBINATIONS="${SEED_COMBINATIONS:-42,43,44,45,47;42,43,44,46,47;42,43,45,46,47;42,44,45,46,47;43,44,45,46,47}"
OUTPUT_ROOT_BASE="${OUTPUT_ROOT_BASE:-$REPO_DIR/format_ablation_outs_5formats_seed_combinations}"

IFS=';' read -r -a combinations <<< "$SEED_COMBINATIONS"
for seeds in "${combinations[@]}"; do
  if [[ -z "$seeds" ]]; then
    continue
  fi
  tag="seed_${seeds//,/_}"
  echo "============================================================"
  echo "[5-format Self-Consistency seed combination] seeds=$seeds"
  echo "output=$OUTPUT_ROOT_BASE/$tag"
  echo "============================================================"
  SELF_CONSISTENCY_SEEDS="$seeds" \
    OUTPUT_ROOT="$OUTPUT_ROOT_BASE/$tag" \
    bash "$SCRIPT_DIR/_run.sh" self_consistency
done
