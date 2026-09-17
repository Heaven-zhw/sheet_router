#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SELF_CONSISTENCY_SEEDS="${SELF_CONSISTENCY_SEEDS:-42,43,44,45,46}" \
  exec bash "$SCRIPT_DIR/_run.sh" self_consistency "$@"
