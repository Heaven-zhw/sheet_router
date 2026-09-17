#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VOTE_METHOD=vote_fixed exec bash "$SCRIPT_DIR/aggregate_sheetflex_vote_4formats.sh" "$@"
