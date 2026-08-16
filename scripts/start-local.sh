#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO/.tmcra/venv/bin/python" -m tmcra_local start \
  --config "$REPO/.tmcra/config/runtime/local-runtime.json" "$@"
