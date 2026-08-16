#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$REPO/.tmcra"
[[ "$TARGET" == "$REPO/.tmcra" && "$TARGET" != / && "$TARGET" != "$HOME" ]] || {
  echo 'Refusing to remove an unsafe path.' >&2
  exit 2
}
if [[ ! -e "$TARGET" ]]; then
  echo 'TMCRA local data is already absent.'
  exit 0
fi
if [[ "${1:-}" == --purge-data ]]; then
  rm -rf -- "$TARGET"
  echo 'Removed runtime, models, credentials, and local memory databases.'
else
  rm -rf -- "$TARGET/venv"
  echo "Removed the Python environment. Memory, models, config, and credentials remain in: $TARGET"
  echo 'Run again with --purge-data only when you intend to erase all local TMCRA data.'
fi
