#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA="${DATA:-$ROOT/fixtures/tiny_longmemeval.json}"
OUT="${OUT:-$ROOT/runs/offline_fixture_check}"
mkdir -p "$OUT"

python scripts/prepare_longmemeval.py \
  --data "$DATA" \
  --qid-output "$OUT/qids.txt" \
  --summary-output "$OUT/input_summary.json"
python -m pytest

echo "Offline fixture and unit tests passed."
echo "This smoke test does not call Writer, Answer, or Judge providers."
