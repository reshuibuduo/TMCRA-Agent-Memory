#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_ROOT="$REPO/.tmcra"
VENV="$LOCAL_ROOT/venv"
CONFIG_ROOT="$LOCAL_ROOT/config"
CONFIG="$CONFIG_ROOT/runtime/local-runtime.json"
MODE="${TMCRA_INSTALL_MODE:-byok}"
EMBEDDING="${TMCRA_EMBEDDING_PROFILE:-balanced-multilingual}"
EMBEDDING_DEVICE="${TMCRA_EMBEDDING_DEVICE:-auto}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
TORCH_CHANNEL="${TMCRA_TORCH_CHANNEL:-auto}"

ask() {
  local prompt="$1" default="${2:-}" value=""
  if [[ -t 0 ]]; then
    read -r -p "$prompt${default:+ [$default]}: " value
    printf '%s' "${value:-$default}"
  elif [[ -n "$default" ]]; then
    printf '%s' "$default"
  else
    echo "Missing required non-interactive value: $prompt" >&2
    exit 2
  fi
}

"$PYTHON_BIN" -c "import sys; assert sys.version_info[:2] == (3, 12), 'TMCRA local requires Python 3.12'"
mkdir -p "$LOCAL_ROOT"
chmod 700 "$LOCAL_ROOT"
test -x "$VENV/bin/python" || "$PYTHON_BIN" -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade 'pip>=26.1.2' wheel

if [[ "$TORCH_CHANNEL" == auto ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then TORCH_CHANNEL=cu128; else TORCH_CHANNEL=cpu; fi
fi
if [[ "$TORCH_CHANNEL" == cu128 ]]; then
  "$PY" -m pip install 'torch==2.11.0+cu128' --extra-index-url https://download.pytorch.org/whl/cu128
elif [[ "$TORCH_CHANNEL" == cpu && "$(uname -s)" == Darwin ]]; then
  "$PY" -m pip install 'torch==2.11.0'
elif [[ "$TORCH_CHANNEL" == cpu ]]; then
  "$PY" -m pip install 'torch==2.11.0+cpu' --index-url https://download.pytorch.org/whl/cpu
else
  "$PY" -c 'import torch; print(torch.__version__)'
fi
"$PY" -m pip install -e "$REPO/runtime"

command -v git >/dev/null 2>&1 || {
  echo 'Git with Git LFS is required. Clone this repository with Git before running the installer.' >&2
  exit 2
}
git lfs version >/dev/null
git -C "$REPO" lfs pull --include='models/tmcra_v4_longmemeval_s500_20260715/**'

if [[ "$MODE" == byok ]]; then
  PROVIDER="${TMCRA_BYOK_PROVIDER:-$(ask 'OpenAI-compatible provider name' 'openai-compatible')}"
  BASE_URL="${TMCRA_BYOK_BASE_URL:-$(ask 'Credential-free /v1 base URL' 'https://api.deepseek.com/v1')}"
  MODEL="${TMCRA_BYOK_MODEL:-$(ask 'Model id' 'deepseek-chat')}"
  KEY_FILE="$CONFIG_ROOT/runtime/secrets/byok-api.key"
  "$PY" -m tmcra_local configure --embedding "$EMBEDDING" \
    --embedding-device "$EMBEDDING_DEVICE" --llm-policy byok \
    --byok-provider "$PROVIDER" --byok-base-url "$BASE_URL" \
    --byok-model "$MODEL" --byok-api-key-file "$KEY_FILE" \
    --config-root "$CONFIG_ROOT"
  API_KEY="${TMCRA_BYOK_API_KEY:-}"
  if [[ -z "$API_KEY" ]]; then
    if [[ ! -t 0 ]]; then
      echo 'Set TMCRA_BYOK_API_KEY for a non-interactive install.' >&2
      exit 2
    fi
    read -r -s -p 'BYOK API key (stored only in .tmcra/config/runtime/secrets): ' API_KEY
    echo
  fi
  TMCRA_INSTALL_API_KEY="$API_KEY" "$PY" -m tmcra_local set-key \
    --config "$CONFIG" --from-env TMCRA_INSTALL_API_KEY
  unset API_KEY TMCRA_BYOK_API_KEY || true
elif [[ "$MODE" == local-model ]]; then
  LLAMA_SERVER="${TMCRA_LLAMA_SERVER:-}"
  if [[ -z "$LLAMA_SERVER" || ! -x "$LLAMA_SERVER" ]]; then
    echo 'Local-model mode requires TMCRA_LLAMA_SERVER pointing to an executable llama-server.' >&2
    exit 2
  fi
  if [[ "${TMCRA_ACCEPT_LARGE_MODEL:-0}" != 1 ]]; then
    if [[ ! -t 0 ]]; then
      echo 'Set TMCRA_ACCEPT_LARGE_MODEL=1 for the 12.74 GiB local generation download.' >&2
      exit 2
    fi
    read -r -p 'Download the suggested 12.74 GiB Qwen3.6 model for 32K context? [y/N]: ' answer
    [[ "$answer" =~ ^[Yy]([Ee][Ss])?$ ]] || { echo 'Local model download cancelled.' >&2; exit 2; }
  fi
  "$PY" -m tmcra_local configure --embedding "$EMBEDDING" \
    --embedding-device "$EMBEDDING_DEVICE" --llm-policy local-model \
    --generation-profile recommended-qwen36 \
    --generation-runtime-executable "$LLAMA_SERVER" --config-root "$CONFIG_ROOT"
  "$PY" -m tmcra_local download-model --generation recommended-qwen36 \
    --models-root "$CONFIG_ROOT/models" --execute
else
  echo "TMCRA_INSTALL_MODE must be byok or local-model, got: $MODE" >&2
  exit 2
fi

"$PY" -m tmcra_local download-model --embedding "$EMBEDDING" \
  --models-root "$CONFIG_ROOT/models" --execute
DOCTOR=("$PY" -m tmcra_local doctor --config "$CONFIG" --probe-models --json)
if [[ "${TMCRA_SKIP_GENERATION_PROBE:-0}" != 1 ]]; then DOCTOR+=(--probe-generation); fi
"${DOCTOR[@]}"
"$PY" -m tmcra_local token --config "$CONFIG"
echo "TMCRA local install is ready. Run: $REPO/scripts/start-local.sh"
