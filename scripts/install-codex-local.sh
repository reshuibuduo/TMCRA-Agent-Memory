#!/usr/bin/env sh
set -eu

REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PLUGIN="$REPO/integrations/local-agent-hooks"
RUNTIME_CONFIG=${TMCRA_RUNTIME_CONFIG:-"$REPO/.tmcra/config/runtime/local-runtime.json"}
NODE=${NODE:-node}
CODEX=${CODEX:-codex}

"$NODE" --version
"$NODE" "$PLUGIN/scripts/configure.mjs" --runtime-config "$RUNTIME_CONFIG"

if [ "${TMCRA_SKIP_CODEX_PLUGIN_INSTALL:-0}" != "1" ]; then
  "$CODEX" plugin marketplace --help >/dev/null
  "$CODEX" features enable hooks >/dev/null
  "$CODEX" plugin marketplace remove tmcra-owner-local --json >/dev/null 2>&1 || true
  "$CODEX" plugin marketplace add "$REPO" --json
  "$CODEX" plugin add tmcra-local-memory@tmcra-owner-local --json
fi

printf '%s\n' 'TMCRA owner-local hooks are configured.'
printf '%s\n' 'Restart Codex, open /hooks, review the four TMCRA Local Memory hooks, and trust them.'
