#!/usr/bin/env sh
set -eu

REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PACKAGE_ROOT="$REPO/integrations/deepseek-harness-local"
RUNTIME_CONFIG=${TMCRA_RUNTIME_CONFIG:-"$REPO/.tmcra/config/runtime/local-runtime.json"}
PROFILE=${DSH_PROFILE:-web}
PACKAGE_DIRECTORY=${TMCRA_DSH_PACKAGE_DIRECTORY:-"$HOME/.tmcra/packages"}
NODE=${NODE:-node}
NPM=${NPM:-npm}
DSH=${DSH:-dsh}

case "$PACKAGE_DIRECTORY" in
  *[! -~]*|*' '*)
    printf '%s\n' 'DeepSeek Harness preview can mis-handle package paths containing spaces or non-ASCII characters.' >&2
    printf '%s\n' 'Set TMCRA_DSH_PACKAGE_DIRECTORY to a short ASCII-only path.' >&2
    exit 2
    ;;
esac

mkdir -p "$PACKAGE_DIRECTORY"
"$NODE" --version
"$NODE" "$REPO/integrations/local-agent-hooks/scripts/configure.mjs" --runtime-config "$RUNTIME_CONFIG"

cd "$PACKAGE_ROOT"
"$NPM" ci --ignore-scripts --no-audit --no-fund
"$NPM" run typecheck
if [ "${TMCRA_SKIP_INTEGRATION_TESTS:-0}" != "1" ]; then
  "$NPM" test
fi
"$NPM" run build
"$NPM" pack --pack-destination "$PACKAGE_DIRECTORY"

NAME=$($NODE -p "require('./package.json').name")
VERSION=$($NODE -p "require('./package.json').version")
TARBALL="$PACKAGE_DIRECTORY/$NAME-$VERSION.tgz"
test -f "$TARBALL"
if [ "${TMCRA_SKIP_DSH_PLUGIN_INSTALL:-0}" = "1" ]; then
  printf '%s\n' "DeepSeek Harness package verified at: $TARBALL"
else
  "$DSH" plugin --profile "$PROFILE" add "$TARBALL"
  printf '%s\n' "TMCRA owner-local memory was added to DeepSeek Harness profile '$PROFILE'."
  printf '%s\n' 'Start the local TMCRA API before starting Harness.'
fi
