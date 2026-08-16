# TMCRA owner-local memory for DeepSeek Harness

This DSH plugin connects the native `agent/pre-step` and `session/event` lifecycle seams to a TMCRA instance running on the same computer.

It requires the owner-local runtime and the shared integration config created by:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-local.ps1
node .\integrations\local-agent-hooks\scripts\configure.mjs --runtime-config .\.tmcra\config\runtime\local-runtime.json
```

Install and test the local package, then add it to the selected Harness profile:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-deepseek-harness-local.ps1 `
  -PackageDirectory D:\tmcra-packages
```

Shell:

```bash
TMCRA_DSH_PACKAGE_DIRECTORY="$HOME/tmcra-packages" \
bash scripts/install-deepseek-harness-local.sh
```

The installer runs `npm ci`, type checking, the Harness lifecycle test, the build, `npm pack`, and `dsh plugin --profile web add`. The plugin reads `~/.tmcra/local-integration.json` by default. `configPath` can point to another owner-local integration config. DeepSeek Harness preview releases can mis-handle tarball paths containing spaces or non-ASCII characters, so the packaging directory must be a short ASCII-only path.

For every admitted human turn, the plugin:

- resolves a project identity shared with the other local TMCRA adapters;
- recalls user-global and current-project evidence before the first model step;
- injects that evidence as an explicit plugin-sourced, untrusted user message;
- stores the human prompt with `user` provenance;
- stores the visible completed answer with the exact Harness agent/subagent identity;
- queues failed writes locally and retries them on the next turn.

No TMCRA account, subscription, device authorization, hosted scope, or production API endpoint is part of this package.

The lifecycle and package contents have been accepted against `@deepseek-ai/dsh-agent-loop` `0.1.0-rc.6`. The final `dsh plugin add` command still requires a locally installed DSH CLI; the installer fails if it is missing.
