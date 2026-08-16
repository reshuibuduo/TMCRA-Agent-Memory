# Local tool integrations

Every automatic integration in this repository targets the same loopback API and the same project identity contract. No adapter needs a TMCRA account or a TMCRA-issued API key.

## Shared turn order

```text
current human prompt
  -> resolve project identity
  -> retry pending local writes
  -> recall owner-global + current-project memory
  -> inject evidence as untrusted data
  -> store the redacted USER source record
  -> host model/tool loop
  -> store the visible ASSISTANT source record
```

The project ID is derived from `.tmcra/project.json`, Git origin, Git root, or the canonical working directory, in that order. Session IDs remain provenance inside the project. Two tools opened in the same repository therefore share project memory while retaining separate application, session, role, and agent identities.

## Support status

| Host | Automation | Installation | Current evidence |
| --- | --- | --- | --- |
| Codex | Recall before answer; user/assistant writeback | One command | Node contract plus real local FastAPI cross-tool E2E |
| DeepSeek Harness | Native `agent/pre-step` recall and `session/event` (`turn/end`) writeback | Source-build installer; technical preview | Real Harness AgentLoop two-session test, typecheck, build, package audit |
| Claude Code | Shared hook implementation and manifest | Manual host registration | Shared hook contract plus real local FastAPI cross-tool E2E |
| ZCode | Shared hook implementation and manifest | Manual host registration | Shared hook contract test; current host packaging still needs acceptance |
| Other hosts | Local REST API | Host-specific | Use the documented turn order; do not claim automatic integration without a tested lifecycle seam |

## Codex

Start TMCRA in one terminal. In another terminal:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-codex-local.ps1
```

Shell:

```bash
bash scripts/install-codex-local.sh
```

The installer creates `~/.tmcra/local-integration.json`, which stores a path to the local bearer-token file, not the token value. It then registers this checkout as a local Codex marketplace, enables Hooks, and installs `tmcra-local-memory`.

Restart Codex, open `/hooks`, inspect the four commands, and grant trust explicitly. The installer cannot grant hook trust for the user.

## DeepSeek Harness

Requirements: a local DSH CLI, Node.js `22.19.0` or newer, and npm.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-deepseek-harness-local.ps1 `
  -PackageDirectory D:\tmcra-packages
```

The script configures the shared loopback client, installs pinned package dependencies, runs type checking and the Harness lifecycle test, builds a tarball, audits the package contents through `npm pack`, and calls `dsh plugin --profile web add`.

To validate and build the package on a machine that does not yet have the DSH CLI, add `-SkipDshInstall` on PowerShell or set `TMCRA_SKIP_DSH_PLUGIN_INSTALL=1` for the shell script. This runs every package check and stops after writing the audited tarball.

DeepSeek Harness is still a preview dependency. The plugin is deliberately labelled technical preview even though the lifecycle test passes against `0.1.0-rc.6`.

## Claude Code and ZCode

The reusable implementation and manifests are:

- `integrations/local-agent-hooks/hooks/claude-hooks.json`
- `integrations/local-agent-hooks/hooks/zcode-hooks.json`
- `integrations/local-agent-hooks/hooks/run_hook.mjs`

They use the same owner-local config and data contract as Codex. This repository does not advertise one-click installation for either host until its current public plugin packaging flow is tested on a clean installation.

## Failure behavior

- Recall failures let the host continue by default.
- Failed message writes are stored in `~/.tmcra/integrations/outbox` and retried by the next lifecycle event.
- Prompts, responses, bearer tokens, and API keys are omitted from hook diagnostics.
- The client rejects non-loopback URLs before reading the local bearer token.
- Pending-turn, outbox, and diagnostics directories and files are restricted to the current OS user.
