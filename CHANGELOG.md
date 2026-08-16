# Changelog

## 2026.08.16

- Added an owner-local TMCRA runtime with a loopback-only FastAPI service,
  SQLite memory state, project and owner-global scopes, Source/Fast/Slow recall,
  Visual Atlas, Personal Knowledge projection, usage accounting, and physical
  message/project deletion.
- Added two explicit generation policies: user-supplied OpenAI-compatible BYOK
  and a fully local `llama-server` route. Provider credentials are stored in
  owner-only local files and omitted from config, logs, API errors, and child
  process environments.
- Added one-command local installation, startup, diagnostics, smoke testing,
  and uninstall entry points for Windows and POSIX systems.
- Added automatic owner-local Codex hooks and a tested DeepSeek Harness
  technical preview. Claude Code and ZCode share the audited hook contract and
  remain manual integrations until their current host packaging is accepted.
- Added Linux and Windows CI, release-manifest enforcement, current-tree and
  Git-history secret scanning, application dependency audits, wheel/package
  audits, and real local API
  write/recall/graph/knowledge/usage/delete end-to-end tests.
- Removed private training-run checkpoints, logs, and launch state from the
  release tip. The public inference scorers required by the runtime remain.

## 2026.07.18

- Repositioned TMCRA as an Agent Memory Engine with a clear Source / Fast /
  Slow layered architecture and scope isolation.
- Added the maintained LongMemEval-S500 reproduction package.
- Published the 82.2% scorecard with six task-level results.
- Added the latest node scorer, path scorer, and retrieval reranker weights.
- Hardened checkpoint loading with `weights_only=True`, published an
  inference-only reranker checkpoint without training-machine paths, pinned
  third-party model revisions, and added offline CI and gold-isolation checks.
- Versioned the maintained answer contract as `evidence_operation_bound_v6`
  and made compatible interrupted benchmark stages resumable.
- Switched the project license to Apache-2.0.
- Credited Yu Haoxin and OpenAI Codex for their respective development roles.
