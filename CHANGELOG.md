# Changelog

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
