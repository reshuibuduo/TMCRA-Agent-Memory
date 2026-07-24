# TMCRA LongMemEval-S frozen 82.2% result

This directory contains sanitized review artifacts for the frozen TMCRA
LongMemEval-S run reported as **411/500 (82.2%)**.

## Frozen configuration

| Item | Value |
|---|---|
| Dataset | `longmemeval_s_cleaned`, 500 questions |
| Answer model | `gpt-5.4` |
| Answer protocol | `evidence_operation_bound_v5` |
| Judge | LongMemEval official prompt with `gpt-4o-2024-08-06` |
| Dataset SHA256 | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| Frozen judge artifact SHA256 | `671cfef6fe161dcbb89c6dfb09b798a0581e91ef6ad1134e8fb9e3626243fb9a` |

## Artifacts

- `artifacts/hypotheses.jsonl`: the official two-field `question_id` +
  `hypothesis` submission format.
- `artifacts/judge_labels.jsonl`: sanitized official-judge labels and raw
  Yes/No responses.
- `artifacts/retrieval_session_ids.jsonl`: ranked Source candidate session IDs,
  final answer-facing session IDs, and immutable evidence hashes. It contains no
  conversation text or Gold labels.
- `artifacts/metrics_for_submission.json`: end-to-end and explicitly defined
  retrieval metrics for the full 500 rows and the official 470-row retrieval
  scope.
- `SHA256SUMS`: checksums for all four artifacts.

## Retrieval scope

The official retrieval scope excludes 30 abstention questions. On the remaining
470 questions, the final answer-facing evidence packet (up to eight unique
session windows) has:

| Metric | Result |
|---|---:|
| Any required session | 470/470 (100.00%) |
| All required sessions | 416/470 (88.51%) |
| Macro session recall | 95.79% |
| Micro session recall | 822/890 (92.36%) |

Candidate-stage Top-K and final answer-facing Top-8 are separate stages. Their
definitions and complete results are recorded in `metrics_for_submission.json`.

## Disclosure

- Four empty first-attempt judge transport responses were quarantined, and only
  those four judge calls were retried.
- Answer generation used validation/repair attempts where necessary: 485 rows
  completed in one attempt, 14 in two attempts, and 1 in three attempts.
- The published score is a frozen v5 result. The current public harness uses a
  security-hardened v6 answer protocol, so this directory supports result review
  and metric recomputation but does not claim bit-for-bit regeneration of the
  proprietary-model run.
- Gold answers and Gold session IDs were isolated from Writer, retrieval,
  evidence compilation, and answer generation. `question_type` was retained for
  reporting and was not used for routing or answer-prompt input.
