# Benchmark results / Benchmark 成绩说明

## Current result / 当前成绩

The current published TMCRA result is **411 / 500 = 82.2%** on LongMemEval S500. The machine-readable scorecard is [`latest_benchmark.json`](latest_benchmark.json), and the maintained reproduction pipeline is [`../benchmarks/longmemeval/`](../benchmarks/longmemeval/README.md).

TMCRA 当前发布的 LongMemEval S500 成绩为 **411 / 500 = 82.2%**。机器可读成绩见 [`latest_benchmark.json`](latest_benchmark.json)，持续维护的复现链路见 [`../benchmarks/longmemeval/`](../benchmarks/longmemeval/README.zh-CN.md)。

| Task / 任务 | Correct / Total | Accuracy |
| --- | ---: | ---: |
| Knowledge Update / 信息更新 | 71 / 78 | 91.0% |
| Multi-session / 跨会话整合 | 90 / 133 | 67.7% |
| Single-session Assistant / 单会话助手信息 | 55 / 56 | 98.2% |
| Single-session Preference / 单会话偏好 | 27 / 30 | 90.0% |
| Single-session User / 单会话用户信息 | 67 / 70 | 95.7% |
| Temporal Reasoning / 时间推理 | 101 / 133 | 75.9% |
| **Overall / 总成绩** | **411 / 500** | **82.2%** |

## Retained historical baseline / 保留的历史基线

The following files belong to the frozen 2026-05-25 baseline, which scored **310 / 500 = 62.0%**. They are retained for longitudinal comparison and must not be interpreted as the latest result:

以下文件属于 2026-05-25 冻结的 **310 / 500 = 62.0%** 历史基线。它们仅用于纵向对比，不代表当前成绩：

```text
predictions.jsonl
judge_gpt4o_alias_vectorengine.jsonl
judge_gpt4o_alias_vectorengine.jsonl.summary.json
lme_s500_frozen_baseline38_full10_20260525_results.tar.gz
```

The detailed historical record remains in [`../docs/BASELINE_S500_20260525.md`](../docs/BASELINE_S500_20260525.md).

历史基线的详细记录仍保留在 [`../docs/BASELINE_S500_20260525.md`](../docs/BASELINE_S500_20260525.md)。
