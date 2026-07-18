#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for file in configs/writer.env configs/answer.env configs/judge.env configs/run.env; do
  if [[ ! -f "$file" ]]; then
    echo "missing $file; copy the matching .example file and fill in your values" >&2
    exit 2
  fi
done

set -a
source configs/run.env
set +a

LEGACY="$ROOT/src/tmcra_benchmark/legacy"
VENDOR="$ROOT/src/tmcra_benchmark/vendor/tmcra_integrated"
HARNESS="$ROOT/src/tmcra_benchmark/harness/run_lme_s10_native_tmcra.py"
JUDGE="$ROOT/src/tmcra_benchmark/harness/evaluate_qa_official_judge.py"
export PYTHONPATH="$LEGACY:$VENDOR:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

: "${DATA:?DATA is required in configs/run.env}"
: "${QIDS:?QIDS is required in configs/run.env}"
: "${RUN:?RUN is required in configs/run.env}"
: "${MODELS:?MODELS is required in configs/run.env}"
: "${BGE_MODELS:?BGE_MODELS is required in configs/run.env}"

python scripts/prepare_longmemeval.py \
  --data "$DATA" \
  --qid-output "$QIDS" \
  --summary-output "$RUN.input_summary.json"

if [[ ! -f "$RUN/BUILD_COMPLETE" ]]; then
  build_resume_args=()
  if [[ -d "$RUN" ]]; then
    build_resume_args+=(--resume)
  fi
  python "$LEGACY/run_tmcra_v4_build.py" \
    "${build_resume_args[@]}" \
    --out-dir "$RUN" \
    --qid-list "$QIDS" \
    --data "$DATA" \
    --repo "$VENDOR" \
    --writer-env configs/writer.env \
    --embedding-model "$BGE_MODELS/bge-m3" \
    --writer-concurrency "${WRITER_CONCURRENCY:-20}" \
    --slow-concurrency "${SLOW_CONCURRENCY:-20}" \
    --index-batch-size 16 \
    --device "${RETRIEVAL_DEVICE:-cuda}"
fi
python scripts/validate_stage.py build --run-dir "$RUN" --expected-qids "$QIDS"

export TMCRA_NODE_MODEL_PATH="$MODELS/node_scorer.pt"
export TMCRA_PATH_MODEL_PATH="$MODELS/path_scorer.pt"
export TMCRA_BGE_M3_PATH="$BGE_MODELS/bge-m3"
export TMCRA_BGE_RERANKER_PATH="$BGE_MODELS/bge-reranker-v2-m3"

if [[ ! -f "$RUN/retrieval_s500/RETRIEVAL_COMPLETE" ]]; then
  retrieval_resume_args=()
  if [[ -d "$RUN/retrieval_s500" ]]; then
    retrieval_resume_args+=(--resume)
  fi
  python "$LEGACY/run_tmcra_v4_retrieve.py" \
    "${retrieval_resume_args[@]}" \
    --run-dir "$RUN" \
    --tag retrieval_s500 \
    --qid-list "$QIDS" \
    --writer-env configs/writer.env \
    --repo "$VENDOR" \
    --harness "$HARNESS" \
    --node-model "$MODELS/node_scorer.pt" \
    --path-model "$MODELS/path_scorer.pt" \
    --checkpoint "$MODELS/tmcra_v3_reranker.pt" \
    --embedding-model "$BGE_MODELS/bge-m3" \
    --cross-model "$BGE_MODELS/bge-reranker-v2-m3" \
    --device "${RETRIEVAL_DEVICE:-cuda}" \
    --graph-device "${GRAPH_DEVICE:-cuda}" \
    --dense-k 32 \
    --slow-dense-k 24 \
    --graph-k 24 \
    --composition-mode layered \
    --packing-budget-mode fixed \
    --top-k 8 \
    --cross-batch-size 24
fi
python scripts/validate_stage.py retrieve --run-dir "$RUN" --tag retrieval_s500 --expected-qids "$QIDS"

if [[ ! -f "$RUN/retrieval_s500_compiled/COMPILE_COMPLETE" ]]; then
  python "$LEGACY/run_tmcra_v4_compile_evidence.py" \
    --evidence "$RUN/retrieval_s500/evidence_windows.jsonl" \
    --retrieval-debug "$RUN/retrieval_s500/retrieval_debug.jsonl" \
    --require-retrieval-debug \
    --out-dir "$RUN/retrieval_s500_compiled" \
    --qid-list "$QIDS" \
    --writer-env configs/writer.env \
    --planner-provider deepseek \
    --planner-model deepseek-v4-pro \
    --workers 4 \
    --timeout 180
fi
python scripts/validate_stage.py compile --run-dir "$RUN/retrieval_s500_compiled" --expected-qids "$QIDS"

set -a
source configs/answer.env
set +a
# The answer runner resumes from its validated answers.jsonl. Running it again
# also repairs an incomplete report without repeating completed provider calls.
python "$LEGACY/run_tmcra_v4_gpt54_answers.py" \
  --evidence "$RUN/retrieval_s500_compiled/evidence_windows.jsonl" \
  --harness "$HARNESS" \
  --out-dir "$RUN/answers_s500" \
  --qid-list "$QIDS" \
  --workers "${ANSWER_WORKERS:-12}" \
  --attempts 3 \
  --answer-window-limit 8 \
  --lane production
python scripts/validate_stage.py answer --run-dir "$RUN/answers_s500" --expected-qids "$QIDS"

set -a
source configs/judge.env
set +a
: "${TMCRA_JUDGE_BASE_URL:?TMCRA_JUDGE_BASE_URL is required}"
: "${TMCRA_JUDGE_API_KEY:?TMCRA_JUDGE_API_KEY is required}"

python "$LEGACY/run_tmcra_v4_parallel_official_judge.py" \
  --judge "$JUDGE" \
  --metric-model gpt-4o \
  --hyp-file "$RUN/answers_s500/answers.jsonl" \
  --ref-file "$DATA" \
  --result-file "$RUN/official_judge_gpt4o_20240806.jsonl" \
  --work-dir "$RUN/official_judge_work" \
  --base-url "$TMCRA_JUDGE_BASE_URL" \
  --api-key-env TMCRA_JUDGE_API_KEY \
  --workers "${JUDGE_WORKERS:-8}" \
  --timeout 180 \
  --process-timeout 240 \
  --max-retries 0 \
  --resume

python scripts/validate_stage.py judge \
  --run-dir "$RUN" \
  --expected-qids "$QIDS" \
  --judge-results "$RUN/official_judge_gpt4o_20240806.jsonl"

python scripts/summarize_official_judge.py \
  --judge-results "$RUN/official_judge_gpt4o_20240806.jsonl" \
  --dataset "$DATA" \
  --output "$RUN/scorecard.json"

echo "TMCRA LongMemEval run complete: $RUN/scorecard.json"
