#!/usr/bin/env python3
"""LongMemEval official-prompt QA judge.

This is an implementation-compatible wrapper around the public LongMemEval
`src/evaluation/evaluate_qa.py` logic. The scoring prompt and yes/no rule are
kept equivalent, while the runner adds resume support and avoids extra Python
package dependencies by using urllib.

For official reporting, use:
  --metric-model gpt-4o
with OPENAI_API_KEY set. The resolved model is gpt-4o-2024-08-06, matching the
official evaluator's model alias.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import mean
from typing import Any


MODEL_ZOO = {
    "gpt5.4": ("gpt-5.4", "openai"),
    "gpt-5.4": ("gpt-5.4", "openai"),
    "gpt-4o-mini": ("gpt-4o-mini-2024-07-18", "openai"),
    "gpt-4o": ("gpt-4o-2024-08-06", "openai"),
    "gpt-4o-2024-08-06": ("gpt-4o-2024-08-06", "openai"),
    "llama-3.1-70b-instruct": ("meta-llama/Meta-Llama-3.1-70B-Instruct", "local"),
}


def get_anscheck_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. "
                "If the response only contains a subset of the information required by the answer, answer no. \n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "temporal-reasoning":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. "
                "If the response only contains a subset of the information required by the answer, answer no. "
                "In addition, do not penalize off-by-one errors for the number of days. "
                "If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "knowledge-update":
            template = (
                "I will give you a question, a correct answer, and a response from a model. "
                "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                "If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\n"
                "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        if task == "single-session-preference":
            template = (
                "I will give you a question, a rubric for desired personalized response, and a response from a model. "
                "Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
                "The model does not need to reflect all the points in the rubric. "
                "The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\n"
                "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
                "Is the model response correct? Answer yes or no only."
            )
            return template.format(question, answer, response)
        raise NotImplementedError(f"Unsupported question_type: {task}")

    template = (
        "I will give you an unanswerable question, an explanation, and a response from a model. "
        "Please answer yes if the model correctly identifies the question as unanswerable. "
        "The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\n"
        "Question: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
        "Does the model correctly identify the question as unanswerable? Answer yes or no only."
    )
    return template.format(question, answer, response)


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError(f"{path} JSON root is not a list")
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def call_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: int,
    max_retries: int,
) -> str:
    wire_api = os.environ.get("OPENAI_WIRE_API", "").strip().lower()
    if wire_api == "responses":
        url = base_url.rstrip("/") + "/responses"
        payload = {
            "model": model,
            "input": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_output_tokens": 10,
        }
    else:
        url = base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "n": 1,
            "temperature": 0,
            "max_tokens": 10,
        }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if wire_api == "responses":
                output_text = str(data.get("output_text", "")).strip()
                if output_text:
                    return output_text
                chunks: list[str] = []
                for item in data.get("output", []) or []:
                    for content in item.get("content", []) or []:
                        text = content.get("text") or content.get("content") or ""
                        if text:
                            chunks.append(str(text))
                return "\n".join(chunks).strip()
            return data["choices"][0]["message"]["content"].strip()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"chat completion failed after {max_retries + 1} attempts: {last_error}")


def read_existing_labels(result_file: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not result_file.exists():
        return done
    with result_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            qid = str(item.get("question_id", ""))
            if qid:
                done[qid] = item
    return done


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-model", default="gpt-4o", choices=sorted(MODEL_ZOO))
    parser.add_argument("--hyp-file", required=True, type=Path)
    parser.add_argument("--ref-file", required=True, type=Path)
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    model, source = MODEL_ZOO[args.metric_model]
    if source == "openai":
        base_url = args.base_url or "https://api.openai.com/v1"
    else:
        base_url = args.base_url or "http://localhost:8001/v1"

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print(f"Missing API key env: {args.api_key_env}", file=sys.stderr)
        return 2

    hypotheses = load_json_or_jsonl(args.hyp_file)
    references = load_json_or_jsonl(args.ref_file)
    qid2qdata = {str(entry["question_id"]): entry for entry in references}
    qid2qtype = {str(entry["question_id"]): entry["question_type"] for entry in references}

    if args.limit and args.limit > 0:
        hypotheses = hypotheses[: args.limit]

    result_file = args.result_file or Path(str(args.hyp_file) + f".eval-results-{args.metric_model}")
    result_file.parent.mkdir(parents=True, exist_ok=True)
    existing = read_existing_labels(result_file) if args.resume else {}

    logs: list[dict[str, Any]] = list(existing.values())
    qtype2acc: dict[str, list[int]] = {}
    for item in logs:
        qid = str(item.get("question_id"))
        if qid in qid2qtype and item.get("autoeval_label"):
            qtype2acc.setdefault(qid2qtype[qid], []).append(1 if item["autoeval_label"].get("label") else 0)

    mode = "a" if args.resume and result_file.exists() else "w"
    with result_file.open(mode, encoding="utf-8") as out_f:
        for idx, entry in enumerate(hypotheses, 1):
            qid = str(entry.get("question_id", ""))
            if not qid or qid not in qid2qdata:
                print(f"Warning: skipping unknown question_id={qid}", file=sys.stderr)
                continue
            if qid in existing:
                continue
            qtype = qid2qtype[qid]
            qdata = qid2qdata[qid]
            prompt = get_anscheck_prompt(
                qtype,
                str(qdata["question"]),
                str(qdata["answer"]),
                str(entry.get("hypothesis", "")),
                abstention="_abs" in qid,
            )
            eval_response = call_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            label = "yes" in eval_response.lower()
            output = dict(entry)
            output["autoeval_label"] = {"model": model, "label": label, "raw_response": eval_response}
            print(json.dumps(output, ensure_ascii=False), file=out_f, flush=True)
            logs.append(output)
            qtype2acc.setdefault(qtype, []).append(1 if label else 0)
            if not args.quiet:
                print(
                    json.dumps(
                        {
                            "index": idx,
                            "question_id": qid,
                            "label": label,
                            "raw_response": eval_response,
                            "question": qdata["question"],
                            "answer": qdata["answer"],
                            "hypothesis": entry.get("hypothesis", ""),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    labels = [1 if x.get("autoeval_label", {}).get("label") else 0 for x in logs if x.get("autoeval_label")]
    summary = {
        "metric_model_alias": args.metric_model,
        "metric_model": model,
        "base_url": base_url,
        "hyp_file": str(args.hyp_file),
        "ref_file": str(args.ref_file),
        "result_file": str(result_file),
        "evaluated_count": len(labels),
        "accuracy": round(mean(labels), 4) if labels else None,
        "by_question_type": {
            k: {"accuracy": round(mean(v), 4), "count": len(v)} for k, v in sorted(qtype2acc.items()) if v
        },
    }
    summary_file = Path(str(result_file) + ".summary.json")
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
