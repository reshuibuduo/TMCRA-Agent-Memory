#!/usr/bin/env python3
"""Zero-API promotion gate for paired TMCRA official-judge runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "tmcra.v4.regression-gate.1"
MOVEMENTS = ("kept_correct", "improved", "regressed", "kept_wrong")
DEFAULT_CASES = Path(__file__).with_name("tmcra_v4_regression_cases.json")


class GateInputError(ValueError):
    """Raised internally for malformed or non-pairable gate inputs."""


def _qid(row: Mapping[str, Any], source: str, line: int | None = None) -> str:
    qid_value = row.get("qid")
    question_id = row.get("question_id")
    if qid_value is not None and question_id is not None:
        if str(qid_value).strip() != str(question_id).strip():
            location = f" line {line}" if line is not None else ""
            raise GateInputError(f"{source}{location}: qid and question_id disagree")
    value = qid_value if qid_value is not None else question_id
    result = str(value or "").strip()
    if not result:
        location = f" line {line}" if line is not None else ""
        raise GateInputError(f"{source}{location}: missing qid")
    return result


def _read_jsonl(path: Path, source: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise GateInputError(f"cannot read {source}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GateInputError(f"{source} line {line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise GateInputError(f"{source} line {line_number}: expected an object")
        current_qid = _qid(row, source, line_number)
        if current_qid in seen:
            raise GateInputError(f"{source}: duplicate qid {current_qid}")
        seen.add(current_qid)
        rows.append({**row, "qid": current_qid})
    if not rows:
        raise GateInputError(f"{source}: no rows")
    return rows


def _reference_rows(value: Any, source: str) -> list[dict[str, Any]]:
    if isinstance(value, list):
        raw_rows = value
    elif isinstance(value, dict):
        raw_rows: Any = None
        for key in ("questions", "data", "rows", "references"):
            if key in value:
                raw_rows = value[key]
                break
        if raw_rows is None and all(isinstance(item, dict) for item in value.values()):
            raw_rows = [dict(item, qid=key) for key, item in value.items()]
        if raw_rows is None:
            raw_rows = [value] if ("qid" in value or "question_id" in value) else None
        if not isinstance(raw_rows, list):
            raise GateInputError(f"{source}: expected a list of reference rows")
    else:
        raise GateInputError(f"{source}: expected a JSON object or list")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(raw_rows, 1):
        if not isinstance(row, dict):
            raise GateInputError(f"{source} row {index}: expected an object")
        current_qid = _qid(row, source, index)
        if current_qid in seen:
            raise GateInputError(f"{source}: duplicate qid {current_qid}")
        answer_sessions = row.get("answer_session_ids")
        if not isinstance(answer_sessions, list) or not answer_sessions:
            raise GateInputError(
                f"{source} row {index} ({current_qid}): answer_session_ids must be non-empty"
            )
        normalized_sessions = {str(item).strip() for item in answer_sessions if str(item).strip()}
        if len(normalized_sessions) != len(answer_sessions):
            raise GateInputError(
                f"{source} row {index} ({current_qid}): answer_session_ids contains blank or duplicate values"
            )
        question_type = str(row.get("question_type") or row.get("type") or "unknown").strip()
        rows.append(
            {
                **row,
                "qid": current_qid,
                "answer_session_ids": sorted(normalized_sessions),
                "question_type": question_type or "unknown",
            }
        )
        seen.add(current_qid)
    if not rows:
        raise GateInputError(f"{source}: no rows")
    return rows


def _read_reference(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GateInputError(f"cannot read reference: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GateInputError("reference: invalid JSON") from exc
    return _reference_rows(value, "reference")


def _read_cases(path: Path) -> dict[str, dict[str, list[str]]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GateInputError(f"cannot read cases: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GateInputError("cases: invalid JSON") from exc
    if not isinstance(value, dict) or value.get("schema_version") != "tmcra.v4.regression-cases.1":
        raise GateInputError("cases: unsupported schema_version")
    corpora = value.get("corpora")
    if not isinstance(corpora, dict) or not corpora:
        raise GateInputError("cases: corpora must be a non-empty object")
    result: dict[str, dict[str, list[str]]] = {}
    for name, corpus in corpora.items():
        if not isinstance(corpus, dict):
            raise GateInputError(f"cases corpus {name}: expected an object")
        parsed: dict[str, list[str]] = {}
        for key in ("regressed_qids", "improved_qids"):
            qids = corpus.get(key)
            if not isinstance(qids, list) or any(not str(qid).strip() for qid in qids):
                raise GateInputError(f"cases corpus {name}: {key} must be a list of qids")
            normalized = [str(qid).strip() for qid in qids]
            if len(normalized) != len(set(normalized)):
                raise GateInputError(f"cases corpus {name}: {key} contains duplicates")
            parsed[key] = normalized
        if set(parsed["regressed_qids"]) & set(parsed["improved_qids"]):
            raise GateInputError(f"cases corpus {name}: movement lists overlap")
        result[str(name)] = parsed
    return result


def official_judge_correct(row: Mapping[str, Any], source: str = "official judge") -> bool:
    if str(row.get("error") or "").strip():
        raise GateInputError(f"{source} {row.get('qid', '')}: row contains an error")
    label = row.get("autoeval_label")
    if not isinstance(label, Mapping) or type(label.get("label")) is not bool:
        raise GateInputError(
            f"{source} {row.get('qid', '')}: autoeval_label.label must be boolean"
        )
    return bool(label["label"])


def _index(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row["qid"]): row for row in rows}


def _pairing_report(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
) -> tuple[bool, dict[str, Any], set[str]]:
    sets = {
        "baseline_judge": set(row["qid"] for row in baseline),
        "candidate_judge": set(row["qid"] for row in candidate),
        "candidate_evidence": set(row["qid"] for row in evidence),
        "reference": set(row["qid"] for row in reference),
    }
    expected = sets["baseline_judge"]
    compared_sources = ("candidate_judge", "candidate_evidence")
    exact = all(sets[name] == expected for name in compared_sources) and expected.issubset(
        sets["reference"]
    )
    report = {
        "exact": exact,
        "counts": {name: len(qids) for name, qids in sets.items()},
        "missing_by_source": {
            name: sorted(expected - qids)
            for name, qids in sets.items()
            if expected - qids
        },
        "extra_by_source": {
            name: sorted(qids - expected)
            for name, qids in sets.items()
            if name != "reference" and qids - expected
        },
        "reference_is_superset": expected.issubset(sets["reference"]),
    }
    return exact, report, expected


def _empty_report(error: str | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "promotion_eligible": False,
        "errors": [error] if error else [],
    }
    if error:
        report["checks"] = {
            "exact_qid_pairing": False,
            "regressed_source_coverage": False,
            "candidate_accuracy": False,
            "max_regressions": False,
            "min_improvements": False,
        }
    return report


def evaluate_gate(
    baseline_judge_path: Path,
    candidate_judge_path: Path,
    candidate_evidence_path: Path,
    reference_path: Path,
    *,
    cases_path: Path = DEFAULT_CASES,
    min_candidate_accuracy: float | None = None,
    max_regressions: int = 0,
    min_improvements: int = 0,
) -> dict[str, Any]:
    try:
        if max_regressions < 0 or min_improvements < 0:
            raise GateInputError("promotion thresholds must be non-negative")
        if min_candidate_accuracy is not None and not 0 <= min_candidate_accuracy <= 1:
            raise GateInputError("min_candidate_accuracy must be between 0 and 1")
        baseline = _read_jsonl(baseline_judge_path, "baseline judge")
        candidate = _read_jsonl(candidate_judge_path, "candidate judge")
        evidence = _read_jsonl(candidate_evidence_path, "candidate evidence")
        reference = _read_reference(reference_path)
        cases = _read_cases(cases_path)
        baseline_by_qid = _index(baseline)
        candidate_by_qid = _index(candidate)
        evidence_by_qid = _index(evidence)
        reference_by_qid = _index(reference)
        exact, pairing, qids = _pairing_report(baseline, candidate, evidence, reference)
        if not exact:
            report = _empty_report("inputs do not have exactly matching qid sets")
            report["pairing"] = pairing
            report["known_corpus_outcomes"] = _known_corpus_outcomes(
                cases, baseline_by_qid, candidate_by_qid, reference_by_qid
            )
            return report

        rows: list[dict[str, Any]] = []
        for qid in sorted(qids):
            baseline_correct = official_judge_correct(baseline_by_qid[qid], "baseline judge")
            candidate_correct = official_judge_correct(candidate_by_qid[qid], "candidate judge")
            movement = _movement(baseline_correct, candidate_correct)
            selected = evidence_by_qid[qid].get("selected_session_ids")
            if not isinstance(selected, list):
                raise GateInputError(f"candidate evidence {qid}: selected_session_ids must be a list")
            selected_values = [str(item).strip() for item in selected]
            if any(not item for item in selected_values) or len(set(selected_values)) != len(selected_values):
                raise GateInputError(
                    f"candidate evidence {qid}: selected_session_ids contains blank or duplicate values"
                )
            selected_ids = set(selected_values)
            answer_ids = set(reference_by_qid[qid]["answer_session_ids"])
            missing_answer_session_ids = sorted(answer_ids - selected_ids)
            covered_answer_session_count = len(answer_ids) - len(missing_answer_session_ids)
            session_coverage = {
                "complete": not missing_answer_session_ids,
                "expected_session_count": len(answer_ids),
                "selected_session_count": len(selected_ids),
                "covered_session_count": covered_answer_session_count,
                "hit_rate": round(covered_answer_session_count / len(answer_ids), 6),
                "missing_session_ids": missing_answer_session_ids,
            }
            rows.append(
                {
                    "qid": qid,
                    "question_type": reference_by_qid[qid]["question_type"],
                    "movement": movement,
                    "baseline_correct": baseline_correct,
                    "candidate_correct": candidate_correct,
                    "answer_session_ids": sorted(answer_ids),
                    "selected_session_ids": sorted(selected_ids),
                    "missing_answer_session_ids": missing_answer_session_ids,
                    "session_coverage": session_coverage,
                    "source_coverage_complete": session_coverage["complete"],
                }
            )

        counts = {movement: sum(row["movement"] == movement for row in rows) for movement in MOVEMENTS}
        total = len(rows)
        baseline_correct_count = counts["kept_correct"] + counts["regressed"]
        candidate_correct_count = counts["kept_correct"] + counts["improved"]
        baseline_accuracy = baseline_correct_count / total
        candidate_accuracy = candidate_correct_count / total
        by_type: dict[str, dict[str, Any]] = {}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row["question_type"]].append(row)
        for question_type, values in sorted(grouped.items()):
            type_counts = {
                movement: sum(row["movement"] == movement for row in values)
                for movement in MOVEMENTS
            }
            type_total = len(values)
            by_type[question_type] = {
                "count": type_total,
                "baseline_correct": sum(row["baseline_correct"] for row in values),
                "candidate_correct": sum(row["candidate_correct"] for row in values),
                "baseline_accuracy": round(sum(row["baseline_correct"] for row in values) / type_total, 6),
                "candidate_accuracy": round(sum(row["candidate_correct"] for row in values) / type_total, 6),
                **type_counts,
            }

        regressed = [row for row in rows if row["movement"] == "regressed"]
        coverage_failures = [
            {"qid": row["qid"], "missing_answer_session_ids": sorted(
                set(row["answer_session_ids"]) - set(row["selected_session_ids"])
            )}
            for row in regressed
            if not row["source_coverage_complete"]
        ]
        accuracy_floor = baseline_accuracy if min_candidate_accuracy is None else min_candidate_accuracy
        checks = {
            "exact_qid_pairing": True,
            "regressed_source_coverage": not coverage_failures,
            "candidate_accuracy": candidate_accuracy >= accuracy_floor,
            "max_regressions": counts["regressed"] <= max_regressions,
            "min_improvements": counts["improved"] >= min_improvements,
        }
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed" if all(checks.values()) else "blocked",
            "promotion_eligible": all(checks.values()),
            "errors": [],
            "pairing": pairing,
            "thresholds": {
                "candidate_accuracy_min": accuracy_floor,
                "candidate_accuracy_min_source": "baseline" if min_candidate_accuracy is None else "cli",
                "max_regressions": max_regressions,
                "min_improvements": min_improvements,
            },
            "checks": checks,
            "metrics": {
                "evaluated_count": total,
                "baseline_correct": baseline_correct_count,
                "candidate_correct": candidate_correct_count,
                "baseline_accuracy": round(baseline_accuracy, 6),
                "candidate_accuracy": round(candidate_accuracy, 6),
                **counts,
            },
            "by_question_type": by_type,
            "source_coverage": {
                "reference_backed": True,
                "reference_source": "reference.answer_session_ids",
                "regressed_count": len(regressed),
                "regressed_complete_count": len(regressed) - len(coverage_failures),
                "regressed_complete_rate": round(
                    (len(regressed) - len(coverage_failures)) / len(regressed), 6
                ) if regressed else 1.0,
                "regressed_failures": coverage_failures,
                "all_qids_complete_count": sum(row["source_coverage_complete"] for row in rows),
                "all_qids_complete_rate": round(
                    sum(row["source_coverage_complete"] for row in rows) / total, 6
                ),
                "missing_session_ids_by_qid": {
                    row["qid"]: row["missing_answer_session_ids"]
                    for row in rows
                    if row["missing_answer_session_ids"]
                },
            },
            "reference_backed_session_coverage": {
                "reference_source": "reference.answer_session_ids",
                "evaluated_count": total,
                "complete_count": sum(row["source_coverage_complete"] for row in rows),
                "complete_rate": round(
                    sum(row["source_coverage_complete"] for row in rows) / total, 6
                ),
                "covered_session_count": sum(
                    row["session_coverage"]["covered_session_count"] for row in rows
                ),
                "expected_session_count": sum(
                    row["session_coverage"]["expected_session_count"] for row in rows
                ),
                "missing_session_ids_by_qid": {
                    row["qid"]: row["missing_answer_session_ids"]
                    for row in rows
                    if row["missing_answer_session_ids"]
                },
            },
            "known_corpus_outcomes": _known_corpus_outcomes(
                cases, baseline_by_qid, candidate_by_qid, reference_by_qid
            ),
        }
        return report
    except GateInputError as exc:
        return _empty_report(str(exc))


def _movement(baseline_correct: bool, candidate_correct: bool) -> str:
    if baseline_correct and candidate_correct:
        return "kept_correct"
    if not baseline_correct and candidate_correct:
        return "improved"
    if baseline_correct and not candidate_correct:
        return "regressed"
    return "kept_wrong"


def _known_corpus_outcomes(
    cases: Mapping[str, Mapping[str, Sequence[str]]],
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    outcomes: dict[str, Any] = {}
    for name, corpus in sorted(cases.items()):
        expected_by_qid = {
            qid: "regressed" for qid in corpus["regressed_qids"]
        }
        expected_by_qid.update({qid: "improved" for qid in corpus["improved_qids"]})
        present = sorted(set(expected_by_qid) & set(baseline) & set(candidate))
        observed = []
        for qid in present:
            try:
                baseline_correct = official_judge_correct(baseline[qid], "baseline judge")
                candidate_correct = official_judge_correct(candidate[qid], "candidate judge")
            except GateInputError:
                continue
            observed.append({
                "qid": qid,
                "expected_movement": expected_by_qid[qid],
                "actual_movement": _movement(baseline_correct, candidate_correct),
                "matches_expected": _movement(baseline_correct, candidate_correct) == expected_by_qid[qid],
                "question_type": reference.get(qid, {}).get("question_type", "unknown"),
            })
        outcomes[name] = {
            "expected_regressed_count": len(corpus["regressed_qids"]),
            "expected_improved_count": len(corpus["improved_qids"]),
            "present_count": len(present),
            "missing_qids": sorted(set(expected_by_qid) - set(present)),
            "observed": observed,
        }
    return outcomes


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the zero-API TMCRA V4 regression promotion gate")
    parser.add_argument("--baseline-judge", "--baseline", dest="baseline_judge", type=Path, required=True)
    parser.add_argument("--candidate-judge", "--candidate", dest="candidate_judge", type=Path, required=True)
    parser.add_argument("--candidate-evidence", "--evidence", dest="candidate_evidence", type=Path, required=True)
    parser.add_argument("--reference", "--reference-json", dest="reference", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--min-candidate-accuracy", "--candidate-accuracy-min", dest="min_candidate_accuracy", type=float)
    parser.add_argument("--max-regressions", type=int, default=0)
    parser.add_argument("--min-improvements", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate_gate(
        args.baseline_judge,
        args.candidate_judge,
        args.candidate_evidence,
        args.reference,
        cases_path=args.cases,
        min_candidate_accuracy=args.min_candidate_accuracy,
        max_regressions=args.max_regressions,
        min_improvements=args.min_improvements,
    )
    _write_report(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
