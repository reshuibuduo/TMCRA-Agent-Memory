#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MILLION = 1_000_000


@dataclass(frozen=True)
class ModelRate:
    cache_hit_input_cny_per_million: float
    cache_miss_input_cny_per_million: float
    output_cny_per_million: float


OFFICIAL_DEEPSEEK_RATES = {
    "deepseek-v4-flash": ModelRate(0.02, 1.0, 2.0),
    "deepseek-v4-pro": ModelRate(0.025, 3.0, 6.0),
}

USAGE_KEYS = {
    "prompt_tokens",
    "completion_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "cached_tokens",
}
PHYSICAL_LIST_KEYS = (
    "requests",
    "physical_requests",
    "api_attempts",
    "tier_calls",
)
SQLITE_CALL_TABLES = {
    "slow_graph_archived_attempts",
    "slow_graph_attempts",
    "v4_slow_graph_attempts",
    "v4_reconciliation_attempts",
    "v4_batch_journal",
    "v4_reconciliation_jobs",
    "v4_subject_attribution_audits",
}
SLOW_INTERRUPTION_ERROR = (
    "claim lease expired; external call outcome uncertain; explicit resume required"
)


class CostReportError(RuntimeError):
    pass


def _text(value: Any) -> str:
    return str(value or "").strip()


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise CostReportError(f"{label} must be an integer")
    try:
        output = int(value or 0)
    except (TypeError, ValueError) as exc:
        raise CostReportError(f"{label} must be an integer") from exc
    if output < 0:
        raise CostReportError(f"{label} cannot be negative")
    return output


def _model(record: Mapping[str, Any], inherited: str) -> str:
    direct = _text(record.get("model"))
    if direct:
        return direct
    request = record.get("request")
    if isinstance(request, Mapping):
        requested = _text(request.get("model"))
        if requested:
            return requested
    model_config = record.get("model_config")
    if isinstance(model_config, Mapping):
        configured = _text(model_config.get("model"))
        if configured:
            return configured
    return inherited


def _stage(record: Mapping[str, Any], inherited: str) -> str:
    for key in ("tier_stage", "stage", "route", "routing_reason", "call_kind"):
        value = _text(record.get(key))
        if value:
            return value
    if _text(record.get("planner_version")):
        return "recall_planner"
    return inherited or "unknown"


def _usage(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = record.get("usage")
    if isinstance(value, Mapping) and USAGE_KEYS.intersection(value):
        return value
    if USAGE_KEYS.intersection(record):
        return record
    return None


def _call_id(record: Mapping[str, Any], *, source: str, path: str) -> str:
    for key in ("physical_call_id", "response_id", "request_id", "id"):
        value = _text(record.get(key))
        if value:
            return f"{key}:{value}"
    material = {
        "source": source,
        "path": path,
        "request_sha256": _text(record.get("request_sha256")),
        "started_at": record.get("started_at"),
        "completed_at": record.get("completed_at"),
    }
    return "derived:" + hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def extract_physical_calls(
    value: Any,
    *,
    source: str,
    path: str = "root",
    inherited_model: str = "",
    inherited_stage: str = "",
) -> list[dict[str, Any]]:
    if isinstance(value, list):
        output: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            output.extend(
                extract_physical_calls(
                    item,
                    source=source,
                    path=f"{path}[{index}]",
                    inherited_model=inherited_model,
                    inherited_stage=inherited_stage,
                )
            )
        return output
    if not isinstance(value, Mapping):
        return []

    model = _model(value, inherited_model)
    stage = _stage(value, inherited_stage)
    for key in PHYSICAL_LIST_KEYS:
        children = value.get(key)
        if isinstance(children, list) and children:
            return extract_physical_calls(
                children,
                source=source,
                path=f"{path}.{key}",
                inherited_model=model,
                inherited_stage=stage,
            )

    explicitly_nonphysical = (
        value.get("physical_api_call") is False
        and int(value.get("physical_api_calls", 0) or 0) == 0
        and not _text(value.get("physical_call_id"))
    )
    if explicitly_nonphysical:
        return []

    usage = _usage(value)
    if usage is not None:
        return [
            {
                "call_id": _call_id(value, source=source, path=path),
                "source": source,
                "path": path,
                "model": model or "unknown",
                "stage": stage,
                "status": _text(value.get("status")) or "usage_recorded",
                "usage": dict(usage),
                "usage_recorded": True,
                "external_call_outcome_unknown": bool(
                    value.get("external_call_outcome_unknown")
                ),
            }
        ]

    physical_without_usage = bool(value.get("physical_api_call")) or bool(
        _text(value.get("physical_call_id"))
    )
    if physical_without_usage:
        return [
            {
                "call_id": _call_id(value, source=source, path=path),
                "source": source,
                "path": path,
                "model": model or "unknown",
                "stage": stage,
                "status": _text(value.get("status")) or "usage_missing",
                "usage": {},
                "usage_recorded": False,
                "external_call_outcome_unknown": bool(
                    value.get("external_call_outcome_unknown")
                ),
            }
        ]

    output = []
    for key, child in value.items():
        if key in {"request", "model_config"}:
            continue
        if isinstance(child, (Mapping, list)):
            output.extend(
                extract_physical_calls(
                    child,
                    source=source,
                    path=f"{path}.{key}",
                    inherited_model=model,
                    inherited_stage=stage,
                )
            )
    return output


def read_json_records(path: Path) -> list[Any]:
    if path.suffix.lower() == ".jsonl":
        values = []
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    values.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise CostReportError(f"invalid JSON at {path}:{line_number}") from exc
        return values
    try:
        return [json.loads(path.read_text(encoding="utf-8"))]
    except json.JSONDecodeError as exc:
        raise CostReportError(f"invalid JSON: {path}") from exc


def sqlite_call_metadata(path: Path) -> list[tuple[str, Any]]:
    output: list[tuple[str, Any]] = []
    with closing(sqlite3.connect(path)) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        for table in sorted(tables):
            if table not in SQLITE_CALL_TABLES:
                continue
            columns = {
                str(row[1])
                for row in connection.execute(f'PRAGMA table_info("{table}")')
            }
            for column in (
                "call_metadata_json",
                "api_metadata_json",
                "response_metadata_json",
            ):
                if column not in columns:
                    continue
                query = f'SELECT rowid,"{column}" FROM "{table}" WHERE "{column}" IS NOT NULL AND "{column}" != \'\''
                for rowid, raw in connection.execute(query):
                    try:
                        output.append((f"{table}:{rowid}:{column}", json.loads(raw)))
                    except json.JSONDecodeError as exc:
                        raise CostReportError(
                            f"invalid call metadata JSON in {path}:{table}:{rowid}"
                        ) from exc
            if table in {
                "slow_graph_archived_attempts",
                "slow_graph_attempts",
                "v4_slow_graph_attempts",
            } and {
                "attempt_id",
                "status",
                "call_metadata_json",
                "error",
            }.issubset(columns):
                for rowid, attempt_id in connection.execute(
                    f'SELECT rowid,"attempt_id" FROM "{table}" '
                    "WHERE status='expired' AND error=? "
                    "AND (call_metadata_json='{}' OR call_metadata_json='')",
                    (SLOW_INTERRUPTION_ERROR,),
                ):
                    output.append(
                        (
                            f"{table}:{rowid}:interrupted_external_call",
                            {
                                "physical_api_call": True,
                                "physical_call_id": f"slow-interrupted:{attempt_id}",
                                "model": "unknown",
                                "stage": "slow_graph_interrupted",
                                "status": "external_call_outcome_unknown",
                                "external_call_outcome_unknown": True,
                            },
                        )
                    )
    return output


def normalize_usage(call: Mapping[str, Any]) -> dict[str, Any]:
    usage = dict(call["usage"])
    prompt = _integer(usage.get("prompt_tokens"), label="prompt_tokens")
    completion = _integer(
        usage.get("completion_tokens"), label="completion_tokens"
    )
    hit_value = usage.get(
        "prompt_cache_hit_tokens",
        usage.get("cache_read_input_tokens", usage.get("cached_tokens")),
    )
    miss_value = usage.get(
        "prompt_cache_miss_tokens", usage.get("cache_miss_input_tokens")
    )
    has_hit = hit_value is not None
    has_miss = miss_value is not None
    hit = _integer(hit_value, label="prompt_cache_hit_tokens") if has_hit else 0
    miss = _integer(miss_value, label="prompt_cache_miss_tokens") if has_miss else 0
    if has_hit and has_miss and hit + miss != prompt:
        raise CostReportError(
            f"{call['call_id']}: cache hit+miss tokens do not equal prompt_tokens"
        )
    if has_hit and not has_miss:
        if hit > prompt:
            raise CostReportError(f"{call['call_id']}: cache hit tokens exceed prompt")
        miss = prompt - hit
        has_miss = True
    if has_miss and not has_hit:
        if miss > prompt:
            raise CostReportError(f"{call['call_id']}: cache miss tokens exceed prompt")
        hit = prompt - miss
        has_hit = True
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cache_hit_tokens": hit,
        "cache_miss_tokens": miss,
        "cache_breakdown_exact": has_hit and has_miss,
        "usage_recorded": bool(call.get("usage_recorded", True)),
    }


def price_usage(model: str, usage: Mapping[str, Any]) -> dict[str, Any]:
    rate = OFFICIAL_DEEPSEEK_RATES.get(model)
    if rate is None:
        return {
            "priced": False,
            "exact_cost_cny": None,
            "min_cost_cny": None,
            "max_cost_cny": None,
        }
    output_cost = (
        int(usage["completion_tokens"]) * rate.output_cny_per_million / MILLION
    )
    if bool(usage["cache_breakdown_exact"]):
        input_cost = (
            int(usage["cache_hit_tokens"])
            * rate.cache_hit_input_cny_per_million
            + int(usage["cache_miss_tokens"])
            * rate.cache_miss_input_cny_per_million
        ) / MILLION
        exact = input_cost + output_cost
        return {
            "priced": True,
            "exact_cost_cny": exact,
            "min_cost_cny": exact,
            "max_cost_cny": exact,
        }
    prompt = int(usage["prompt_tokens"])
    return {
        "priced": True,
        "exact_cost_cny": None,
        "min_cost_cny": output_cost
        + prompt * rate.cache_hit_input_cny_per_million / MILLION,
        "max_cost_cny": output_cost
        + prompt * rate.cache_miss_input_cny_per_million / MILLION,
    }


def build_report(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    unique: dict[str, Mapping[str, Any]] = {}
    duplicate_observations = 0
    for call in calls:
        call_id = _text(call.get("call_id"))
        if not call_id:
            raise CostReportError("physical call lacks call_id")
        existing = unique.get(call_id)
        if existing is not None:
            existing_has_usage = bool(existing.get("usage_recorded", True))
            current_has_usage = bool(call.get("usage_recorded", True))
            if current_has_usage and not existing_has_usage:
                unique[call_id] = call
                duplicate_observations += 1
                continue
            if existing_has_usage and not current_has_usage:
                duplicate_observations += 1
                continue
            same_physical_call = (
                _text(existing.get("model")) == _text(call.get("model"))
                and normalize_usage(existing) == normalize_usage(call)
            )
            if not same_physical_call:
                raise CostReportError(f"physical call ID collision: {call_id}")
            duplicate_observations += 1
            continue
        unique[call_id] = call

    buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "physical_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
            "calls_without_cache_breakdown": 0,
            "calls_without_usage": 0,
            "exact_cost_cny": 0.0,
            "min_cost_cny": 0.0,
            "max_cost_cny": 0.0,
            "unpriced_calls": 0,
        }
    )
    exact_total = 0.0
    min_total = 0.0
    max_total = 0.0
    all_priced = True
    all_exact = True
    normalized_calls = []
    unknown_outcome_call_count = 0
    for call_id, call in sorted(unique.items()):
        usage = normalize_usage(call)
        model = _text(call.get("model")) or "unknown"
        stage = _text(call.get("stage")) or "unknown"
        price = price_usage(model, usage)
        bucket = buckets[(stage, model)]
        bucket["physical_calls"] += 1
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "cache_hit_tokens",
            "cache_miss_tokens",
        ):
            bucket[key] += int(usage[key])
        if not usage["cache_breakdown_exact"]:
            bucket["calls_without_cache_breakdown"] += 1
            all_exact = False
        if not usage["usage_recorded"]:
            bucket["calls_without_usage"] += 1
            all_priced = False
            all_exact = False
        if bool(call.get("external_call_outcome_unknown")):
            unknown_outcome_call_count += 1
        if not price["priced"]:
            bucket["unpriced_calls"] += 1
            all_priced = False
            all_exact = False
        else:
            bucket["min_cost_cny"] += float(price["min_cost_cny"])
            bucket["max_cost_cny"] += float(price["max_cost_cny"])
            min_total += float(price["min_cost_cny"])
            max_total += float(price["max_cost_cny"])
            if price["exact_cost_cny"] is not None:
                bucket["exact_cost_cny"] += float(price["exact_cost_cny"])
                exact_total += float(price["exact_cost_cny"])
        normalized_calls.append(
            {
                **{key: value for key, value in call.items() if key != "usage"},
                "usage": usage,
                "price": price,
            }
        )

    by_stage_model = []
    for (stage, model), bucket in sorted(buckets.items()):
        item = {"stage": stage, "model": model, **bucket}
        for key in ("exact_cost_cny", "min_cost_cny", "max_cost_cny"):
            item[key] = round(float(item[key]), 9)
        if (
            item["unpriced_calls"]
            or item["calls_without_cache_breakdown"]
            or item["calls_without_usage"]
        ):
            item["exact_cost_cny"] = None
        by_stage_model.append(item)
    return {
        "schema_version": "tmcra.v4.cost-report.1",
        "pricing_basis": "DeepSeek official CNY token rates configured in report",
        "physical_call_count": len(unique),
        "definite_physical_call_count": len(unique) - unknown_outcome_call_count,
        "unknown_outcome_call_count": unknown_outcome_call_count,
        "duplicate_observation_count": duplicate_observations,
        "all_calls_priced": all_priced,
        "cache_breakdown_complete": all_exact and all_priced,
        "exact_cost_cny": round(exact_total, 9) if all_exact and all_priced else None,
        "known_priced_exact_component_cny": round(exact_total, 9),
        "known_priced_min_cost_cny": round(min_total, 9),
        "known_priced_max_cost_cny": round(max_total, 9),
        "min_cost_cny": round(min_total, 9) if all_priced else None,
        "max_cost_cny": round(max_total, 9) if all_priced else None,
        "by_stage_model": by_stage_model,
        "calls": normalized_calls,
    }


def collect_calls(json_paths: Iterable[Path], sqlite_paths: Iterable[Path]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in json_paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        for index, value in enumerate(read_json_records(resolved)):
            output.extend(
                extract_physical_calls(
                    value, source=str(resolved), path=f"record[{index}]"
                )
            )
    for path in sqlite_paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        for location, value in sqlite_call_metadata(resolved):
            output.extend(
                extract_physical_calls(
                    value, source=str(resolved), path=location
                )
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="TMCRA V4 physical API cost audit")
    parser.add_argument("--json", action="append", default=[], type=Path)
    parser.add_argument("--sqlite", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.json and not args.sqlite:
        raise CostReportError("at least one --json or --sqlite input is required")
    report = build_report(collect_calls(args.json, args.sqlite))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({key: report[key] for key in (
        "physical_call_count", "exact_cost_cny", "min_cost_cny", "max_cost_cny"
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
