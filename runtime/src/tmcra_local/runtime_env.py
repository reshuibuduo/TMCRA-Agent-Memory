from __future__ import annotations

import ipaddress
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from .config import LOCAL_CONFIG_SCHEMA_VERSION


class LocalRuntimeEnvironmentError(ValueError):
    pass


SUPPORTED_CONFIG_SCHEMAS = {LOCAL_CONFIG_SCHEMA_VERSION}


def load_local_runtime_config(path: Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LocalRuntimeEnvironmentError(f"local runtime config is missing: {source}") from exc
    except json.JSONDecodeError as exc:
        raise LocalRuntimeEnvironmentError(f"local runtime config is invalid JSON: {source}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") not in SUPPORTED_CONFIG_SCHEMAS:
        raise LocalRuntimeEnvironmentError("unsupported local runtime config schema")
    return payload


def _loopback_network(payload: Mapping[str, Any]) -> tuple[str, int]:
    network = payload.get("network")
    if not isinstance(network, Mapping) or network.get("allow_non_loopback") is not False:
        raise LocalRuntimeEnvironmentError("local runtime must explicitly forbid non-loopback binding")
    host = str(network.get("bind_host") or "").strip()
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise LocalRuntimeEnvironmentError("local runtime bind_host must be an IP loopback address") from exc
    if not address.is_loopback:
        raise LocalRuntimeEnvironmentError("local runtime bind_host must be loopback")
    try:
        port = int(network.get("bind_port"))
    except (TypeError, ValueError) as exc:
        raise LocalRuntimeEnvironmentError("local runtime bind_port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise LocalRuntimeEnvironmentError("local runtime bind_port must be between 1 and 65535")
    return host, port


def _model_contract(payload: Mapping[str, Any], *, require_model: bool) -> dict[str, str]:
    embedding = payload.get("embedding")
    if not isinstance(embedding, Mapping):
        raise LocalRuntimeEnvironmentError("local runtime embedding configuration is missing")
    model_path = Path(str(embedding.get("model_path") or "")).expanduser().resolve()
    signature = str(embedding.get("index_signature") or "").strip().lower()
    if len(signature) != 64 or any(character not in "0123456789abcdef" for character in signature):
        raise LocalRuntimeEnvironmentError("embedding index signature is invalid")
    if require_model:
        manifest_path = model_path / "TMCRA_MODEL_MANIFEST.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise LocalRuntimeEnvironmentError(
                f"selected embedding model is not verified: {manifest_path}"
            ) from exc
        installed_signature = str(
            (manifest.get("profile") or {}).get("index_signature")
            if isinstance(manifest, Mapping)
            else ""
        ).lower()
        if installed_signature != signature:
            raise LocalRuntimeEnvironmentError(
                "selected embedding model manifest does not match the configured index signature"
            )
        required = manifest.get("required_files") if isinstance(manifest, Mapping) else None
        if not isinstance(required, Mapping) or not required or not all(
            bool(value) and (model_path / str(name)).is_file()
            for name, value in required.items()
        ):
            raise LocalRuntimeEnvironmentError("selected embedding model files are incomplete")
    return {
        "model_path": str(model_path),
        "profile_id": str(embedding.get("profile_id") or ""),
        "index_signature": signature,
        "dimension": str(int(embedding.get("dimension") or 0)),
        "max_length": str(int(embedding.get("max_length") or 0)),
        "pooling": str(embedding.get("pooling") or ""),
        "query_prefix": str(embedding.get("query_prefix") or ""),
        "document_prefix": str(embedding.get("document_prefix") or ""),
        "subchunk_chars": str(int(embedding.get("subchunk_chars") or 0)),
        "device_preference": str(embedding.get("device_preference") or "auto").lower(),
    }


def _reranker_contract(
    payload: Mapping[str, Any], *, require_model: bool
) -> dict[str, str]:
    reranker = payload.get("reranker")
    if not isinstance(reranker, Mapping):
        # Version 1/2 configs predate selectable rerankers and preserve the
        # original fusion-chain behavior.
        return {
            "profile_id": "current-fusion-reranker",
            "runtime_mode": "fusion",
            "model_path": "",
            "checkpoint_path": "",
        }
    runtime_mode = str(reranker.get("runtime_mode") or "fusion").strip().lower()
    if runtime_mode not in {"dense-only", "semantic-only", "fusion"}:
        raise LocalRuntimeEnvironmentError("local reranker runtime mode is invalid")
    model_path_text = str(reranker.get("model_path") or "").strip()
    checkpoint_text = str(reranker.get("checkpoint_path") or "").strip()
    requires_cross = runtime_mode in {"semantic-only", "fusion"}
    requires_checkpoint = runtime_mode == "fusion"
    if requires_cross and not model_path_text:
        raise LocalRuntimeEnvironmentError("selected reranker requires a model path")
    if requires_checkpoint and not checkpoint_text:
        raise LocalRuntimeEnvironmentError("fusion reranker requires a checkpoint path")
    if require_model and requires_cross:
        model_path = Path(model_path_text).expanduser().resolve()
        manifest_path = model_path / "TMCRA_MODEL_MANIFEST.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise LocalRuntimeEnvironmentError(
                f"selected reranker model is not verified: {manifest_path}"
            ) from exc
        manifest_profile = manifest.get("profile") if isinstance(manifest, Mapping) else None
        if not isinstance(manifest_profile, Mapping) or str(
            manifest_profile.get("id") or ""
        ) != str(reranker.get("profile_id") or ""):
            raise LocalRuntimeEnvironmentError(
                "selected reranker manifest does not match the configured profile"
            )
        manifest_model = manifest_profile.get("model")
        installed_revision = str(
            manifest_model.get("revision")
            if isinstance(manifest_model, Mapping)
            else ""
        )
        if installed_revision != str(reranker.get("revision") or ""):
            raise LocalRuntimeEnvironmentError(
                "selected reranker manifest does not match the configured revision"
            )
        required = manifest.get("required_files") if isinstance(manifest, Mapping) else None
        if not isinstance(required, Mapping) or not required or not all(
            bool(value) and (model_path / str(name)).is_file()
            for name, value in required.items()
        ):
            raise LocalRuntimeEnvironmentError("selected reranker model files are incomplete")
    if require_model and requires_checkpoint:
        checkpoint_path = Path(checkpoint_text).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise LocalRuntimeEnvironmentError(
                f"fusion reranker checkpoint is missing: {checkpoint_path}"
            )
    return {
        "profile_id": str(reranker.get("profile_id") or ""),
        "runtime_mode": runtime_mode,
        "model_path": (
            str(Path(model_path_text).expanduser().resolve())
            if model_path_text
            else ""
        ),
        "checkpoint_path": (
            str(Path(checkpoint_text).expanduser().resolve())
            if checkpoint_text
            else ""
        ),
    }


def resolve_compute_device(preference: str) -> str:
    normalized = str(preference or "auto").strip().lower()
    if normalized in {"cpu", "cuda", "mps"}:
        return normalized
    if normalized != "auto":
        raise LocalRuntimeEnvironmentError("embedding device preference is invalid")
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except (ImportError, RuntimeError):
        pass
    return "cpu"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_graph_assets(model_root: Path) -> tuple[Path, Path]:
    """Verify released graph scorers against the public manifest."""

    manifest_path = model_root / "TMCRA_MODEL_MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise LocalRuntimeEnvironmentError(
            f"TMCRA graph-scoring manifest is missing or invalid: {manifest_path}"
        ) from exc
    entries = manifest.get("files") if isinstance(manifest, Mapping) else None
    if not isinstance(entries, list):
        raise LocalRuntimeEnvironmentError("TMCRA graph-scoring manifest has no file list")
    by_name = {
        str(item.get("name") or ""): item
        for item in entries
        if isinstance(item, Mapping)
    }
    verified: list[Path] = []
    for name in ("node_scorer.pt", "path_scorer.pt"):
        path = model_root / name
        entry = by_name.get(name)
        if not path.is_file() or not isinstance(entry, Mapping):
            raise LocalRuntimeEnvironmentError(
                f"TMCRA graph-scoring asset is missing from the release: {path}"
            )
        try:
            with path.open("rb") as handle:
                prefix = handle.read(128)
        except OSError as exc:
            raise LocalRuntimeEnvironmentError(
                f"TMCRA graph-scoring asset cannot be read: {path}"
            ) from exc
        if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise LocalRuntimeEnvironmentError(
                f"TMCRA graph-scoring asset is a Git LFS pointer; run git lfs pull: {path}"
            )
        expected_bytes = int(entry.get("bytes") or -1)
        expected_sha = str(entry.get("sha256") or "").strip().lower()
        if path.stat().st_size != expected_bytes or _sha256(path) != expected_sha:
            raise LocalRuntimeEnvironmentError(
                f"TMCRA graph-scoring asset failed manifest verification: {path}"
            )
        verified.append(path)
    return verified[0], verified[1]


def _graph_scorer_contract(*, require_model: bool) -> dict[str, str]:
    release_root = Path(__file__).resolve().parents[3]
    model_root = release_root / "models" / "tmcra_v4_longmemeval_s500_20260715"
    node_model = model_root / "node_scorer.pt"
    path_model = model_root / "path_scorer.pt"
    if require_model:
        node_model, path_model = _validate_graph_assets(model_root)
    return {
        "retrieval_mode": "hybrid_node_scored",
        "node_model_path": str(node_model.resolve()),
        "path_model_path": str(path_model.resolve()),
    }


def _validated_openai_base_url(value: Any, *, local_only: bool) -> str:
    base_url = str(value or "").strip().rstrip("/")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/v1"
    ):
        raise LocalRuntimeEnvironmentError(
            "generation endpoint must be a credential-free HTTP(S) /v1 base URL"
        )
    try:
        loopback = ipaddress.ip_address(str(parsed.hostname)).is_loopback
    except ValueError:
        loopback = str(parsed.hostname).lower() == "localhost"
    if local_only and (parsed.scheme != "http" or not loopback):
        raise LocalRuntimeEnvironmentError(
            "bundled generation engine must use an exact loopback HTTP /v1 endpoint"
        )
    if not local_only and parsed.scheme == "http" and not loopback:
        raise LocalRuntimeEnvironmentError(
            "plain HTTP BYOK generation endpoints must use an exact loopback host"
        )
    return base_url


def _generation_contract(
    payload: Mapping[str, Any], *, require_runtime: bool
) -> dict[str, str]:
    generation = payload.get("generation")
    if not isinstance(generation, Mapping):
        raise LocalRuntimeEnvironmentError(
            "local runtime generation configuration is missing; retrieval-only "
            "configuration cannot start as complete TMCRA"
        )
    routes = generation.get("task_routes")
    if not isinstance(routes, Mapping):
        raise LocalRuntimeEnvironmentError("generation task routes are missing")
    required_tasks = ("memory_writer",)
    policies: list[str] = []
    for task_id in required_tasks:
        route = routes.get(task_id)
        if not isinstance(route, Mapping):
            raise LocalRuntimeEnvironmentError(
                f"required generation task is missing: {task_id}"
            )
        policy = str(route.get("policy_id") or "").strip()
        if policy not in {"local-model", "byok"}:
            raise LocalRuntimeEnvironmentError(
                f"required generation task {task_id} has no executable local/BYOK route"
            )
        policies.append(policy)
    if len(set(policies)) != 1:
        raise LocalRuntimeEnvironmentError(
            "required generation tasks must share one execution source"
        )
    knowledge = routes.get("personal_knowledge")
    if isinstance(knowledge, Mapping):
        knowledge_policy = str(knowledge.get("policy_id") or "").strip()
        if knowledge_policy not in {policies[0], "disabled"}:
            raise LocalRuntimeEnvironmentError(
                "personal knowledge must share the generation engine or be disabled"
            )
    source = policies[0]
    if source == "local-model":
        engine = generation.get("shared_local_engine")
        if not isinstance(engine, Mapping):
            raise LocalRuntimeEnvironmentError("shared local generation engine is missing")
        base_url = _validated_openai_base_url(
            engine.get("base_url"), local_only=True
        )
        model_id = str(engine.get("model_id") or "").strip()
        profile_id = str(engine.get("profile_id") or "").strip()
        model_path_text = str(engine.get("model_path") or "").strip()
        model_root_text = str(engine.get("model_root") or "").strip()
        runtime_executable_text = str(engine.get("runtime_executable") or "").strip()
        key_file_text = str(engine.get("api_key_file") or "").strip()
        if not all(
            (model_path_text, model_root_text, runtime_executable_text, key_file_text)
        ):
            raise LocalRuntimeEnvironmentError(
                "shared local generation paths are incomplete"
            )
        model_path = Path(model_path_text).expanduser().resolve()
        model_root = Path(model_root_text).expanduser().resolve()
        runtime_executable = Path(
            runtime_executable_text
        ).expanduser().resolve()
        key_file = Path(key_file_text).expanduser().resolve()
        try:
            context_tokens = int(engine.get("context_tokens") or 0)
            max_output_tokens = int(engine.get("max_output_tokens") or 0)
        except (TypeError, ValueError) as exc:
            raise LocalRuntimeEnvironmentError(
                "local generation context contract is invalid"
            ) from exc
        if not profile_id or not model_id or context_tokens < 32_768:
            raise LocalRuntimeEnvironmentError(
                "local generation profile must provide a model and at least 32K context"
            )
        if max_output_tokens <= 0 or max_output_tokens >= context_tokens:
            raise LocalRuntimeEnvironmentError(
                "local generation output budget is invalid"
            )
        key = ""
        if key_file.is_file():
            key = key_file.read_text(encoding="utf-8").strip()
        if require_runtime:
            manifest_path = model_root / "TMCRA_MODEL_MANIFEST.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                raise LocalRuntimeEnvironmentError(
                    f"shared generation model is not verified: {manifest_path}"
                ) from exc
            manifest_profile = (
                manifest.get("profile") if isinstance(manifest, Mapping) else None
            )
            required = (
                manifest.get("required_files")
                if isinstance(manifest, Mapping)
                else None
            )
            if (
                not isinstance(manifest_profile, Mapping)
                or str(manifest_profile.get("id") or "") != profile_id
                or str(manifest_profile.get("revision") or "")
                != str(engine.get("revision") or "")
                or not isinstance(required, Mapping)
                or not required
                or not all(
                    bool(value) and (model_root / str(name)).is_file()
                    for name, value in required.items()
                )
                or not model_path.is_file()
            ):
                raise LocalRuntimeEnvironmentError(
                    "shared generation model manifest is incomplete or mismatched"
                )
            if not runtime_executable.is_file():
                raise LocalRuntimeEnvironmentError(
                    f"llama.cpp runtime is missing: {runtime_executable}"
                )
            if not key:
                raise LocalRuntimeEnvironmentError(
                    f"local generation loopback key is missing: {key_file}"
                )
        return {
            "source": source,
            "provider": "local-qwen",
            "base_url": base_url,
            "model_id": model_id,
            "api_key": key,
            "profile_id": profile_id,
            "runtime_executable": str(runtime_executable),
            "model_path": str(model_path),
            "api_key_file": str(key_file),
            "context_tokens": str(context_tokens),
            "max_output_tokens": str(max_output_tokens),
            "writer_adapter": "qwen-local-v1",
            "reviewer_adapter": "qwen-local-reconciliation-v1",
        }
    engine = generation.get("byok_engine")
    if not isinstance(engine, Mapping):
        raise LocalRuntimeEnvironmentError("BYOK generation engine is missing")
    base_url = _validated_openai_base_url(engine.get("base_url"), local_only=False)
    model_id = str(engine.get("model_id") or "").strip()
    key_env = str(engine.get("api_key_env") or "").strip()
    key_file_text = str(engine.get("api_key_file") or "").strip()
    key_file = Path(key_file_text).expanduser().resolve() if key_file_text else None
    key = str(os.getenv(key_env) or "").strip() if key_env else ""
    if not key and key_file is not None and key_file.is_file():
        key = key_file.read_text(encoding="utf-8").strip()
    try:
        context_tokens = int(engine.get("context_tokens") or 0)
        max_output_tokens = int(engine.get("max_output_tokens") or 0)
    except (TypeError, ValueError) as exc:
        raise LocalRuntimeEnvironmentError("BYOK generation limits are invalid") from exc
    if not model_id or (not key_env and key_file is None):
        raise LocalRuntimeEnvironmentError("BYOK generation route is incomplete")
    if context_tokens < 32_768 or max_output_tokens <= 0 or max_output_tokens >= context_tokens:
        raise LocalRuntimeEnvironmentError(
            "BYOK generation route must provide at least 32K context and a valid output budget"
        )
    if require_runtime and not key:
        raise LocalRuntimeEnvironmentError(
            "BYOK credential is empty; set the configured environment variable "
            "or local key file"
        )
    return {
        "source": source,
        "provider": "openai-compatible",
        "base_url": base_url,
        "model_id": model_id,
        "api_key": key,
        "profile_id": "user-declared",
        "runtime_executable": "",
        "model_path": "",
        "api_key_file": "",
        "credential_file": str(key_file) if key_file is not None else "",
        "context_tokens": str(context_tokens),
        "max_output_tokens": str(max_output_tokens),
        "writer_adapter": "openai-memory-v1",
        "reviewer_adapter": "openai-memory-reconciliation-v1",
    }


def build_service_environment(
    payload: Mapping[str, Any],
    *,
    config_path: Path,
    require_model: bool = True,
) -> dict[str, str]:
    if payload.get("schema_version") not in SUPPORTED_CONFIG_SCHEMAS:
        raise LocalRuntimeEnvironmentError("unsupported local runtime config schema")
    if payload.get("mode") != "local-first":
        raise LocalRuntimeEnvironmentError("local runtime config mode must be local-first")
    host, port = _loopback_network(payload)
    model = _model_contract(payload, require_model=require_model)
    reranker = _reranker_contract(payload, require_model=require_model)
    generation = _generation_contract(payload, require_runtime=require_model)
    graph_scorer = _graph_scorer_contract(require_model=require_model)
    installation = payload.get("installation")
    if not isinstance(installation, Mapping):
        raise LocalRuntimeEnvironmentError("local runtime installation metadata is missing")
    config_root = Path(str(installation.get("config_root") or "")).expanduser().resolve()
    state_dir = config_root / "state"
    host_for_url = f"[{host}]" if ":" in host else host
    device = resolve_compute_device(model["device_preference"])
    environment = {
        "TMCRA_LOCAL_RUNTIME_CONFIG": str(Path(config_path).expanduser().resolve()),
        "TMCRA_SERVICE_DEPLOYMENT_MODE": "local",
        "TMCRA_SERVICE_BIND_HOST": host,
        "TMCRA_SERVICE_BIND_PORT": str(port),
        "TMCRA_SERVICE_PUBLIC_BASE_URL": f"http://{host_for_url}:{port}",
        "TMCRA_SERVICE_STATE_DIR": str(state_dir),
        "TMCRA_SERVICE_CONTROL_DB": str(state_dir / "control.sqlite3"),
        "TMCRA_EMBEDDING_MODEL": model["model_path"],
        "TMCRA_EMBEDDING_PROFILE_ID": model["profile_id"],
        "TMCRA_EMBEDDING_INDEX_SIGNATURE": model["index_signature"],
        "TMCRA_EMBEDDING_DIMENSION": model["dimension"],
        "TMCRA_EMBEDDING_MAX_LENGTH": model["max_length"],
        "TMCRA_EMBEDDING_POOLING": model["pooling"],
        "TMCRA_EMBEDDING_QUERY_PREFIX": model["query_prefix"],
        "TMCRA_EMBEDDING_DOCUMENT_PREFIX": model["document_prefix"],
        "TMCRA_EMBEDDING_SUBCHUNK_CHARS": model["subchunk_chars"],
        "TMCRA_EMBEDDER_MODEL_PATH": model["model_path"],
        "TMCRA_EMBEDDER_DEVICE": device,
        "TMCRA_EMBEDDER_MODEL_MAX_LENGTH": model["max_length"],
        "TMCRA_EMBEDDER_POOLING": model["pooling"],
        "TMCRA_EMBEDDER_QUERY_PREFIX": model["query_prefix"],
        "TMCRA_EMBEDDER_DOCUMENT_PREFIX": model["document_prefix"],
        "TMCRA_WRITE_EMBEDDER_INDEX_MODE": "local_transformers",
        "TMCRA_EMBEDDER_INDEX_RECALL_MODE": "local_transformers",
        "TMCRA_EMBEDDER_INDEX_RECALL_K": "48",
        "TMCRA_EMBEDDER_PRE_RECALL_MODE": "auto",
        "TMCRA_EMBEDDER_PRE_RECALL_K": "48",
        "TMCRA_EMBEDDER_FUSION_MODE": "on",
        "TMCRA_EMBEDDER_FUSION_TOP_K": "16",
        "TMCRA_EMBEDDER_FUSION_SELECT_K": "4",
        "TMCRA_GRAPH_RETRIEVAL_MODE": graph_scorer["retrieval_mode"],
        "TMCRA_GRAPH_NODE_MODEL_PATH": graph_scorer["node_model_path"],
        "TMCRA_GRAPH_PATH_MODEL_PATH": graph_scorer["path_model_path"],
        "TMCRA_SERVICE_DEVICE": device,
        "TMCRA_SERVICE_GRAPH_DEVICE": device,
        "TMCRA_RERANKER_PROFILE_ID": reranker["profile_id"],
        "TMCRA_RERANKER_MODE": reranker["runtime_mode"],
        "TMCRA_WRITER_PROVIDER": generation["provider"],
        "TMCRA_WRITER_BASE_URL": generation["base_url"],
        "TMCRA_WRITER_MODEL": generation["model_id"],
        "TMCRA_WRITER_API_KEY_POOL": generation["api_key"],
        "TMCRA_WRITER_PROMPT_ADAPTER": generation["writer_adapter"],
        "TMCRA_WRITER_REVIEWER_PROVIDER": generation["provider"],
        "TMCRA_WRITER_REVIEWER_BASE_URL": generation["base_url"],
        "TMCRA_WRITER_REVIEWER_MODEL": generation["model_id"],
        "TMCRA_WRITER_REVIEWER_API_KEY_POOL": generation["api_key"],
        "TMCRA_WRITER_REVIEWER_PROMPT_ADAPTER": generation["reviewer_adapter"],
        "TMCRA_WRITER_MAX_TOKENS": generation["max_output_tokens"],
        "TMCRA_WRITER_TIMEOUT_SECONDS": (
            "900" if generation["source"] == "local-model" else "180"
        ),
        "TMCRA_WRITER_POOL_REQUEST_TIMEOUT_SECONDS": (
            "1200" if generation["source"] == "local-model" else "900"
        ),
        "TMCRA_LOCAL_GENERATION_SOURCE": generation["source"],
        "TMCRA_LOCAL_GENERATION_PROFILE_ID": generation["profile_id"],
        "TMCRA_LOCAL_GENERATION_RUNTIME": generation["runtime_executable"],
        "TMCRA_LOCAL_GENERATION_MODEL_PATH": generation["model_path"],
        "TMCRA_LOCAL_GENERATION_CONTEXT_TOKENS": generation["context_tokens"],
        "TMCRA_PROJECTION_RESERVED_PRODUCTION_SLOTS": "0",
    }
    if reranker["model_path"]:
        environment["TMCRA_CROSS_MODEL"] = reranker["model_path"]
    if reranker["checkpoint_path"]:
        environment["TMCRA_CHECKPOINT"] = reranker["checkpoint_path"]
    if generation["api_key_file"]:
        environment.update(
            {
                "TMCRA_LOCAL_WRITER_API_KEY_FILE": generation["api_key_file"],
                "TMCRA_LOCAL_REVIEWER_API_KEY_FILE": generation["api_key_file"],
            }
        )
    llm = payload.get("llm")
    if not isinstance(llm, Mapping):
        raise LocalRuntimeEnvironmentError("local runtime LLM policy is missing")
    task_policies = llm.get("task_policies")
    if not isinstance(task_policies, Mapping):
        default_policy = str(llm.get("policy_id") or "local-model")
        task_policies = {
            name: {"policy_id": default_policy}
            for name in (
                "memory_writer",
                "personal_knowledge",
            )
        }
    for task_id, item in task_policies.items():
        if not isinstance(item, Mapping):
            raise LocalRuntimeEnvironmentError(f"invalid generation task policy: {task_id}")
        name = str(task_id).upper()
        environment[f"TMCRA_LOCAL_{name}_POLICY"] = str(item.get("policy_id") or "")
    task_models = llm.get("task_models")
    if isinstance(task_models, Mapping):
        for task_id, item in task_models.items():
            if not isinstance(item, Mapping):
                raise LocalRuntimeEnvironmentError(
                    f"invalid generation task model: {task_id}"
                )
            environment[f"TMCRA_LOCAL_{str(task_id).upper()}_MODEL"] = str(
                item.get("model_id") or ""
            )
    knowledge = task_policies.get("personal_knowledge")
    if isinstance(knowledge, Mapping) and knowledge.get("policy_id") == "disabled":
        environment["TMCRA_LOCAL_PERSONAL_KNOWLEDGE_ENABLED"] = "0"
    else:
        environment["TMCRA_LOCAL_PERSONAL_KNOWLEDGE_ENABLED"] = "1"
    byok = llm.get("byok")
    if isinstance(byok, Mapping) and str(byok.get("api_key_env") or ""):
        environment.update(
            {
                "TMCRA_LOCAL_BYOK_PROVIDER": str(byok.get("provider") or ""),
                "TMCRA_LOCAL_BYOK_BASE_URL": str(byok.get("base_url") or ""),
                "TMCRA_LOCAL_BYOK_MODEL": str(byok.get("model") or ""),
                "TMCRA_LOCAL_BYOK_API_KEY_ENV": str(byok.get("api_key_env") or ""),
            }
        )
    return environment
