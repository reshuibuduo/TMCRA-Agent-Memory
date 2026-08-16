from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


CATALOG_SCHEMA_VERSION = "tmcra.local-model-catalog.6"
INDEX_SIGNATURE_VERSION = "tmcra.embedding-index-signature.1"
RUNTIME_DEPENDENCY_CONTRACT = {
    "schema_version": "tmcra.local-runtime-dependencies.1",
    "distribution": "installer-managed-venv",
    "validated_stack": {
        "python": "3.12",
        "torch": "2.11.0",
        "transformers": "5.10.2",
    },
    "notes": [
        "The installer must verify real inference before reporting ready.",
        "The installer creates an isolated Python 3.12 environment.",
        "This stack passed compact embedding plus released node/path scorer inference on 2026-08-16.",
    ],
}


class ModelSelectionError(ValueError):
    """A user-actionable local model selection error."""


@dataclass(frozen=True)
class EffectEvidence:
    status: str
    summary_zh: str
    summary_en: str
    source: str
    metric: str | None = None
    value: float | None = None


@dataclass(frozen=True)
class EmbeddingProfile:
    id: str
    label_zh: str
    label_en: str
    hf_repo: str
    revision: str
    license_spdx: str
    parameter_count: int
    weight_bytes: int
    dimension: int
    pooling: str
    query_prefix: str
    document_prefix: str
    max_length: int
    subchunk_chars: int
    languages: tuple[str, ...]
    min_ram_gib: float
    recommended_ram_gib: float
    recommended_device: str
    selectable: bool
    channel: str
    validation_status: str
    effect: EffectEvidence
    limitations_zh: tuple[str, ...]
    limitations_en: tuple[str, ...]
    download_includes: tuple[str, ...]
    required_files: tuple[str, ...]

    @property
    def weight_gib(self) -> float:
        return round(self.weight_bytes / 1024**3, 2)

    def index_signature_payload(self) -> dict[str, Any]:
        return {
            "schema_version": INDEX_SIGNATURE_VERSION,
            "profile_id": self.id,
            "hf_repo": self.hf_repo,
            "revision": self.revision,
            "dimension": self.dimension,
            "pooling": self.pooling,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "max_length": self.max_length,
            "subchunk_chars": self.subchunk_chars,
        }

    @property
    def index_signature(self) -> str:
        encoded = json.dumps(
            self.index_signature_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["weight_gib"] = self.weight_gib
        payload["index_signature"] = self.index_signature
        payload["requires_index_rebuild_on_change"] = True
        return payload


@dataclass(frozen=True)
class GenerationProfile:
    id: str
    label_zh: str
    label_en: str
    hf_repo: str
    revision: str
    filename: str
    model_alias: str
    license_spdx: str
    parameter_count: int
    weight_bytes: int
    native_context_tokens: int
    configured_context_tokens: int
    max_output_tokens: int
    quantization: str
    min_ram_gib: float
    recommended_ram_gib: float
    recommended_device: str
    selectable: bool
    channel: str
    validation_status: str
    effect: EffectEvidence
    limitations_zh: tuple[str, ...]
    limitations_en: tuple[str, ...]
    metadata_files: tuple[str, ...]

    @property
    def weight_gib(self) -> float:
        return round(self.weight_bytes / 1024**3, 2)

    def as_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["weight_gib"] = self.weight_gib
        payload["runtime"] = "llama.cpp"
        payload["shared_tasks"] = [
            "memory_writer",
            "recall_planner",
            "graph_organizer",
            "personal_knowledge",
        ]
        return payload


_COMMON_SAFE_FILES = (
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "vocab.txt",
    "merges.txt",
    "1_Pooling/*",
)


EMBEDDING_PROFILES: tuple[EmbeddingProfile, ...] = (
    EmbeddingProfile(
        id="compact-zh",
        label_zh="轻量中文",
        label_en="Compact Chinese",
        hf_repo="BAAI/bge-small-zh-v1.5",
        revision="7999e1d3359715c523056ef9478215996d62a620",
        license_spdx="MIT",
        parameter_count=23_954_432,
        weight_bytes=95_800_000,
        dimension=512,
        pooling="cls",
        query_prefix="为这个句子生成表示以用于检索相关文章：",
        document_prefix="",
        max_length=512,
        subchunk_chars=360,
        languages=("zh",),
        min_ram_gib=4.0,
        recommended_ram_gib=8.0,
        recommended_device="CPU",
        selectable=True,
        channel="candidate",
        validation_status="runtime-inference-verified;tmcra-ab-pending",
        effect=EffectEvidence(
            status="official-model-benchmark-only",
            summary_zh="中文检索的轻量候选；尚未完成 TMCRA 同链路消融。",
            summary_en="Compact Chinese retrieval candidate; TMCRA chain ablation is pending.",
            source="BAAI model card / C-MTEB",
            metric="C-MTEB retrieval",
            value=61.77,
        ),
        limitations_zh=(
            "面向中文；多语言项目不应默认选择。",
            "512 token 上限要求本地索引采用更短的子块。",
        ),
        limitations_en=(
            "Chinese-focused; it is not the default for multilingual projects.",
            "The 512-token limit requires shorter local index subchunks.",
        ),
        download_includes=(*_COMMON_SAFE_FILES, "model.safetensors"),
        required_files=("config.json", "tokenizer.json", "model.safetensors"),
    ),
    EmbeddingProfile(
        id="balanced-multilingual",
        label_zh="均衡多语言",
        label_en="Balanced Multilingual",
        hf_repo="intfloat/multilingual-e5-small",
        revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
        license_spdx="MIT",
        parameter_count=117_654_272,
        weight_bytes=470_600_000,
        dimension=384,
        pooling="mean",
        query_prefix="query: ",
        document_prefix="passage: ",
        max_length=512,
        subchunk_chars=360,
        languages=("multilingual", "zh", "en"),
        min_ram_gib=6.0,
        recommended_ram_gib=12.0,
        recommended_device="CPU or GPU",
        selectable=True,
        channel="candidate",
        validation_status="runtime-inference-verified;tmcra-ab-pending",
        effect=EffectEvidence(
            status="official-model-benchmark-only",
            summary_zh="跨中英文软件与项目的默认候选；尚未完成 TMCRA 同链路消融。",
            summary_en="Default candidate for cross-language tools and projects; TMCRA chain ablation is pending.",
            source="BAAI FlagEmbedding model comparison / C-MTEB",
            metric="C-MTEB retrieval",
            value=59.95,
        ),
        limitations_zh=(
            "查询和文档必须使用不同前缀。",
            "512 token 上限要求本地索引采用更短的子块。",
        ),
        limitations_en=(
            "Queries and passages require distinct prefixes.",
            "The 512-token limit requires shorter local index subchunks.",
        ),
        download_includes=(*_COMMON_SAFE_FILES, "model.safetensors"),
        required_files=("config.json", "tokenizer.json", "model.safetensors"),
    ),
    EmbeddingProfile(
        id="enhanced-multilingual",
        label_zh="增强多语言",
        label_en="Enhanced Multilingual",
        hf_repo="intfloat/multilingual-e5-base",
        revision="d13f1b27baf31030b7fd040960d60d909913633f",
        license_spdx="MIT",
        parameter_count=278_043_648,
        weight_bytes=1_134_354_391,
        dimension=768,
        pooling="mean",
        query_prefix="query: ",
        document_prefix="passage: ",
        max_length=512,
        subchunk_chars=360,
        languages=("multilingual", "zh", "en"),
        min_ram_gib=10.0,
        recommended_ram_gib=16.0,
        recommended_device="GPU preferred; CPU supported",
        selectable=True,
        channel="candidate",
        validation_status="runtime-inference-verified;tmcra-ab-pending",
        effect=EffectEvidence(
            status="official-model-benchmark-only",
            summary_zh="官方 E5 表格的 BEIR 分数为 48.9，同系列 small 为 46.6；这不是 TMCRA 成绩，TMCRA 同链路消融待完成。",
            summary_en="The official E5 table reports 48.9 on BEIR versus 46.6 for the small sibling. This is not a TMCRA score; TMCRA chain ablation is pending.",
            source="Microsoft unilm/e5 official model table",
            metric="BEIR",
            value=48.9,
        ),
        limitations_zh=(
            "下载约 1.13 GB，CPU 可运行但速度明显低于均衡档。",
            "512 token 上限要求本地索引采用更短的子块。",
        ),
        limitations_en=(
            "Approximately 1.13 GB to download; CPU is supported but slower than the balanced tier.",
            "The 512-token limit requires shorter local index subchunks.",
        ),
        download_includes=(*_COMMON_SAFE_FILES, "model.safetensors"),
        required_files=("config.json", "tokenizer.json", "model.safetensors"),
    ),
    EmbeddingProfile(
        id="quality-qwen3-0.6b",
        label_zh="高质量多语言（预览）",
        label_en="High-quality Multilingual (Preview)",
        hf_repo="Qwen/Qwen3-Embedding-0.6B",
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        license_spdx="Apache-2.0",
        parameter_count=595_776_512,
        weight_bytes=1_200_000_000,
        dimension=1024,
        pooling="last_token",
        query_prefix="Instruct: Given a user request, retrieve relevant memories that help continue the work.\nQuery:",
        document_prefix="",
        max_length=8192,
        subchunk_chars=1800,
        languages=("multilingual", "zh", "en", "code"),
        min_ram_gib=10.0,
        recommended_ram_gib=16.0,
        recommended_device="GPU preferred",
        selectable=False,
        channel="preview",
        validation_status="adapter-implemented;dependency-and-tmcra-ab-pending",
        effect=EffectEvidence(
            status="official-specification-only",
            summary_zh="32K/多语言能力有潜力，需完成依赖、速度和 TMCRA 消融后开放。",
            summary_en="Promising 32K multilingual option; dependency, speed, and TMCRA ablations are required before release.",
            source="Qwen model card",
        ),
        limitations_zh=(
            "需要 transformers>=4.51.0。",
            "当前不允许稳定安装通道选择。",
        ),
        limitations_en=(
            "Requires transformers>=4.51.0.",
            "Not selectable in the stable installer channel yet.",
        ),
        download_includes=(*_COMMON_SAFE_FILES, "model.safetensors"),
        required_files=("config.json", "tokenizer.json", "model.safetensors"),
    ),
    EmbeddingProfile(
        id="reference-bge-m3",
        label_zh="现有链路参考",
        label_en="Existing-chain Reference",
        hf_repo="BAAI/bge-m3",
        revision="5617a9f61b028005a4858fdac845db406aefb181",
        license_spdx="MIT",
        parameter_count=567_000_000,
        weight_bytes=2_300_000_000,
        dimension=1024,
        pooling="cls",
        query_prefix="",
        document_prefix="",
        max_length=8192,
        subchunk_chars=1800,
        languages=("multilingual", "zh", "en"),
        min_ram_gib=10.0,
        recommended_ram_gib=16.0,
        recommended_device="GPU preferred",
        selectable=False,
        channel="reference",
        validation_status="existing-tmcra-chain-reference",
        effect=EffectEvidence(
            status="tmcra-system-chain-reference-not-isolated",
            summary_zh="用于现有 TMCRA 链路复现；系统总成绩不能归因给该模型。",
            summary_en="Reproduces the existing TMCRA chain; system-level scores are not attributable to this model alone.",
            source="Public TMCRA reproduction profile in this repository",
        ),
        limitations_zh=(
            "权重约 2.3 GB，不作为普通本地安装默认项。",
            "现有结果没有隔离 Embedding 变量。",
        ),
        limitations_en=(
            "Approximately 2.3 GB of weights; not the default local install.",
            "Existing results do not isolate the embedding variable.",
        ),
        download_includes=(*_COMMON_SAFE_FILES, "pytorch_model.bin"),
        required_files=(
            "config.json",
            "tokenizer.json",
            "sentencepiece.bpe.model",
            "pytorch_model.bin",
        ),
    ),
)


GENERATION_PROFILES: tuple[GenerationProfile, ...] = (
    GenerationProfile(
        id="recommended-qwen36",
        label_zh="推荐完整质量本地模型",
        label_en="Recommended full-quality local model",
        hf_repo="unsloth/Qwen3.6-35B-A3B-GGUF",
        revision="a483e9e6cbd595906af30beda3187c2663a1118c",
        filename="Qwen3.6-35B-A3B-UD-IQ3_S.gguf",
        model_alias="tmcra-qwen3.6-35b-a3b-iq3s",
        license_spdx="Apache-2.0",
        parameter_count=35_000_000_000,
        weight_bytes=13_676_723_168,
        native_context_tokens=262_144,
        configured_context_tokens=32_768,
        max_output_tokens=16_384,
        quantization="UD-IQ3_S",
        min_ram_gib=32.0,
        recommended_ram_gib=64.0,
        recommended_device="NVIDIA RTX 5090D 32 GB or better",
        selectable=True,
        channel="suggested",
        validation_status="public-local-runtime-contract-verified",
        effect=EffectEvidence(
            status="public-local-runtime-contract-verified",
            summary_zh=(
                "这是公开本地运行时为记忆写入、重整与个人知识整理推荐的"
                "完整质量模型。召回本身使用本地 Embedding 和图打分，不调用该模型。"
                "发布前仍需在目标 GPU 上完成独立性能验收。"
            ),
            summary_en=(
                "This is the public local runtime's recommended full-quality "
                "model for memory writing, reconciliation, and personal-knowledge "
                "generation. Recall itself uses local embeddings and graph scoring "
                "without calling this model. Independent performance qualification "
                "on the target GPU is still required."
            ),
            source="TMCRA public local runtime model policy",
        ),
        limitations_zh=(
            "权重约 12.74 GiB；32K 上下文还需要额外的 KV Cache 和运行时显存。",
            "RTX 5090D 32 GB 是建议起点，不是任何工作负载下的吞吐保证。",
            "这只是未指定模型时的建议项；用户确认前不会自动下载。",
        ),
        limitations_en=(
            "Approximately 12.74 GiB of weights; a 32K context also requires KV-cache and runtime VRAM.",
            "An RTX 5090D with 32 GB is the suggested starting point, not a throughput guarantee for every workload.",
            "This is only the suggestion when no model is specified; it is never downloaded before confirmation.",
        ),
        metadata_files=("README.md",),
    ),
    GenerationProfile(
        id="official-local-small",
        label_zh="TMCRA 低资源本地模型（预览）",
        label_en="TMCRA Low-resource Local Model (Preview)",
        hf_repo="Qwen/Qwen3-4B-GGUF",
        revision="bc640142c66e1fdd12af0bd68f40445458f3869b",
        filename="Qwen3-4B-Q4_K_M.gguf",
        model_alias="tmcra-qwen3-4b-q4km",
        license_spdx="Apache-2.0",
        parameter_count=4_022_468_096,
        weight_bytes=2_497_280_256,
        native_context_tokens=40_960,
        configured_context_tokens=32_768,
        max_output_tokens=16_384,
        quantization="Q4_K_M",
        min_ram_gib=8.0,
        recommended_ram_gib=16.0,
        recommended_device="CPU or GPU; GPU offload preferred",
        selectable=False,
        channel="preview",
        validation_status=(
            "runtime-inference-verified;tmcra-quality-gate-partial;"
            "writer-failed;recall-planner-failed;slow-graph-experimental"
        ),
        effect=EffectEvidence(
            status="tmcra-quality-gate-partial",
            summary_zh=(
                "真实全链测试通过 Session Graph、Visual Atlas 与个人知识整理；"
                "Writer、Recall Planner 未通过，Slow Graph 仍有过度拆分。"
            ),
            summary_en=(
                "The real-chain gate passed Session Graph, Visual Atlas, and personal "
                "knowledge. Writer and Recall Planner failed; Slow Graph still "
                "over-splits semantic groups."
            ),
            source="TMCRA local generation quality gate, 2026-08-15",
        ),
        limitations_zh=(
            "约 2.5 GB 权重，32K 上下文还会额外占用 KV Cache。",
            "仅供后台投影预览；当前未获准承担前台 Writer 与 Recall Planner。",
        ),
        limitations_en=(
            "Approximately 2.5 GB of weights; a 32K context also consumes KV-cache memory.",
            "Preview-only for background projection; it is not approved as the foreground Writer or Recall Planner.",
        ),
        metadata_files=("LICENSE", "README.md"),
    ),
)


LLM_POLICIES: tuple[dict[str, Any], ...] = (
    {
        "id": "local-model",
        "label_zh": "使用本地模型",
        "label_en": "Use a local model",
        "selectable": True,
        "privacy": "local-only",
        "billing": "no-provider-charge",
        "fallback": "queue-until-local-engine-available",
        "local_weight_bytes": 13_676_723_168,
        "resource_behavior": "shared-local-generation-runtime",
        "effect_status": "public-full-quality-model-suggested;user-configurable",
        "note_zh": "未指定模型时建议 Qwen3.6-35B-A3B UD-IQ3_S；不会静默下载，也不会静默回退云端。",
    },
    {
        "id": "host-model",
        "label_zh": "沿用当前软件模型",
        "label_en": "Use the current host model",
        "selectable": False,
        "privacy": "host-policy",
        "billing": "host-subscription-or-host-api",
        "fallback": "queue-until-host-available",
        "local_weight_bytes": 0,
        "resource_behavior": "uses-host-runtime",
        "effect_status": "host-dependent",
        "note_zh": "宿主提供可调用模型接口时使用；不进行静默云端回退。",
    },
    {
        "id": "byok",
        "label_zh": "使用自己的 API Key",
        "label_en": "Bring your own API key",
        "selectable": True,
        "privacy": "provider-policy",
        "billing": "provider-direct",
        "fallback": "fail-closed",
        "local_weight_bytes": 0,
        "resource_behavior": "remote-provider-or-user-loopback-runtime",
        "effect_status": "user-declared;tmcra-ab-unknown",
        "note_zh": "Key 只从系统凭据或环境变量读取，配置文件仅保存变量名。",
    },
    {
        "id": "prefer-host-then-byok",
        "label_zh": "优先当前软件，必要时使用自有 Key",
        "label_en": "Prefer host, then BYOK",
        "selectable": False,
        "privacy": "mixed-explicit-policy",
        "billing": "host-or-provider-direct",
        "fallback": "explicit-byok-only",
        "local_weight_bytes": 0,
        "resource_behavior": "uses-host-then-explicit-provider",
        "effect_status": "source-dependent",
        "note_zh": "切换条件会写入本地事件账本，方便用户核对来源与费用。",
    },
)


GENERATION_TASKS: tuple[dict[str, Any], ...] = (
    {
        "id": "memory_writer",
        "label_zh": "记忆写入",
        "label_en": "Memory writer",
        "required": True,
        "allows_disabled": False,
        "importance": "critical",
        "effect_zh": "决定用户与 Agent 内容如何被抽取、分角色并写入记忆。该项对最终记忆质量影响最大。",
        "effect_en": "Extracts user and agent content, preserves actor identity, and writes memory. This has the largest direct effect on memory quality.",
        "evidence": {
            "status": "tmcra-controlled-experiment",
            "summary_zh": "MemReader-4B 受控 50 题实验为 31/50，且 Assistant 主体覆盖不足，因此未进入可选稳定通道。",
            "summary_en": "MemReader-4B scored 31/50 in the controlled run and under-covered assistant actors, so it is not a stable selectable option.",
            "source": "docs/MULTI_AGENT_MEMORY_OBSIDIAN_PLAN_20260723.md",
        },
    },
    {
        "id": "personal_knowledge",
        "label_zh": "个人知识库整理",
        "label_en": "Personal knowledge organizer",
        "required": False,
        "allows_disabled": True,
        "importance": "optional",
        "effect_zh": "把已提交、可追溯的记忆整理成人可阅读的知识页面。关闭后不影响基础写入与召回。",
        "effect_en": "Turns committed, traceable memories into human-readable knowledge pages. Disabling it does not stop core write or recall.",
        "evidence": {"status": "tmcra-ab-pending"},
    },
)


RERANKER_POLICIES: tuple[dict[str, Any], ...] = (
    {
        "id": "local-dense-only",
        "label_zh": "轻量本地召回",
        "label_en": "Lightweight local recall",
        "selectable": True,
        "channel": "stable",
        "configuration_mode": "no-cross-model",
        "runtime_mode": "dense-only",
        "validation_status": "runtime-inference-verified;tmcra-ab-pending",
        "model": None,
        "parameter_count": 0,
        "approx_weight_bytes": 0,
        "min_ram_gib": 4.0,
        "recommended_ram_gib": 8.0,
        "recommended_device": "CPU",
        "effect": {
            "status": "tmcra-ab-pending",
            "summary_zh": "只使用向量相似度与候选作用域，不下载 Cross Encoder。资源占用最低；复杂歧义查询可能损失精排质量，TMCRA 同链路消融待完成。",
            "summary_en": "Uses dense similarity and candidate scopes without a cross encoder. It has the smallest footprint; hard ambiguous queries may lose ranking quality and the TMCRA ablation is pending.",
            "source": "TMCRA local runtime contract",
        },
        "limitations_zh": [
            "不复现现有融合重排链路。",
            "安装器不得把现有 TMCRA 系统总分显示为该档位的成绩。",
        ],
        "download_includes": [],
        "required_files": [],
    },
    {
        "id": "compact-cross-reranker",
        "label_zh": "轻量中英精排",
        "label_en": "Compact Chinese-English reranker",
        "selectable": True,
        "channel": "candidate",
        "configuration_mode": "semantic-logit-only",
        "runtime_mode": "semantic-only",
        "validation_status": "runtime-inference-verified;tmcra-ab-pending",
        "model": {
            "hf_repo": "BAAI/bge-reranker-base",
            "revision": "2cfc18c9415c912f9d8155881c133215df768a70",
            "parameter_count": 278_044_931,
            "approx_weight_bytes": 1_112_177_724,
            "languages": ["zh", "en"],
        },
        "parameter_count": 278_044_931,
        "approx_weight_bytes": 1_112_177_724,
        "min_ram_gib": 6.0,
        "recommended_ram_gib": 12.0,
        "recommended_device": "CPU or GPU",
        "effect": {
            "status": "official-model-benchmark-only",
            "summary_zh": "使用模型原始相关性分数进行精排，不加载 TMCRA 融合 checkpoint。官方模型卡的 C-MTEB Reranking 平均分为 65.42；这不是 TMCRA 成绩。",
            "summary_en": "Uses the model relevance logit without the TMCRA fusion checkpoint. The official model card reports a 65.42 C-MTEB reranking average; this is not a TMCRA score.",
            "source": "BAAI/bge-reranker-base model card",
            "metric": "C-MTEB reranking average",
            "value": 65.42,
        },
        "limitations_zh": [
            "面向中文和英文，不作为多语种默认项。",
            "尚未完成固定候选集上的 TMCRA 受控消融。",
        ],
        "download_includes": [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
            "model.safetensors",
        ],
        "required_files": ["config.json", "tokenizer.json", "model.safetensors"],
    },
    {
        "id": "current-fusion-reranker",
        "label_zh": "现有融合重排链路",
        "label_en": "Current fusion reranker",
        "selectable": False,
        "channel": "reference",
        "configuration_mode": "locked-compatible-set",
        "runtime_mode": "fusion",
        "validation_status": "existing-tmcra-chain-reference",
        "model": {
            "hf_repo": "BAAI/bge-reranker-v2-m3",
            "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "parameter_count": 567_755_777,
            "approx_weight_bytes": 2_271_023_108,
            "languages": ["multilingual"],
        },
        "parameter_count": 567_755_777,
        "approx_weight_bytes": 2_271_023_108,
        "min_ram_gib": 10.0,
        "recommended_ram_gib": 16.0,
        "recommended_device": "GPU preferred",
        "effect": {
            "status": "tmcra-system-chain-reference-not-isolated",
            "summary_zh": "复用现有 TMCRA Cross Encoder 与融合 checkpoint。现有系统成绩没有隔离该模型变量。",
            "summary_en": "Reuses the existing TMCRA cross encoder and fusion checkpoint. Existing system results do not isolate this model variable.",
            "source": "Public TMCRA reproduction profile in this repository",
        },
        "note_zh": "需要与当前 cross encoder 和融合 checkpoint 成套使用。",
        "effect_zh": "负责候选记忆的最终相关性重排。现有 TMCRA 成绩没有隔离该模型变量。",
        "download_includes": [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
            "model.safetensors",
        ],
        "required_files": ["config.json", "tokenizer.json", "model.safetensors"],
    },
    {
        "id": "preview-multilingual-cross-reranker",
        "label_zh": "多语言精排（预览）",
        "label_en": "Multilingual reranker (preview)",
        "selectable": False,
        "channel": "preview",
        "configuration_mode": "semantic-logit-only",
        "runtime_mode": "semantic-only",
        "validation_status": "download-and-tmcra-ab-pending",
        "effect": {
            "status": "tmcra-ab-pending",
            "summary_zh": "该预览项只用于后续隔离融合层影响的实验，尚无 TMCRA 受控消融结果。",
            "summary_en": "This preview exists for a future fusion-layer ablation; no controlled TMCRA result is available yet.",
            "source": "TMCRA public local runtime model policy",
        },
        "candidate": {
            "hf_repo": "BAAI/bge-reranker-v2-m3",
            "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "parameter_count": 567_755_777,
            "approx_weight_bytes": 2_271_023_108,
            "languages": ["multilingual"],
        },
        "note_zh": "与现有 Cross Encoder 权重相同，但跳过融合 checkpoint；资源不比参考档低，因此稳定安装不开放。",
        "effect_zh": "用于隔离融合层影响的实验候选，不是普通用户的轻量方案。",
    },
)


def iter_reranker_profiles(*, include_preview: bool = False) -> Iterable[dict[str, Any]]:
    for profile in RERANKER_POLICIES:
        if include_preview or bool(profile.get("selectable")):
            yield dict(profile)


def resolve_reranker_profile(
    profile_id: str, *, allow_preview: bool = False
) -> dict[str, Any]:
    normalized = str(profile_id or "").strip().lower()
    for profile in RERANKER_POLICIES:
        if str(profile.get("id")) != normalized:
            continue
        if not bool(profile.get("selectable")) and not allow_preview:
            raise ModelSelectionError(
                f"reranker profile {normalized!r} is not selectable: "
                f"{profile.get('validation_status')}"
            )
        return dict(profile)
    raise ModelSelectionError(f"unknown reranker profile: {profile_id!r}")


def recommend_reranker_profile(
    *, ram_gib: float, vram_gib: float = 0.0, language: str = "multilingual"
) -> dict[str, Any]:
    if ram_gib <= 0 or vram_gib < 0:
        raise ModelSelectionError("RAM must be positive and VRAM cannot be negative")
    normalized_language = str(language or "multilingual").strip().lower()
    if ram_gib >= 12 and normalized_language in {"zh", "en", "chinese", "english"}:
        return resolve_reranker_profile("compact-cross-reranker")
    return resolve_reranker_profile("local-dense-only")


def resolve_llm_policy(policy_id: str, *, allow_disabled: bool = False) -> dict[str, Any]:
    normalized = str(policy_id or "").strip().lower()
    if allow_disabled and normalized == "disabled":
        return {
            "id": "disabled",
            "label_zh": "关闭",
            "label_en": "Disabled",
            "selectable": True,
            "fallback": "none",
        }
    for policy in LLM_POLICIES:
        if policy["id"] == normalized:
            return dict(policy)
    raise ModelSelectionError(f"unknown generation policy: {policy_id!r}")


def resolve_generation_task_policies(
    *,
    default_policy_id: str,
    overrides: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve per-task generation policy without inventing an implicit fallback."""

    default_policy = resolve_llm_policy(default_policy_id)
    supplied = {
        str(key).strip(): str(value).strip().lower()
        for key, value in (overrides or {}).items()
        if str(value or "").strip() not in {"", "inherit"}
    }
    known_tasks = {str(item["id"]): item for item in GENERATION_TASKS}
    unknown = sorted(set(supplied) - set(known_tasks))
    if unknown:
        raise ModelSelectionError(
            "unknown generation task override(s): " + ", ".join(unknown)
        )
    result: dict[str, dict[str, Any]] = {}
    for task in GENERATION_TASKS:
        task_id = str(task["id"])
        policy_id = supplied.get(task_id, str(default_policy["id"]))
        policy = resolve_llm_policy(
            policy_id,
            allow_disabled=bool(task.get("allows_disabled")),
        )
        if policy["id"] == "disabled" and bool(task.get("required")):
            raise ModelSelectionError(f"generation task {task_id!r} cannot be disabled")
        result[task_id] = {
            "task_id": task_id,
            "policy_id": str(policy["id"]),
            "required": bool(task.get("required")),
            "selection_source": "override" if task_id in supplied else "default",
        }
    return result


def iter_embedding_profiles(*, include_preview: bool = False) -> Iterable[EmbeddingProfile]:
    for profile in EMBEDDING_PROFILES:
        if include_preview or profile.selectable:
            yield profile


def resolve_embedding_profile(profile_id: str, *, allow_preview: bool = False) -> EmbeddingProfile:
    normalized = str(profile_id or "").strip().lower()
    for profile in EMBEDDING_PROFILES:
        if profile.id != normalized:
            continue
        if not profile.selectable and not allow_preview:
            raise ModelSelectionError(
                f"embedding profile {profile.id!r} is not selectable: {profile.validation_status}"
            )
        return profile
    raise ModelSelectionError(f"unknown embedding profile: {profile_id!r}")


def iter_generation_profiles(
    *, include_preview: bool = False
) -> Iterable[GenerationProfile]:
    for profile in GENERATION_PROFILES:
        if include_preview or profile.selectable:
            yield profile


def resolve_generation_profile(
    profile_id: str, *, allow_preview: bool = False
) -> GenerationProfile:
    normalized = str(profile_id or "").strip().lower()
    for profile in GENERATION_PROFILES:
        if profile.id != normalized:
            continue
        if not profile.selectable and not allow_preview:
            raise ModelSelectionError(
                f"generation profile {profile.id!r} is not selectable: "
                f"{profile.validation_status}"
            )
        return profile
    raise ModelSelectionError(f"unknown generation profile: {profile_id!r}")


def recommend_embedding_profile(
    *,
    ram_gib: float,
    vram_gib: float = 0.0,
    language: str = "multilingual",
) -> EmbeddingProfile:
    if ram_gib <= 0 or vram_gib < 0:
        raise ModelSelectionError("RAM must be positive and VRAM cannot be negative")
    language = str(language or "multilingual").strip().lower()
    if ram_gib < 6 or (language in {"zh", "chinese"} and ram_gib < 10):
        return resolve_embedding_profile("compact-zh")
    if ram_gib >= 16 and vram_gib >= 4:
        return resolve_embedding_profile("enhanced-multilingual")
    return resolve_embedding_profile("balanced-multilingual")


def model_directory(models_root: Path, profile: EmbeddingProfile) -> Path:
    safe_repo = profile.hf_repo.replace("/", "--")
    return Path(models_root).expanduser().resolve() / "embedding" / f"{safe_repo}@{profile.revision[:12]}"


def generation_model_directory(
    models_root: Path, profile: GenerationProfile
) -> Path:
    safe_repo = profile.hf_repo.replace("/", "--")
    return (
        Path(models_root).expanduser().resolve()
        / "generation"
        / f"{safe_repo}@{profile.revision[:12]}"
    )


def reranker_model_directory(models_root: Path, profile: dict[str, Any]) -> Path | None:
    model = profile.get("model") or profile.get("candidate")
    if not isinstance(model, dict) or not str(model.get("hf_repo") or "").strip():
        return None
    repository = str(model["hf_repo"])
    revision = str(model["revision"])
    safe_repo = repository.replace("/", "--")
    return (
        Path(models_root).expanduser().resolve()
        / "reranker"
        / f"{safe_repo}@{revision[:12]}"
    )


def hf_download_command(profile: EmbeddingProfile, destination: Path) -> list[str]:
    command = [
        "hf",
        "download",
        profile.hf_repo,
        "--revision",
        profile.revision,
        "--local-dir",
        str(Path(destination).expanduser().resolve()),
        "--max-workers",
        "4",
    ]
    for pattern in profile.download_includes:
        command.extend(["--include", pattern])
    return command


def hf_verify_command(profile: EmbeddingProfile, destination: Path) -> list[str]:
    return [
        "hf",
        "cache",
        "verify",
        profile.hf_repo,
        "--revision",
        profile.revision,
        "--local-dir",
        str(Path(destination).expanduser().resolve()),
    ]


def hf_reranker_download_command(
    profile: dict[str, Any], destination: Path
) -> list[str]:
    model = profile.get("model") or profile.get("candidate")
    if not isinstance(model, dict):
        return []
    command = [
        "hf",
        "download",
        str(model["hf_repo"]),
        "--revision",
        str(model["revision"]),
        "--local-dir",
        str(Path(destination).expanduser().resolve()),
        "--max-workers",
        "4",
    ]
    for pattern in list(profile.get("download_includes") or []):
        command.extend(["--include", str(pattern)])
    return command


def hf_reranker_verify_command(
    profile: dict[str, Any], destination: Path
) -> list[str]:
    model = profile.get("model") or profile.get("candidate")
    if not isinstance(model, dict):
        return []
    return [
        "hf",
        "cache",
        "verify",
        str(model["hf_repo"]),
        "--revision",
        str(model["revision"]),
        "--local-dir",
        str(Path(destination).expanduser().resolve()),
    ]


def hf_generation_download_command(
    profile: GenerationProfile, destination: Path
) -> list[str]:
    return [
        "hf",
        "download",
        profile.hf_repo,
        profile.filename,
        *profile.metadata_files,
        "--revision",
        profile.revision,
        "--local-dir",
        str(Path(destination).expanduser().resolve()),
        "--max-workers",
        "4",
    ]


def hf_generation_verify_command(
    profile: GenerationProfile, destination: Path
) -> list[str]:
    return [
        "hf",
        "cache",
        "verify",
        profile.hf_repo,
        "--revision",
        profile.revision,
        "--local-dir",
        str(Path(destination).expanduser().resolve()),
    ]


def validate_local_model_files(
    profile: EmbeddingProfile, destination: Path
) -> dict[str, Any]:
    root = Path(destination).expanduser().resolve()
    files = {
        name: (root / name).is_file()
        for name in profile.required_files
    }
    return {
        "root": str(root),
        "required_files": files,
        "complete": bool(files) and all(files.values()),
    }


def validate_local_reranker_files(
    profile: dict[str, Any], destination: Path | None
) -> dict[str, Any]:
    required_names = [str(name) for name in list(profile.get("required_files") or [])]
    if not required_names:
        return {
            "root": "",
            "required_files": {},
            "complete": True,
        }
    if destination is None:
        return {
            "root": "",
            "required_files": {name: False for name in required_names},
            "complete": False,
        }
    root = Path(destination).expanduser().resolve()
    files = {name: (root / name).is_file() for name in required_names}
    return {
        "root": str(root),
        "required_files": files,
        "complete": bool(files) and all(files.values()),
    }


def validate_local_generation_files(
    profile: GenerationProfile, destination: Path
) -> dict[str, Any]:
    root = Path(destination).expanduser().resolve()
    required = (profile.filename, *profile.metadata_files)
    files = {name: (root / name).is_file() for name in required}
    return {
        "root": str(root),
        "required_files": files,
        "complete": bool(files) and all(files.values()),
    }


def catalog_payload(*, include_preview: bool = False) -> dict[str, Any]:
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "runtime_dependency_contract": dict(RUNTIME_DEPENDENCY_CONTRACT),
        "effect_contract": {
            "official-model-benchmark-only": "Model-card result; not a TMCRA result.",
            "official-specification-only": "Official model specification; no TMCRA score.",
            "tmcra-system-chain-reference-not-isolated": "TMCRA chain reference; no single-model attribution.",
            "tmcra-ab-verified": "A controlled TMCRA ablation result.",
            "tmcra-ab-pending": "No controlled TMCRA ablation result is available yet.",
            "public-local-runtime-contract-verified": "The public local runtime contract has been validated; target-device performance qualification remains separate.",
            "tmcra-quality-gate-partial": "Only a subset of the local generation quality gate passed; this profile is preview-only.",
            "public-full-quality-model-suggested;user-configurable": "Default local-generation recommendation; users may choose BYOK instead.",
            "host-dependent": "Effect and resource use depend on the current host model.",
            "user-declared;tmcra-ab-unknown": "User-selected provider model; no TMCRA result is inferred.",
            "source-dependent": "Effect depends on the selected host or explicit provider model.",
        },
        "embedding_profiles": [
            profile.as_public_dict()
            for profile in iter_embedding_profiles(include_preview=include_preview)
        ],
        "generation_profiles": [
            profile.as_public_dict()
            for profile in iter_generation_profiles(include_preview=include_preview)
        ],
        "reranker_policies": list(iter_reranker_profiles(include_preview=include_preview)),
        "llm_policies": [
            dict(policy)
            for policy in LLM_POLICIES
            if include_preview or bool(policy.get("selectable"))
        ],
        "generation_tasks": list(GENERATION_TASKS),
    }
