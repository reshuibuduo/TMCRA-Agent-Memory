from .base import (
    AdapterResponse,
    EvalCase,
    FailureRecord,
    LeaderboardRecord,
    LLMProfile,
    LongDialogProfile,
    LongDialogProbe,
    MemoryAdapter,
    MemoryHit,
    MemoryRetrieval,
    ReasoningAdapter,
    ScenarioProfile,
)

_MEMORY_EXPORTS = {
    "FullHistoryMemoryAdapter",
    "GraphSessionMemoryAdapter",
    "NullMemoryAdapter",
    "SummaryWindowMemoryAdapter",
    "VectorRAGMemoryAdapter",
}
_REASONER_EXPORTS = {
    "DirectExtractionReasoner",
    "OpenAICompatCoTReasoner",
    "OpenAICompatDirectReasoner",
    "OpenAICompatFullContextReasoner",
    "TriMazeIsolatedReasoner",
}

__all__ = [
    "AdapterResponse",
    "EvalCase",
    "FailureRecord",
    "LeaderboardRecord",
    "LLMProfile",
    "LongDialogProfile",
    "LongDialogProbe",
    "MemoryAdapter",
    "MemoryHit",
    "MemoryRetrieval",
    "ReasoningAdapter",
    "ScenarioProfile",
    *_MEMORY_EXPORTS,
    *_REASONER_EXPORTS,
]


def __getattr__(name: str):
    if name in _MEMORY_EXPORTS:
        from . import memory_adapters as _memory_adapters

        value = getattr(_memory_adapters, name)
        globals()[name] = value
        return value
    if name in _REASONER_EXPORTS:
        from . import reasoning_adapters as _reasoning_adapters

        value = getattr(_reasoning_adapters, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
