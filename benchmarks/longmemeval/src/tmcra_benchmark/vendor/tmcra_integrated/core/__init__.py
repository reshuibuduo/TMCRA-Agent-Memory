"""TMCRA core package exports.

Keep package-level imports best-effort so focused submodules such as the
object-sketch trainer can run without pulling in every optional runtime
dependency from the broader app stack.
"""

__all__ = []

try:
    from .maze_engine import MazeEdge, MazeNode, MazePath, TriMazeEngine

    __all__.extend(["MazeNode", "MazeEdge", "MazePath", "TriMazeEngine"])
except Exception:  # pragma: no cover - optional dependency
    MazeNode = None
    MazeEdge = None
    MazePath = None
    TriMazeEngine = None

try:
    from .native_concept_extractor import NativeConceptExtractor, NativeExtractorConfig

    __all__.extend(["NativeConceptExtractor", "NativeExtractorConfig"])
except Exception:  # pragma: no cover - optional dependency
    NativeConceptExtractor = None
    NativeExtractorConfig = None

try:
    from .concept_graph import ConceptGraph

    __all__.append("ConceptGraph")
except Exception:  # pragma: no cover - optional dependency
    ConceptGraph = None

try:
    from .concept_memory import ConceptMemory

    __all__.append("ConceptMemory")
except Exception:  # pragma: no cover - optional dependency
    ConceptMemory = None

try:
    from .query_understanding import QueryUnderstandingLayer

    __all__.append("QueryUnderstandingLayer")
except Exception:  # pragma: no cover - optional dependency
    QueryUnderstandingLayer = None

try:
    from .gru_text_generator import GRUTextGenerator

    __all__.append("GRUTextGenerator")
except Exception:  # pragma: no cover - optional dependency
    GRUTextGenerator = None

try:
    from .policy_network import EdgePolicy

    __all__.append("EdgePolicy")
except Exception:  # pragma: no cover - optional dependency
    EdgePolicy = None

try:
    from .policy_dataset import (
        CurriculumConfig,
        PolicyBatch,
        PolicyStepDataset,
        PolicyStepRecord,
        PolicyVocabulary,
    )
    from .policy_network import PolicyModelConfig

    __all__.extend(
        [
            "CurriculumConfig",
            "PolicyBatch",
            "PolicyModelConfig",
            "PolicyStepDataset",
            "PolicyStepRecord",
            "PolicyVocabulary",
        ]
    )
except Exception:  # pragma: no cover - optional dependency
    CurriculumConfig = None
    PolicyBatch = None
    PolicyModelConfig = None
    PolicyStepDataset = None
    PolicyStepRecord = None
    PolicyVocabulary = None
