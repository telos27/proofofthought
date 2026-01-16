"""Entity management for link-based knowledge architecture.

This package provides entity storage and linking for commonsense reasoning.
Instead of merging similar entities, it keeps them separate and connects
them via pre-computed similarity links.

Key components:
- Entity: A concept/process/thing in the knowledge graph
- EntityLink: Pre-computed similarity between entities
- EntityStore: SQLite-backed storage for entities and links
- VectorIndex: FAISS-based ANN search for entity resolution (optional, requires faiss-cpu)
- EntityLinker: Multi-level entity resolution with linking
- PrecomputationPipeline: Batch pre-computation of embeddings and links
- QueryExpander: Expand queries using entity similarity links
- QueryEngine: Full query pipeline with expansion, matching, and evidence combination
"""

from z3adapter.ikr.entities.schema import (
    Entity,
    EntityEmbedding,
    EntityLink,
    LinkType,
    SurfaceForm,
)
from z3adapter.ikr.entities.store import EntityStore
from z3adapter.ikr.entities.linker import EntityLinker, LinkResult
from z3adapter.ikr.entities.precompute import PrecomputationPipeline, PrecomputationStats
from z3adapter.ikr.entities.query_expander import (
    ExpandedPattern,
    QueryExpander,
    QueryTriple,
    PREDICATE_SIMILARITY,
)
from z3adapter.ikr.entities.query_engine import (
    EvidenceCombiner,
    PatternMatch,
    PatternMatcher,
    QueryEngine,
    QueryResult,
)

__all__ = [
    "Entity",
    "EntityEmbedding",
    "EntityLink",
    "EntityLinker",
    "EntityStore",
    "EvidenceCombiner",
    "ExpandedPattern",
    "LinkResult",
    "LinkType",
    "PatternMatch",
    "PatternMatcher",
    "PREDICATE_SIMILARITY",
    "PrecomputationPipeline",
    "PrecomputationStats",
    "QueryEngine",
    "QueryExpander",
    "QueryResult",
    "QueryTriple",
    "SurfaceForm",
]

# VectorIndex is optional (requires faiss-cpu)
try:
    from z3adapter.ikr.entities.vector_index import VectorIndex, FAISS_AVAILABLE

    __all__.append("VectorIndex")
    __all__.append("FAISS_AVAILABLE")
except ImportError:
    FAISS_AVAILABLE = False
    __all__.append("FAISS_AVAILABLE")
