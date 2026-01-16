"""Entity management for link-based knowledge architecture.

This package provides entity storage and linking for commonsense reasoning.
Instead of merging similar entities, it keeps them separate and connects
them via pre-computed similarity links.

Key components:
- Entity: A concept/process/thing in the knowledge graph
- EntityLink: Pre-computed similarity between entities
- EntityStore: SQLite-backed storage for entities and links
- VectorIndex: FAISS-based ANN search for entity resolution (optional, requires faiss-cpu)
"""

from z3adapter.ikr.entities.schema import (
    Entity,
    EntityEmbedding,
    EntityLink,
    LinkType,
    SurfaceForm,
)
from z3adapter.ikr.entities.store import EntityStore

__all__ = [
    "Entity",
    "EntityEmbedding",
    "EntityLink",
    "EntityStore",
    "LinkType",
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
