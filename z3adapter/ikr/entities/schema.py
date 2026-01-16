"""Entity schema for link-based knowledge architecture.

Entities are the nodes in the knowledge graph. Unlike the merge-based approach
where similar entities are combined, this architecture keeps entities separate
and connects them via pre-computed similarity links.

Key concepts:
- Entity: A concept, process, or thing with a canonical name
- EntityLink: Pre-computed similarity between two entities
- LinkType: Type of relationship between linked entities
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4


class LinkType(str, Enum):
    """Type of relationship between linked entities.

    SIMILAR_TO: Semantic similarity (most common, from embeddings)
    IS_A: Taxonomy (entity is a type of target)
    PART_OF: Meronymy (entity is part of target)
    SAME_AS: Exact equivalence (rare, for confirmed duplicates)
    """

    SIMILAR_TO = "similar_to"
    IS_A = "is_a"
    PART_OF = "part_of"
    SAME_AS = "same_as"


@dataclass
class Entity:
    """An entity in the knowledge graph.

    Entities represent concepts, processes, disorders, or any other
    semantic unit extracted from text. Each entity has a canonical
    name (lowercase with underscores) and optional metadata.

    Attributes:
        id: Unique identifier (UUID)
        name: Canonical name (lowercase, underscores)
        entity_type: Optional type (e.g., "concept", "process", "disorder")
        description: Optional description or definition
        external_ids: External references (e.g., Wikidata QID)
        source: First source that introduced this entity
        created_at: Timestamp when entity was created

    Example:
        entity = Entity(
            name="anxiety_disorder",
            entity_type="disorder",
            description="A mental disorder characterized by excessive worry",
            external_ids={"wikidata": "Q175629"},
            source="DSM-5"
        )
    """

    name: str  # Canonical name (lowercase, underscores)
    id: str = field(default_factory=lambda: str(uuid4()))
    entity_type: Optional[str] = None
    description: Optional[str] = None
    external_ids: dict[str, str] = field(default_factory=dict)
    source: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Normalize entity name to canonical form."""
        self.name = self._normalize_name(self.name)

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize name to lowercase with underscores."""
        return name.lower().strip().replace(" ", "_").replace("-", "_")

    def __hash__(self) -> int:
        """Hash based on ID for set operations."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on ID."""
        if not isinstance(other, Entity):
            return False
        return self.id == other.id


@dataclass
class EntityLink:
    """Pre-computed similarity link between two entities.

    Links connect entities based on semantic similarity without merging them.
    This allows queries to expand via links while keeping entities distinct.

    Attributes:
        source_id: ID of the source entity
        target_id: ID of the target entity
        link_type: Type of relationship
        score: Similarity score [0, 1]
        method: How the link was computed (e.g., "embedding", "lexical", "manual")
        computed_at: Timestamp when link was computed

    Example:
        link = EntityLink(
            source_id="abc123",
            target_id="def456",
            link_type=LinkType.SIMILAR_TO,
            score=0.85,
            method="embedding"
        )
    """

    source_id: str
    target_id: str
    link_type: LinkType
    score: float
    method: str = "embedding"
    computed_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate score is in [0, 1]."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be in [0, 1], got {self.score}")

    def __hash__(self) -> int:
        """Hash based on source, target, and type for set operations."""
        return hash((self.source_id, self.target_id, self.link_type))

    def __eq__(self, other: object) -> bool:
        """Equality based on source, target, and type."""
        if not isinstance(other, EntityLink):
            return False
        return (
            self.source_id == other.source_id
            and self.target_id == other.target_id
            and self.link_type == other.link_type
        )


@dataclass
class EntityEmbedding:
    """Embedding vector for an entity.

    Stored separately to allow lazy loading and efficient bulk operations.

    Attributes:
        entity_id: ID of the entity this embedding belongs to
        embedding: Vector representation (e.g., 1536 dims for OpenAI)
        model: Model used to generate embedding
        computed_at: Timestamp when embedding was computed
    """

    entity_id: str
    embedding: list[float]
    model: str
    computed_at: datetime = field(default_factory=datetime.now)


@dataclass
class SurfaceForm:
    """Learned mapping from surface form to entity.

    Surface forms are alternative names or phrasings that map to an entity.
    These are learned during entity resolution and cached for O(1) lookup.

    Attributes:
        form: The surface form text (normalized)
        entity_id: ID of the entity this form maps to
        score: Confidence score of the mapping [0, 1]
        source: How the mapping was learned (e.g., "exact", "embedding", "manual")
        created_at: Timestamp when mapping was created

    Example:
        # "WM" maps to "working_memory" entity
        surface = SurfaceForm(
            form="wm",
            entity_id="abc123",
            score=0.95,
            source="manual"
        )
    """

    form: str
    entity_id: str
    score: float
    source: str = "exact"
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Normalize form and validate score."""
        self.form = self.form.lower().strip()
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be in [0, 1], got {self.score}")
