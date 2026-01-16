# Design: Link-Based Knowledge Architecture

## Overview

A scalable architecture for commonsense reasoning that:
- **Links** entities instead of merging them
- **Pre-computes** similarity for fast queries
- **Expands** queries via links for semantic matching
- Keeps **inference symbolic** (fast)

## Scope: Proof of Concept

- **Domain**: Psychology
- **Scale**: 1-3 psychology books (~10K-100K triples, ~5K-20K entities)
- **Storage**: SQLite + FAISS
- **Goal**: Validate architecture before scaling to Wikidata

---

## Data Model

### 1. Entity

```python
@dataclass
class Entity:
    id: str                      # UUID or sequential
    name: str                    # Canonical name (lowercase, underscores)
    entity_type: str             # e.g., "concept", "process", "disorder"
    description: Optional[str]   # From source or generated

    # External references (for future Wikidata integration)
    external_ids: dict[str, str] = field(default_factory=dict)
    # e.g., {"wikidata": "Q12345", "mesh": "D001008"}

    # Metadata
    source: Optional[str]        # First source that introduced this entity
    created_at: datetime
```

### 2. Triple (existing, minor changes)

```python
@dataclass
class Triple:
    id: str
    subject_id: str              # Changed: reference to Entity.id
    predicate: Predicate         # One of 7 predicates
    object_id: str               # Changed: reference to Entity.id

    # Keep existing fields
    negated: bool = False
    truth: Optional[TruthValue] = None
    source: Optional[str] = None
    surface_form: Optional[str] = None
```

### 3. EntityLink (NEW)

```python
@dataclass
class EntityLink:
    """Pre-computed similarity between entities."""
    source_id: str               # Entity.id
    target_id: str               # Entity.id
    link_type: LinkType          # SIMILAR_TO, IS_A, PART_OF, etc.
    score: float                 # Similarity score [0, 1]

    # Metadata
    method: str                  # "embedding", "lexical", "wikidata", "manual"
    computed_at: datetime

class LinkType(Enum):
    SIMILAR_TO = "similar_to"    # Semantic similarity
    IS_A = "is_a"                # Taxonomy (entity is type of target)
    PART_OF = "part_of"          # Meronymy
    SAME_AS = "same_as"          # Exact equivalence (rare)
```

### 4. EntityEmbedding

```python
@dataclass
class EntityEmbedding:
    entity_id: str
    embedding: list[float]       # Vector (1536 dims for OpenAI)
    model: str                   # "text-embedding-3-small"
    computed_at: datetime
```

---

## SQLite Schema

```sql
-- Entities
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    entity_type TEXT,
    description TEXT,
    external_ids TEXT,           -- JSON: {"wikidata": "Q123", ...}
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_entities_name ON entities(name);
CREATE INDEX idx_entities_type ON entities(entity_type);

-- Triples (modified to reference entities)
CREATE TABLE triples (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES entities(id),
    predicate TEXT NOT NULL,
    object_id TEXT NOT NULL REFERENCES entities(id),
    negated BOOLEAN DEFAULT FALSE,
    frequency REAL,
    confidence REAL,
    source TEXT,
    surface_form TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_triples_subject ON triples(subject_id);
CREATE INDEX idx_triples_predicate ON triples(predicate);
CREATE INDEX idx_triples_object ON triples(object_id);

-- Entity Links (pre-computed similarities)
CREATE TABLE entity_links (
    source_id TEXT NOT NULL REFERENCES entities(id),
    target_id TEXT NOT NULL REFERENCES entities(id),
    link_type TEXT NOT NULL,
    score REAL NOT NULL,
    method TEXT,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source_id, target_id, link_type)
);

CREATE INDEX idx_links_source ON entity_links(source_id);
CREATE INDEX idx_links_target ON entity_links(target_id);
CREATE INDEX idx_links_score ON entity_links(score);

-- Entity Embeddings
CREATE TABLE entity_embeddings (
    entity_id TEXT PRIMARY KEY REFERENCES entities(id),
    embedding BLOB NOT NULL,     -- numpy array serialized
    model TEXT NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Surface Forms (learned mappings for O(1) lookup)
CREATE TABLE surface_forms (
    surface_form TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL REFERENCES entities(id),
    score REAL NOT NULL,
    source TEXT,                 -- "exact", "embedding", "manual"
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Components

### 1. EntityStore

```python
class EntityStore:
    """Manages entities with linking support."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    # CRUD
    def add(self, entity: Entity) -> str: ...
    def get(self, entity_id: str) -> Optional[Entity]: ...
    def get_by_name(self, name: str) -> Optional[Entity]: ...
    def search(self, pattern: str, limit: int = 10) -> list[Entity]: ...

    # Links
    def add_link(self, link: EntityLink) -> None: ...
    def get_links(self, entity_id: str, link_type: Optional[LinkType] = None) -> list[EntityLink]: ...
    def get_similar(self, entity_id: str, min_score: float = 0.5, limit: int = 10) -> list[tuple[Entity, float]]: ...

    # Surface forms
    def add_surface_form(self, form: str, entity_id: str, score: float, source: str): ...
    def lookup_surface_form(self, form: str) -> Optional[tuple[str, float]]: ...

    # Embeddings
    def save_embedding(self, entity_id: str, embedding: list[float], model: str): ...
    def get_embedding(self, entity_id: str) -> Optional[list[float]]: ...
    def get_all_embeddings(self) -> dict[str, list[float]]: ...
```

### 2. VectorIndex

```python
class VectorIndex:
    """FAISS-based vector index for entity resolution."""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.index = None           # FAISS index
        self.id_map: list[str] = [] # Position → entity_id

    def build(self, embeddings: dict[str, list[float]]) -> None:
        """Build index from entity embeddings."""
        ...

    def add(self, entity_id: str, embedding: list[float]) -> None:
        """Add single embedding (for incremental updates)."""
        ...

    def search(self, query_embedding: list[float], k: int = 10) -> list[tuple[str, float]]:
        """Find k nearest entities. Returns (entity_id, score) pairs."""
        ...

    def save(self, path: str) -> None:
        """Persist index to disk."""
        ...

    def load(self, path: str) -> None:
        """Load index from disk."""
        ...
```

### 3. EntityLinker (replaces EntityResolver)

```python
class EntityLinker:
    """Link mentions to entities using multi-level resolution."""

    def __init__(
        self,
        entity_store: EntityStore,
        vector_index: VectorIndex,
        embed_fn: Callable[[str], list[float]],
        link_threshold: float = 0.7,      # Create link if above this
        identity_threshold: float = 0.95,  # Use existing entity if above this
    ):
        self.entity_store = entity_store
        self.vector_index = vector_index
        self.embed_fn = embed_fn
        self.link_threshold = link_threshold
        self.identity_threshold = identity_threshold

    def link(self, mention: str, context: Optional[str] = None) -> LinkResult:
        """
        Link a mention to an entity.

        Resolution order:
        1. Exact match (O(1) hash)
        2. Surface form lookup (O(1) learned mappings)
        3. Vector search (O(log n) ANN)

        Returns:
            LinkResult with:
            - entity: The resolved or newly created entity
            - is_new: Whether a new entity was created
            - links: Similarity links to create
        """
        # 1. Exact match
        normalized = self._normalize(mention)
        entity = self.entity_store.get_by_name(normalized)
        if entity:
            return LinkResult(entity=entity, is_new=False, links=[])

        # 2. Surface form lookup
        lookup = self.entity_store.lookup_surface_form(normalized)
        if lookup:
            entity_id, score = lookup
            entity = self.entity_store.get(entity_id)
            return LinkResult(entity=entity, is_new=False, links=[])

        # 3. Vector search
        embedding = self.embed_fn(mention)
        candidates = self.vector_index.search(embedding, k=10)

        if candidates:
            best_id, best_score = candidates[0]

            if best_score >= self.identity_threshold:
                # High confidence: use existing entity
                entity = self.entity_store.get(best_id)
                # Learn surface form for next time
                self.entity_store.add_surface_form(normalized, best_id, best_score, "embedding")
                return LinkResult(entity=entity, is_new=False, links=[])

            elif best_score >= self.link_threshold:
                # Moderate confidence: create new entity, link to existing
                entity = Entity(id=uuid4(), name=normalized, ...)
                links = [
                    EntityLink(source_id=entity.id, target_id=cid,
                              link_type=LinkType.SIMILAR_TO, score=score)
                    for cid, score in candidates if score >= self.link_threshold
                ]
                return LinkResult(entity=entity, is_new=True, links=links)

        # 4. No match: create new entity
        entity = Entity(id=uuid4(), name=normalized, ...)
        return LinkResult(entity=entity, is_new=True, links=[])

@dataclass
class LinkResult:
    entity: Entity
    is_new: bool
    links: list[EntityLink]
```

### 4. QueryExpander

```python
class QueryExpander:
    """Expand queries using entity links."""

    def __init__(
        self,
        entity_store: EntityStore,
        min_score: float = 0.5,
        max_expansions: int = 20,
    ):
        self.entity_store = entity_store
        self.min_score = min_score
        self.max_expansions = max_expansions

    def expand(self, triple: Triple) -> list[ExpandedPattern]:
        """
        Expand a query triple into multiple patterns.

        For query (A, predicate, B):
        - Find entities similar to A
        - Find entities similar to B
        - Generate cross-product of patterns
        - Compute combined confidence for each
        """
        patterns = []

        # Get similar entities for subject
        subject_expansions = [(triple.subject_id, 1.0)]  # Include original
        for entity, score in self.entity_store.get_similar(triple.subject_id, self.min_score):
            subject_expansions.append((entity.id, score))

        # Get similar entities for object
        object_expansions = [(triple.object_id, 1.0)]
        for entity, score in self.entity_store.get_similar(triple.object_id, self.min_score):
            object_expansions.append((entity.id, score))

        # Generate patterns
        for subj_id, subj_score in subject_expansions:
            for obj_id, obj_score in object_expansions:
                combined_score = subj_score * obj_score
                if combined_score >= self.min_score:
                    patterns.append(ExpandedPattern(
                        subject_id=subj_id,
                        predicate=triple.predicate,
                        object_id=obj_id,
                        confidence=combined_score,
                    ))

        # Sort by confidence, limit
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        return patterns[:self.max_expansions]

@dataclass
class ExpandedPattern:
    subject_id: str
    predicate: Predicate
    object_id: str
    confidence: float  # Decay from link scores
```

### 5. Pre-computation Pipeline

```python
class PrecomputationPipeline:
    """Batch pre-computation of embeddings and links."""

    def __init__(
        self,
        entity_store: EntityStore,
        vector_index: VectorIndex,
        embed_fn: Callable[[list[str]], list[list[float]]],  # Batch
    ):
        ...

    def compute_embeddings(self, batch_size: int = 100) -> int:
        """Compute embeddings for entities missing them."""
        missing = self.entity_store.get_entities_without_embeddings()

        for batch in chunks(missing, batch_size):
            texts = [e.name.replace("_", " ") for e in batch]
            embeddings = self.embed_fn(texts)

            for entity, emb in zip(batch, embeddings):
                self.entity_store.save_embedding(entity.id, emb, model="...")

        return len(missing)

    def rebuild_vector_index(self) -> None:
        """Rebuild FAISS index from all embeddings."""
        embeddings = self.entity_store.get_all_embeddings()
        self.vector_index.build(embeddings)
        self.vector_index.save("index.faiss")

    def compute_links(self, k: int = 50, min_score: float = 0.5) -> int:
        """Pre-compute top-k similar links for each entity."""
        count = 0

        for entity_id, embedding in self.entity_store.get_all_embeddings().items():
            candidates = self.vector_index.search(embedding, k=k+1)  # +1 to exclude self

            for target_id, score in candidates:
                if target_id != entity_id and score >= min_score:
                    link = EntityLink(
                        source_id=entity_id,
                        target_id=target_id,
                        link_type=LinkType.SIMILAR_TO,
                        score=score,
                        method="embedding",
                    )
                    self.entity_store.add_link(link)
                    count += 1

        return count
```

---

## Query Flow

```
Query: "Does anxiety affect memory?"

1. PARSE
   └─ LLM extracts: (anxiety, affects, memory)

2. ENTITY RESOLUTION
   ├─ "anxiety" → EntityLinker.link() → Entity(id="e123", name="anxiety")
   └─ "memory" → EntityLinker.link() → Entity(id="e456", name="memory")

3. QUERY EXPANSION
   └─ QueryExpander.expand(Triple(e123, CAUSES, e456))
      Returns:
        - (anxiety, causes, memory)           conf=1.0
        - (anxiety, causes, working_memory)   conf=0.85
        - (stress, causes, memory)            conf=0.75
        - (fear, causes, memory)              conf=0.70
        - (stress, causes, working_memory)    conf=0.64
        ...

4. PATTERN MATCHING
   └─ For each pattern, query TripleStore (exact match)
      Found: (chronic_stress, impairs, working_memory)

      Match:
        - chronic_stress has link to stress (0.88)
        - stress is in our expansion (0.75)
        - working_memory is in our expansion (0.85)
        - impairs ~ causes via predicate similarity (0.80)

      Combined: 0.88 * 0.75 * 0.85 * 0.80 = 0.45

5. EVIDENCE COMBINATION
   └─ NARS revision across all matches
      Result: SUPPORTED with confidence 0.45

6. EXPLANATION
   └─ "Found related evidence: chronic_stress impairs working_memory
       (chronic_stress ≈ anxiety via stress, working_memory ≈ memory)"
```

---

## Implementation Plan

### Phase 1: Data Model & Storage (Week 1)
- [ ] Entity dataclass and SQLite schema
- [ ] EntityStore with CRUD + links + surface forms
- [ ] Migrate Triple to use entity IDs
- [ ] Unit tests

### Phase 2: Vector Index (Week 1)
- [ ] FAISS integration (basic IVF index)
- [ ] VectorIndex class with build/search/save/load
- [ ] Integration with EntityStore
- [ ] Unit tests

### Phase 3: Entity Linker (Week 2)
- [ ] EntityLinker with multi-level resolution
- [ ] Surface form learning
- [ ] LinkResult with similarity links
- [ ] Unit tests

### Phase 4: Pre-computation (Week 2)
- [ ] Batch embedding computation
- [ ] Link pre-computation
- [ ] Index rebuild pipeline
- [ ] CLI commands

### Phase 5: Query Expansion (Week 3)
- [ ] QueryExpander implementation
- [ ] Predicate similarity handling
- [ ] Evidence combination with NARS
- [ ] Unit tests

### Phase 6: Integration (Week 3)
- [ ] Updated ingestion pipeline
- [ ] Updated query pipeline
- [ ] End-to-end tests with psychology text

### Phase 7: Evaluation (Week 4)
- [ ] Ingest 1-3 psychology books
- [ ] Benchmark query accuracy
- [ ] Benchmark performance
- [ ] Iterate on thresholds

---

## Open Questions

1. **Predicate similarity**: Should we also expand predicates?
   - "affects" ~ "causes" ~ "influences" ~ "impacts"
   - Could use same embedding approach

2. **Negative links**: Should we track dissimilarity?
   - "anxiety" is NOT similar to "relaxation"
   - Useful for contradiction detection

3. **Hierarchical types**: Should entity types form a hierarchy?
   - "anxiety_disorder" IS_A "mental_disorder" IS_A "disorder"
   - Enables type-based filtering

4. **Confidence decay**: How should link scores decay through chains?
   - A ~ B (0.8), B ~ C (0.9) → A ~ C = ?
   - Options: multiply, min, custom formula

5. **Update strategy**: When to recompute links?
   - After each ingestion? Batch nightly?
   - Incremental vs full rebuild?

---

## File Structure

```
z3adapter/ikr/
├── triples/
│   ├── schema.py              # Triple, Predicate (existing)
│   ├── extractor.py           # TripleExtractor (existing)
│   └── ...
├── entities/                   # NEW package
│   ├── __init__.py
│   ├── schema.py              # Entity, EntityLink, LinkType
│   ├── store.py               # EntityStore (SQLite)
│   ├── vector_index.py        # VectorIndex (FAISS)
│   ├── linker.py              # EntityLinker
│   └── precompute.py          # PrecomputationPipeline
├── query/                      # NEW package
│   ├── __init__.py
│   ├── expander.py            # QueryExpander
│   ├── matcher.py             # Pattern matching
│   └── engine.py              # Main query engine
└── pipeline.py                 # Updated end-to-end pipeline
```

---

## Dependencies

```
# Existing
z3-solver
openai
pydantic

# New
faiss-cpu          # Vector similarity search
numpy              # Embedding operations
```

---

## Success Criteria

1. **Accuracy**: Query expansion finds relevant matches that exact matching misses
2. **Performance**: Query latency < 1 second for 100K triples
3. **Scalability**: Architecture supports future Wikidata integration
4. **Explainability**: Can trace why a query matched (which links, which patterns)
