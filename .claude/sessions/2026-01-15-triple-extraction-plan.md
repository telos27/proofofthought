# Implementation Plan: Triple Extraction Pipeline

## Overview

Add a text-to-triples extraction pipeline to proofofthought, enabling knowledge extraction from books and documents. The design follows Wikidata's philosophy: **predicates are fixed schema, entities emerge from content**.

### Design Decisions (from discussion)

1. **7 generic predicates** (not domain-specific)
2. **Triple references** for multi-level beliefs (reification)
3. **Negated flag** for handling negations
4. **No pre-built KB** - vocabulary emerges from text + LLM common sense
5. **Entity resolution** is the core challenge (not predicate classification)
6. **NARS truth values** for uncertainty

---

## Phase 1: Core Data Model

### 1.1 Triple Schema

Create `z3adapter/ikr/triples/schema.py`:

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from z3adapter.ikr.schema import TruthValue


class Predicate(str, Enum):
    """Fixed predicate vocabulary (Wikidata-inspired)."""
    IS_A = "is_a"           # Taxonomy: X is a type of Y
    PART_OF = "part_of"     # Structure: X is part of Y
    HAS = "has"             # Attributes: X has property Y
    CAUSES = "causes"       # Causation: X causes Y
    PREVENTS = "prevents"   # Negative causation: X prevents Y
    BELIEVES = "believes"   # Epistemic: X believes Y
    RELATED_TO = "related_to"  # Catch-all


# Predicate opposites for contradiction detection
PREDICATE_OPPOSITES = {
    Predicate.CAUSES: Predicate.PREVENTS,
    Predicate.PREVENTS: Predicate.CAUSES,
}


@dataclass
class Triple:
    """A semantic triple with optional reification support.

    Subject and object can reference:
    - Entities: bare strings like "stress", "memory"
    - Other triples: prefixed with "t:" like "t:t1"
    """
    id: str                           # Unique ID (e.g., "t1", "t2")
    subject: str                      # Entity or triple reference (t:xxx)
    predicate: Predicate              # One of 7 predicates
    object: str                       # Entity or triple reference (t:xxx)

    # Negation
    negated: bool = False             # "X does NOT predicate Y"

    # Uncertainty (NARS)
    truth: Optional[TruthValue] = None

    # Provenance
    source: Optional[str] = None      # "Zimbardo 2017 p.42"
    surface_form: Optional[str] = None  # Original text

    def is_triple_reference(self, ref: str) -> bool:
        """Check if reference points to another triple."""
        return ref.startswith("t:")

    def get_triple_id(self, ref: str) -> Optional[str]:
        """Extract triple ID from reference."""
        if self.is_triple_reference(ref):
            return ref[2:]
        return None

    @property
    def subject_is_triple(self) -> bool:
        return self.is_triple_reference(self.subject)

    @property
    def object_is_triple(self) -> bool:
        return self.is_triple_reference(self.object)


@dataclass
class TripleStore:
    """In-memory store for triples with indexing."""
    triples: dict[str, Triple] = field(default_factory=dict)

    # Indexes for fast lookup
    by_subject: dict[str, list[str]] = field(default_factory=dict)
    by_predicate: dict[Predicate, list[str]] = field(default_factory=dict)
    by_object: dict[str, list[str]] = field(default_factory=dict)

    def add(self, triple: Triple) -> None:
        """Add triple and update indexes."""
        self.triples[triple.id] = triple

        # Update indexes
        self.by_subject.setdefault(triple.subject, []).append(triple.id)
        self.by_predicate.setdefault(triple.predicate, []).append(triple.id)
        self.by_object.setdefault(triple.object, []).append(triple.id)

    def get(self, triple_id: str) -> Optional[Triple]:
        """Get triple by ID."""
        return self.triples.get(triple_id)

    def resolve(self, ref: str) -> Triple | str:
        """Resolve reference to triple or return entity string."""
        if ref.startswith("t:"):
            return self.triples.get(ref[2:], ref)
        return ref

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[Predicate] = None,
        object: Optional[str] = None,
    ) -> list[Triple]:
        """Query triples by subject/predicate/object pattern."""
        candidates = set(self.triples.keys())

        if subject is not None:
            candidates &= set(self.by_subject.get(subject, []))
        if predicate is not None:
            candidates &= set(self.by_predicate.get(predicate, []))
        if object is not None:
            candidates &= set(self.by_object.get(object, []))

        return [self.triples[tid] for tid in candidates]
```

### 1.2 Files to Create

| File | Purpose |
|------|---------|
| `z3adapter/ikr/triples/__init__.py` | Package exports |
| `z3adapter/ikr/triples/schema.py` | Triple, TripleStore, Predicate |
| `tests/unit/test_triple_schema.py` | Unit tests for data model |

### 1.3 Acceptance Criteria

- [ ] Triple dataclass with all fields
- [ ] Predicate enum with 7 values
- [ ] TripleStore with add/get/query/resolve
- [ ] Triple reference detection (t: prefix)
- [ ] Unit tests passing

---

## Phase 2: Triple Extraction

### 2.1 Extractor Design

Create `z3adapter/ikr/triples/extractor.py`:

```python
from dataclasses import dataclass
from typing import Protocol, Optional
from z3adapter.ikr.triples.schema import Triple, Predicate, TruthValue


class LLMClient(Protocol):
    """Protocol for LLM client (OpenAI-compatible)."""
    def chat_completions_create(self, messages: list, **kwargs) -> Any: ...


@dataclass
class ExtractionResult:
    """Result of triple extraction from text."""
    triples: list[Triple]
    raw_response: str
    source: Optional[str] = None


class TripleExtractor:
    """Extract triples from text using LLM."""

    SYSTEM_PROMPT = '''Extract semantic triples from the text.

Use ONLY these predicates:
- is_a: X is a type/kind of Y (taxonomy)
- part_of: X is part of Y (structure)
- has: X has property/attribute Y
- causes: X causes/leads to Y
- prevents: X prevents/stops Y
- believes: X believes/claims Y (for attributed statements)
- related_to: X is related to Y (use when others don't fit)

Output format (JSON):
{
  "triples": [
    {
      "id": "t1",
      "subject": "entity or t:triple_id",
      "predicate": "one of 7 predicates",
      "object": "entity or t:triple_id",
      "negated": false,
      "surface_form": "original text snippet"
    }
  ]
}

Rules:
1. Entity names: lowercase, underscores for spaces (working_memory, not "Working Memory")
2. For beliefs about beliefs, use triple references: {"subject": "alice", "predicate": "believes", "object": "t:t1"}
3. Set negated=true for negations ("X does NOT cause Y")
4. Prefer specific predicates over related_to
5. Extract ALL meaningful relationships, not just the main one
'''

    def __init__(self, llm_client: LLMClient, model: str = "gpt-4o"):
        self.client = llm_client
        self.model = model

    def extract(
        self,
        text: str,
        source: Optional[str] = None,
        context: Optional[str] = None,
    ) -> ExtractionResult:
        """Extract triples from text."""

        user_prompt = f"Extract triples from:\n\n{text}"
        if context:
            user_prompt = f"Context: {context}\n\n{user_prompt}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        triples = []
        for t in data.get("triples", []):
            triple = Triple(
                id=t["id"],
                subject=t["subject"],
                predicate=Predicate(t["predicate"]),
                object=t["object"],
                negated=t.get("negated", False),
                surface_form=t.get("surface_form"),
                source=source,
            )
            triples.append(triple)

        return ExtractionResult(triples=triples, raw_response=raw, source=source)
```

### 2.2 Files to Create

| File | Purpose |
|------|---------|
| `z3adapter/ikr/triples/extractor.py` | TripleExtractor class |
| `tests/unit/test_triple_extractor.py` | Unit tests (mocked LLM) |
| `tests/integration/test_triple_extractor_live.py` | Integration tests (real LLM, skipped in CI) |

### 2.3 Acceptance Criteria

- [ ] TripleExtractor with configurable LLM client
- [ ] JSON output parsing
- [ ] Multi-level belief extraction (triple references)
- [ ] Negation handling
- [ ] Source/provenance tracking
- [ ] Unit tests with mocked responses
- [ ] Integration test with real LLM

---

## Phase 3: Entity Resolution

### 3.1 Entity Resolver Design

Create `z3adapter/ikr/triples/entity_resolver.py`:

```python
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class EntityMatch:
    """Result of entity matching."""
    canonical: str          # Canonical entity name
    surface_form: str       # Original surface form
    similarity: float       # Match confidence [0, 1]


class EntityResolver:
    """Resolve surface forms to canonical entities."""

    def __init__(
        self,
        similarity_fn: Callable[[str, str], float] = None,
        threshold: float = 0.8,
    ):
        self.similarity_fn = similarity_fn or combined_lexical_similarity
        self.threshold = threshold
        self.entities: dict[str, set[str]] = {}  # canonical -> surface forms

    def add_entity(self, canonical: str, surface_forms: list[str] = None) -> None:
        """Register entity with optional surface forms."""
        if canonical not in self.entities:
            self.entities[canonical] = set()
        if surface_forms:
            self.entities[canonical].update(surface_forms)

    def resolve(self, surface_form: str) -> EntityMatch:
        """Find canonical entity for surface form."""
        normalized = self._normalize(surface_form)

        # Exact match
        if normalized in self.entities:
            return EntityMatch(normalized, surface_form, 1.0)

        # Fuzzy match
        best_match = None
        best_score = 0.0

        for canonical, forms in self.entities.items():
            # Check canonical name
            score = self.similarity_fn(normalized, canonical)
            if score > best_score:
                best_match, best_score = canonical, score

            # Check known surface forms
            for form in forms:
                score = self.similarity_fn(normalized, form)
                if score > best_score:
                    best_match, best_score = canonical, score

        if best_match and best_score >= self.threshold:
            # Add new surface form
            self.entities[best_match].add(normalized)
            return EntityMatch(best_match, surface_form, best_score)

        # New entity
        self.add_entity(normalized)
        return EntityMatch(normalized, surface_form, 1.0)

    def _normalize(self, text: str) -> str:
        """Normalize entity name."""
        return text.lower().strip().replace(" ", "_").replace("-", "_")
```

### 3.2 Integration with Existing Fuzzy Matching

Reuse from `z3adapter/ikr/fuzzy_nars.py`:
- `lexical_similarity()`
- `jaccard_word_similarity()`
- `combined_lexical_similarity()`
- `make_embedding_similarity()` (optional, for semantic matching)

### 3.3 Files to Create

| File | Purpose |
|------|---------|
| `z3adapter/ikr/triples/entity_resolver.py` | EntityResolver class |
| `tests/unit/test_entity_resolver.py` | Unit tests |

### 3.4 Acceptance Criteria

- [ ] EntityResolver with pluggable similarity function
- [ ] Fuzzy matching with configurable threshold
- [ ] Surface form tracking
- [ ] Normalization (lowercase, underscores)
- [ ] Unit tests

---

## Phase 4: Verification Integration

### 4.1 Connect to Existing Fuzzy-NARS

The existing `fuzzy_nars.py` already supports triple verification. Adapt it to work with new Triple schema:

```python
# In z3adapter/ikr/triples/verification.py

def triple_to_verification(triple: Triple) -> VerificationTriple:
    """Convert Triple to VerificationTriple for fuzzy-NARS."""
    return VerificationTriple(
        subject=triple.subject,
        predicate=triple.predicate.value,
        obj=triple.object,
        truth=triple.truth,
    )


def verify_triple_against_store(
    query: Triple,
    store: TripleStore,
    sim_fn: Callable[[str, str], float] = combined_lexical_similarity,
) -> VerificationResult:
    """Verify a triple against the store using fuzzy-NARS."""
    kb_triples = [
        triple_to_verification(t)
        for t in store.triples.values()
    ]
    query_vt = triple_to_verification(query)
    return verify_triple(query_vt, kb_triples, sim_fn)
```

### 4.2 Files to Create/Modify

| File | Purpose |
|------|---------|
| `z3adapter/ikr/triples/verification.py` | Verification bridge |
| `tests/unit/test_triple_verification.py` | Unit tests |

### 4.3 Acceptance Criteria

- [ ] Convert Triple ↔ VerificationTriple
- [ ] Verify against TripleStore
- [ ] Predicate polarity (causes ↔ prevents) works
- [ ] Unit tests

---

## Phase 5: Persistence (SQLite)

### 5.1 Database Schema

Create `z3adapter/ikr/triples/storage.py`:

```sql
-- Entities (Layer 1)
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    canonical_name TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE entity_surface_forms (
    entity_id TEXT REFERENCES entities(id),
    surface_form TEXT NOT NULL,
    PRIMARY KEY (entity_id, surface_form)
);

-- Triples (Layer 2)
CREATE TABLE triples (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,          -- Entity ID or triple reference (t:xxx)
    predicate TEXT NOT NULL,        -- One of 7 predicates
    object TEXT NOT NULL,           -- Entity ID or triple reference (t:xxx)
    negated BOOLEAN DEFAULT FALSE,
    frequency REAL,                 -- NARS truth value
    confidence REAL,
    source TEXT,
    surface_form TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_triples_subject ON triples(subject);
CREATE INDEX idx_triples_predicate ON triples(predicate);
CREATE INDEX idx_triples_object ON triples(object);
```

### 5.2 Files to Create

| File | Purpose |
|------|---------|
| `z3adapter/ikr/triples/storage.py` | SQLite persistence |
| `tests/unit/test_triple_storage.py` | Unit tests |

### 5.3 Acceptance Criteria

- [ ] SQLite schema creation
- [ ] CRUD operations for triples
- [ ] CRUD operations for entities
- [ ] Surface form tracking
- [ ] Query by subject/predicate/object
- [ ] Unit tests

---

## Phase 6: End-to-End Pipeline

### 6.1 Pipeline Integration

Create `z3adapter/ikr/triples/pipeline.py`:

```python
@dataclass
class ExtractionPipeline:
    """End-to-end text → verified triples pipeline."""

    extractor: TripleExtractor
    resolver: EntityResolver
    store: TripleStore

    def ingest(self, text: str, source: str = None) -> list[Triple]:
        """Extract, resolve, and store triples from text."""
        # 1. Extract raw triples
        result = self.extractor.extract(text, source=source)

        # 2. Resolve entities
        resolved_triples = []
        for triple in result.triples:
            resolved = Triple(
                id=triple.id,
                subject=self.resolver.resolve(triple.subject).canonical,
                predicate=triple.predicate,
                object=self.resolver.resolve(triple.object).canonical,
                negated=triple.negated,
                truth=triple.truth,
                source=triple.source,
                surface_form=triple.surface_form,
            )
            resolved_triples.append(resolved)

        # 3. Store
        for triple in resolved_triples:
            self.store.add(triple)

        return resolved_triples

    def query(self, question: str) -> VerificationResult:
        """Answer question by extracting query triple and verifying."""
        # Extract query as triple
        result = self.extractor.extract(question)
        if not result.triples:
            return None

        query_triple = result.triples[0]

        # Verify against store
        return verify_triple_against_store(query_triple, self.store)
```

### 6.2 Files to Create

| File | Purpose |
|------|---------|
| `z3adapter/ikr/triples/pipeline.py` | ExtractionPipeline |
| `tests/integration/test_pipeline.py` | Integration tests |

### 6.3 Acceptance Criteria

- [ ] End-to-end ingest: text → stored triples
- [ ] End-to-end query: question → verification result
- [ ] Entity resolution in pipeline
- [ ] Integration tests

---

## Phase 7: Cleanup

### 7.1 Remove Domain-Specific KBs

Delete or deprecate:
- `z3adapter/ikr/nars_datalog/kb/psychology.json`
- `z3adapter/ikr/nars_datalog/kb/biology.json`
- `z3adapter/ikr/kb/social.json`
- `z3adapter/ikr/kb/food.json`

Keep:
- `z3adapter/ikr/nars_datalog/kb/commonsense.json` (if generic enough)

### 7.2 Update Exports

Update `z3adapter/ikr/__init__.py` with new exports:
- `Triple`, `TripleStore`, `Predicate`
- `TripleExtractor`, `ExtractionResult`
- `EntityResolver`, `EntityMatch`
- `ExtractionPipeline`

### 7.3 Documentation

Update `CLAUDE.md` with:
- Triple extraction architecture
- 7 predicates explanation
- Usage examples

---

## Summary: File Structure

```
z3adapter/ikr/triples/
├── __init__.py           # Package exports
├── schema.py             # Triple, TripleStore, Predicate (Phase 1)
├── extractor.py          # TripleExtractor (Phase 2)
├── entity_resolver.py    # EntityResolver (Phase 3)
├── verification.py       # Fuzzy-NARS bridge (Phase 4)
├── storage.py            # SQLite persistence (Phase 5)
└── pipeline.py           # ExtractionPipeline (Phase 6)

tests/unit/
├── test_triple_schema.py
├── test_triple_extractor.py
├── test_entity_resolver.py
├── test_triple_verification.py
└── test_triple_storage.py

tests/integration/
├── test_triple_extractor_live.py
└── test_pipeline.py
```

---

## Implementation Order

| Phase | Focus | Dependencies | Estimated Complexity |
|-------|-------|--------------|---------------------|
| 1 | Data Model | None | Low |
| 2 | Extraction | Phase 1 | Medium |
| 3 | Entity Resolution | Phase 1, fuzzy_nars.py | Medium |
| 4 | Verification | Phase 1, 3, fuzzy_nars.py | Low |
| 5 | Persistence | Phase 1 | Medium |
| 6 | Pipeline | Phase 1-5 | Low |
| 7 | Cleanup | Phase 1-6 | Low |

---

## Open Questions

1. **Embedding-based entity resolution**: Use `make_embedding_similarity()` for semantic matching? Adds dependency on embedding API but improves matching quality.

2. **Batch extraction**: For book ingestion, need sliding window / chunking strategy. Port from pysem's `SlidingWindowChunk`?

3. **Conflict detection**: When two triples contradict, how to handle? NARS revision? Flag for human review?

4. **Triple ID generation**: UUID vs sequential (t1, t2, ...)? UUIDs are safer for distributed systems.

---

## Next Session

Start with **Phase 1: Core Data Model**
- Create `z3adapter/ikr/triples/` package
- Implement `schema.py`
- Write unit tests
