# Session: Link-Based Architecture Phase 3 - EntityLinker

**Date**: 2026-01-16

## Summary

Implemented Phase 3 of the link-based knowledge architecture: EntityLinker for multi-level entity resolution with automatic linking.

## Work Done

### 1. Created `EntityLinker` Class

New class in `z3adapter/ikr/entities/linker.py`:

- **Multi-level resolution** (cascading strategy):
  1. Exact match by canonical name (O(1))
  2. Surface form lookup (O(1))
  3. Vector similarity search (O(log n))

- **Resolution behavior based on similarity score**:
  - High confidence (≥identity_threshold, default 0.9): Use existing entity
  - Moderate confidence (≥link_threshold, default 0.5): Create new entity with similarity links
  - Low confidence: Create new entity without links

- **Surface form learning**: High-confidence vector matches learn surface forms for O(1) future lookups

Key methods:
```python
class EntityLinker:
    def link(self, mention: str, entity_type=None, source=None) -> LinkResult
    def link_and_store(self, mention: str, ...) -> LinkResult  # Convenience method
    def link_batch(self, mentions: list[str], ...) -> list[LinkResult]

    @property
    def has_vector_search(self) -> bool
```

### 2. Created `LinkResult` Dataclass

Result object containing:
- `entity`: The resolved or newly created Entity
- `is_new`: Whether a new entity was created
- `links`: List of similarity links to create (EntityLink objects)
- `resolution_method`: How resolved ("exact", "surface_form", "vector_high", "vector_link", "new")
- `score`: Similarity score (1.0 for exact, 0.0 for new)

### 3. Comprehensive Test Suite

Created `tests/unit/test_entity_linker.py` with 36 tests:
- LinkResult creation tests
- EntityLinker initialization tests
- Exact match resolution tests
- Surface form resolution tests
- Vector search resolution tests (high/moderate/low confidence)
- Entity metadata propagation tests
- link_and_store convenience method tests
- Batch linking tests
- Integration tests (resolution cascade, psychology domain)

## Files Created/Modified

**Created:**
- `z3adapter/ikr/entities/linker.py` - EntityLinker and LinkResult classes
- `tests/unit/test_entity_linker.py` - 36 unit tests
- `.claude/sessions/2026-01-16-entity-linker-phase3.md` - This session note

**Modified:**
- `z3adapter/ikr/entities/__init__.py` - Added EntityLinker and LinkResult exports

## Test Results

- 36 new EntityLinker tests: All pass
- 37 existing entity tests: All pass
- 35 existing vector index tests: All pass

Total: 108 tests passing

## Usage Example

```python
from z3adapter.ikr.entities import EntityStore, EntityLinker, VectorIndex

# Setup
store = EntityStore("knowledge.db")
index = VectorIndex(dimension=1536)

def embed_fn(text: str) -> list[float]:
    return openai.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding

linker = EntityLinker(
    entity_store=store,
    vector_index=index,
    embed_fn=embed_fn,
    link_threshold=0.5,
    identity_threshold=0.9,
)

# Link a mention
result = linker.link("working memory")

if result.is_new:
    # Persist new entity and links
    store.add(result.entity)
    for link in result.links:
        store.add_link(link)
else:
    print(f"Resolved to: {result.entity.name}")
    print(f"Method: {result.resolution_method}")

# Or use convenience method
result = linker.link_and_store("cognitive load", entity_type="concept")
```

## Next Steps (Phase 4)

1. **PrecomputationPipeline**: Batch embedding and link computation
   - Compute embeddings for entities missing them
   - Rebuild vector index from all embeddings
   - Pre-compute top-k similarity links for each entity

2. **CLI commands** for pre-computation:
   - `compute-embeddings`: Batch embed entities
   - `rebuild-index`: Rebuild FAISS index
   - `compute-links`: Pre-compute similarity links

## Architecture Notes

### Resolution Cascade
```
mention
  │
  ▼
Exact match? ──yes──▶ Return existing entity (method="exact")
  │no
  ▼
Surface form? ──yes──▶ Return existing entity (method="surface_form")
  │no
  ▼
Vector search
  │
  ├── score ≥ 0.9 ──▶ Return existing + learn surface form (method="vector_high")
  │
  ├── score ≥ 0.5 ──▶ Return new entity + links (method="vector_link")
  │
  └── score < 0.5 ──▶ Return new entity (method="new")
```

### Surface Form Learning
When vector search finds a high-confidence match (≥identity_threshold):
- The normalized mention is saved as a surface form
- Future lookups for the same mention resolve in O(1)
- This learns aliases like "WM" → "working_memory"
