# Session: Link-Based Architecture Phase 4 - PrecomputationPipeline

**Date**: 2026-01-16

## Summary

Implemented Phase 4 of the link-based knowledge architecture: PrecomputationPipeline for batch embedding and link computation.

## Work Done

### 1. Created `PrecomputationPipeline` Class

New class in `z3adapter/ikr/entities/precompute.py`:

- **Batch embedding computation**: Compute embeddings for entities missing them
- **Vector index rebuild**: Rebuild FAISS index from stored embeddings
- **Similarity link pre-computation**: Pre-compute top-k similarity links for each entity
- **Progress callbacks**: Optional callbacks for monitoring long-running operations
- **Full pipeline**: Run all three stages in sequence

Key methods:
```python
class PrecomputationPipeline:
    def compute_embeddings(self, batch_size: int = 100) -> int
    def rebuild_vector_index(self, index_path: Optional[str] = None) -> int
    def compute_links(self, k: int = 50, min_score: float = 0.5, clear_existing: bool = False) -> int
    def run_full_pipeline(...) -> PrecomputationStats

    # Utilities
    def get_entities_needing_embeddings(self) -> list[Entity]
    def get_embedding_coverage(self) -> tuple[int, int]
```

### 2. Created `PrecomputationStats` Dataclass

Statistics returned from pipeline runs:
- `entities_processed`: Total entities in store
- `embeddings_computed`: New embeddings computed
- `links_created`: Similarity links created
- `index_size`: Vectors in the rebuilt index

### 3. Comprehensive Test Suite

Created `tests/unit/test_precomputation.py` with 32 tests:
- PrecomputationStats tests
- Pipeline initialization tests
- Embedding computation tests (batching, skipping existing, progress)
- Vector index rebuild tests
- Link computation tests (min_score, self-exclusion, clear existing)
- Full pipeline tests
- Utility method tests
- Integration tests (psychology domain, incremental updates)

## Files Created/Modified

**Created:**
- `z3adapter/ikr/entities/precompute.py` - PrecomputationPipeline and PrecomputationStats
- `tests/unit/test_precomputation.py` - 32 unit tests
- `.claude/sessions/2026-01-16-precomputation-phase4.md` - This session note

**Modified:**
- `z3adapter/ikr/entities/__init__.py` - Added PrecomputationPipeline and PrecomputationStats exports

## Test Results

- 32 new PrecomputationPipeline tests: All pass
- 36 EntityLinker tests: All pass
- 37 Entity tests: All pass
- 35 VectorIndex tests: All pass

Total: 140 tests passing

## Usage Example

```python
from z3adapter.ikr.entities import EntityStore, VectorIndex, PrecomputationPipeline

# Setup
store = EntityStore("knowledge.db")
index = VectorIndex(dimension=1536)

def batch_embed(texts: list[str]) -> list[list[float]]:
    response = openai.embeddings.create(input=texts, model="text-embedding-3-small")
    return [d.embedding for d in response.data]

pipeline = PrecomputationPipeline(
    entity_store=store,
    vector_index=index,
    batch_embed_fn=batch_embed,
    model_name="text-embedding-3-small",
)

# Full pipeline run
stats = pipeline.run_full_pipeline(
    embedding_batch_size=100,
    link_k=50,
    link_min_score=0.5,
    index_path="knowledge_index",
)

print(f"Entities: {stats.entities_processed}")
print(f"Embeddings computed: {stats.embeddings_computed}")
print(f"Index size: {stats.index_size}")
print(f"Links created: {stats.links_created}")

# Or run stages individually
n_embeddings = pipeline.compute_embeddings(batch_size=50)
n_vectors = pipeline.rebuild_vector_index(index_path="my_index")
n_links = pipeline.compute_links(k=20, min_score=0.6)

# With progress callback
def on_progress(operation: str, current: int, total: int):
    print(f"{operation}: {current}/{total}")

pipeline.progress_callback = on_progress
pipeline.run_full_pipeline()
```

## Next Steps (Phase 5)

**QueryExpander**: Expand queries using entity links
- Expand query subject/object via similarity links
- Generate cross-product of patterns
- Compute combined confidence scores
- Integrate with NARS truth propagation

## Architecture Notes

### Pipeline Stages

```
1. compute_embeddings()
   └─ For each entity without embedding:
      ├─ Convert name to text (replace _ with space)
      ├─ Include description if available
      ├─ Call batch_embed_fn
      └─ Save to entity_store

2. rebuild_vector_index()
   └─ Get all embeddings from store
   └─ Build FAISS index
   └─ Optionally save to disk

3. compute_links()
   └─ For each entity:
      ├─ Search for k nearest neighbors
      ├─ Filter by min_score
      ├─ Exclude self
      └─ Create EntityLink objects
```

### Entity to Text Conversion
- Simple names: `working_memory` → `"working memory"`
- With description: `anxiety + "Feeling of worry"` → `"anxiety: Feeling of worry"`

### Incremental Updates
The pipeline supports incremental updates:
- `compute_embeddings()` skips entities that already have embeddings
- `run_full_pipeline()` can be run repeatedly; only new entities get embeddings
- `clear_existing=True` in `compute_links()` clears old links before computing new ones
