# Session: Link-Based Architecture Phase 2 - VectorIndex

**Date**: 2026-01-16

## Summary

Implemented Phase 2 of the link-based knowledge architecture: FAISS-based vector index for approximate nearest neighbor (ANN) search.

## Work Done

### 1. Created `VectorIndex` Class

New class in `z3adapter/ikr/entities/vector_index.py`:

- **FAISS integration**: Uses `faiss-cpu` for fast similarity search
- **Adaptive index type**:
  - Flat index (exact search) for small datasets (<1000)
  - IVF index (approximate) for large datasets
- **ID mapping**: FAISS integer indices → entity ID strings
- **Cosine similarity**: Vectors normalized for cosine similarity via inner product
- **Persistence**: Save/load index and ID map to disk

Key methods:
```python
class VectorIndex:
    def build(self, embeddings: dict[str, list[float]]) -> None
    def add(self, entity_id: str, embedding: list[float]) -> None
    def search(self, query: list[float], k: int = 10) -> list[tuple[str, float]]
    def search_batch(self, queries: list[list[float]], k: int) -> list[list[tuple[str, float]]]
    def remove(self, entity_id: str) -> bool
    def save(self, path: str) -> None
    def load(self, path: str) -> None
```

### 2. Graceful Degradation

- Optional import: If FAISS not installed, `FAISS_AVAILABLE = False`
- Tests skip gracefully with `pytest.importorskip("faiss")`
- Package still works for non-vector operations

### 3. Comprehensive Test Suite

Created `tests/unit/test_vector_index.py` with 35 tests:
- Creation (default/custom dimension)
- Build (from dict, empty, wrong dimension, flat vs IVF)
- Search (basic, similar entities, empty index, batch)
- Add (incremental, to existing index)
- Remove (with exclusion from search)
- Persistence (save/load roundtrip)
- Integration (psychology domain, large scale)

## Files Created/Modified

**Created:**
- `z3adapter/ikr/entities/vector_index.py`
- `tests/unit/test_vector_index.py`
- `.claude/sessions/2026-01-16-vector-index-phase2.md`

**Modified:**
- `z3adapter/ikr/entities/__init__.py` (added VectorIndex export)

## Dependencies

Added `faiss-cpu` to optional dependencies. Install with:
```bash
pip install faiss-cpu
```

## Test Results

- 35 new VectorIndex tests: All pass
- 37 existing entity tests: All pass

## Next Steps (Phase 3)

1. **EntityLinker**: Multi-level entity resolution
   - Exact match → Surface form lookup → Vector search
   - Integration with EntityStore and VectorIndex
   - Surface form learning (learn mappings from successful resolutions)

2. **Integration with EntityStore**:
   - Method to build VectorIndex from stored embeddings
   - Automatic index updates on entity add

## Architecture Notes

### Index Selection
- <1000 vectors: Flat index (exact, O(n))
- ≥1000 vectors: IVF with sqrt(n) clusters (approximate, O(log n))

### Similarity Scores
- Vectors L2-normalized before indexing
- Inner product = cosine similarity for normalized vectors
- Scores clamped to [0, 1]

### Persistence Format
- `{path}.index`: FAISS binary index
- `{path}.idmap`: JSON with dimension, ID map, metadata
