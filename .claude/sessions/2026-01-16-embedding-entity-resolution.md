# Session: Embedding-Based Entity Resolution

## Summary

Added embedding-based semantic similarity to the entity resolution system, enabling better matching for synonyms and semantically related terms.

## Files Created

| File | Purpose |
|------|---------|
| `z3adapter/ikr/triples/embeddings.py` | Embedding backends and similarity functions |
| `tests/unit/test_embeddings.py` | 35 unit tests for embeddings |

## Files Modified

| File | Change |
|------|--------|
| `z3adapter/ikr/triples/entity_resolver.py` | Added `with_embeddings()` and `with_hybrid_similarity()` factory methods |
| `z3adapter/ikr/triples/pipeline.py` | Added `create_with_embeddings()` factory method |
| `z3adapter/ikr/triples/__init__.py` | Added embedding exports |

## Components Added

### EmbeddingBackend Abstraction

```python
from z3adapter.ikr.triples.embeddings import (
    EmbeddingBackend,  # Abstract base class
    OpenAIEmbedding,   # Uses OpenAI API (text-embedding-3-small)
    MockEmbedding,     # Deterministic for testing, no API calls
    EmbeddingCache,    # Caches embeddings for reuse
)
```

### Similarity Functions

```python
from z3adapter.ikr.triples.embeddings import (
    cosine_similarity,           # Vector similarity
    make_embedding_similarity,   # Factory for embedding-based similarity
    make_hybrid_similarity,      # Lexical + embedding hybrid
)
```

### EntityResolver Factory Methods

```python
# Pure embedding similarity
resolver = EntityResolver.with_embeddings(use_mock=True)

# Hybrid (lexical + embedding)
resolver = EntityResolver.with_hybrid_similarity(lexical_weight=0.3)
```

### Pipeline Factory Method

```python
# Pipeline with embedding-based resolution
pipeline = ExtractionPipeline.create_with_embeddings(
    client,
    model="gpt-4o",
    use_mock_embeddings=True,  # For testing
    use_hybrid=True,           # Combine lexical + embedding
)
```

## Design Decisions

1. **MockEmbedding for testing**: Uses character trigram features to create deterministic embeddings where similar texts have similar embeddings. No API calls needed.

2. **Pluggable similarity**: Both EntityResolver and ExtractionPipeline accept custom similarity functions.

3. **Hybrid similarity**: Combines lexical (handles typos, variants) with embedding (semantic understanding). Default: 30% lexical, 70% embedding.

4. **Lower threshold for embeddings**: Default threshold is 0.7 for embeddings (vs 0.8 for lexical) since semantic matches are inherently fuzzier.

## Usage Examples

```python
# Testing (no API calls)
resolver = EntityResolver.with_embeddings(use_mock=True, threshold=0.5)
resolver.add_entity("anxiety_disorder")
match = resolver.resolve("worry and fear")  # Semantic match!

# Production (requires OPENAI_API_KEY)
resolver = EntityResolver.with_embeddings()
resolver.add_entity("cognitive_behavioral_therapy", ["CBT"])
match = resolver.resolve("talk therapy for anxiety")

# Pipeline with embeddings
pipeline = ExtractionPipeline.create_with_embeddings(
    client, model="gpt-4o", use_mock_embeddings=True
)
pipeline.ingest("Stress causes anxiety.")
result = pipeline.query("Does worry cause stress?")
```

## Test Coverage

- 35 new tests for embeddings module
- All 573 project tests pass

## Ported From

- `pysem/psychology-knowledge/retrieval/embedding_manager.py`
- `pysem/psychology-knowledge/knowledge_store/entity_resolver.py`
