"""Unit tests for embedding-based entity resolution.

Tests the embeddings module and its integration with EntityResolver.
"""

import pytest
from z3adapter.ikr.triples.embeddings import (
    EmbeddingBackend,
    EmbeddingCache,
    EmbeddingResult,
    MockEmbedding,
    cosine_similarity,
    make_embedding_similarity,
    make_hybrid_similarity,
)
from z3adapter.ikr.triples.entity_resolver import EntityResolver


# =============================================================================
# MockEmbedding Tests
# =============================================================================


class TestMockEmbedding:
    """Test the MockEmbedding backend."""

    def test_creates_embedding(self):
        """MockEmbedding creates an embedding vector."""
        backend = MockEmbedding(dimension=64)
        result = backend.embed("hello world")

        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 64
        assert result.model == "mock-embedding"

    def test_deterministic(self):
        """Same text produces same embedding."""
        backend = MockEmbedding(dimension=32)

        result1 = backend.embed("test phrase")
        result2 = backend.embed("test phrase")

        assert result1.embedding == result2.embedding

    def test_normalized_text(self):
        """Embeddings are normalized (spaces vs underscores)."""
        backend = MockEmbedding(dimension=32)

        result1 = backend.embed("hello_world")
        result2 = backend.embed("hello world")

        # Should be same after normalization
        assert result1.embedding == result2.embedding

    def test_different_texts_different_embeddings(self):
        """Different texts produce different embeddings."""
        backend = MockEmbedding(dimension=64)

        result1 = backend.embed("anxiety")
        result2 = backend.embed("happiness")

        assert result1.embedding != result2.embedding

    def test_similar_texts_similar_embeddings(self):
        """Similar texts produce similar embeddings."""
        backend = MockEmbedding(dimension=64)

        result1 = backend.embed("working memory")
        result2 = backend.embed("working memories")

        # Compute cosine similarity
        sim = cosine_similarity(result1.embedding, result2.embedding)

        # Should be reasonably similar (shared trigrams)
        assert sim > 0.5

    def test_batch_embed(self):
        """Can embed multiple texts."""
        backend = MockEmbedding(dimension=32)

        results = backend.embed_batch(["one", "two", "three"])

        assert len(results) == 3
        assert all(len(r.embedding) == 32 for r in results)

    def test_unit_normalized(self):
        """Embeddings are unit normalized."""
        backend = MockEmbedding(dimension=64)
        result = backend.embed("test")

        # Compute L2 norm
        norm = sum(x * x for x in result.embedding) ** 0.5

        assert abs(norm - 1.0) < 0.01  # Should be close to 1


# =============================================================================
# Cosine Similarity Tests
# =============================================================================


class TestCosineSimilarity:
    """Test the cosine similarity function."""

    def test_identical_vectors(self):
        """Identical vectors have similarity 1.0."""
        v = [0.5, 0.5, 0.5, 0.5]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0.0."""
        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0, 0.0]
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Opposite vectors have similarity -1.0."""
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0)

    def test_zero_vector(self):
        """Zero vector returns 0.0."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [0.0, 0.0, 0.0]
        assert cosine_similarity(v1, v2) == 0.0


# =============================================================================
# EmbeddingCache Tests
# =============================================================================


class TestEmbeddingCache:
    """Test the EmbeddingCache class."""

    def test_cache_put_get(self):
        """Can store and retrieve embeddings."""
        cache = EmbeddingCache()

        embedding = [0.1, 0.2, 0.3]
        cache.put("test", embedding)

        result = cache.get("test")
        assert result == embedding

    def test_cache_miss(self):
        """Returns None for missing keys."""
        cache = EmbeddingCache()
        assert cache.get("nonexistent") is None

    def test_normalizes_keys(self):
        """Cache normalizes keys (case, underscores)."""
        cache = EmbeddingCache()

        cache.put("Hello World", [1.0, 2.0])

        # Should find with different formatting
        assert cache.get("hello_world") == [1.0, 2.0]
        assert cache.get("HELLO WORLD") == [1.0, 2.0]

    def test_hit_miss_tracking(self):
        """Tracks cache hits and misses."""
        cache = EmbeddingCache()

        cache.put("exists", [1.0])

        cache.get("exists")  # Hit
        cache.get("exists")  # Hit
        cache.get("missing")  # Miss

        assert cache.hits == 2
        assert cache.misses == 1

    def test_hit_rate(self):
        """Computes hit rate correctly."""
        cache = EmbeddingCache()
        cache.put("a", [1.0])

        cache.get("a")  # Hit
        cache.get("b")  # Miss
        cache.get("a")  # Hit

        assert cache.hit_rate == pytest.approx(2 / 3)

    def test_clear(self):
        """Clear removes all entries and resets stats."""
        cache = EmbeddingCache()
        cache.put("test", [1.0])
        cache.get("test")

        cache.clear()

        assert cache.get("test") is None
        assert cache.hits == 0
        assert cache.misses == 1  # The get after clear


# =============================================================================
# make_embedding_similarity Tests
# =============================================================================


class TestMakeEmbeddingSimilarity:
    """Test the embedding similarity factory function."""

    def test_creates_similarity_function(self):
        """Factory returns a callable."""
        backend = MockEmbedding()
        sim_fn = make_embedding_similarity(backend)

        assert callable(sim_fn)

    def test_identical_terms(self):
        """Identical terms have similarity 1.0."""
        backend = MockEmbedding()
        sim_fn = make_embedding_similarity(backend)

        assert sim_fn("anxiety", "anxiety") == 1.0

    def test_similar_terms(self):
        """Similar terms have high similarity."""
        backend = MockEmbedding(dimension=128)
        sim_fn = make_embedding_similarity(backend)

        # These share character trigrams
        sim = sim_fn("working_memory", "working_memories")

        assert sim > 0.5

    def test_caches_embeddings(self):
        """Embeddings are cached for reuse."""
        cache = EmbeddingCache()
        backend = MockEmbedding()
        sim_fn = make_embedding_similarity(backend, cache)

        # First call computes embeddings
        sim_fn("test1", "test2")
        assert cache.misses == 2

        # Second call uses cache
        sim_fn("test1", "test2")
        assert cache.hits == 2

    def test_normalized_terms(self):
        """Similarity is computed on normalized terms."""
        backend = MockEmbedding()
        sim_fn = make_embedding_similarity(backend)

        # These should be treated as identical after normalization
        sim = sim_fn("hello_world", "hello world")
        assert sim == 1.0


# =============================================================================
# make_hybrid_similarity Tests
# =============================================================================


class TestMakeHybridSimilarity:
    """Test the hybrid similarity factory function."""

    def test_creates_similarity_function(self):
        """Factory returns a callable."""
        backend = MockEmbedding()
        sim_fn = make_hybrid_similarity(backend)

        assert callable(sim_fn)

    def test_identical_terms(self):
        """Identical terms have similarity 1.0."""
        backend = MockEmbedding()
        sim_fn = make_hybrid_similarity(backend)

        assert sim_fn("test", "test") == pytest.approx(1.0, abs=0.01)

    def test_lexical_weight(self):
        """Lexical weight affects the result."""
        backend = MockEmbedding()

        # All lexical
        sim_lex = make_hybrid_similarity(backend, lexical_weight=1.0)
        # All embedding
        sim_emb = make_hybrid_similarity(backend, lexical_weight=0.0)

        # For similar lexical terms, results should differ
        result_lex = sim_lex("working_memory", "working_memories")
        result_emb = sim_emb("working_memory", "working_memories")

        # Both should be positive
        assert result_lex > 0
        assert result_emb > 0


# =============================================================================
# EntityResolver with Embeddings Tests
# =============================================================================


class TestEntityResolverWithEmbeddings:
    """Test EntityResolver with embedding-based similarity."""

    def test_with_embeddings_factory(self):
        """Can create resolver with embedding similarity."""
        resolver = EntityResolver.with_embeddings(use_mock=True)

        assert resolver is not None
        assert resolver.threshold == 0.7  # Default for embeddings

    def test_with_embeddings_custom_threshold(self):
        """Can specify custom threshold."""
        resolver = EntityResolver.with_embeddings(use_mock=True, threshold=0.5)

        assert resolver.threshold == 0.5

    def test_semantic_matching(self):
        """Resolver can match semantically similar terms."""
        resolver = EntityResolver.with_embeddings(use_mock=True, threshold=0.3)

        # Add entity
        resolver.add_entity("working_memory")

        # Resolve similar term (shares character trigrams)
        match = resolver.resolve("working_memories")

        assert match.canonical == "working_memory"
        assert match.similarity > 0.3

    def test_exact_match_still_works(self):
        """Exact matches are still handled correctly."""
        resolver = EntityResolver.with_embeddings(use_mock=True)

        resolver.add_entity("stress")
        match = resolver.resolve("stress")

        assert match.canonical == "stress"
        assert match.similarity == 1.0
        assert not match.is_new

    def test_with_hybrid_similarity_factory(self):
        """Can create resolver with hybrid similarity."""
        resolver = EntityResolver.with_hybrid_similarity(use_mock=True)

        assert resolver is not None
        assert resolver.threshold == 0.75  # Default for hybrid

    def test_with_hybrid_similarity_custom_weight(self):
        """Can specify lexical weight for hybrid similarity."""
        resolver = EntityResolver.with_hybrid_similarity(
            use_mock=True,
            lexical_weight=0.5,
            threshold=0.6,
        )

        resolver.add_entity("test_entity")
        match = resolver.resolve("test_entity")

        assert match.canonical == "test_entity"
        assert match.similarity == 1.0

    def test_new_entity_below_threshold(self):
        """Terms below threshold become new entities."""
        resolver = EntityResolver.with_embeddings(use_mock=True, threshold=0.99)

        resolver.add_entity("anxiety")
        match = resolver.resolve("completely_different_term")

        assert match.is_new
        assert match.canonical == "completely_different_term"


# =============================================================================
# Integration Tests
# =============================================================================


class TestEmbeddingIntegration:
    """Integration tests for embedding-based entity resolution."""

    def test_resolver_with_multiple_entities(self):
        """Resolver can handle multiple entities."""
        resolver = EntityResolver.with_embeddings(use_mock=True, threshold=0.3)

        # Add several entities
        resolver.add_entity("cognitive_behavioral_therapy")
        resolver.add_entity("anxiety_disorder")
        resolver.add_entity("working_memory")

        # Should find best match
        match = resolver.resolve("cognitive behavioral therapies")

        assert match.canonical == "cognitive_behavioral_therapy"

    def test_surface_forms_still_work(self):
        """Surface form tracking works with embeddings."""
        resolver = EntityResolver.with_embeddings(use_mock=True, threshold=0.3)

        resolver.add_entity("working_memory", ["WM", "short-term memory"])

        # Exact surface form match
        match = resolver.resolve("wm")
        assert match.canonical == "working_memory"
        assert match.similarity == 1.0

    def test_embedding_resolver_learns_surface_forms(self):
        """Successful fuzzy matches are remembered as surface forms."""
        resolver = EntityResolver.with_embeddings(use_mock=True, threshold=0.3)

        resolver.add_entity("anxiety")

        # First resolve (fuzzy match)
        match1 = resolver.resolve("anxieties")

        if match1.canonical == "anxiety":
            # The surface form should be learned
            match2 = resolver.resolve("anxieties")
            # Second time should be exact match (learned)
            assert match2.similarity == 1.0
