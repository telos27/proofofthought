"""Unit tests for VectorIndex (FAISS-based ANN search)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip all tests if FAISS is not installed
faiss = pytest.importorskip("faiss", reason="FAISS not installed")

from z3adapter.ikr.entities.vector_index import VectorIndex, FAISS_AVAILABLE


# Test fixtures
@pytest.fixture
def sample_embeddings() -> dict[str, list[float]]:
    """Create sample embeddings for testing."""
    np.random.seed(42)
    return {
        "anxiety": list(np.random.randn(128).astype(np.float32)),
        "stress": list(np.random.randn(128).astype(np.float32)),
        "fear": list(np.random.randn(128).astype(np.float32)),
        "happiness": list(np.random.randn(128).astype(np.float32)),
        "joy": list(np.random.randn(128).astype(np.float32)),
    }


@pytest.fixture
def similar_embeddings() -> dict[str, list[float]]:
    """Create embeddings where some are intentionally similar."""
    # Base vectors
    base_anxiety = np.array([1.0, 0.0, 0.0, 0.0] + [0.0] * 124, dtype=np.float32)
    base_happiness = np.array([0.0, 1.0, 0.0, 0.0] + [0.0] * 124, dtype=np.float32)

    # Create similar vectors by adding small noise
    stress = base_anxiety + np.random.randn(128).astype(np.float32) * 0.1
    fear = base_anxiety + np.random.randn(128).astype(np.float32) * 0.1
    joy = base_happiness + np.random.randn(128).astype(np.float32) * 0.1

    return {
        "anxiety": list(base_anxiety),
        "stress": list(stress),
        "fear": list(fear),
        "happiness": list(base_happiness),
        "joy": list(joy),
    }


@pytest.fixture
def index_128d() -> VectorIndex:
    """Create an empty 128-dimensional index."""
    return VectorIndex(dimension=128)


class TestVectorIndexCreation:
    """Tests for VectorIndex initialization."""

    def test_create_default_dimension(self):
        """Test creating index with default dimension (1536)."""
        index = VectorIndex()
        assert index.dimension == 1536
        assert index.is_empty
        assert len(index) == 0

    def test_create_custom_dimension(self):
        """Test creating index with custom dimension."""
        index = VectorIndex(dimension=128)
        assert index.dimension == 128
        assert index.is_empty

    def test_faiss_available(self):
        """Test that FAISS is available."""
        assert FAISS_AVAILABLE


class TestVectorIndexBuild:
    """Tests for building the index."""

    def test_build_from_embeddings(self, index_128d, sample_embeddings):
        """Test building index from embedding dict."""
        index_128d.build(sample_embeddings)

        assert len(index_128d) == 5
        assert not index_128d.is_empty
        assert set(index_128d.get_entity_ids()) == set(sample_embeddings.keys())

    def test_build_empty_embeddings_raises(self, index_128d):
        """Test that building from empty dict raises error."""
        with pytest.raises(ValueError, match="empty"):
            index_128d.build({})

    def test_build_wrong_dimension_raises(self, index_128d):
        """Test that wrong dimension raises error."""
        wrong_dim = {"entity1": [0.1] * 64}  # 64 instead of 128
        with pytest.raises(ValueError, match="dimension mismatch"):
            index_128d.build(wrong_dim)

    def test_build_uses_flat_index_for_small_data(self, index_128d, sample_embeddings):
        """Test that small datasets use flat index."""
        index_128d.build(sample_embeddings, use_ivf=True)
        # With only 5 vectors, should use flat index
        assert isinstance(index_128d.index, faiss.IndexFlatIP)

    def test_build_large_dataset_uses_ivf(self):
        """Test that large datasets use IVF index."""
        index = VectorIndex(dimension=32)
        np.random.seed(42)

        # Create 2000 random embeddings
        embeddings = {
            f"entity_{i}": list(np.random.randn(32).astype(np.float32))
            for i in range(2000)
        }

        index.build(embeddings, use_ivf=True)
        assert len(index) == 2000
        # Should use IVF for large dataset
        assert isinstance(index.index, faiss.IndexIVFFlat)

    def test_build_force_flat_index(self):
        """Test forcing flat index even with large dataset."""
        index = VectorIndex(dimension=32)
        np.random.seed(42)

        embeddings = {
            f"entity_{i}": list(np.random.randn(32).astype(np.float32))
            for i in range(2000)
        }

        index.build(embeddings, use_ivf=False)
        assert isinstance(index.index, faiss.IndexFlatIP)


class TestVectorIndexSearch:
    """Tests for searching the index."""

    def test_search_returns_results(self, index_128d, sample_embeddings):
        """Test basic search returns results."""
        index_128d.build(sample_embeddings)

        query = sample_embeddings["anxiety"]
        results = index_128d.search(query, k=3)

        assert len(results) == 3
        # First result should be exact match (highest score)
        assert results[0][0] == "anxiety"
        assert results[0][1] > 0.99  # Should be ~1.0 for exact match

    def test_search_similar_entities(self, similar_embeddings):
        """Test that similar embeddings are found together."""
        index = VectorIndex(dimension=128)
        index.build(similar_embeddings)

        # Search for something similar to anxiety
        query = similar_embeddings["anxiety"]
        results = index.search(query, k=5)

        # anxiety, stress, and fear should be in top 3
        top_3_ids = [r[0] for r in results[:3]]
        assert "anxiety" in top_3_ids
        assert "stress" in top_3_ids or "fear" in top_3_ids

    def test_search_empty_index_returns_empty(self, index_128d):
        """Test searching empty index returns empty list."""
        query = [0.1] * 128
        results = index_128d.search(query, k=10)
        assert results == []

    def test_search_k_larger_than_index(self, index_128d, sample_embeddings):
        """Test searching with k larger than index size."""
        index_128d.build(sample_embeddings)

        query = sample_embeddings["anxiety"]
        results = index_128d.search(query, k=100)  # More than 5 entities

        assert len(results) == 5  # Should return all available

    def test_search_wrong_dimension_raises(self, index_128d, sample_embeddings):
        """Test that wrong query dimension raises error."""
        index_128d.build(sample_embeddings)

        wrong_query = [0.1] * 64  # Wrong dimension
        with pytest.raises(ValueError, match="dimension mismatch"):
            index_128d.search(wrong_query)

    def test_search_scores_in_valid_range(self, index_128d, sample_embeddings):
        """Test that search scores are in [0, 1]."""
        index_128d.build(sample_embeddings)

        query = sample_embeddings["anxiety"]
        results = index_128d.search(query, k=5)

        for entity_id, score in results:
            assert 0.0 <= score <= 1.0

    def test_search_batch(self, index_128d, sample_embeddings):
        """Test batch search."""
        index_128d.build(sample_embeddings)

        queries = [
            sample_embeddings["anxiety"],
            sample_embeddings["happiness"],
        ]
        results = index_128d.search_batch(queries, k=3)

        assert len(results) == 2
        assert len(results[0]) == 3
        assert len(results[1]) == 3

        # Each query should find itself first
        assert results[0][0][0] == "anxiety"
        assert results[1][0][0] == "happiness"


class TestVectorIndexAdd:
    """Tests for incrementally adding to the index."""

    def test_add_to_empty_index(self, index_128d):
        """Test adding to empty index creates flat index."""
        embedding = [0.1] * 128
        index_128d.add("entity1", embedding)

        assert len(index_128d) == 1
        assert index_128d.contains("entity1")

    def test_add_multiple_entities(self, index_128d):
        """Test adding multiple entities incrementally."""
        for i in range(10):
            embedding = [float(i) / 10] * 128
            index_128d.add(f"entity_{i}", embedding)

        assert len(index_128d) == 10
        assert all(index_128d.contains(f"entity_{i}") for i in range(10))

    def test_add_to_existing_index(self, index_128d, sample_embeddings):
        """Test adding to already-built index."""
        index_128d.build(sample_embeddings)
        initial_size = len(index_128d)

        new_embedding = [0.5] * 128
        index_128d.add("new_entity", new_embedding)

        assert len(index_128d) == initial_size + 1
        assert index_128d.contains("new_entity")

    def test_add_wrong_dimension_raises(self, index_128d):
        """Test that wrong dimension raises error."""
        # Add one valid embedding first
        index_128d.add("entity1", [0.1] * 128)

        # Try to add wrong dimension
        with pytest.raises(ValueError, match="dimension mismatch"):
            index_128d.add("entity2", [0.1] * 64)


class TestVectorIndexRemove:
    """Tests for removing entities from the index."""

    def test_remove_existing_entity(self, index_128d, sample_embeddings):
        """Test removing an existing entity."""
        index_128d.build(sample_embeddings)

        result = index_128d.remove("anxiety")
        assert result is True
        assert not index_128d.contains("anxiety")

    def test_remove_nonexistent_entity(self, index_128d, sample_embeddings):
        """Test removing non-existent entity returns False."""
        index_128d.build(sample_embeddings)

        result = index_128d.remove("nonexistent")
        assert result is False

    def test_removed_entity_not_in_search(self, index_128d, sample_embeddings):
        """Test that removed entity doesn't appear in search results."""
        index_128d.build(sample_embeddings)
        index_128d.remove("anxiety")

        query = sample_embeddings["anxiety"]
        results = index_128d.search(query, k=5)

        result_ids = [r[0] for r in results]
        assert "anxiety" not in result_ids


class TestVectorIndexPersistence:
    """Tests for saving and loading the index."""

    def test_save_and_load(self, index_128d, sample_embeddings):
        """Test saving and loading index."""
        index_128d.build(sample_embeddings)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_index"
            index_128d.save(str(path))

            # Check files were created
            assert (path.with_suffix(".index")).exists()
            assert (path.with_suffix(".idmap")).exists()

            # Load into new index
            loaded_index = VectorIndex(dimension=128)
            loaded_index.load(str(path))

            assert len(loaded_index) == len(index_128d)
            assert loaded_index.dimension == index_128d.dimension
            assert set(loaded_index.get_entity_ids()) == set(index_128d.get_entity_ids())

    def test_loaded_index_search_works(self, index_128d, sample_embeddings):
        """Test that search works on loaded index."""
        index_128d.build(sample_embeddings)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_index"
            index_128d.save(str(path))

            loaded_index = VectorIndex(dimension=128)
            loaded_index.load(str(path))

            # Search should work
            query = sample_embeddings["anxiety"]
            results = loaded_index.search(query, k=3)

            assert len(results) == 3
            assert results[0][0] == "anxiety"

    def test_save_empty_index_raises(self, index_128d):
        """Test that saving empty index raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_index"
            with pytest.raises(RuntimeError, match="empty"):
                index_128d.save(str(path))

    def test_load_missing_file_raises(self, index_128d):
        """Test that loading missing files raises error."""
        with pytest.raises(FileNotFoundError):
            index_128d.load("/nonexistent/path")

    def test_save_creates_parent_dirs(self, index_128d, sample_embeddings):
        """Test that save creates parent directories."""
        index_128d.build(sample_embeddings)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dirs" / "index"
            index_128d.save(str(path))

            assert path.with_suffix(".index").exists()


class TestVectorIndexUtilities:
    """Tests for utility methods."""

    def test_get_entity_ids(self, index_128d, sample_embeddings):
        """Test getting all entity IDs."""
        index_128d.build(sample_embeddings)

        ids = index_128d.get_entity_ids()
        assert set(ids) == set(sample_embeddings.keys())

    def test_get_entity_ids_excludes_removed(self, index_128d, sample_embeddings):
        """Test that removed entities are excluded from get_entity_ids."""
        index_128d.build(sample_embeddings)
        index_128d.remove("anxiety")

        ids = index_128d.get_entity_ids()
        assert "anxiety" not in ids
        assert len(ids) == 4

    def test_contains(self, index_128d, sample_embeddings):
        """Test contains method."""
        index_128d.build(sample_embeddings)

        assert index_128d.contains("anxiety")
        assert index_128d.contains("stress")
        assert not index_128d.contains("nonexistent")

    def test_len(self, index_128d, sample_embeddings):
        """Test __len__ method."""
        assert len(index_128d) == 0

        index_128d.build(sample_embeddings)
        assert len(index_128d) == 5


class TestVectorIndexIntegration:
    """Integration tests for realistic scenarios."""

    def test_psychology_domain_search(self):
        """Test searching in a psychology-like domain."""
        index = VectorIndex(dimension=64)

        # Create embeddings that simulate semantic relationships
        np.random.seed(42)

        # Emotion cluster
        base_emotion = np.random.randn(64).astype(np.float32)
        embeddings = {
            "anxiety": list(base_emotion + np.random.randn(64).astype(np.float32) * 0.1),
            "fear": list(base_emotion + np.random.randn(64).astype(np.float32) * 0.1),
            "worry": list(base_emotion + np.random.randn(64).astype(np.float32) * 0.1),
        }

        # Cognitive cluster
        base_cognitive = np.random.randn(64).astype(np.float32)
        embeddings.update({
            "memory": list(base_cognitive + np.random.randn(64).astype(np.float32) * 0.1),
            "attention": list(base_cognitive + np.random.randn(64).astype(np.float32) * 0.1),
            "learning": list(base_cognitive + np.random.randn(64).astype(np.float32) * 0.1),
        })

        index.build(embeddings)

        # Search for anxiety-related concepts
        results = index.search(embeddings["anxiety"], k=3)
        top_ids = [r[0] for r in results]

        # Should find other emotions in top 3
        assert "anxiety" in top_ids
        assert any(e in top_ids for e in ["fear", "worry"])

    def test_incremental_knowledge_base(self):
        """Test building knowledge base incrementally."""
        index = VectorIndex(dimension=32)
        np.random.seed(42)

        # Simulate extracting entities from chapters
        chapters = [
            ["stress", "cortisol", "anxiety"],
            ["memory", "hippocampus", "learning"],
            ["emotion", "amygdala", "fear"],
        ]

        for chapter in chapters:
            for entity in chapter:
                embedding = list(np.random.randn(32).astype(np.float32))
                index.add(entity, embedding)

        assert len(index) == 9

        # Search should work
        results = index.search([0.0] * 32, k=5)
        assert len(results) == 5

    def test_large_scale_performance(self):
        """Test performance with larger dataset."""
        index = VectorIndex(dimension=64)
        np.random.seed(42)

        # Create 5000 entities
        embeddings = {
            f"entity_{i}": list(np.random.randn(64).astype(np.float32))
            for i in range(5000)
        }

        # Build should complete
        index.build(embeddings)
        assert len(index) == 5000

        # Search should be fast
        query = list(np.random.randn(64).astype(np.float32))
        results = index.search(query, k=10)
        assert len(results) == 10

        # Save/load should work
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "large_index"
            index.save(str(path))

            loaded = VectorIndex(dimension=64)
            loaded.load(str(path))
            assert len(loaded) == 5000
