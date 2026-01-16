"""Tests for PrecomputationPipeline (batch embedding and link computation).

Tests the core functionality:
- Batch embedding computation
- Vector index rebuilding
- Similarity link pre-computation
- Full pipeline execution
- Progress callbacks
"""

import pytest
from unittest.mock import MagicMock, call

from z3adapter.ikr.entities import (
    Entity,
    EntityStore,
    LinkType,
    PrecomputationPipeline,
    PrecomputationStats,
)

# Skip all tests if FAISS is not available
faiss = pytest.importorskip("faiss")
from z3adapter.ikr.entities import VectorIndex


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def store():
    """Create an in-memory entity store."""
    return EntityStore()


@pytest.fixture
def index():
    """Create a vector index with small dimension for testing."""
    return VectorIndex(dimension=8)


@pytest.fixture
def sample_entities(store):
    """Add sample entities to the store and return them."""
    entities = {
        "anxiety": Entity(name="anxiety", entity_type="emotion"),
        "stress": Entity(name="stress", entity_type="state"),
        "depression": Entity(name="depression", entity_type="disorder"),
        "fear": Entity(name="fear", entity_type="emotion"),
        "worry": Entity(name="worry", entity_type="emotion"),
    }
    for entity in entities.values():
        store.add(entity)
    return entities


@pytest.fixture
def mock_embed_fn():
    """Create a mock embedding function that returns deterministic embeddings."""
    def embed_fn(texts: list[str]) -> list[list[float]]:
        # Generate deterministic embeddings based on text content
        embeddings = []
        for text in texts:
            # Simple hash-based embedding for testing
            h = hash(text) % 1000
            embedding = [(h + i) / 1000.0 for i in range(8)]
            embeddings.append(embedding)
        return embeddings
    return embed_fn


@pytest.fixture
def similar_embed_fn():
    """Create embedding function that produces similar vectors for related concepts."""
    def embed_fn(texts: list[str]) -> list[list[float]]:
        # Create embeddings where similar concepts have similar vectors
        base_vectors = {
            "anxiety": [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "stress": [0.85, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Similar to anxiety
            "fear": [0.88, 0.12, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Similar to anxiety
            "worry": [0.87, 0.13, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Similar to anxiety
            "depression": [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # Different
        }
        embeddings = []
        for text in texts:
            # Look up base vector or generate random
            key = text.lower().replace(" ", "_")
            if key in base_vectors:
                embeddings.append(base_vectors[key])
            else:
                h = hash(text) % 1000
                embeddings.append([(h + i) / 1000.0 for i in range(8)])
        return embeddings
    return embed_fn


# =============================================================================
# PrecomputationStats Tests
# =============================================================================


class TestPrecomputationStats:
    """Tests for PrecomputationStats dataclass."""

    def test_default_values(self):
        """Test default values are zero."""
        stats = PrecomputationStats()
        assert stats.entities_processed == 0
        assert stats.embeddings_computed == 0
        assert stats.links_created == 0
        assert stats.index_size == 0

    def test_custom_values(self):
        """Test setting custom values."""
        stats = PrecomputationStats(
            entities_processed=100,
            embeddings_computed=50,
            links_created=500,
            index_size=100,
        )
        assert stats.entities_processed == 100
        assert stats.embeddings_computed == 50
        assert stats.links_created == 500
        assert stats.index_size == 100


# =============================================================================
# PrecomputationPipeline Initialization Tests
# =============================================================================


class TestPipelineInit:
    """Tests for PrecomputationPipeline initialization."""

    def test_basic_creation(self, store, index, mock_embed_fn):
        """Test basic pipeline creation."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=mock_embed_fn,
        )
        assert pipeline.entity_store == store
        assert pipeline.vector_index == index
        assert pipeline.batch_embed_fn == mock_embed_fn
        assert pipeline.model_name == "embedding"
        assert pipeline.progress_callback is None

    def test_creation_with_model_name(self, store, index, mock_embed_fn):
        """Test pipeline creation with custom model name."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=mock_embed_fn,
            model_name="text-embedding-3-small",
        )
        assert pipeline.model_name == "text-embedding-3-small"

    def test_creation_with_progress_callback(self, store, index, mock_embed_fn):
        """Test pipeline creation with progress callback."""
        callback = MagicMock()
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=mock_embed_fn,
            progress_callback=callback,
        )
        assert pipeline.progress_callback == callback


# =============================================================================
# Embedding Computation Tests
# =============================================================================


class TestComputeEmbeddings:
    """Tests for compute_embeddings method."""

    def test_compute_embeddings_empty_store(self, store, index, mock_embed_fn):
        """Test computing embeddings with no entities."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=mock_embed_fn,
        )
        result = pipeline.compute_embeddings()
        assert result == 0

    def test_compute_embeddings_all_entities(self, store, index, mock_embed_fn, sample_entities):
        """Test computing embeddings for all entities."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=mock_embed_fn,
        )
        result = pipeline.compute_embeddings()
        assert result == 5  # All 5 entities

        # Verify embeddings were saved
        for entity in sample_entities.values():
            embedding = store.get_embedding(entity.id)
            assert embedding is not None
            assert len(embedding.embedding) == 8

    def test_compute_embeddings_skips_existing(self, store, index, mock_embed_fn, sample_entities):
        """Test that entities with embeddings are skipped."""
        # Add embedding for one entity
        store.save_embedding(
            sample_entities["anxiety"].id,
            [0.1] * 8,
            "existing-model",
        )

        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=mock_embed_fn,
        )
        result = pipeline.compute_embeddings()
        assert result == 4  # Only 4 entities (anxiety already has embedding)

    def test_compute_embeddings_batch_size(self, store, index, sample_entities):
        """Test that embeddings are computed in batches."""
        call_counts = []

        def tracking_embed_fn(texts: list[str]) -> list[list[float]]:
            call_counts.append(len(texts))
            return [[0.1] * 8 for _ in texts]

        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=tracking_embed_fn,
        )
        pipeline.compute_embeddings(batch_size=2)

        # With 5 entities and batch_size=2, should have 3 batches: [2, 2, 1]
        assert len(call_counts) == 3
        assert call_counts == [2, 2, 1]

    def test_compute_embeddings_model_name_saved(self, store, index, mock_embed_fn, sample_entities):
        """Test that model name is saved with embeddings."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=mock_embed_fn,
            model_name="test-model-v1",
        )
        pipeline.compute_embeddings()

        embedding = store.get_embedding(sample_entities["anxiety"].id)
        assert embedding.model == "test-model-v1"

    def test_compute_embeddings_progress_callback(self, store, index, mock_embed_fn, sample_entities):
        """Test progress callback is called during embedding computation."""
        callback = MagicMock()
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=mock_embed_fn,
            progress_callback=callback,
        )
        pipeline.compute_embeddings(batch_size=2)

        # Should be called after each batch
        assert callback.call_count == 3
        # Check the operation name
        for c in callback.call_args_list:
            assert c[0][0] == "compute_embeddings"

    def test_entity_description_included(self, store, index, sample_entities):
        """Test that entity description is included in embedding text."""
        # Add description to an entity
        entity = Entity(
            name="ptsd",
            entity_type="disorder",
            description="Post-traumatic stress disorder",
        )
        store.add(entity)

        captured_texts = []

        def capturing_embed_fn(texts: list[str]) -> list[list[float]]:
            captured_texts.extend(texts)
            return [[0.1] * 8 for _ in texts]

        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=capturing_embed_fn,
        )
        pipeline.compute_embeddings()

        # Find the text for ptsd
        ptsd_text = [t for t in captured_texts if "ptsd" in t.lower()][0]
        assert "Post-traumatic stress disorder" in ptsd_text


# =============================================================================
# Vector Index Rebuild Tests
# =============================================================================


class TestRebuildVectorIndex:
    """Tests for rebuild_vector_index method."""

    def test_rebuild_empty_store(self, store, index, mock_embed_fn):
        """Test rebuilding index with no embeddings."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=mock_embed_fn,
        )
        result = pipeline.rebuild_vector_index()
        assert result == 0

    def test_rebuild_with_embeddings(self, store, index, mock_embed_fn, sample_entities):
        """Test rebuilding index from stored embeddings."""
        # First compute embeddings
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=mock_embed_fn,
        )
        pipeline.compute_embeddings()

        # Then rebuild index
        result = pipeline.rebuild_vector_index()
        assert result == 5

        # Verify index has the vectors
        assert len(index) == 5

    def test_rebuild_saves_to_path(self, store, index, mock_embed_fn, sample_entities, tmp_path):
        """Test rebuilding index and saving to disk."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=mock_embed_fn,
        )
        pipeline.compute_embeddings()

        index_path = str(tmp_path / "test_index")
        pipeline.rebuild_vector_index(index_path=index_path)

        # Verify files were created
        assert (tmp_path / "test_index.index").exists()
        assert (tmp_path / "test_index.idmap").exists()

    def test_rebuild_progress_callback(self, store, index, mock_embed_fn, sample_entities):
        """Test progress callback during index rebuild."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=mock_embed_fn,
        )
        pipeline.compute_embeddings()

        callback = MagicMock()
        pipeline.progress_callback = callback
        pipeline.rebuild_vector_index()

        # Should be called twice: start (0/1) and end (1/1)
        assert callback.call_count == 2
        callback.assert_any_call("rebuild_index", 0, 1)
        callback.assert_any_call("rebuild_index", 1, 1)


# =============================================================================
# Link Computation Tests
# =============================================================================


class TestComputeLinks:
    """Tests for compute_links method."""

    def test_compute_links_empty_store(self, store, index, mock_embed_fn):
        """Test computing links with no embeddings."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=mock_embed_fn,
        )
        result = pipeline.compute_links()
        assert result == 0

    def test_compute_links_creates_similarity_links(self, store, index, similar_embed_fn, sample_entities):
        """Test that similarity links are created for similar entities."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=similar_embed_fn,
        )
        pipeline.compute_embeddings()
        pipeline.rebuild_vector_index()

        result = pipeline.compute_links(k=10, min_score=0.5)
        assert result > 0

        # Check that anxiety has links to similar entities
        links = store.get_links(sample_entities["anxiety"].id, direction="outgoing")
        assert len(links) > 0
        for link in links:
            assert link.link_type == LinkType.SIMILAR_TO
            assert link.score >= 0.5

    def test_compute_links_excludes_self(self, store, index, similar_embed_fn, sample_entities):
        """Test that entities don't link to themselves."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=similar_embed_fn,
        )
        pipeline.compute_embeddings()
        pipeline.rebuild_vector_index()
        pipeline.compute_links()

        for entity in sample_entities.values():
            links = store.get_links(entity.id, direction="outgoing")
            self_links = [l for l in links if l.target_id == entity.id]
            assert len(self_links) == 0

    def test_compute_links_respects_min_score(self, store, index, similar_embed_fn, sample_entities):
        """Test that links below min_score are not created."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=similar_embed_fn,
        )
        pipeline.compute_embeddings()
        pipeline.rebuild_vector_index()

        # High threshold should create fewer links
        result_high = pipeline.compute_links(k=10, min_score=0.99)

        # Get all links
        all_links = []
        for entity in sample_entities.values():
            all_links.extend(store.get_links(entity.id, direction="outgoing"))

        # All links should be above threshold
        for link in all_links:
            assert link.score >= 0.99

    def test_compute_links_clear_existing(self, store, index, similar_embed_fn, sample_entities):
        """Test clearing existing links before computing new ones."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=similar_embed_fn,
        )
        pipeline.compute_embeddings()
        pipeline.rebuild_vector_index()

        # Compute links first time
        result1 = pipeline.compute_links(k=5, min_score=0.5)

        # Compute links again with clear_existing=True
        result2 = pipeline.compute_links(k=5, min_score=0.5, clear_existing=True)

        # Results should be similar (not doubled)
        assert result1 == result2

    def test_compute_links_progress_callback(self, store, index, similar_embed_fn, sample_entities):
        """Test progress callback during link computation."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=similar_embed_fn,
        )
        pipeline.compute_embeddings()
        pipeline.rebuild_vector_index()

        callback = MagicMock()
        pipeline.progress_callback = callback
        pipeline.compute_links()

        # Should be called once per entity
        assert callback.call_count == 5
        for c in callback.call_args_list:
            assert c[0][0] == "compute_links"


# =============================================================================
# Full Pipeline Tests
# =============================================================================


class TestFullPipeline:
    """Tests for run_full_pipeline method."""

    def test_full_pipeline_empty_store(self, store, index, mock_embed_fn):
        """Test full pipeline with empty store."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=mock_embed_fn,
        )
        stats = pipeline.run_full_pipeline()

        assert stats.entities_processed == 0
        assert stats.embeddings_computed == 0
        assert stats.index_size == 0
        assert stats.links_created == 0

    def test_full_pipeline_with_entities(self, store, index, similar_embed_fn, sample_entities):
        """Test full pipeline with entities."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=similar_embed_fn,
        )
        stats = pipeline.run_full_pipeline(
            embedding_batch_size=2,
            link_k=10,
            link_min_score=0.5,
        )

        assert stats.entities_processed == 5
        assert stats.embeddings_computed == 5
        assert stats.index_size == 5
        assert stats.links_created > 0

    def test_full_pipeline_saves_index(self, store, index, similar_embed_fn, sample_entities, tmp_path):
        """Test full pipeline saves index to disk."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=similar_embed_fn,
        )

        index_path = str(tmp_path / "pipeline_index")
        pipeline.run_full_pipeline(index_path=index_path)

        assert (tmp_path / "pipeline_index.index").exists()
        assert (tmp_path / "pipeline_index.idmap").exists()

    def test_full_pipeline_idempotent(self, store, index, similar_embed_fn, sample_entities):
        """Test that running pipeline twice produces same results."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=similar_embed_fn,
        )

        stats1 = pipeline.run_full_pipeline()
        stats2 = pipeline.run_full_pipeline()

        # Second run should compute no new embeddings (all exist)
        assert stats2.embeddings_computed == 0
        # But should still rebuild index and links
        assert stats2.index_size == stats1.index_size


# =============================================================================
# Utility Method Tests
# =============================================================================


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_get_entities_needing_embeddings(self, store, index, mock_embed_fn, sample_entities):
        """Test getting entities without embeddings."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=mock_embed_fn,
        )

        # Initially all entities need embeddings
        needing = pipeline.get_entities_needing_embeddings()
        assert len(needing) == 5

        # Add embedding for one
        store.save_embedding(sample_entities["anxiety"].id, [0.1] * 8, "test")

        needing = pipeline.get_entities_needing_embeddings()
        assert len(needing) == 4

    def test_get_embedding_coverage(self, store, index, mock_embed_fn, sample_entities):
        """Test getting embedding coverage statistics."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=mock_embed_fn,
        )

        # Initially no coverage
        with_emb, total = pipeline.get_embedding_coverage()
        assert with_emb == 0
        assert total == 5

        # Add some embeddings
        store.save_embedding(sample_entities["anxiety"].id, [0.1] * 8, "test")
        store.save_embedding(sample_entities["stress"].id, [0.1] * 8, "test")

        with_emb, total = pipeline.get_embedding_coverage()
        assert with_emb == 2
        assert total == 5

    def test_entity_to_text_simple(self, store, index, mock_embed_fn):
        """Test entity to text conversion."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=mock_embed_fn,
        )

        entity = Entity(name="working_memory")
        text = pipeline._entity_to_text(entity)
        assert text == "working memory"

    def test_entity_to_text_with_description(self, store, index, mock_embed_fn):
        """Test entity to text with description."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=mock_embed_fn,
        )

        entity = Entity(
            name="anxiety_disorder",
            description="A mental disorder characterized by excessive worry",
        )
        text = pipeline._entity_to_text(entity)
        assert "anxiety disorder" in text
        assert "A mental disorder" in text


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for PrecomputationPipeline."""

    def test_psychology_domain_pipeline(self, store, index):
        """Test realistic psychology domain scenario."""
        # Create entities
        entities = [
            Entity(name="anxiety", entity_type="emotion", description="Feeling of worry"),
            Entity(name="depression", entity_type="disorder", description="Persistent sadness"),
            Entity(name="stress", entity_type="state", description="Response to pressure"),
            Entity(name="cbt", entity_type="treatment", description="Cognitive behavioral therapy"),
            Entity(name="mindfulness", entity_type="technique", description="Present moment awareness"),
        ]
        for e in entities:
            store.add(e)

        # Create embedding function that produces related vectors
        def domain_embed_fn(texts: list[str]) -> list[list[float]]:
            base = {
                "anxiety": [0.9, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                "depression": [0.85, 0.15, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                "stress": [0.8, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                "cbt": [0.1, 0.1, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0],
                "mindfulness": [0.1, 0.1, 0.1, 0.7, 0.3, 0.0, 0.0, 0.0],
            }
            embeddings = []
            for text in texts:
                key = text.split(":")[0].strip().replace(" ", "_").lower()
                if key in base:
                    embeddings.append(base[key])
                else:
                    embeddings.append([0.5] * 8)
            return embeddings

        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=domain_embed_fn,
            model_name="test-domain-model",
        )

        stats = pipeline.run_full_pipeline(
            embedding_batch_size=2,
            link_k=5,
            link_min_score=0.5,
        )

        # Verify stats
        assert stats.entities_processed == 5
        assert stats.embeddings_computed == 5
        assert stats.index_size == 5
        assert stats.links_created > 0

        # Verify similar entities are linked
        # Anxiety, depression, and stress should be linked
        anxiety_entity = store.get_by_name("anxiety")
        anxiety_links = store.get_links(anxiety_entity.id, direction="outgoing")
        linked_ids = {l.target_id for l in anxiety_links}

        depression_entity = store.get_by_name("depression")
        stress_entity = store.get_by_name("stress")

        # At least depression and stress should be similar to anxiety
        assert depression_entity.id in linked_ids or stress_entity.id in linked_ids

    def test_incremental_updates(self, store, index, similar_embed_fn, sample_entities):
        """Test incremental entity additions."""
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=similar_embed_fn,
        )

        # Initial pipeline run
        stats1 = pipeline.run_full_pipeline()
        assert stats1.embeddings_computed == 5

        # Add new entity
        new_entity = Entity(name="panic", entity_type="emotion")
        store.add(new_entity)

        # Run pipeline again
        stats2 = pipeline.run_full_pipeline()
        assert stats2.embeddings_computed == 1  # Only new entity
        assert stats2.index_size == 6  # All 6 entities

        # Verify new entity has embedding
        embedding = store.get_embedding(new_entity.id)
        assert embedding is not None
