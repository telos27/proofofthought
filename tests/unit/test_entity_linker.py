"""Tests for EntityLinker (multi-level entity resolution).

Tests the core functionality:
- Exact match resolution
- Surface form lookup resolution
- Vector search resolution with linking
- Surface form learning
- LinkResult dataclass
"""

import pytest
from unittest.mock import MagicMock

from z3adapter.ikr.entities import (
    Entity,
    EntityLink,
    EntityLinker,
    EntityStore,
    LinkResult,
    LinkType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def store():
    """Create an in-memory entity store."""
    return EntityStore()


@pytest.fixture
def sample_entities(store):
    """Add sample entities to the store and return them."""
    entities = {
        "anxiety": Entity(name="anxiety", entity_type="emotion"),
        "stress": Entity(name="stress", entity_type="state"),
        "depression": Entity(name="depression", entity_type="disorder"),
        "working_memory": Entity(name="working_memory", entity_type="concept"),
        "long_term_memory": Entity(name="long_term_memory", entity_type="concept"),
    }
    for entity in entities.values():
        store.add(entity)
    return entities


# =============================================================================
# LinkResult Tests
# =============================================================================


class TestLinkResult:
    """Tests for LinkResult dataclass."""

    def test_basic_creation(self):
        """Test basic LinkResult creation."""
        entity = Entity(name="test")
        result = LinkResult(entity=entity, is_new=True)
        assert result.entity == entity
        assert result.is_new is True
        assert result.links == []
        assert result.resolution_method == "new"
        assert result.score == 0.0

    def test_creation_with_all_fields(self):
        """Test LinkResult with all fields."""
        entity = Entity(name="test")
        link = EntityLink(
            source_id=entity.id,
            target_id="other",
            link_type=LinkType.SIMILAR_TO,
            score=0.8,
        )
        result = LinkResult(
            entity=entity,
            is_new=True,
            links=[link],
            resolution_method="vector_link",
            score=0.8,
        )
        assert result.entity == entity
        assert result.is_new is True
        assert len(result.links) == 1
        assert result.links[0].score == 0.8
        assert result.resolution_method == "vector_link"
        assert result.score == 0.8

    def test_existing_entity_result(self):
        """Test LinkResult for existing entity."""
        entity = Entity(name="anxiety")
        result = LinkResult(
            entity=entity,
            is_new=False,
            resolution_method="exact",
            score=1.0,
        )
        assert result.is_new is False
        assert result.links == []
        assert result.resolution_method == "exact"
        assert result.score == 1.0


# =============================================================================
# EntityLinker Initialization Tests
# =============================================================================


class TestEntityLinkerInit:
    """Tests for EntityLinker initialization."""

    def test_basic_creation(self, store):
        """Test basic linker creation without vector index."""
        linker = EntityLinker(entity_store=store)
        assert linker.entity_store == store
        assert linker.vector_index is None
        assert linker.embed_fn is None
        assert linker.link_threshold == 0.5
        assert linker.identity_threshold == 0.9
        assert linker.learn_surface_forms is True
        assert linker.max_links == 10

    def test_creation_with_custom_thresholds(self, store):
        """Test linker creation with custom thresholds."""
        linker = EntityLinker(
            entity_store=store,
            link_threshold=0.7,
            identity_threshold=0.95,
            learn_surface_forms=False,
            max_links=5,
        )
        assert linker.link_threshold == 0.7
        assert linker.identity_threshold == 0.95
        assert linker.learn_surface_forms is False
        assert linker.max_links == 5

    def test_vector_index_requires_embed_fn(self, store):
        """Test that vector_index requires embed_fn."""
        mock_index = MagicMock()
        with pytest.raises(ValueError, match="embed_fn is required"):
            EntityLinker(entity_store=store, vector_index=mock_index)

    def test_creation_with_vector_index(self, store):
        """Test linker creation with vector index."""
        mock_index = MagicMock()
        mock_embed_fn = MagicMock()
        linker = EntityLinker(
            entity_store=store,
            vector_index=mock_index,
            embed_fn=mock_embed_fn,
        )
        assert linker.vector_index == mock_index
        assert linker.embed_fn == mock_embed_fn
        assert linker.has_vector_search is True

    def test_has_vector_search_property(self, store):
        """Test has_vector_search property."""
        linker_no_vector = EntityLinker(entity_store=store)
        assert linker_no_vector.has_vector_search is False

        mock_index = MagicMock()
        mock_embed_fn = MagicMock()
        linker_with_vector = EntityLinker(
            entity_store=store,
            vector_index=mock_index,
            embed_fn=mock_embed_fn,
        )
        assert linker_with_vector.has_vector_search is True


# =============================================================================
# Exact Match Resolution Tests
# =============================================================================


class TestExactMatchResolution:
    """Tests for exact match resolution (O(1) hash lookup)."""

    def test_exact_match_by_name(self, store, sample_entities):
        """Test exact match by canonical name."""
        linker = EntityLinker(entity_store=store)
        result = linker.link("anxiety")

        assert result.is_new is False
        assert result.entity.id == sample_entities["anxiety"].id
        assert result.entity.name == "anxiety"
        assert result.resolution_method == "exact"
        assert result.score == 1.0
        assert result.links == []

    def test_exact_match_with_normalization(self, store, sample_entities):
        """Test exact match normalizes input."""
        linker = EntityLinker(entity_store=store)

        # Test various normalizations
        result1 = linker.link("Working Memory")  # Uppercase
        assert result1.is_new is False
        assert result1.entity.id == sample_entities["working_memory"].id

        result2 = linker.link("working-memory")  # Hyphens
        assert result2.is_new is False
        assert result2.entity.id == sample_entities["working_memory"].id

        result3 = linker.link("  stress  ")  # Whitespace
        assert result3.is_new is False
        assert result3.entity.id == sample_entities["stress"].id

    def test_exact_match_different_entities(self, store, sample_entities):
        """Test exact match for different entities."""
        linker = EntityLinker(entity_store=store)

        result1 = linker.link("anxiety")
        result2 = linker.link("depression")

        assert result1.entity.id != result2.entity.id
        assert result1.entity.name == "anxiety"
        assert result2.entity.name == "depression"

    def test_no_exact_match_creates_new(self, store, sample_entities):
        """Test that missing entity creates new one."""
        linker = EntityLinker(entity_store=store)
        result = linker.link("phobia")  # Not in store

        assert result.is_new is True
        assert result.entity.name == "phobia"
        assert result.resolution_method == "new"
        assert result.score == 0.0
        assert result.links == []


# =============================================================================
# Surface Form Resolution Tests
# =============================================================================


class TestSurfaceFormResolution:
    """Tests for surface form lookup resolution (O(1) learned mappings)."""

    def test_surface_form_lookup(self, store, sample_entities):
        """Test resolution via surface form."""
        linker = EntityLinker(entity_store=store)

        # Add surface form mapping
        store.add_surface_form(
            form="wm",
            entity_id=sample_entities["working_memory"].id,
            score=0.95,
            source="manual",
        )

        result = linker.link("WM")  # Will be normalized to "wm"

        assert result.is_new is False
        assert result.entity.id == sample_entities["working_memory"].id
        assert result.resolution_method == "surface_form"
        assert result.score == 0.95

    def test_surface_form_takes_priority_over_vector(self, store, sample_entities):
        """Test that surface form lookup is checked before vector search."""
        mock_index = MagicMock()
        mock_embed_fn = MagicMock()
        linker = EntityLinker(
            entity_store=store,
            vector_index=mock_index,
            embed_fn=mock_embed_fn,
        )

        # Add surface form mapping
        store.add_surface_form(
            form="ltm",
            entity_id=sample_entities["long_term_memory"].id,
            score=0.9,
            source="manual",
        )

        result = linker.link("LTM")

        assert result.is_new is False
        assert result.entity.id == sample_entities["long_term_memory"].id
        assert result.resolution_method == "surface_form"
        # Vector search should not have been called
        mock_embed_fn.assert_not_called()

    def test_surface_form_not_found_continues_to_vector(self, store, sample_entities):
        """Test that missing surface form falls through to vector search."""
        mock_index = MagicMock()
        mock_index.search.return_value = []
        mock_embed_fn = MagicMock(return_value=[0.1] * 10)
        linker = EntityLinker(
            entity_store=store,
            vector_index=mock_index,
            embed_fn=mock_embed_fn,
        )

        result = linker.link("unknown_term")

        # Should have tried vector search
        mock_embed_fn.assert_called_once_with("unknown_term")
        mock_index.search.assert_called_once()


# =============================================================================
# Vector Search Resolution Tests
# =============================================================================


class TestVectorSearchResolution:
    """Tests for vector search resolution (O(log n) ANN)."""

    def test_high_confidence_uses_existing(self, store, sample_entities):
        """Test high confidence match uses existing entity."""
        mock_index = MagicMock()
        mock_index.search.return_value = [
            (sample_entities["anxiety"].id, 0.95),  # Above identity threshold
        ]
        mock_embed_fn = MagicMock(return_value=[0.1] * 10)
        linker = EntityLinker(
            entity_store=store,
            vector_index=mock_index,
            embed_fn=mock_embed_fn,
            identity_threshold=0.9,
        )

        result = linker.link("worry")  # Not an exact match

        assert result.is_new is False
        assert result.entity.id == sample_entities["anxiety"].id
        assert result.resolution_method == "vector_high"
        assert result.score == 0.95
        assert result.links == []

    def test_high_confidence_learns_surface_form(self, store, sample_entities):
        """Test that high confidence match learns surface form."""
        mock_index = MagicMock()
        mock_index.search.return_value = [
            (sample_entities["anxiety"].id, 0.95),
        ]
        mock_embed_fn = MagicMock(return_value=[0.1] * 10)
        linker = EntityLinker(
            entity_store=store,
            vector_index=mock_index,
            embed_fn=mock_embed_fn,
            identity_threshold=0.9,
            learn_surface_forms=True,
        )

        linker.link("worry")

        # Check surface form was learned
        lookup = store.lookup_surface_form("worry")
        assert lookup is not None
        entity_id, score = lookup
        assert entity_id == sample_entities["anxiety"].id
        assert score == 0.95

    def test_high_confidence_no_surface_form_learning(self, store, sample_entities):
        """Test that surface form learning can be disabled."""
        mock_index = MagicMock()
        mock_index.search.return_value = [
            (sample_entities["anxiety"].id, 0.95),
        ]
        mock_embed_fn = MagicMock(return_value=[0.1] * 10)
        linker = EntityLinker(
            entity_store=store,
            vector_index=mock_index,
            embed_fn=mock_embed_fn,
            learn_surface_forms=False,
        )

        linker.link("worry")

        # Surface form should NOT be learned
        lookup = store.lookup_surface_form("worry")
        assert lookup is None

    def test_moderate_confidence_creates_with_links(self, store, sample_entities):
        """Test moderate confidence creates new entity with similarity links."""
        mock_index = MagicMock()
        mock_index.search.return_value = [
            (sample_entities["anxiety"].id, 0.75),  # Above link, below identity
            (sample_entities["stress"].id, 0.60),  # Above link
            (sample_entities["depression"].id, 0.40),  # Below link threshold
        ]
        mock_embed_fn = MagicMock(return_value=[0.1] * 10)
        linker = EntityLinker(
            entity_store=store,
            vector_index=mock_index,
            embed_fn=mock_embed_fn,
            link_threshold=0.5,
            identity_threshold=0.9,
        )

        result = linker.link("fear")

        assert result.is_new is True
        assert result.entity.name == "fear"
        assert result.resolution_method == "vector_link"
        assert result.score == 0.75

        # Should have 2 links (anxiety and stress, not depression)
        assert len(result.links) == 2
        link_targets = {link.target_id for link in result.links}
        assert sample_entities["anxiety"].id in link_targets
        assert sample_entities["stress"].id in link_targets
        assert sample_entities["depression"].id not in link_targets

        # Check link properties
        for link in result.links:
            assert link.source_id == result.entity.id
            assert link.link_type == LinkType.SIMILAR_TO
            assert link.method == "embedding"

    def test_low_confidence_creates_no_links(self, store, sample_entities):
        """Test low confidence creates new entity without links."""
        mock_index = MagicMock()
        mock_index.search.return_value = [
            (sample_entities["anxiety"].id, 0.3),  # Below link threshold
        ]
        mock_embed_fn = MagicMock(return_value=[0.1] * 10)
        linker = EntityLinker(
            entity_store=store,
            vector_index=mock_index,
            embed_fn=mock_embed_fn,
            link_threshold=0.5,
        )

        result = linker.link("completely_unrelated")

        assert result.is_new is True
        assert result.entity.name == "completely_unrelated"
        assert result.resolution_method == "new"
        assert result.score == 0.0
        assert result.links == []

    def test_no_candidates_creates_new(self, store, sample_entities):
        """Test no candidates found creates new entity."""
        mock_index = MagicMock()
        mock_index.search.return_value = []
        mock_embed_fn = MagicMock(return_value=[0.1] * 10)
        linker = EntityLinker(
            entity_store=store,
            vector_index=mock_index,
            embed_fn=mock_embed_fn,
        )

        result = linker.link("novel_concept")

        assert result.is_new is True
        assert result.entity.name == "novel_concept"
        assert result.resolution_method == "new"

    def test_max_links_limit(self, store, sample_entities):
        """Test that max_links limits the number of similarity links."""
        mock_index = MagicMock()
        mock_index.search.return_value = [
            (sample_entities["anxiety"].id, 0.80),
            (sample_entities["stress"].id, 0.75),
            (sample_entities["depression"].id, 0.70),
            (sample_entities["working_memory"].id, 0.65),
            (sample_entities["long_term_memory"].id, 0.60),
        ]
        mock_embed_fn = MagicMock(return_value=[0.1] * 10)
        linker = EntityLinker(
            entity_store=store,
            vector_index=mock_index,
            embed_fn=mock_embed_fn,
            link_threshold=0.5,
            identity_threshold=0.9,
            max_links=2,  # Limit to 2 links
        )

        result = linker.link("emotional_state")

        assert result.is_new is True
        assert len(result.links) == 2  # Limited by max_links


# =============================================================================
# Entity Type and Source Tests
# =============================================================================


class TestEntityMetadata:
    """Tests for entity type and source propagation."""

    def test_new_entity_with_type(self, store):
        """Test new entity gets specified type."""
        linker = EntityLinker(entity_store=store)
        result = linker.link("phobia", entity_type="disorder")

        assert result.is_new is True
        assert result.entity.entity_type == "disorder"

    def test_new_entity_with_source(self, store):
        """Test new entity gets specified source."""
        linker = EntityLinker(entity_store=store)
        result = linker.link("cognitive_load", source="Psychology 101")

        assert result.is_new is True
        assert result.entity.source == "Psychology 101"

    def test_new_entity_with_type_and_source(self, store):
        """Test new entity gets both type and source."""
        linker = EntityLinker(entity_store=store)
        result = linker.link(
            "executive_function",
            entity_type="concept",
            source="Neuroscience Text",
        )

        assert result.is_new is True
        assert result.entity.entity_type == "concept"
        assert result.entity.source == "Neuroscience Text"

    def test_existing_entity_ignores_type(self, store, sample_entities):
        """Test that existing entity keeps its original type."""
        linker = EntityLinker(entity_store=store)
        result = linker.link("anxiety", entity_type="different_type")

        assert result.is_new is False
        assert result.entity.entity_type == "emotion"  # Original type


# =============================================================================
# link_and_store Tests
# =============================================================================


class TestLinkAndStore:
    """Tests for link_and_store convenience method."""

    def test_link_and_store_existing(self, store, sample_entities):
        """Test link_and_store with existing entity."""
        linker = EntityLinker(entity_store=store)
        initial_count = store.count()

        result = linker.link_and_store("anxiety")

        assert result.is_new is False
        assert store.count() == initial_count  # No new entity added

    def test_link_and_store_new(self, store, sample_entities):
        """Test link_and_store with new entity."""
        linker = EntityLinker(entity_store=store)
        initial_count = store.count()

        result = linker.link_and_store("phobia", entity_type="disorder")

        assert result.is_new is True
        assert store.count() == initial_count + 1

        # Verify entity was stored
        stored = store.get_by_name("phobia")
        assert stored is not None
        assert stored.id == result.entity.id

    def test_link_and_store_with_links(self, store, sample_entities):
        """Test link_and_store persists similarity links."""
        mock_index = MagicMock()
        mock_index.search.return_value = [
            (sample_entities["anxiety"].id, 0.75),
        ]
        mock_embed_fn = MagicMock(return_value=[0.1] * 10)
        mock_index.add = MagicMock()
        linker = EntityLinker(
            entity_store=store,
            vector_index=mock_index,
            embed_fn=mock_embed_fn,
            link_threshold=0.5,
            identity_threshold=0.9,
        )

        result = linker.link_and_store("nervousness")

        assert result.is_new is True
        assert len(result.links) == 1

        # Verify link was stored
        links = store.get_links(result.entity.id, direction="outgoing")
        assert len(links) == 1
        assert links[0].target_id == sample_entities["anxiety"].id

    def test_link_and_store_adds_to_index(self, store, sample_entities):
        """Test link_and_store adds new entity to vector index."""
        mock_index = MagicMock()
        mock_index.search.return_value = []  # No matches
        mock_embed_fn = MagicMock(return_value=[0.1] * 10)
        linker = EntityLinker(
            entity_store=store,
            vector_index=mock_index,
            embed_fn=mock_embed_fn,
        )

        result = linker.link_and_store("new_concept")

        assert result.is_new is True
        # Verify entity was added to index
        mock_index.add.assert_called_once_with(result.entity.id, [0.1] * 10)

    def test_link_and_store_saves_embedding(self, store, sample_entities):
        """Test link_and_store saves embedding to entity store."""
        mock_index = MagicMock()
        mock_index.search.return_value = []
        embedding = [0.1] * 1536
        mock_embed_fn = MagicMock(return_value=embedding)
        linker = EntityLinker(
            entity_store=store,
            vector_index=mock_index,
            embed_fn=mock_embed_fn,
        )

        result = linker.link_and_store("new_concept")

        assert result.is_new is True
        # Verify embedding was saved
        stored_embedding = store.get_embedding(result.entity.id)
        assert stored_embedding is not None
        # Use approximate comparison due to float32 storage precision
        assert len(stored_embedding.embedding) == len(embedding)
        assert all(
            abs(a - b) < 1e-6
            for a, b in zip(stored_embedding.embedding, embedding)
        )


# =============================================================================
# link_batch Tests
# =============================================================================


class TestLinkBatch:
    """Tests for batch linking."""

    def test_batch_link_empty(self, store):
        """Test batch linking with empty list."""
        linker = EntityLinker(entity_store=store)
        results = linker.link_batch([])
        assert results == []

    def test_batch_link_multiple(self, store, sample_entities):
        """Test batch linking with multiple mentions."""
        linker = EntityLinker(entity_store=store)
        results = linker.link_batch(["anxiety", "stress", "phobia"])

        assert len(results) == 3
        assert results[0].is_new is False  # anxiety exists
        assert results[0].entity.name == "anxiety"
        assert results[1].is_new is False  # stress exists
        assert results[1].entity.name == "stress"
        assert results[2].is_new is True  # phobia is new
        assert results[2].entity.name == "phobia"

    def test_batch_link_with_type(self, store, sample_entities):
        """Test batch linking with entity type."""
        linker = EntityLinker(entity_store=store)
        results = linker.link_batch(
            ["phobia", "trauma"],
            entity_type="disorder",
        )

        assert all(r.is_new for r in results)
        assert all(r.entity.entity_type == "disorder" for r in results)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for EntityLinker."""

    def test_resolution_cascade(self, store):
        """Test complete resolution cascade: exact → surface → vector → new."""
        # Setup: Add entity with surface form
        anxiety = Entity(name="anxiety", entity_type="emotion")
        store.add(anxiety)
        store.add_surface_form("worry", anxiety.id, 0.9, "manual")

        # Setup: Mock vector index
        mock_index = MagicMock()
        mock_index.search.return_value = [(anxiety.id, 0.75)]
        mock_embed_fn = MagicMock(return_value=[0.1] * 10)

        linker = EntityLinker(
            entity_store=store,
            vector_index=mock_index,
            embed_fn=mock_embed_fn,
            link_threshold=0.5,
            identity_threshold=0.9,
        )

        # Test exact match
        r1 = linker.link("anxiety")
        assert r1.resolution_method == "exact"
        mock_embed_fn.assert_not_called()

        # Test surface form
        r2 = linker.link("worry")
        assert r2.resolution_method == "surface_form"
        mock_embed_fn.assert_not_called()

        # Test vector (moderate confidence)
        r3 = linker.link("nervousness")  # Not exact, not surface form
        assert r3.resolution_method == "vector_link"
        mock_embed_fn.assert_called()

        # Test new entity
        mock_index.search.return_value = []
        r4 = linker.link("completely_unknown")
        assert r4.resolution_method == "new"

    def test_psychology_domain_scenario(self, store):
        """Test realistic psychology domain scenario."""
        # Setup initial entities
        entities = {
            "anxiety": Entity(name="anxiety", entity_type="emotion"),
            "depression": Entity(name="depression", entity_type="disorder"),
            "cognitive_behavioral_therapy": Entity(name="cognitive_behavioral_therapy", entity_type="treatment"),
        }
        for e in entities.values():
            store.add(e)

        # Setup vector index mock
        mock_index = MagicMock()
        mock_embed_fn = MagicMock(return_value=[0.1] * 10)

        linker = EntityLinker(
            entity_store=store,
            vector_index=mock_index,
            embed_fn=mock_embed_fn,
            link_threshold=0.5,
            identity_threshold=0.9,
        )

        # Test 1: Exact match
        result = linker.link("anxiety")
        assert result.is_new is False
        assert result.entity.id == entities["anxiety"].id

        # Test 2: Abbreviation via surface form
        store.add_surface_form("cbt", entities["cognitive_behavioral_therapy"].id, 0.95, "manual")
        result = linker.link("CBT")
        assert result.is_new is False
        assert result.entity.id == entities["cognitive_behavioral_therapy"].id

        # Test 3: Similar concept via vector
        mock_index.search.return_value = [(entities["anxiety"].id, 0.85)]
        result = linker.link("generalized anxiety disorder")
        # Below identity but above link
        assert result.is_new is True
        assert len(result.links) == 1
        assert result.links[0].target_id == entities["anxiety"].id

        # Test 4: High confidence identity
        mock_index.search.return_value = [(entities["depression"].id, 0.95)]
        result = linker.link("clinical depression")
        assert result.is_new is False
        assert result.entity.id == entities["depression"].id
