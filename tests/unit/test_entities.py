"""Tests for entities module (link-based architecture).

Tests the core data model and storage:
- Entity dataclass
- EntityLink dataclass
- LinkType enum
- SurfaceForm dataclass
- EntityStore with SQLite backend
"""

import pytest
from datetime import datetime

from z3adapter.ikr.entities import (
    Entity,
    EntityEmbedding,
    EntityLink,
    EntityStore,
    LinkType,
    SurfaceForm,
)


# =============================================================================
# Entity Tests
# =============================================================================


class TestEntity:
    """Tests for Entity dataclass."""

    def test_basic_creation(self):
        """Test basic entity creation."""
        entity = Entity(name="anxiety")
        assert entity.name == "anxiety"
        assert entity.id is not None  # UUID generated
        assert entity.entity_type is None
        assert entity.description is None
        assert entity.external_ids == {}
        assert entity.source is None
        assert isinstance(entity.created_at, datetime)

    def test_creation_with_all_fields(self):
        """Test entity creation with all optional fields."""
        entity = Entity(
            name="anxiety_disorder",
            id="custom-id",
            entity_type="disorder",
            description="A mental disorder characterized by excessive worry",
            external_ids={"wikidata": "Q175629", "mesh": "D001008"},
            source="DSM-5",
        )
        assert entity.name == "anxiety_disorder"
        assert entity.id == "custom-id"
        assert entity.entity_type == "disorder"
        assert entity.description == "A mental disorder characterized by excessive worry"
        assert entity.external_ids == {"wikidata": "Q175629", "mesh": "D001008"}
        assert entity.source == "DSM-5"

    def test_name_normalization(self):
        """Test that entity names are normalized."""
        # Spaces become underscores
        e1 = Entity(name="working memory")
        assert e1.name == "working_memory"

        # Uppercase becomes lowercase
        e2 = Entity(name="Working Memory")
        assert e2.name == "working_memory"

        # Hyphens become underscores
        e3 = Entity(name="short-term-memory")
        assert e3.name == "short_term_memory"

        # Leading/trailing whitespace removed
        e4 = Entity(name="  anxiety  ")
        assert e4.name == "anxiety"

    def test_hash_and_equality(self):
        """Test that entities hash and compare by ID."""
        e1 = Entity(name="anxiety", id="id1")
        e2 = Entity(name="different", id="id1")  # Same ID
        e3 = Entity(name="anxiety", id="id2")  # Same name, different ID

        # Same ID = equal
        assert e1 == e2
        assert hash(e1) == hash(e2)

        # Different ID = not equal
        assert e1 != e3

    def test_equality_with_non_entity(self):
        """Test equality comparison with non-Entity objects."""
        entity = Entity(name="anxiety", id="id1")
        assert entity != "anxiety"
        assert entity != {"name": "anxiety"}
        assert entity != None

    def test_entity_in_set(self):
        """Test that entities can be stored in sets (hashable)."""
        e1 = Entity(name="a", id="id1")
        e2 = Entity(name="b", id="id1")  # Same ID
        e3 = Entity(name="a", id="id2")  # Different ID

        s = {e1, e2, e3}
        assert len(s) == 2  # id1 appears once


# =============================================================================
# EntityLink Tests
# =============================================================================


class TestEntityLink:
    """Tests for EntityLink dataclass."""

    def test_basic_creation(self):
        """Test basic link creation."""
        link = EntityLink(
            source_id="abc",
            target_id="def",
            link_type=LinkType.SIMILAR_TO,
            score=0.85,
        )
        assert link.source_id == "abc"
        assert link.target_id == "def"
        assert link.link_type == LinkType.SIMILAR_TO
        assert link.score == 0.85
        assert link.method == "embedding"
        assert isinstance(link.computed_at, datetime)

    def test_creation_with_all_fields(self):
        """Test link creation with all optional fields."""
        link = EntityLink(
            source_id="abc",
            target_id="def",
            link_type=LinkType.IS_A,
            score=0.95,
            method="manual",
        )
        assert link.link_type == LinkType.IS_A
        assert link.method == "manual"

    def test_score_validation(self):
        """Test that score must be in [0, 1]."""
        # Valid scores
        EntityLink(source_id="a", target_id="b", link_type=LinkType.SIMILAR_TO, score=0.0)
        EntityLink(source_id="a", target_id="b", link_type=LinkType.SIMILAR_TO, score=1.0)
        EntityLink(source_id="a", target_id="b", link_type=LinkType.SIMILAR_TO, score=0.5)

        # Invalid scores
        with pytest.raises(ValueError, match="Score must be in"):
            EntityLink(source_id="a", target_id="b", link_type=LinkType.SIMILAR_TO, score=-0.1)
        with pytest.raises(ValueError, match="Score must be in"):
            EntityLink(source_id="a", target_id="b", link_type=LinkType.SIMILAR_TO, score=1.1)

    def test_link_types(self):
        """Test all link types."""
        assert LinkType.SIMILAR_TO.value == "similar_to"
        assert LinkType.IS_A.value == "is_a"
        assert LinkType.PART_OF.value == "part_of"
        assert LinkType.SAME_AS.value == "same_as"

    def test_hash_and_equality(self):
        """Test that links hash and compare by source, target, and type."""
        l1 = EntityLink(source_id="a", target_id="b", link_type=LinkType.SIMILAR_TO, score=0.8)
        l2 = EntityLink(source_id="a", target_id="b", link_type=LinkType.SIMILAR_TO, score=0.9)
        l3 = EntityLink(source_id="a", target_id="b", link_type=LinkType.IS_A, score=0.8)

        # Same source/target/type = equal (score doesn't matter)
        assert l1 == l2
        assert hash(l1) == hash(l2)

        # Different type = not equal
        assert l1 != l3


# =============================================================================
# SurfaceForm Tests
# =============================================================================


class TestSurfaceForm:
    """Tests for SurfaceForm dataclass."""

    def test_basic_creation(self):
        """Test basic surface form creation."""
        sf = SurfaceForm(form="wm", entity_id="abc", score=0.95)
        assert sf.form == "wm"
        assert sf.entity_id == "abc"
        assert sf.score == 0.95
        assert sf.source == "exact"

    def test_form_normalization(self):
        """Test that forms are normalized."""
        sf = SurfaceForm(form="  Working Memory  ", entity_id="abc", score=0.9)
        assert sf.form == "working memory"

    def test_score_validation(self):
        """Test that score must be in [0, 1]."""
        with pytest.raises(ValueError, match="Score must be in"):
            SurfaceForm(form="test", entity_id="abc", score=1.5)


# =============================================================================
# EntityStore Tests
# =============================================================================


class TestEntityStore:
    """Tests for EntityStore class."""

    def test_create_in_memory(self):
        """Test creating in-memory store."""
        store = EntityStore()
        assert store.db_path == ":memory:"
        assert store.count() == 0

    def test_add_and_get(self):
        """Test adding and retrieving entities."""
        store = EntityStore()
        entity = Entity(name="anxiety", entity_type="emotion")

        entity_id = store.add(entity)

        assert entity_id == entity.id
        retrieved = store.get(entity.id)
        assert retrieved.name == "anxiety"
        assert retrieved.entity_type == "emotion"

    def test_get_nonexistent(self):
        """Test getting nonexistent entity returns None."""
        store = EntityStore()
        assert store.get("nonexistent") is None

    def test_get_by_name(self):
        """Test looking up entity by name."""
        store = EntityStore()
        entity = Entity(name="working memory", entity_type="concept")
        store.add(entity)

        # Exact match (normalized)
        found = store.get_by_name("working_memory")
        assert found.id == entity.id

        # With original form
        found = store.get_by_name("working memory")
        assert found.id == entity.id

        # Nonexistent
        assert store.get_by_name("nonexistent") is None

    def test_get_or_create_existing(self):
        """Test get_or_create returns existing entity."""
        store = EntityStore()
        entity = Entity(name="anxiety")
        store.add(entity)

        found, is_new = store.get_or_create("anxiety")
        assert found.id == entity.id
        assert is_new is False

    def test_get_or_create_new(self):
        """Test get_or_create creates new entity."""
        store = EntityStore()

        entity, is_new = store.get_or_create("anxiety", entity_type="emotion")
        assert entity.name == "anxiety"
        assert entity.entity_type == "emotion"
        assert is_new is True
        assert store.count() == 1

    def test_search(self):
        """Test searching entities by pattern."""
        store = EntityStore()
        store.add(Entity(name="anxiety"))
        store.add(Entity(name="anxiety_disorder"))
        store.add(Entity(name="stress"))

        # Prefix search
        results = store.search("anxiety%")
        assert len(results) == 2
        names = {e.name for e in results}
        assert names == {"anxiety", "anxiety_disorder"}

        # No match
        results = store.search("nonexistent%")
        assert len(results) == 0

    def test_all_entities(self):
        """Test getting all entities."""
        store = EntityStore()
        store.add(Entity(name="a"))
        store.add(Entity(name="b"))
        store.add(Entity(name="c"))

        all_entities = store.all_entities()
        assert len(all_entities) == 3
        names = {e.name for e in all_entities}
        assert names == {"a", "b", "c"}

    def test_delete_entity(self):
        """Test deleting an entity."""
        store = EntityStore()
        entity = Entity(name="anxiety")
        store.add(entity)

        # Add some associated data
        store.add_surface_form("worry", entity.id, 0.8)
        store.save_embedding(entity.id, [0.1, 0.2, 0.3], "test-model")

        # Create another entity and link
        other = Entity(name="stress")
        store.add(other)
        store.add_link(EntityLink(
            source_id=entity.id,
            target_id=other.id,
            link_type=LinkType.SIMILAR_TO,
            score=0.75,
        ))

        # Delete
        assert store.delete_entity(entity.id) is True
        assert store.get(entity.id) is None
        assert store.count() == 1  # Only "stress" remains

        # Delete nonexistent
        assert store.delete_entity("nonexistent") is False

    def test_clear(self):
        """Test clearing all data."""
        store = EntityStore()
        store.add(Entity(name="a"))
        store.add(Entity(name="b"))

        store.clear()
        assert store.count() == 0


class TestEntityStoreLinks:
    """Tests for EntityStore link operations."""

    @pytest.fixture
    def store_with_entities(self):
        """Create store with sample entities."""
        store = EntityStore()
        store.add(Entity(name="anxiety", id="e1"))
        store.add(Entity(name="stress", id="e2"))
        store.add(Entity(name="fear", id="e3"))
        store.add(Entity(name="worry", id="e4"))
        return store

    def test_add_and_get_links(self, store_with_entities):
        """Test adding and retrieving links."""
        store = store_with_entities

        link = EntityLink(
            source_id="e1",
            target_id="e2",
            link_type=LinkType.SIMILAR_TO,
            score=0.85,
        )
        store.add_link(link)

        # Get outgoing links
        links = store.get_links("e1", direction="outgoing")
        assert len(links) == 1
        assert links[0].target_id == "e2"
        assert links[0].score == 0.85

        # Get incoming links
        links = store.get_links("e2", direction="incoming")
        assert len(links) == 1
        assert links[0].source_id == "e1"

    def test_get_links_by_type(self, store_with_entities):
        """Test filtering links by type."""
        store = store_with_entities

        store.add_link(EntityLink(source_id="e1", target_id="e2", link_type=LinkType.SIMILAR_TO, score=0.8))
        store.add_link(EntityLink(source_id="e1", target_id="e3", link_type=LinkType.IS_A, score=0.9))

        # Filter by type
        similar = store.get_links("e1", link_type=LinkType.SIMILAR_TO)
        assert len(similar) == 1
        assert similar[0].target_id == "e2"

        is_a = store.get_links("e1", link_type=LinkType.IS_A)
        assert len(is_a) == 1
        assert is_a[0].target_id == "e3"

    def test_get_similar(self, store_with_entities):
        """Test getting similar entities."""
        store = store_with_entities

        # Add similarity links
        store.add_link(EntityLink(source_id="e1", target_id="e2", link_type=LinkType.SIMILAR_TO, score=0.85))
        store.add_link(EntityLink(source_id="e1", target_id="e3", link_type=LinkType.SIMILAR_TO, score=0.75))
        store.add_link(EntityLink(source_id="e1", target_id="e4", link_type=LinkType.SIMILAR_TO, score=0.45))

        # Get similar with default threshold
        similar = store.get_similar("e1", min_score=0.5)
        assert len(similar) == 2  # e4 excluded (0.45 < 0.5)

        # Check sorted by score
        assert similar[0][0].id == "e2"  # 0.85
        assert similar[1][0].id == "e3"  # 0.75

    def test_add_link_replaces_existing(self, store_with_entities):
        """Test that adding link with same key replaces existing."""
        store = store_with_entities

        link1 = EntityLink(source_id="e1", target_id="e2", link_type=LinkType.SIMILAR_TO, score=0.5)
        link2 = EntityLink(source_id="e1", target_id="e2", link_type=LinkType.SIMILAR_TO, score=0.9)

        store.add_link(link1)
        store.add_link(link2)

        links = store.get_links("e1")
        assert len(links) == 1
        assert links[0].score == 0.9


class TestEntityStoreSurfaceForms:
    """Tests for EntityStore surface form operations."""

    def test_add_and_lookup(self):
        """Test adding and looking up surface forms."""
        store = EntityStore()
        entity = Entity(name="working_memory", id="e1")
        store.add(entity)

        store.add_surface_form("WM", "e1", 0.95, source="manual")
        store.add_surface_form("short-term memory", "e1", 0.8, source="embedding")

        # Lookup
        result = store.lookup_surface_form("wm")  # Normalized
        assert result is not None
        entity_id, score = result
        assert entity_id == "e1"
        assert score == 0.95

        result = store.lookup_surface_form("short-term memory")
        assert result is not None
        assert result[0] == "e1"

        # Nonexistent
        assert store.lookup_surface_form("nonexistent") is None

    def test_get_surface_forms(self):
        """Test getting all surface forms for an entity."""
        store = EntityStore()
        entity = Entity(name="working_memory", id="e1")
        store.add(entity)

        store.add_surface_form("WM", "e1", 0.95)
        store.add_surface_form("STM", "e1", 0.85)

        forms = store.get_surface_forms("e1")
        assert len(forms) == 2
        form_texts = {sf.form for sf in forms}
        assert form_texts == {"wm", "stm"}


class TestEntityStoreEmbeddings:
    """Tests for EntityStore embedding operations."""

    def test_save_and_get_embedding(self):
        """Test saving and retrieving embeddings."""
        store = EntityStore()
        entity = Entity(name="anxiety", id="e1")
        store.add(entity)

        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        store.save_embedding("e1", embedding, model="test-model")

        result = store.get_embedding("e1")
        assert result is not None
        assert result.entity_id == "e1"
        assert result.model == "test-model"
        assert len(result.embedding) == 5
        assert abs(result.embedding[0] - 0.1) < 1e-6

    def test_get_embedding_nonexistent(self):
        """Test getting embedding for nonexistent entity."""
        store = EntityStore()
        assert store.get_embedding("nonexistent") is None

    def test_get_all_embeddings(self):
        """Test getting all embeddings."""
        store = EntityStore()
        store.add(Entity(name="a", id="e1"))
        store.add(Entity(name="b", id="e2"))
        store.add(Entity(name="c", id="e3"))

        store.save_embedding("e1", [0.1, 0.2], "model")
        store.save_embedding("e2", [0.3, 0.4], "model")
        # e3 has no embedding

        embeddings = store.get_all_embeddings()
        assert len(embeddings) == 2
        assert "e1" in embeddings
        assert "e2" in embeddings
        assert "e3" not in embeddings

    def test_get_entities_without_embeddings(self):
        """Test getting entities that lack embeddings."""
        store = EntityStore()
        store.add(Entity(name="a", id="e1"))
        store.add(Entity(name="b", id="e2"))
        store.add(Entity(name="c", id="e3"))

        store.save_embedding("e1", [0.1, 0.2], "model")

        missing = store.get_entities_without_embeddings()
        assert len(missing) == 2
        ids = {e.id for e in missing}
        assert ids == {"e2", "e3"}

    def test_embedding_blob_conversion(self):
        """Test that embedding blob conversion preserves precision."""
        original = [0.123456789, -0.987654321, 1e-10, 1e10]
        blob = EntityStore._embedding_to_blob(original)
        recovered = EntityStore._blob_to_embedding(blob)

        assert len(recovered) == len(original)
        for o, r in zip(original, recovered):
            assert abs(o - r) < 1e-6


# =============================================================================
# Integration Tests
# =============================================================================


class TestEntityStoreIntegration:
    """Integration tests with realistic scenarios."""

    def test_psychology_domain_example(self):
        """Test modeling psychology concepts with links."""
        store = EntityStore()

        # Add entities
        anxiety = Entity(name="anxiety", entity_type="emotion", id="e1")
        stress = Entity(name="stress", entity_type="state", id="e2")
        fear = Entity(name="fear", entity_type="emotion", id="e3")
        worry = Entity(name="worry", entity_type="cognitive_process", id="e4")
        anxiety_disorder = Entity(name="anxiety_disorder", entity_type="disorder", id="e5")

        for e in [anxiety, stress, fear, worry, anxiety_disorder]:
            store.add(e)

        # Add similarity links
        links = [
            EntityLink(source_id="e1", target_id="e3", link_type=LinkType.SIMILAR_TO, score=0.85),
            EntityLink(source_id="e1", target_id="e4", link_type=LinkType.SIMILAR_TO, score=0.80),
            EntityLink(source_id="e1", target_id="e2", link_type=LinkType.SIMILAR_TO, score=0.70),
        ]
        for link in links:
            store.add_link(link)

        # Add taxonomy link
        store.add_link(EntityLink(
            source_id="e1",
            target_id="e5",
            link_type=LinkType.IS_A,
            score=0.60,  # Anxiety can be a symptom of anxiety disorder
        ))

        # Add surface forms
        store.add_surface_form("nervousness", "e1", 0.9)
        store.add_surface_form("anxiousness", "e1", 0.95)

        # Query: What is similar to anxiety?
        similar = store.get_similar("e1", min_score=0.6)
        assert len(similar) == 3
        similar_names = {e.name for e, _ in similar}
        assert similar_names == {"fear", "worry", "stress"}

        # Query: Lookup surface form
        result = store.lookup_surface_form("nervousness")
        assert result is not None
        entity_id, _ = result
        entity = store.get(entity_id)
        assert entity.name == "anxiety"

    def test_persistence_roundtrip(self, tmp_path):
        """Test that data persists across store instances."""
        db_path = str(tmp_path / "test.db")

        # Create and populate store
        store1 = EntityStore(db_path)
        store1.add(Entity(name="anxiety", id="e1"))
        store1.add_link(EntityLink(source_id="e1", target_id="e1", link_type=LinkType.SAME_AS, score=1.0))
        store1.add_surface_form("worry", "e1", 0.8)
        store1.save_embedding("e1", [0.1, 0.2, 0.3], "test-model")
        store1.close()

        # Reopen store and verify data
        store2 = EntityStore(db_path)
        assert store2.count() == 1

        entity = store2.get("e1")
        assert entity.name == "anxiety"

        links = store2.get_links("e1")
        assert len(links) == 1

        surface = store2.lookup_surface_form("worry")
        assert surface is not None

        embedding = store2.get_embedding("e1")
        assert embedding is not None
        assert len(embedding.embedding) == 3

        store2.close()
