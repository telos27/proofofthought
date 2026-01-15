"""Tests for Triple SQLite storage.

Tests the SQLite persistence layer:
- Triple CRUD operations
- Entity CRUD with surface forms
- Query and count operations
- Import/export to in-memory stores
"""

import tempfile
from pathlib import Path

import pytest

from z3adapter.ikr.schema import TruthValue
from z3adapter.ikr.triples import (
    Predicate,
    Triple,
    TripleStore,
    EntityResolver,
)
from z3adapter.ikr.triples.storage import SQLiteTripleStorage


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def storage():
    """Create an in-memory SQLite storage for testing."""
    return SQLiteTripleStorage(":memory:")


@pytest.fixture
def file_storage(tmp_path):
    """Create a file-based SQLite storage for testing."""
    db_path = tmp_path / "test.db"
    return SQLiteTripleStorage(db_path)


@pytest.fixture
def sample_triples():
    """Create sample triples for testing."""
    return [
        Triple(id="t1", subject="cat", predicate=Predicate.IS_A, object="mammal"),
        Triple(id="t2", subject="mammal", predicate=Predicate.IS_A, object="animal"),
        Triple(id="t3", subject="stress", predicate=Predicate.CAUSES, object="anxiety"),
        Triple(
            id="t4",
            subject="exercise",
            predicate=Predicate.PREVENTS,
            object="anxiety",
            negated=False,
        ),
        Triple(
            id="t5",
            subject="relaxation",
            predicate=Predicate.CAUSES,
            object="stress",
            negated=True,
        ),
    ]


# =============================================================================
# Initialization Tests
# =============================================================================


class TestStorageInitialization:
    """Tests for storage initialization."""

    def test_in_memory_storage(self):
        """Test creating in-memory storage."""
        storage = SQLiteTripleStorage(":memory:")
        assert storage is not None
        assert len(storage) == 0

    def test_file_storage(self, tmp_path):
        """Test creating file-based storage."""
        db_path = tmp_path / "test.db"
        storage = SQLiteTripleStorage(db_path)
        assert storage is not None
        assert db_path.exists()

    def test_reopen_file_storage(self, tmp_path):
        """Test reopening file-based storage preserves data."""
        db_path = tmp_path / "test.db"

        # Create and add data
        storage1 = SQLiteTripleStorage(db_path)
        triple = Triple(id="t1", subject="a", predicate=Predicate.IS_A, object="b")
        storage1.add_triple(triple)

        # Reopen and verify data
        storage2 = SQLiteTripleStorage(db_path)
        retrieved = storage2.get_triple("t1")
        assert retrieved is not None
        assert retrieved.subject == "a"


# =============================================================================
# Triple CRUD Tests
# =============================================================================


class TestTripleCRUD:
    """Tests for triple CRUD operations."""

    def test_add_triple(self, storage):
        """Test adding a triple."""
        triple = Triple(id="t1", subject="cat", predicate=Predicate.IS_A, object="mammal")
        storage.add_triple(triple)
        assert len(storage) == 1

    def test_add_triple_with_truth(self, storage):
        """Test adding a triple with truth value."""
        triple = Triple(
            id="t1",
            subject="stress",
            predicate=Predicate.CAUSES,
            object="anxiety",
            truth=TruthValue(frequency=0.8, confidence=0.9),
        )
        storage.add_triple(triple)
        retrieved = storage.get_triple("t1")
        assert retrieved.truth is not None
        assert retrieved.truth.frequency == 0.8
        assert retrieved.truth.confidence == 0.9

    def test_add_triple_with_metadata(self, storage):
        """Test adding a triple with source and surface form."""
        triple = Triple(
            id="t1",
            subject="stress",
            predicate=Predicate.CAUSES,
            object="anxiety",
            source="Psychology 101 p.42",
            surface_form="Stress causes anxiety",
        )
        storage.add_triple(triple)
        retrieved = storage.get_triple("t1")
        assert retrieved.source == "Psychology 101 p.42"
        assert retrieved.surface_form == "Stress causes anxiety"

    def test_add_negated_triple(self, storage):
        """Test adding a negated triple."""
        triple = Triple(
            id="t1",
            subject="relaxation",
            predicate=Predicate.CAUSES,
            object="stress",
            negated=True,
        )
        storage.add_triple(triple)
        retrieved = storage.get_triple("t1")
        assert retrieved.negated is True

    def test_add_triples_batch(self, storage, sample_triples):
        """Test batch adding triples."""
        storage.add_triples(sample_triples)
        assert len(storage) == 5

    def test_get_triple(self, storage):
        """Test getting a triple by ID."""
        triple = Triple(id="t1", subject="cat", predicate=Predicate.IS_A, object="mammal")
        storage.add_triple(triple)
        retrieved = storage.get_triple("t1")
        assert retrieved is not None
        assert retrieved.id == "t1"
        assert retrieved.subject == "cat"
        assert retrieved.predicate == Predicate.IS_A
        assert retrieved.object == "mammal"

    def test_get_triple_not_found(self, storage):
        """Test getting a non-existent triple."""
        result = storage.get_triple("nonexistent")
        assert result is None

    def test_remove_triple(self, storage):
        """Test removing a triple."""
        triple = Triple(id="t1", subject="cat", predicate=Predicate.IS_A, object="mammal")
        storage.add_triple(triple)
        assert len(storage) == 1

        removed = storage.remove_triple("t1")
        assert removed is True
        assert len(storage) == 0

    def test_remove_triple_not_found(self, storage):
        """Test removing a non-existent triple."""
        removed = storage.remove_triple("nonexistent")
        assert removed is False

    def test_update_triple(self, storage):
        """Test updating a triple by re-adding with same ID."""
        triple1 = Triple(id="t1", subject="cat", predicate=Predicate.IS_A, object="mammal")
        storage.add_triple(triple1)

        triple2 = Triple(id="t1", subject="dog", predicate=Predicate.IS_A, object="mammal")
        storage.add_triple(triple2)

        retrieved = storage.get_triple("t1")
        assert retrieved.subject == "dog"
        assert len(storage) == 1

    def test_contains(self, storage):
        """Test __contains__ operator."""
        triple = Triple(id="t1", subject="cat", predicate=Predicate.IS_A, object="mammal")
        storage.add_triple(triple)

        assert "t1" in storage
        assert "nonexistent" not in storage


# =============================================================================
# Triple Query Tests
# =============================================================================


class TestTripleQuery:
    """Tests for triple query operations."""

    def test_query_all(self, storage, sample_triples):
        """Test querying all triples."""
        storage.add_triples(sample_triples)
        result = storage.query_triples()
        assert len(result) == 5

    def test_query_by_subject(self, storage, sample_triples):
        """Test querying by subject."""
        storage.add_triples(sample_triples)
        result = storage.query_triples(subject="cat")
        assert len(result) == 1
        assert result[0].id == "t1"

    def test_query_by_predicate(self, storage, sample_triples):
        """Test querying by predicate."""
        storage.add_triples(sample_triples)
        result = storage.query_triples(predicate=Predicate.CAUSES)
        assert len(result) == 2  # t3 and t5
        ids = {t.id for t in result}
        assert ids == {"t3", "t5"}

    def test_query_by_object(self, storage, sample_triples):
        """Test querying by object."""
        storage.add_triples(sample_triples)
        result = storage.query_triples(obj="anxiety")
        assert len(result) == 2  # t3 and t4
        ids = {t.id for t in result}
        assert ids == {"t3", "t4"}

    def test_query_combined(self, storage, sample_triples):
        """Test querying with multiple filters."""
        storage.add_triples(sample_triples)
        result = storage.query_triples(predicate=Predicate.CAUSES, obj="anxiety")
        assert len(result) == 1
        assert result[0].id == "t3"

    def test_query_no_match(self, storage, sample_triples):
        """Test query with no matches."""
        storage.add_triples(sample_triples)
        result = storage.query_triples(subject="nonexistent")
        assert len(result) == 0

    def test_query_with_limit(self, storage, sample_triples):
        """Test query with limit."""
        storage.add_triples(sample_triples)
        result = storage.query_triples(limit=2)
        assert len(result) == 2

    def test_query_with_offset(self, storage, sample_triples):
        """Test query with offset."""
        storage.add_triples(sample_triples)
        all_results = storage.query_triples()
        offset_results = storage.query_triples(limit=2, offset=2)
        assert len(offset_results) == 2
        # Results should be different due to offset
        all_ids = {t.id for t in all_results[:2]}
        offset_ids = {t.id for t in offset_results}
        assert all_ids.isdisjoint(offset_ids)

    def test_count_triples(self, storage, sample_triples):
        """Test counting triples."""
        storage.add_triples(sample_triples)
        assert storage.count_triples() == 5
        assert storage.count_triples(predicate=Predicate.IS_A) == 2
        assert storage.count_triples(subject="cat") == 1


# =============================================================================
# Entity CRUD Tests
# =============================================================================


class TestEntityCRUD:
    """Tests for entity CRUD operations."""

    def test_add_entity(self, storage):
        """Test adding an entity."""
        entity_id = storage.add_entity("working_memory")
        assert entity_id is not None
        assert storage.count_entities() == 1

    def test_add_entity_with_surface_forms(self, storage):
        """Test adding an entity with surface forms."""
        storage.add_entity("working_memory", ["WM", "short-term memory"])
        entity = storage.get_entity("working_memory")
        assert entity is not None
        assert set(entity["surface_forms"]) == {"WM", "short-term memory"}

    def test_add_entity_with_custom_id(self, storage):
        """Test adding an entity with custom ID."""
        entity_id = storage.add_entity("anxiety", entity_id="custom-id")
        assert entity_id == "custom-id"

    def test_get_entity(self, storage):
        """Test getting an entity."""
        storage.add_entity("anxiety", ["anxious", "anxiety disorder"])
        entity = storage.get_entity("anxiety")
        assert entity is not None
        assert entity["canonical_name"] == "anxiety"
        assert "anxious" in entity["surface_forms"]

    def test_get_entity_not_found(self, storage):
        """Test getting a non-existent entity."""
        entity = storage.get_entity("nonexistent")
        assert entity is None

    def test_get_entity_by_id(self, storage):
        """Test getting an entity by ID."""
        entity_id = storage.add_entity("depression")
        entity = storage.get_entity_by_id(entity_id)
        assert entity is not None
        assert entity["canonical_name"] == "depression"

    def test_remove_entity(self, storage):
        """Test removing an entity."""
        storage.add_entity("anxiety", ["anxious"])
        assert storage.count_entities() == 1

        removed = storage.remove_entity("anxiety")
        assert removed is True
        assert storage.count_entities() == 0

    def test_remove_entity_cascades_surface_forms(self, storage):
        """Test that removing entity cascades to surface forms."""
        storage.add_entity("anxiety", ["anxious", "worried"])
        storage.remove_entity("anxiety")
        # Surface forms should be gone too
        assert storage.find_entity_by_surface_form("anxious") is None

    def test_remove_entity_not_found(self, storage):
        """Test removing a non-existent entity."""
        removed = storage.remove_entity("nonexistent")
        assert removed is False

    def test_add_surface_form(self, storage):
        """Test adding a surface form to existing entity."""
        storage.add_entity("anxiety")
        result = storage.add_surface_form("anxiety", "worried")
        assert result is True

        entity = storage.get_entity("anxiety")
        assert "worried" in entity["surface_forms"]

    def test_add_surface_form_entity_not_found(self, storage):
        """Test adding surface form to non-existent entity."""
        result = storage.add_surface_form("nonexistent", "form")
        assert result is False

    def test_find_entity_by_surface_form(self, storage):
        """Test finding entity by surface form."""
        storage.add_entity("working_memory", ["WM", "short-term memory"])
        canonical = storage.find_entity_by_surface_form("WM")
        assert canonical == "working_memory"

    def test_find_entity_by_surface_form_not_found(self, storage):
        """Test finding entity by non-existent surface form."""
        storage.add_entity("anxiety")
        canonical = storage.find_entity_by_surface_form("nonexistent")
        assert canonical is None

    def test_list_entities(self, storage):
        """Test listing all entities."""
        storage.add_entity("anxiety")
        storage.add_entity("depression")
        storage.add_entity("stress")

        entities = storage.list_entities()
        assert set(entities) == {"anxiety", "depression", "stress"}

    def test_list_entities_with_limit(self, storage):
        """Test listing entities with limit."""
        storage.add_entity("anxiety")
        storage.add_entity("depression")
        storage.add_entity("stress")

        entities = storage.list_entities(limit=2)
        assert len(entities) == 2

    def test_count_entities(self, storage):
        """Test counting entities."""
        assert storage.count_entities() == 0
        storage.add_entity("anxiety")
        storage.add_entity("depression")
        assert storage.count_entities() == 2

    def test_duplicate_entity(self, storage):
        """Test adding duplicate entity is idempotent."""
        storage.add_entity("anxiety", ["worried"])
        storage.add_entity("anxiety", ["anxious"])

        assert storage.count_entities() == 1
        entity = storage.get_entity("anxiety")
        # Both surface forms should be present
        assert "worried" in entity["surface_forms"]
        assert "anxious" in entity["surface_forms"]


# =============================================================================
# Import/Export Tests
# =============================================================================


class TestImportExport:
    """Tests for import/export operations."""

    def test_to_triple_store(self, storage, sample_triples):
        """Test exporting to TripleStore."""
        storage.add_triples(sample_triples)
        store = storage.to_triple_store()
        assert len(store) == 5
        assert store.get("t1") is not None

    def test_from_triple_store(self, storage):
        """Test importing from TripleStore."""
        store = TripleStore()
        store.add(Triple(id="t1", subject="a", predicate=Predicate.IS_A, object="b"))
        store.add(Triple(id="t2", subject="c", predicate=Predicate.IS_A, object="d"))

        count = storage.from_triple_store(store)
        assert count == 2
        assert len(storage) == 2

    def test_from_triple_store_clear_existing(self, storage, sample_triples):
        """Test importing with clear_existing flag."""
        storage.add_triples(sample_triples)
        assert len(storage) == 5

        store = TripleStore()
        store.add(Triple(id="new", subject="x", predicate=Predicate.IS_A, object="y"))

        storage.from_triple_store(store, clear_existing=True)
        assert len(storage) == 1
        assert storage.get_triple("new") is not None
        assert storage.get_triple("t1") is None

    def test_to_entity_resolver(self, storage):
        """Test exporting to EntityResolver."""
        storage.add_entity("anxiety", ["worried", "anxious"])
        storage.add_entity("depression", ["sad"])

        resolver = storage.to_entity_resolver()
        assert len(resolver) == 2
        assert "anxiety" in resolver
        assert "worried" in resolver.get_surface_forms("anxiety")

    def test_from_entity_resolver(self, storage):
        """Test importing from EntityResolver."""
        resolver = EntityResolver()
        resolver.add_entity("anxiety", ["worried"])
        resolver.add_entity("depression", ["sad"])

        count = storage.from_entity_resolver(resolver)
        assert count == 2
        assert storage.count_entities() == 2

    def test_from_entity_resolver_clear_existing(self, storage):
        """Test importing with clear_existing flag."""
        storage.add_entity("stress")
        assert storage.count_entities() == 1

        resolver = EntityResolver()
        resolver.add_entity("anxiety", ["worried"])

        storage.from_entity_resolver(resolver, clear_existing=True)
        assert storage.count_entities() == 1
        assert storage.get_entity("anxiety") is not None
        assert storage.get_entity("stress") is None

    def test_roundtrip_triple_store(self, storage, sample_triples):
        """Test roundtrip: TripleStore -> SQLite -> TripleStore."""
        # Create original store
        original = TripleStore()
        for t in sample_triples:
            original.add(t)

        # Import to SQLite
        storage.from_triple_store(original)

        # Export back to TripleStore
        exported = storage.to_triple_store()

        # Verify
        assert len(exported) == len(original)
        for tid in original.triples:
            orig = original.get(tid)
            exp = exported.get(tid)
            assert exp is not None
            assert orig.subject == exp.subject
            assert orig.predicate == exp.predicate
            assert orig.object == exp.object

    def test_roundtrip_entity_resolver(self, storage):
        """Test roundtrip: EntityResolver -> SQLite -> EntityResolver."""
        # Create original resolver
        original = EntityResolver()
        original.add_entity("anxiety", ["worried", "anxious"])
        original.add_entity("depression", ["sad", "depressed"])

        # Import to SQLite
        storage.from_entity_resolver(original)

        # Export back to EntityResolver
        exported = storage.to_entity_resolver()

        # Verify
        assert len(exported) == len(original)
        for canonical in original.get_all_entities():
            orig_forms = original.get_surface_forms(canonical)
            exp_forms = exported.get_surface_forms(canonical)
            assert orig_forms == exp_forms


# =============================================================================
# Utility Tests
# =============================================================================


class TestUtility:
    """Tests for utility methods."""

    def test_clear_triples(self, storage, sample_triples):
        """Test clearing all triples."""
        storage.add_triples(sample_triples)
        assert len(storage) == 5

        count = storage.clear_triples()
        assert count == 5
        assert len(storage) == 0

    def test_clear_entities(self, storage):
        """Test clearing all entities."""
        storage.add_entity("anxiety")
        storage.add_entity("depression")
        assert storage.count_entities() == 2

        count = storage.clear_entities()
        assert count == 2
        assert storage.count_entities() == 0

    def test_clear_all(self, storage, sample_triples):
        """Test clearing everything."""
        storage.add_triples(sample_triples)
        storage.add_entity("anxiety")
        storage.add_entity("depression")

        triples, entities = storage.clear_all()
        assert triples == 5
        assert entities == 2
        assert len(storage) == 0
        assert storage.count_entities() == 0

    def test_len(self, storage, sample_triples):
        """Test __len__ returns triple count."""
        assert len(storage) == 0
        storage.add_triples(sample_triples)
        assert len(storage) == 5


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_storage_operations(self, storage):
        """Test operations on empty storage."""
        assert len(storage) == 0
        assert storage.get_triple("t1") is None
        assert storage.query_triples() == []
        assert storage.count_triples() == 0
        assert storage.list_entities() == []

    def test_special_characters_in_strings(self, storage):
        """Test handling special characters."""
        triple = Triple(
            id="t1",
            subject="user's_data",
            predicate=Predicate.HAS,
            object="special \"chars\"",
            surface_form="User's data has special \"chars\"",
        )
        storage.add_triple(triple)
        retrieved = storage.get_triple("t1")
        assert retrieved.subject == "user's_data"
        assert retrieved.object == "special \"chars\""
        assert retrieved.surface_form == "User's data has special \"chars\""

    def test_unicode_strings(self, storage):
        """Test handling unicode strings."""
        triple = Triple(
            id="t1",
            subject="日本語",
            predicate=Predicate.IS_A,
            object="言語",
        )
        storage.add_triple(triple)
        retrieved = storage.get_triple("t1")
        assert retrieved.subject == "日本語"
        assert retrieved.object == "言語"

    def test_very_long_strings(self, storage):
        """Test handling very long strings."""
        long_text = "a" * 10000
        triple = Triple(
            id="t1",
            subject="test",
            predicate=Predicate.HAS,
            object=long_text,
        )
        storage.add_triple(triple)
        retrieved = storage.get_triple("t1")
        assert retrieved.object == long_text

    def test_null_truth_value(self, storage):
        """Test triple with no truth value."""
        triple = Triple(
            id="t1",
            subject="a",
            predicate=Predicate.IS_A,
            object="b",
            truth=None,
        )
        storage.add_triple(triple)
        retrieved = storage.get_triple("t1")
        assert retrieved.truth is None

    def test_concurrent_path_access(self, tmp_path):
        """Test multiple storage instances on same file."""
        db_path = tmp_path / "shared.db"

        # Create first instance
        storage1 = SQLiteTripleStorage(db_path)
        storage1.add_triple(
            Triple(id="t1", subject="a", predicate=Predicate.IS_A, object="b")
        )

        # Create second instance
        storage2 = SQLiteTripleStorage(db_path)
        storage2.add_triple(
            Triple(id="t2", subject="c", predicate=Predicate.IS_A, object="d")
        )

        # Both should see all data
        assert storage1.count_triples() == 2
        assert storage2.count_triples() == 2
