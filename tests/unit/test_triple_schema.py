"""Tests for Triple extraction schema.

Tests the core data model:
- Predicate enum
- Triple dataclass
- TripleStore with indexing
"""

import pytest

from z3adapter.ikr.schema import TruthValue
from z3adapter.ikr.triples import (
    Predicate,
    PREDICATE_OPPOSITES,
    Triple,
    TripleStore,
)


# =============================================================================
# Predicate Tests
# =============================================================================


class TestPredicate:
    """Tests for Predicate enum."""

    def test_all_predicates_defined(self):
        """Test that all 7 predicates are defined."""
        expected = {"is_a", "part_of", "has", "causes", "prevents", "believes", "related_to"}
        actual = {p.value for p in Predicate}
        assert actual == expected

    def test_predicate_is_string_enum(self):
        """Test that predicates can be used as strings."""
        assert Predicate.CAUSES == "causes"
        assert Predicate.IS_A == "is_a"

    def test_predicate_from_string(self):
        """Test creating predicate from string."""
        assert Predicate("causes") == Predicate.CAUSES
        assert Predicate("is_a") == Predicate.IS_A

    def test_invalid_predicate_raises(self):
        """Test that invalid predicate raises ValueError."""
        with pytest.raises(ValueError):
            Predicate("invalid_predicate")


class TestPredicateOpposites:
    """Tests for predicate opposites mapping."""

    def test_causes_prevents_are_opposites(self):
        """Test that causes and prevents are opposites."""
        assert PREDICATE_OPPOSITES[Predicate.CAUSES] == Predicate.PREVENTS
        assert PREDICATE_OPPOSITES[Predicate.PREVENTS] == Predicate.CAUSES

    def test_only_causes_prevents_have_opposites(self):
        """Test that only causes/prevents have defined opposites."""
        assert len(PREDICATE_OPPOSITES) == 2


# =============================================================================
# Triple Tests
# =============================================================================


class TestTriple:
    """Tests for Triple dataclass."""

    def test_basic_creation(self):
        """Test basic triple creation."""
        triple = Triple(
            id="t1",
            subject="stress",
            predicate=Predicate.CAUSES,
            object="anxiety",
        )
        assert triple.id == "t1"
        assert triple.subject == "stress"
        assert triple.predicate == Predicate.CAUSES
        assert triple.object == "anxiety"
        assert triple.negated is False
        assert triple.truth is None
        assert triple.source is None
        assert triple.surface_form is None

    def test_creation_with_all_fields(self):
        """Test triple creation with all optional fields."""
        truth = TruthValue(frequency=0.9, confidence=0.8)
        triple = Triple(
            id="t1",
            subject="stress",
            predicate=Predicate.CAUSES,
            object="anxiety",
            negated=True,
            truth=truth,
            source="Smith 2020 p.42",
            surface_form="Stress causes anxiety",
        )
        assert triple.negated is True
        assert triple.truth == truth
        assert triple.source == "Smith 2020 p.42"
        assert triple.surface_form == "Stress causes anxiety"

    def test_is_triple_reference(self):
        """Test triple reference detection."""
        assert Triple.is_triple_reference("t:t1") is True
        assert Triple.is_triple_reference("t:abc123") is True
        assert Triple.is_triple_reference("stress") is False
        assert Triple.is_triple_reference("t1") is False  # Missing prefix
        assert Triple.is_triple_reference("") is False

    def test_get_triple_id(self):
        """Test extracting triple ID from reference."""
        assert Triple.get_triple_id("t:t1") == "t1"
        assert Triple.get_triple_id("t:abc123") == "abc123"
        assert Triple.get_triple_id("stress") is None
        assert Triple.get_triple_id("t1") is None

    def test_subject_is_triple(self):
        """Test subject_is_triple property."""
        # Subject is entity
        triple1 = Triple(
            id="t1", subject="alice", predicate=Predicate.BELIEVES, object="t:t2"
        )
        assert triple1.subject_is_triple is False

        # Subject is triple reference
        triple2 = Triple(
            id="t2", subject="t:t1", predicate=Predicate.CAUSES, object="fear"
        )
        assert triple2.subject_is_triple is True

    def test_object_is_triple(self):
        """Test object_is_triple property."""
        # Object is entity
        triple1 = Triple(
            id="t1", subject="stress", predicate=Predicate.CAUSES, object="anxiety"
        )
        assert triple1.object_is_triple is False

        # Object is triple reference (belief about another triple)
        triple2 = Triple(
            id="t2", subject="alice", predicate=Predicate.BELIEVES, object="t:t1"
        )
        assert triple2.object_is_triple is True

    def test_hash_and_equality(self):
        """Test that triples hash and compare by ID."""
        triple1 = Triple(id="t1", subject="a", predicate=Predicate.IS_A, object="b")
        triple2 = Triple(id="t1", subject="x", predicate=Predicate.HAS, object="y")
        triple3 = Triple(id="t2", subject="a", predicate=Predicate.IS_A, object="b")

        # Same ID = equal
        assert triple1 == triple2
        assert hash(triple1) == hash(triple2)

        # Different ID = not equal
        assert triple1 != triple3

    def test_equality_with_non_triple(self):
        """Test equality comparison with non-Triple objects."""
        triple = Triple(id="t1", subject="a", predicate=Predicate.IS_A, object="b")
        assert triple != "t1"
        assert triple != {"id": "t1"}
        assert triple != None

    def test_triple_in_set(self):
        """Test that triples can be stored in sets (hashable)."""
        triple1 = Triple(id="t1", subject="a", predicate=Predicate.IS_A, object="b")
        triple2 = Triple(id="t1", subject="x", predicate=Predicate.HAS, object="y")
        triple3 = Triple(id="t2", subject="a", predicate=Predicate.IS_A, object="b")

        s = {triple1, triple2, triple3}
        assert len(s) == 2  # t1 appears once (deduped)


# =============================================================================
# TripleStore Tests
# =============================================================================


class TestTripleStore:
    """Tests for TripleStore class."""

    def test_empty_store(self):
        """Test empty store initialization."""
        store = TripleStore()
        assert len(store) == 0
        assert "t1" not in store

    def test_add_and_get(self):
        """Test adding and retrieving triples."""
        store = TripleStore()
        triple = Triple(id="t1", subject="cat", predicate=Predicate.IS_A, object="mammal")

        store.add(triple)

        assert len(store) == 1
        assert "t1" in store
        assert store.get("t1") == triple
        assert store.get("nonexistent") is None

    def test_add_replaces_existing(self):
        """Test that adding with same ID replaces existing triple."""
        store = TripleStore()
        triple1 = Triple(id="t1", subject="cat", predicate=Predicate.IS_A, object="mammal")
        triple2 = Triple(id="t1", subject="dog", predicate=Predicate.IS_A, object="animal")

        store.add(triple1)
        store.add(triple2)

        assert len(store) == 1
        retrieved = store.get("t1")
        assert retrieved.subject == "dog"

    def test_remove(self):
        """Test removing triples."""
        store = TripleStore()
        triple = Triple(id="t1", subject="cat", predicate=Predicate.IS_A, object="mammal")

        store.add(triple)
        removed = store.remove("t1")

        assert removed == triple
        assert len(store) == 0
        assert store.get("t1") is None

    def test_remove_nonexistent(self):
        """Test removing nonexistent triple returns None."""
        store = TripleStore()
        assert store.remove("nonexistent") is None

    def test_iteration(self):
        """Test iterating over store."""
        store = TripleStore()
        t1 = Triple(id="t1", subject="cat", predicate=Predicate.IS_A, object="mammal")
        t2 = Triple(id="t2", subject="dog", predicate=Predicate.IS_A, object="mammal")

        store.add(t1)
        store.add(t2)

        triples = list(store)
        assert len(triples) == 2
        assert t1 in triples
        assert t2 in triples


class TestTripleStoreIndexes:
    """Tests for TripleStore index-based queries."""

    @pytest.fixture
    def populated_store(self):
        """Create a store with sample triples."""
        store = TripleStore()
        store.add(Triple(id="t1", subject="cat", predicate=Predicate.IS_A, object="mammal"))
        store.add(Triple(id="t2", subject="dog", predicate=Predicate.IS_A, object="mammal"))
        store.add(Triple(id="t3", subject="mammal", predicate=Predicate.IS_A, object="animal"))
        store.add(Triple(id="t4", subject="cat", predicate=Predicate.HAS, object="whiskers"))
        store.add(Triple(id="t5", subject="stress", predicate=Predicate.CAUSES, object="anxiety"))
        return store

    def test_query_by_subject(self, populated_store):
        """Test querying by subject."""
        results = populated_store.query(subject="cat")
        assert len(results) == 2
        ids = {t.id for t in results}
        assert ids == {"t1", "t4"}

    def test_query_by_predicate(self, populated_store):
        """Test querying by predicate."""
        results = populated_store.query(predicate=Predicate.IS_A)
        assert len(results) == 3
        ids = {t.id for t in results}
        assert ids == {"t1", "t2", "t3"}

    def test_query_by_object(self, populated_store):
        """Test querying by object."""
        results = populated_store.query(obj="mammal")
        assert len(results) == 2
        ids = {t.id for t in results}
        assert ids == {"t1", "t2"}

    def test_query_combined_filters(self, populated_store):
        """Test querying with multiple filters."""
        results = populated_store.query(subject="cat", predicate=Predicate.IS_A)
        assert len(results) == 1
        assert results[0].id == "t1"

    def test_query_no_matches(self, populated_store):
        """Test query returning empty results."""
        results = populated_store.query(subject="nonexistent")
        assert results == []

    def test_query_all(self, populated_store):
        """Test query with no filters returns all."""
        results = populated_store.query()
        assert len(results) == 5

    def test_indexes_updated_on_replace(self):
        """Test that indexes are updated when triple is replaced."""
        store = TripleStore()
        t1 = Triple(id="t1", subject="cat", predicate=Predicate.IS_A, object="mammal")
        t2 = Triple(id="t1", subject="dog", predicate=Predicate.HAS, object="tail")

        store.add(t1)
        store.add(t2)  # Replace t1

        # Old indexes should not contain t1
        assert len(store.query(subject="cat")) == 0
        assert len(store.query(predicate=Predicate.IS_A)) == 0
        assert len(store.query(obj="mammal")) == 0

        # New indexes should contain t1
        assert len(store.query(subject="dog")) == 1
        assert len(store.query(predicate=Predicate.HAS)) == 1
        assert len(store.query(obj="tail")) == 1

    def test_indexes_updated_on_remove(self, populated_store):
        """Test that indexes are updated when triple is removed."""
        populated_store.remove("t1")

        # cat should only have one triple now (t4)
        results = populated_store.query(subject="cat")
        assert len(results) == 1
        assert results[0].id == "t4"

        # IS_A should have two triples now
        results = populated_store.query(predicate=Predicate.IS_A)
        assert len(results) == 2


class TestTripleStoreResolve:
    """Tests for TripleStore.resolve method."""

    def test_resolve_entity(self):
        """Test resolving entity string returns string."""
        store = TripleStore()
        result = store.resolve("stress")
        assert result == "stress"

    def test_resolve_triple_reference(self):
        """Test resolving triple reference returns Triple."""
        store = TripleStore()
        triple = Triple(id="t1", subject="stress", predicate=Predicate.CAUSES, object="anxiety")
        store.add(triple)

        result = store.resolve("t:t1")
        assert result == triple

    def test_resolve_invalid_reference(self):
        """Test resolving invalid reference returns original string."""
        store = TripleStore()
        result = store.resolve("t:nonexistent")
        assert result == "t:nonexistent"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with realistic examples."""

    def test_nested_beliefs(self):
        """Test modeling nested beliefs (reification)."""
        store = TripleStore()

        # Fact: Stress causes anxiety
        t1 = Triple(id="t1", subject="stress", predicate=Predicate.CAUSES, object="anxiety")
        store.add(t1)

        # Alice believes that stress causes anxiety
        t2 = Triple(id="t2", subject="alice", predicate=Predicate.BELIEVES, object="t:t1")
        store.add(t2)

        # Bob believes that Alice believes stress causes anxiety
        t3 = Triple(id="t3", subject="bob", predicate=Predicate.BELIEVES, object="t:t2")
        store.add(t3)

        # Resolve the belief chain
        bob_belief = store.get("t3")
        assert bob_belief.object_is_triple

        alice_belief = store.resolve(bob_belief.object)
        assert isinstance(alice_belief, Triple)
        assert alice_belief.subject == "alice"

        stress_fact = store.resolve(alice_belief.object)
        assert isinstance(stress_fact, Triple)
        assert stress_fact.subject == "stress"
        assert stress_fact.predicate == Predicate.CAUSES

    def test_negated_facts(self):
        """Test handling negated facts."""
        store = TripleStore()

        # Exercise does NOT cause stress
        t1 = Triple(
            id="t1",
            subject="exercise",
            predicate=Predicate.CAUSES,
            object="stress",
            negated=True,
        )
        store.add(t1)

        # Query causal relations
        results = store.query(predicate=Predicate.CAUSES)
        assert len(results) == 1
        assert results[0].negated is True

    def test_truth_values(self):
        """Test triples with NARS truth values."""
        store = TripleStore()

        # Birds typically fly (but not always)
        t1 = Triple(
            id="t1",
            subject="bird",
            predicate=Predicate.HAS,
            object="flight_ability",
            truth=TruthValue(frequency=0.9, confidence=0.8),
        )
        store.add(t1)

        retrieved = store.get("t1")
        assert retrieved.truth is not None
        assert retrieved.truth.frequency == 0.9
        assert retrieved.truth.confidence == 0.8

    def test_taxonomy_chain(self):
        """Test building a taxonomy with IS_A relations."""
        store = TripleStore()

        # Build taxonomy: penguin -> bird -> animal
        store.add(Triple(id="t1", subject="penguin", predicate=Predicate.IS_A, object="bird"))
        store.add(Triple(id="t2", subject="bird", predicate=Predicate.IS_A, object="animal"))
        store.add(Triple(id="t3", subject="cat", predicate=Predicate.IS_A, object="mammal"))
        store.add(Triple(id="t4", subject="mammal", predicate=Predicate.IS_A, object="animal"))

        # Find all things that are animals
        animals = store.query(obj="animal")
        subjects = {t.subject for t in animals}
        assert subjects == {"bird", "mammal"}

        # Find all IS_A relations
        taxonomies = store.query(predicate=Predicate.IS_A)
        assert len(taxonomies) == 4

    def test_provenance_tracking(self):
        """Test tracking source provenance."""
        store = TripleStore()

        t1 = Triple(
            id="t1",
            subject="phobia",
            predicate=Predicate.IS_A,
            object="anxiety_disorder",
            source="DSM-5 p.197",
            surface_form="A phobia is classified as an anxiety disorder",
        )
        store.add(t1)

        retrieved = store.get("t1")
        assert retrieved.source == "DSM-5 p.197"
        assert "classified" in retrieved.surface_form
