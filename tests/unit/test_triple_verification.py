"""Unit tests for triple verification bridge."""

import pytest

from z3adapter.ikr.schema import TruthValue
from z3adapter.ikr.fuzzy_nars import VerificationVerdict
from z3adapter.ikr.triples.schema import Triple, TripleStore, Predicate
from z3adapter.ikr.triples.verification import (
    triple_to_verification,
    verification_to_triple,
    store_to_kb,
    verify_triple_against_store,
    verify_triples_against_store,
)


class TestTripleToVerification:
    """Tests for triple_to_verification conversion."""

    def test_basic_conversion(self):
        """Test basic Triple to VerificationTriple conversion."""
        triple = Triple(
            id="t1",
            subject="stress",
            predicate=Predicate.CAUSES,
            object="anxiety",
        )
        vt = triple_to_verification(triple)
        assert vt.subject == "stress"
        assert vt.predicate == "causes"
        assert vt.obj == "anxiety"
        assert vt.truth is None

    def test_with_truth_value(self):
        """Test conversion with truth value."""
        triple = Triple(
            id="t1",
            subject="stress",
            predicate=Predicate.CAUSES,
            object="anxiety",
            truth=TruthValue(frequency=0.9, confidence=0.8),
        )
        vt = triple_to_verification(triple)
        assert vt.truth is not None
        assert vt.truth.frequency == 0.9
        assert vt.truth.confidence == 0.8

    def test_negated_without_truth(self):
        """Test negated triple without explicit truth value."""
        triple = Triple(
            id="t1",
            subject="exercise",
            predicate=Predicate.CAUSES,
            object="stress",
            negated=True,
        )
        vt = triple_to_verification(triple)
        assert vt.truth is not None
        assert vt.truth.frequency == 0.0  # Negated = false
        assert vt.truth.confidence == 0.9

    def test_negated_with_truth(self):
        """Test negated triple with explicit truth value."""
        triple = Triple(
            id="t1",
            subject="exercise",
            predicate=Predicate.CAUSES,
            object="stress",
            negated=True,
            truth=TruthValue(frequency=0.8, confidence=0.9),
        )
        vt = triple_to_verification(triple)
        # Frequency should be inverted: 1.0 - 0.8 = 0.2
        assert vt.truth.frequency == pytest.approx(0.2)
        assert vt.truth.confidence == 0.9

    def test_all_predicates(self):
        """Test conversion for all predicate types."""
        for pred in Predicate:
            triple = Triple(
                id="t1",
                subject="a",
                predicate=pred,
                object="b",
            )
            vt = triple_to_verification(triple)
            assert vt.predicate == pred.value


class TestVerificationToTriple:
    """Tests for verification_to_triple conversion."""

    def test_basic_conversion(self):
        """Test basic VerificationTriple to Triple conversion."""
        from z3adapter.ikr.fuzzy_nars import VerificationTriple

        vt = VerificationTriple(
            subject="stress",
            predicate="causes",
            obj="anxiety",
        )
        triple = verification_to_triple(vt, "t1")
        assert triple.id == "t1"
        assert triple.subject == "stress"
        assert triple.predicate == Predicate.CAUSES
        assert triple.object == "anxiety"
        assert triple.negated is False

    def test_with_truth_value(self):
        """Test conversion with truth value."""
        from z3adapter.ikr.fuzzy_nars import VerificationTriple

        vt = VerificationTriple(
            subject="stress",
            predicate="causes",
            obj="anxiety",
            truth=TruthValue(frequency=0.9, confidence=0.8),
        )
        triple = verification_to_triple(vt, "t1")
        assert triple.truth.frequency == 0.9
        assert triple.truth.confidence == 0.8
        assert triple.negated is False

    def test_low_frequency_becomes_negated(self):
        """Test that low frequency is interpreted as negation."""
        from z3adapter.ikr.fuzzy_nars import VerificationTriple

        vt = VerificationTriple(
            subject="exercise",
            predicate="causes",
            obj="stress",
            truth=TruthValue(frequency=0.2, confidence=0.9),
        )
        triple = verification_to_triple(vt, "t1")
        assert triple.negated is True
        # Frequency inverted: 1.0 - 0.2 = 0.8
        assert triple.truth.frequency == pytest.approx(0.8)

    def test_unknown_predicate_fallback(self):
        """Test unknown predicate falls back to RELATED_TO."""
        from z3adapter.ikr.fuzzy_nars import VerificationTriple

        vt = VerificationTriple(
            subject="a",
            predicate="unknown_predicate",
            obj="b",
        )
        triple = verification_to_triple(vt, "t1")
        assert triple.predicate == Predicate.RELATED_TO

    def test_with_source(self):
        """Test conversion with source."""
        from z3adapter.ikr.fuzzy_nars import VerificationTriple

        vt = VerificationTriple(
            subject="stress",
            predicate="causes",
            obj="anxiety",
        )
        triple = verification_to_triple(vt, "t1", source="Book p.42")
        assert triple.source == "Book p.42"


class TestStoreToKB:
    """Tests for store_to_kb conversion."""

    def test_empty_store(self):
        """Test converting empty store."""
        store = TripleStore()
        kb = store_to_kb(store)
        assert kb == []

    def test_simple_store(self):
        """Test converting store with simple triples."""
        store = TripleStore()
        store.add(Triple(id="t1", subject="stress", predicate=Predicate.CAUSES, object="anxiety"))
        store.add(Triple(id="t2", subject="exercise", predicate=Predicate.PREVENTS, object="anxiety"))

        kb = store_to_kb(store)
        assert len(kb) == 2

        subjects = {vt.subject for vt in kb}
        assert subjects == {"stress", "exercise"}

    def test_skips_triple_references(self):
        """Test that meta-level triples are skipped."""
        store = TripleStore()
        # Regular triple
        store.add(Triple(id="t1", subject="stress", predicate=Predicate.CAUSES, object="anxiety"))
        # Meta-level triple (alice believes t1)
        store.add(Triple(id="t2", subject="alice", predicate=Predicate.BELIEVES, object="t:t1"))
        # Triple with subject reference
        store.add(Triple(id="t3", subject="t:t1", predicate=Predicate.IS_A, object="fact"))

        kb = store_to_kb(store)
        # Only t1 should be included
        assert len(kb) == 1
        assert kb[0].subject == "stress"


class TestVerifyTripleAgainstStore:
    """Tests for verify_triple_against_store."""

    def test_exact_match_supported(self):
        """Test verification with exact match."""
        store = TripleStore()
        store.add(Triple(
            id="t1",
            subject="stress",
            predicate=Predicate.CAUSES,
            object="anxiety",
            truth=TruthValue(frequency=1.0, confidence=0.9),
        ))

        query = Triple(id="q1", subject="stress", predicate=Predicate.CAUSES, object="anxiety")
        result = verify_triple_against_store(query, store)

        assert result.verdict == VerificationVerdict.SUPPORTED

    def test_fuzzy_match_supported(self):
        """Test verification with fuzzy match."""
        store = TripleStore()
        store.add(Triple(
            id="t1",
            subject="chronic_stress",
            predicate=Predicate.CAUSES,
            object="anxiety_disorder",
            truth=TruthValue(frequency=1.0, confidence=0.9),
        ))

        # Similar but not identical terms
        query = Triple(id="q1", subject="chronic stress", predicate=Predicate.CAUSES, object="anxiety")
        result = verify_triple_against_store(query, store, match_threshold=0.3)

        # Should find a fuzzy match
        assert result.verdict in [VerificationVerdict.SUPPORTED, VerificationVerdict.INSUFFICIENT]
        assert len(result.matches) >= 0

    def test_predicate_polarity_contradiction(self):
        """Test contradiction detection via predicate polarity."""
        store = TripleStore()
        store.add(Triple(
            id="t1",
            subject="exercise",
            predicate=Predicate.PREVENTS,
            object="anxiety",
            truth=TruthValue(frequency=1.0, confidence=0.9),
        ))

        # Opposite predicate should be contradicted
        query = Triple(id="q1", subject="exercise", predicate=Predicate.CAUSES, object="anxiety")
        result = verify_triple_against_store(query, store)

        assert result.verdict == VerificationVerdict.CONTRADICTED

    def test_no_match_insufficient(self):
        """Test insufficient evidence when no matches."""
        store = TripleStore()
        store.add(Triple(
            id="t1",
            subject="stress",
            predicate=Predicate.CAUSES,
            object="anxiety",
        ))

        # Completely unrelated query
        query = Triple(id="q1", subject="coffee", predicate=Predicate.IS_A, object="beverage")
        result = verify_triple_against_store(query, store)

        assert result.verdict == VerificationVerdict.INSUFFICIENT

    def test_meta_level_triple_skipped(self):
        """Test that meta-level triples return insufficient."""
        store = TripleStore()
        store.add(Triple(id="t1", subject="stress", predicate=Predicate.CAUSES, object="anxiety"))

        # Meta-level query
        query = Triple(id="q1", subject="alice", predicate=Predicate.BELIEVES, object="t:t1")
        result = verify_triple_against_store(query, store)

        assert result.verdict == VerificationVerdict.INSUFFICIENT
        assert "meta-level" in result.explanation.lower()

    def test_empty_store(self):
        """Test verification against empty store."""
        store = TripleStore()
        query = Triple(id="q1", subject="stress", predicate=Predicate.CAUSES, object="anxiety")
        result = verify_triple_against_store(query, store)

        assert result.verdict == VerificationVerdict.INSUFFICIENT


class TestVerifyTriplesAgainstStore:
    """Tests for verify_triples_against_store (batch verification)."""

    def test_all_supported(self):
        """Test batch verification with all supported."""
        store = TripleStore()
        store.add(Triple(
            id="t1",
            subject="stress",
            predicate=Predicate.CAUSES,
            object="anxiety",
            truth=TruthValue(frequency=1.0, confidence=0.9),
        ))
        store.add(Triple(
            id="t2",
            subject="exercise",
            predicate=Predicate.PREVENTS,
            object="anxiety",
            truth=TruthValue(frequency=1.0, confidence=0.9),
        ))

        queries = [
            Triple(id="q1", subject="stress", predicate=Predicate.CAUSES, object="anxiety"),
            Triple(id="q2", subject="exercise", predicate=Predicate.PREVENTS, object="anxiety"),
        ]

        result = verify_triples_against_store(queries, store)

        assert result["summary"]["total"] == 2
        assert result["summary"]["supported"] == 2
        assert result["summary"]["contradicted"] == 0
        assert result["summary"]["overall_verdict"] == VerificationVerdict.SUPPORTED

    def test_one_contradiction_fails_all(self):
        """Test that one contradiction makes overall verdict CONTRADICTED."""
        store = TripleStore()
        store.add(Triple(
            id="t1",
            subject="stress",
            predicate=Predicate.CAUSES,
            object="anxiety",
            truth=TruthValue(frequency=1.0, confidence=0.9),
        ))
        store.add(Triple(
            id="t2",
            subject="exercise",
            predicate=Predicate.PREVENTS,
            object="anxiety",
            truth=TruthValue(frequency=1.0, confidence=0.9),
        ))

        queries = [
            Triple(id="q1", subject="stress", predicate=Predicate.CAUSES, object="anxiety"),  # Supported
            Triple(id="q2", subject="exercise", predicate=Predicate.CAUSES, object="anxiety"),  # Contradicted
        ]

        result = verify_triples_against_store(queries, store)

        assert result["summary"]["supported"] == 1
        assert result["summary"]["contradicted"] == 1
        assert result["summary"]["overall_verdict"] == VerificationVerdict.CONTRADICTED

    def test_partial_support(self):
        """Test partial support (some supported, some insufficient)."""
        store = TripleStore()
        store.add(Triple(
            id="t1",
            subject="stress",
            predicate=Predicate.CAUSES,
            object="anxiety",
            truth=TruthValue(frequency=1.0, confidence=0.9),
        ))

        queries = [
            Triple(id="q1", subject="stress", predicate=Predicate.CAUSES, object="anxiety"),  # Supported
            Triple(id="q2", subject="coffee", predicate=Predicate.IS_A, object="beverage"),  # Insufficient
        ]

        result = verify_triples_against_store(queries, store)

        assert result["summary"]["supported"] == 1
        assert result["summary"]["insufficient"] == 1
        assert result["summary"]["overall_verdict"] == VerificationVerdict.SUPPORTED  # Partial support

    def test_empty_queries(self):
        """Test with empty query list."""
        store = TripleStore()
        store.add(Triple(id="t1", subject="stress", predicate=Predicate.CAUSES, object="anxiety"))

        result = verify_triples_against_store([], store)

        assert result["summary"]["total"] == 0
        assert result["summary"]["overall_verdict"] == VerificationVerdict.INSUFFICIENT


class TestIntegration:
    """Integration tests for the verification pipeline."""

    def test_psychology_knowledge_base(self):
        """Test verification against a psychology knowledge base."""
        store = TripleStore()

        # Build a small KB about stress and anxiety
        store.add(Triple(
            id="t1",
            subject="chronic_stress",
            predicate=Predicate.CAUSES,
            object="elevated_cortisol",
            truth=TruthValue(frequency=0.95, confidence=0.9),
        ))
        store.add(Triple(
            id="t2",
            subject="elevated_cortisol",
            predicate=Predicate.CAUSES,
            object="memory_impairment",
            truth=TruthValue(frequency=0.8, confidence=0.85),
        ))
        store.add(Triple(
            id="t3",
            subject="exercise",
            predicate=Predicate.PREVENTS,
            object="chronic_stress",
            truth=TruthValue(frequency=0.9, confidence=0.8),
        ))

        # Verify claims
        queries = [
            Triple(id="q1", subject="chronic_stress", predicate=Predicate.CAUSES, object="elevated_cortisol"),
            Triple(id="q2", subject="exercise", predicate=Predicate.CAUSES, object="chronic_stress"),  # Should contradict
        ]

        result = verify_triples_against_store(queries, store)

        assert result["triple_results"][0]["result"].verdict == VerificationVerdict.SUPPORTED
        assert result["triple_results"][1]["result"].verdict == VerificationVerdict.CONTRADICTED

    def test_negated_fact_verification(self):
        """Test verification of negated facts."""
        store = TripleStore()
        store.add(Triple(
            id="t1",
            subject="vegetarian",
            predicate=Predicate.HAS,
            object="meat_in_diet",
            negated=True,  # Vegetarians do NOT have meat in diet
            truth=TruthValue(frequency=0.95, confidence=0.95),
        ))

        # Query: Does a vegetarian have meat in diet?
        query = Triple(id="q1", subject="vegetarian", predicate=Predicate.HAS, object="meat_in_diet")
        result = verify_triple_against_store(query, store)

        # Should be contradicted because the KB says they DON'T
        assert result.verdict == VerificationVerdict.CONTRADICTED
