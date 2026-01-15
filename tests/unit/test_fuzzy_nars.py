"""Tests for Fuzzy-NARS unification and verification.

Tests the core algorithms:
- TruthValue data structure (including to_evidence)
- Similarity functions
- Fuzzy-NARS unification
- NARS revision
- Verification pipeline
"""

import pytest

from z3adapter.ikr.schema import TruthValue
from z3adapter.ikr.fuzzy_nars import (
    VerificationTriple,
    UnificationResult,
    VerificationVerdict,
    VerificationResult,
    PREDICATE_OPPOSITES,
    get_predicate_polarity,
    lexical_similarity,
    jaccard_word_similarity,
    combined_lexical_similarity,
    make_embedding_similarity,
    fuzzy_nars_unify,
    revise,
    revise_multiple,
    verify_triple,
    verify_answer,
)


# =============================================================================
# TruthValue Tests
# =============================================================================


class TestTruthValue:
    """Tests for TruthValue dataclass."""

    def test_creation_basic(self):
        """Test basic creation with valid values."""
        truth = TruthValue(frequency=0.8, confidence=0.7)
        assert truth.frequency == 0.8
        assert truth.confidence == 0.7

    def test_from_evidence(self):
        """Test creation from evidence counts."""
        # 8 positive out of 10 total, k=1
        truth = TruthValue.from_evidence(8, 10, k=1.0)
        assert truth.frequency == 0.8
        assert abs(truth.confidence - 10 / 11) < 0.01  # ~0.909

    def test_from_evidence_empty(self):
        """Test with no evidence."""
        truth = TruthValue.from_evidence(0, 0)
        assert truth.frequency == 0.5  # Uncertain
        # confidence should be clamped to minimum

    def test_to_evidence_roundtrip(self):
        """Test that to_evidence inverts from_evidence."""
        original_pos, original_total = 8.0, 10.0
        truth = TruthValue.from_evidence(int(original_pos), int(original_total), k=1.0)
        pos, total = truth.to_evidence(k=1.0)

        assert abs(pos - original_pos) < 0.01
        assert abs(total - original_total) < 0.01

    def test_expectation(self):
        """Test expectation calculation."""
        # High confidence: expectation ≈ frequency
        truth1 = TruthValue(frequency=0.9, confidence=0.9)
        assert abs(truth1.expectation() - (0.9 * 0.9 + 0.1 * 0.5)) < 0.01

        # Low confidence: expectation tends toward 0.5
        truth2 = TruthValue(frequency=0.9, confidence=0.1)
        # expectation = 0.9 * 0.1 + 0.5 * 0.9 = 0.09 + 0.45 = 0.54
        assert abs(truth2.expectation() - 0.54) < 0.01

    def test_negate(self):
        """Test negation preserves confidence."""
        truth = TruthValue(frequency=0.8, confidence=0.7)
        negated = truth.negate()
        assert abs(negated.frequency - 0.2) < 0.001  # Float comparison
        assert negated.confidence == 0.7

    def test_is_positive(self):
        """Test is_positive threshold at 0.5."""
        assert TruthValue(frequency=0.6, confidence=0.5).is_positive()
        assert TruthValue(frequency=0.5, confidence=0.5).is_positive()
        assert not TruthValue(frequency=0.4, confidence=0.5).is_positive()


# =============================================================================
# VerificationTriple Tests
# =============================================================================


class TestVerificationTriple:
    """Tests for VerificationTriple dataclass."""

    def test_creation(self):
        """Test basic creation."""
        triple = VerificationTriple("phobia", "is_a", "anxiety_disorder")
        assert triple.subject == "phobia"
        assert triple.predicate == "is_a"
        assert triple.obj == "anxiety_disorder"
        assert triple.truth is None

    def test_creation_with_truth(self):
        """Test creation with truth value."""
        truth = TruthValue(frequency=0.9, confidence=0.8)
        triple = VerificationTriple("phobia", "is_a", "anxiety_disorder", truth)
        assert triple.truth == truth

    def test_repr(self):
        """Test string representation."""
        triple = VerificationTriple("A", "B", "C")
        assert "(A, B, C)" in repr(triple)


# =============================================================================
# Similarity Function Tests
# =============================================================================


class TestSimilarityFunctions:
    """Tests for similarity functions."""

    def test_lexical_identical(self):
        """Test lexical similarity with identical terms."""
        assert lexical_similarity("phobia", "phobia") == 1.0

    def test_lexical_case_insensitive(self):
        """Test case insensitivity."""
        assert lexical_similarity("Phobia", "phobia") == 1.0

    def test_lexical_underscore_space(self):
        """Test underscore/space equivalence."""
        assert lexical_similarity("fear_conditioning", "fear conditioning") == 1.0

    def test_lexical_similar(self):
        """Test lexical similarity with similar terms."""
        sim = lexical_similarity("phobia", "phobias")
        assert 0.7 < sim < 1.0  # Should be high but not perfect

    def test_lexical_different(self):
        """Test lexical similarity with different terms."""
        sim = lexical_similarity("phobia", "memory")
        assert sim < 0.5  # Should be low

    def test_jaccard_identical(self):
        """Test Jaccard with identical terms."""
        assert (
            jaccard_word_similarity("classical conditioning", "classical conditioning")
            == 1.0
        )

    def test_jaccard_partial_overlap(self):
        """Test Jaccard with partial word overlap."""
        sim = jaccard_word_similarity("classical conditioning", "operant conditioning")
        # 1 word overlap (conditioning), 3 total words
        assert abs(sim - 1 / 3) < 0.01

    def test_jaccard_no_overlap(self):
        """Test Jaccard with no word overlap."""
        sim = jaccard_word_similarity("fear", "memory")
        assert sim == 0.0

    def test_combined_uses_best(self):
        """Test that combined similarity uses the best score."""
        # "phobia" vs "phobias" - high lexical similarity
        sim = combined_lexical_similarity("phobia", "phobias")
        assert sim > 0.7

        # "fear response" vs "fear reaction" - better Jaccard
        sim2 = combined_lexical_similarity("fear response", "fear reaction")
        assert sim2 >= 0.5  # At least Jaccard overlap


class TestPredicatePolarity:
    """Tests for predicate polarity detection."""

    def test_opposite_predicates(self):
        """Test known opposite predicates."""
        sim, pol = get_predicate_polarity("causes", "prevents", 0.3)
        assert sim == 0.9  # High similarity
        assert pol == -1.0  # Opposite polarity

    def test_same_predicate(self):
        """Test same predicate."""
        sim, pol = get_predicate_polarity("causes", "causes", 1.0)
        assert sim == 1.0
        assert pol == 1.0

    def test_unrelated_predicates(self):
        """Test unrelated predicates."""
        sim, pol = get_predicate_polarity("causes", "has_part", 0.3)
        assert sim == 0.3  # Uses base similarity
        assert pol == 1.0  # Default positive polarity

    def test_increases_decreases(self):
        """Test increases/decreases opposition."""
        sim, pol = get_predicate_polarity("increases", "decreases", 0.5)
        assert sim == 0.9
        assert pol == -1.0


# =============================================================================
# Fuzzy-NARS Unification Tests
# =============================================================================


class TestFuzzyNARSUnify:
    """Tests for fuzzy_nars_unify function."""

    def test_perfect_match(self):
        """Test unification with identical triples."""
        query = VerificationTriple("phobia", "is_a", "anxiety_disorder")
        kb = VerificationTriple(
            "phobia", "is_a", "anxiety_disorder", TruthValue(frequency=0.9, confidence=0.8)
        )

        result = fuzzy_nars_unify(query, kb, lexical_similarity, threshold=0.5)

        assert result is not None
        assert result.success
        assert result.match_quality == 1.0
        assert result.effective_truth.frequency == 0.9
        assert result.effective_truth.confidence == 0.8

    def test_fuzzy_match(self):
        """Test unification with similar but not identical triples."""
        query = VerificationTriple("phobias", "caused_by", "trauma")
        kb = VerificationTriple(
            "phobia", "causes", "fear", TruthValue(frequency=0.9, confidence=0.8)
        )

        result = fuzzy_nars_unify(query, kb, lexical_similarity, threshold=0.3)

        # Should match with reduced quality
        if result is not None:
            assert result.match_quality < 1.0
            assert result.effective_truth.confidence < 0.8  # Reduced by match quality

    def test_no_match_below_threshold(self):
        """Test that low-quality matches are rejected."""
        query = VerificationTriple("phobia", "is_a", "disorder")
        kb = VerificationTriple(
            "memory", "has_part", "encoding", TruthValue(frequency=0.9, confidence=0.8)
        )

        result = fuzzy_nars_unify(query, kb, lexical_similarity, threshold=0.5)
        assert result is None

    def test_opposite_predicate_inverts_frequency(self):
        """Test that opposite predicates invert the frequency."""
        query = VerificationTriple("stress", "causes", "anxiety")
        kb = VerificationTriple(
            "stress", "prevents", "anxiety", TruthValue(frequency=0.9, confidence=0.8)
        )

        result = fuzzy_nars_unify(query, kb, lexical_similarity, threshold=0.5)

        assert result is not None
        assert result.polarity == -1.0
        # Frequency should be inverted: 1.0 - 0.9 = 0.1
        assert abs(result.effective_truth.frequency - 0.1) < 0.01

    def test_default_truth_value(self):
        """Test that missing truth value gets default."""
        query = VerificationTriple("phobia", "is_a", "disorder")
        kb = VerificationTriple("phobia", "is_a", "disorder")  # No truth value

        result = fuzzy_nars_unify(query, kb, lexical_similarity, threshold=0.5)

        assert result is not None
        assert result.effective_truth.frequency == 1.0  # Default
        assert result.effective_truth.confidence == 0.9  # Default


# =============================================================================
# NARS Revision Tests
# =============================================================================


class TestNARSRevision:
    """Tests for NARS revision (evidence combination)."""

    def test_revise_same_frequency(self):
        """Test revision with same frequency increases confidence."""
        t1 = TruthValue(frequency=0.8, confidence=0.5)
        t2 = TruthValue(frequency=0.8, confidence=0.5)

        result = revise(t1, t2)

        # Same frequency should remain
        assert abs(result.frequency - 0.8) < 0.01
        # Confidence should increase
        assert result.confidence > 0.5

    def test_revise_different_frequency(self):
        """Test revision with different frequencies averages them."""
        t1 = TruthValue(frequency=0.9, confidence=0.5)
        t2 = TruthValue(frequency=0.5, confidence=0.5)

        result = revise(t1, t2)

        # Frequency should be weighted average (0.7)
        assert abs(result.frequency - 0.7) < 0.1
        # Confidence should increase
        assert result.confidence > 0.5

    def test_revise_high_confidence_dominates(self):
        """Test that high-confidence evidence dominates."""
        t1 = TruthValue(frequency=0.9, confidence=0.9)  # High confidence
        t2 = TruthValue(frequency=0.3, confidence=0.1)  # Low confidence

        result = revise(t1, t2)

        # Result should be closer to t1's frequency
        assert result.frequency > 0.7

    def test_revise_multiple_combines_all(self):
        """Test revise_multiple combines multiple sources."""
        truths = [
            TruthValue(frequency=0.9, confidence=0.5),
            TruthValue(frequency=0.8, confidence=0.5),
            TruthValue(frequency=0.85, confidence=0.5),
        ]

        result = revise_multiple(truths)

        # Should have higher confidence than any individual
        assert result.confidence > 0.5
        # Frequency should be around 0.85
        assert 0.8 < result.frequency < 0.9

    def test_revise_multiple_empty(self):
        """Test revise_multiple with empty list."""
        result = revise_multiple([])
        assert result.frequency == 0.5


# =============================================================================
# Verification Pipeline Tests
# =============================================================================


class TestVerifyTriple:
    """Tests for verify_triple function."""

    @pytest.fixture
    def sample_kb(self):
        """Create a sample knowledge base for testing."""
        return [
            VerificationTriple(
                "phobia", "is_a", "anxiety_disorder", TruthValue(frequency=1.0, confidence=0.9)
            ),
            VerificationTriple(
                "phobia",
                "caused_by",
                "fear_conditioning",
                TruthValue(frequency=0.9, confidence=0.8),
            ),
            VerificationTriple(
                "classical_conditioning",
                "discovered_by",
                "Pavlov",
                TruthValue(frequency=1.0, confidence=0.95),
            ),
            VerificationTriple(
                "memory", "has_part", "encoding", TruthValue(frequency=1.0, confidence=0.85)
            ),
            VerificationTriple(
                "stress",
                "causes",
                "cortisol_release",
                TruthValue(frequency=0.95, confidence=0.9),
            ),
        ]

    def test_supported_claim(self, sample_kb):
        """Test verification of supported claim."""
        query = VerificationTriple("phobia", "is_a", "anxiety_disorder")
        result = verify_triple(query, sample_kb, lexical_similarity)

        assert result.verdict == VerificationVerdict.SUPPORTED
        assert result.combined_truth.frequency > 0.5
        assert len(result.matches) > 0

    def test_insufficient_no_match(self, sample_kb):
        """Test verification when no matches found."""
        query = VerificationTriple("gravity", "causes", "falling")
        result = verify_triple(query, sample_kb, lexical_similarity)

        assert result.verdict == VerificationVerdict.INSUFFICIENT
        assert result.combined_truth is None
        assert len(result.matches) == 0

    def test_contradicted_by_opposite_predicate(self):
        """Test contradiction detection via opposite predicate."""
        kb = [
            VerificationTriple(
                "stress", "prevents", "relaxation", TruthValue(frequency=0.95, confidence=0.9)
            ),
        ]

        query = VerificationTriple("stress", "causes", "relaxation")
        result = verify_triple(query, kb, lexical_similarity, match_threshold=0.5)

        # Should be contradicted because "causes" is opposite of "prevents"
        assert result.verdict == VerificationVerdict.CONTRADICTED
        assert result.combined_truth.frequency < 0.5


class TestVerifyAnswer:
    """Tests for verify_answer function (multiple triples)."""

    @pytest.fixture
    def sample_kb(self):
        """Create a sample knowledge base."""
        return [
            VerificationTriple(
                "phobia", "is_a", "anxiety_disorder", TruthValue(frequency=1.0, confidence=0.9)
            ),
            VerificationTriple(
                "classical_conditioning",
                "discovered_by",
                "Pavlov",
                TruthValue(frequency=1.0, confidence=0.95),
            ),
            VerificationTriple(
                "stress",
                "causes",
                "cortisol_release",
                TruthValue(frequency=0.95, confidence=0.9),
            ),
        ]

    def test_all_supported(self, sample_kb):
        """Test when all answer triples are supported."""
        answer = [
            VerificationTriple("phobia", "is_a", "anxiety_disorder"),
            VerificationTriple("classical_conditioning", "discovered_by", "Pavlov"),
        ]

        result = verify_answer(answer, sample_kb, lexical_similarity)

        assert result["summary"]["overall_verdict"] == VerificationVerdict.SUPPORTED
        assert result["summary"]["supported"] == 2
        assert result["summary"]["contradicted"] == 0

    def test_partial_support(self, sample_kb):
        """Test when some triples are supported, some insufficient."""
        answer = [
            VerificationTriple("phobia", "is_a", "anxiety_disorder"),  # Supported
            VerificationTriple("gravity", "causes", "falling"),  # No match
        ]

        result = verify_answer(answer, sample_kb, lexical_similarity)

        assert result["summary"]["supported"] >= 1
        assert result["summary"]["insufficient"] >= 1

    def test_any_contradiction_fails(self, sample_kb):
        """Test that any contradiction results in overall contradiction."""
        # Add a triple that will contradict
        kb = sample_kb + [
            VerificationTriple(
                "relaxation", "prevents", "stress", TruthValue(frequency=0.95, confidence=0.9)
            ),
        ]

        answer = [
            VerificationTriple("phobia", "is_a", "anxiety_disorder"),  # Supported
            VerificationTriple("relaxation", "causes", "stress"),  # Contradicted
        ]

        result = verify_answer(answer, kb, lexical_similarity)

        assert result["summary"]["overall_verdict"] == VerificationVerdict.CONTRADICTED


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with realistic examples."""

    @pytest.fixture
    def commonsense_kb(self):
        """Create a commonsense knowledge base."""
        return [
            # Food knowledge
            VerificationTriple(
                "vegetarian", "does_not_eat", "meat", TruthValue(frequency=1.0, confidence=0.95)
            ),
            VerificationTriple(
                "plant_burger", "contains", "vegetables", TruthValue(frequency=1.0, confidence=0.9)
            ),
            VerificationTriple(
                "plant_burger", "is_a", "vegetarian_food", TruthValue(frequency=1.0, confidence=0.9)
            ),
            # Animal knowledge
            VerificationTriple(
                "bird", "can", "fly", TruthValue(frequency=0.9, confidence=0.8)
            ),  # Most birds
            VerificationTriple(
                "penguin", "is_a", "bird", TruthValue(frequency=1.0, confidence=0.95)
            ),
            VerificationTriple(
                "penguin", "cannot", "fly", TruthValue(frequency=1.0, confidence=0.95)
            ),
        ]

    def test_verify_correct_fact(self, commonsense_kb):
        """Test verifying a correct commonsense fact."""
        query = VerificationTriple("vegetarian", "does_not_eat", "meat")
        result = verify_triple(query, commonsense_kb, combined_lexical_similarity)

        assert result.verdict == VerificationVerdict.SUPPORTED

    def test_verify_fuzzy_match(self, commonsense_kb):
        """Test verifying with fuzzy term matching."""
        # "vegetarians" should match "vegetarian"
        query = VerificationTriple("vegetarians", "does_not_eat", "meat")
        result = verify_triple(
            query, commonsense_kb, combined_lexical_similarity, match_threshold=0.6
        )

        assert result.verdict == VerificationVerdict.SUPPORTED

    def test_verify_no_evidence(self, commonsense_kb):
        """Test verifying a claim with no evidence."""
        query = VerificationTriple("robot", "needs", "electricity")
        result = verify_triple(query, commonsense_kb, combined_lexical_similarity)

        assert result.verdict == VerificationVerdict.INSUFFICIENT

    def test_embedding_similarity_factory(self):
        """Test that embedding similarity factory works."""
        # Simple mock embedding: just hash the string
        def mock_embedding(term: str) -> list[float]:
            # Create a simple 3D embedding based on hash
            h = hash(term)
            return [
                (h % 100) / 100.0,
                ((h >> 8) % 100) / 100.0,
                ((h >> 16) % 100) / 100.0,
            ]

        sim_fn = make_embedding_similarity(mock_embedding)

        # Same term should have similarity 1.0
        assert sim_fn("phobia", "phobia") == 1.0

        # Different terms should have some similarity (not testing exact value)
        sim = sim_fn("phobia", "fear")
        assert 0.0 <= sim <= 1.0
