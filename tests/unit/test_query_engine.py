"""Unit tests for QueryEngine, PatternMatcher, and EvidenceCombiner."""

import pytest

from z3adapter.ikr.entities import Entity, EntityLink, EntityStore, LinkType
from z3adapter.ikr.entities.query_expander import ExpandedPattern, QueryTriple
from z3adapter.ikr.entities.query_engine import (
    EvidenceCombiner,
    PatternMatch,
    PatternMatcher,
    QueryEngine,
    QueryResult,
)
from z3adapter.ikr.fuzzy_nars import VerificationVerdict
from z3adapter.ikr.schema import TruthValue
from z3adapter.ikr.triples.schema import Predicate, Triple, TripleStore


class TestPatternMatch:
    """Tests for PatternMatch dataclass."""

    def test_create_pattern_match(self):
        """Test creating a pattern match."""
        pattern = ExpandedPattern(
            subject_id="e1", subject_name="stress",
            predicate=Predicate.CAUSES,
            object_id="e2", object_name="anxiety",
            confidence=1.0,
        )
        triple = Triple(
            id="t1", subject="stress",
            predicate=Predicate.CAUSES,
            object="anxiety",
        )
        match = PatternMatch(
            pattern=pattern,
            triple=triple,
            match_score=0.9,
            subject_score=1.0,
            predicate_score=1.0,
            object_score=0.9,
            polarity=1.0,
            effective_truth=TruthValue(frequency=1.0, confidence=0.81),
        )
        assert match.match_score == 0.9
        assert match.polarity == 1.0


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_create_query_result(self):
        """Test creating a query result."""
        result = QueryResult(
            verdict=VerificationVerdict.SUPPORTED,
            combined_truth=TruthValue(frequency=0.9, confidence=0.8),
            matches=[],
            patterns_searched=5,
            explanation="Evidence supports claim",
        )
        assert result.verdict == VerificationVerdict.SUPPORTED
        assert result.patterns_searched == 5


class TestPatternMatcherInit:
    """Tests for PatternMatcher initialization."""

    def test_init_defaults(self):
        """Test default initialization."""
        store = TripleStore()
        matcher = PatternMatcher(store)

        assert matcher.triple_store is store
        assert matcher.entity_store is None
        assert matcher.match_threshold == 0.5

    def test_init_with_entity_store(self):
        """Test initialization with entity store."""
        triple_store = TripleStore()
        entity_store = EntityStore()
        matcher = PatternMatcher(triple_store, entity_store)

        assert matcher.entity_store is entity_store


class TestPatternMatcherMatch:
    """Tests for PatternMatcher.match() method."""

    @pytest.fixture
    def triple_store_with_triples(self):
        """Create triple store with test triples."""
        store = TripleStore()

        # Add triples
        store.add(Triple(
            id="t1", subject="stress",
            predicate=Predicate.CAUSES,
            object="anxiety",
            subject_id="e1", object_id="e2",
        ))
        store.add(Triple(
            id="t2", subject="exercise",
            predicate=Predicate.PREVENTS,
            object="stress",
            subject_id="e3", object_id="e1",
        ))
        store.add(Triple(
            id="t3", subject="chronic_stress",
            predicate=Predicate.CAUSES,
            object="memory_impairment",
            subject_id="e4", object_id="e5",
        ))

        return store

    def test_exact_match(self, triple_store_with_triples):
        """Test exact pattern match."""
        matcher = PatternMatcher(triple_store_with_triples)

        pattern = ExpandedPattern(
            subject_id="e1", subject_name="stress",
            predicate=Predicate.CAUSES,
            object_id="e2", object_name="anxiety",
            confidence=1.0,
        )

        matches = matcher.match(pattern)

        assert len(matches) == 1
        assert matches[0].triple.id == "t1"
        assert matches[0].subject_score == 1.0
        assert matches[0].predicate_score == 1.0
        assert matches[0].object_score == 1.0

    def test_fuzzy_match(self, triple_store_with_triples):
        """Test fuzzy string match when IDs don't match."""
        matcher = PatternMatcher(triple_store_with_triples, match_threshold=0.3)

        pattern = ExpandedPattern(
            subject_id="unknown", subject_name="stress",
            predicate=Predicate.CAUSES,
            object_id="unknown", object_name="anxiousness",  # Similar to "anxiety"
            confidence=1.0,
        )

        matches = matcher.match(pattern)

        # Should find match via fuzzy string matching
        assert len(matches) >= 1
        t1_match = next((m for m in matches if m.triple.id == "t1"), None)
        assert t1_match is not None
        assert t1_match.subject_score == 1.0  # "stress" == "stress"
        assert t1_match.object_score < 1.0  # "anxiousness" ~ "anxiety"

    def test_predicate_opposite_detection(self, triple_store_with_triples):
        """Test detection of opposite predicates."""
        matcher = PatternMatcher(triple_store_with_triples)

        # Query: does exercise cause stress?
        pattern = ExpandedPattern(
            subject_id="e3", subject_name="exercise",
            predicate=Predicate.CAUSES,  # Opposite of PREVENTS
            object_id="e1", object_name="stress",
            confidence=1.0,
        )

        matches = matcher.match(pattern)

        # Should find t2 (exercise PREVENTS stress) with negative polarity
        assert len(matches) == 1
        assert matches[0].triple.id == "t2"
        assert matches[0].polarity == -1.0

    def test_no_match(self, triple_store_with_triples):
        """Test when no matches found."""
        matcher = PatternMatcher(triple_store_with_triples)

        pattern = ExpandedPattern(
            subject_id="unknown", subject_name="completely_unrelated",
            predicate=Predicate.IS_A,
            object_id="unknown", object_name="something_else",
            confidence=1.0,
        )

        matches = matcher.match(pattern)
        assert len(matches) == 0

    def test_match_threshold_filter(self, triple_store_with_triples):
        """Test that match threshold filters low-quality matches."""
        # High threshold
        matcher = PatternMatcher(triple_store_with_triples, match_threshold=0.99)

        pattern = ExpandedPattern(
            subject_id="unknown", subject_name="stress",
            predicate=Predicate.CAUSES,
            object_id="unknown", object_name="anxiousness",  # Fuzzy match
            confidence=1.0,
        )

        matches = matcher.match(pattern)
        # Fuzzy match won't meet 0.99 threshold
        assert len(matches) == 0


class TestPatternMatcherMatchAll:
    """Tests for PatternMatcher.match_all() method."""

    def test_match_all_deduplicates(self):
        """Test that match_all deduplicates by triple ID."""
        store = TripleStore()
        store.add(Triple(
            id="t1", subject="stress",
            predicate=Predicate.CAUSES,
            object="anxiety",
        ))

        matcher = PatternMatcher(store, match_threshold=0.3)

        # Two patterns that both match t1
        patterns = [
            ExpandedPattern(
                subject_id="e1", subject_name="stress",
                predicate=Predicate.CAUSES,
                object_id="e2", object_name="anxiety",
                confidence=1.0,
            ),
            ExpandedPattern(
                subject_id="e1", subject_name="stress",
                predicate=Predicate.CAUSES,
                object_id="e3", object_name="anxiousness",  # Similar
                confidence=0.9,
            ),
        ]

        matches = matcher.match_all(patterns)

        # Should only return t1 once
        triple_ids = [m.triple.id for m in matches]
        assert triple_ids.count("t1") == 1


class TestEvidenceCombinerInit:
    """Tests for EvidenceCombiner initialization."""

    def test_init_defaults(self):
        """Test default initialization."""
        combiner = EvidenceCombiner()

        assert combiner.k == 1.0
        assert combiner.support_threshold == 0.5
        assert combiner.confidence_threshold == 0.3

    def test_init_custom(self):
        """Test custom initialization."""
        combiner = EvidenceCombiner(k=2.0, support_threshold=0.7, confidence_threshold=0.4)

        assert combiner.k == 2.0
        assert combiner.support_threshold == 0.7
        assert combiner.confidence_threshold == 0.4


class TestEvidenceCombinerCombine:
    """Tests for EvidenceCombiner.combine() method."""

    @pytest.fixture
    def sample_matches(self):
        """Create sample pattern matches."""
        pattern = ExpandedPattern(
            subject_id="e1", subject_name="stress",
            predicate=Predicate.CAUSES,
            object_id="e2", object_name="anxiety",
            confidence=1.0,
        )
        triple = Triple(
            id="t1", subject="stress",
            predicate=Predicate.CAUSES,
            object="anxiety",
        )
        return [
            PatternMatch(
                pattern=pattern,
                triple=triple,
                match_score=1.0,
                subject_score=1.0,
                predicate_score=1.0,
                object_score=1.0,
                polarity=1.0,
                effective_truth=TruthValue(frequency=1.0, confidence=0.9),
            )
        ]

    def test_combine_no_matches(self):
        """Test combining with no matches."""
        combiner = EvidenceCombiner()
        result = combiner.combine([])

        assert result.verdict == VerificationVerdict.INSUFFICIENT
        assert result.combined_truth is None

    def test_combine_supported(self, sample_matches):
        """Test combining when evidence supports claim."""
        combiner = EvidenceCombiner()
        result = combiner.combine(sample_matches)

        assert result.verdict == VerificationVerdict.SUPPORTED
        assert result.combined_truth is not None
        assert result.combined_truth.frequency >= 0.5

    def test_combine_contradicted(self):
        """Test combining when evidence contradicts claim."""
        pattern = ExpandedPattern(
            subject_id="e1", subject_name="stress",
            predicate=Predicate.CAUSES,
            object_id="e2", object_name="anxiety",
            confidence=1.0,
        )
        triple = Triple(
            id="t1", subject="stress",
            predicate=Predicate.PREVENTS,  # Opposite
            object="anxiety",
        )
        matches = [
            PatternMatch(
                pattern=pattern,
                triple=triple,
                match_score=1.0,
                subject_score=1.0,
                predicate_score=0.9,
                object_score=1.0,
                polarity=-1.0,  # Contradicting
                effective_truth=TruthValue(frequency=0.0, confidence=0.9),  # Inverted
            )
        ]

        combiner = EvidenceCombiner()
        result = combiner.combine(matches)

        assert result.verdict == VerificationVerdict.CONTRADICTED
        assert result.combined_truth.frequency < 0.5

    def test_combine_insufficient_confidence(self):
        """Test combining with low confidence."""
        pattern = ExpandedPattern(
            subject_id="e1", subject_name="stress",
            predicate=Predicate.CAUSES,
            object_id="e2", object_name="anxiety",
            confidence=1.0,
        )
        triple = Triple(id="t1", subject="stress", predicate=Predicate.CAUSES, object="anxiety")
        matches = [
            PatternMatch(
                pattern=pattern,
                triple=triple,
                match_score=0.3,
                subject_score=0.5,
                predicate_score=1.0,
                object_score=0.6,
                polarity=1.0,
                effective_truth=TruthValue(frequency=0.8, confidence=0.1),  # Low confidence
            )
        ]

        combiner = EvidenceCombiner(confidence_threshold=0.5)
        result = combiner.combine(matches)

        assert result.verdict == VerificationVerdict.INSUFFICIENT

    def test_combine_multiple_matches(self):
        """Test combining multiple matches."""
        pattern = ExpandedPattern(
            subject_id="e1", subject_name="stress",
            predicate=Predicate.CAUSES,
            object_id="e2", object_name="anxiety",
            confidence=1.0,
        )
        matches = [
            PatternMatch(
                pattern=pattern,
                triple=Triple(id="t1", subject="stress", predicate=Predicate.CAUSES, object="anxiety"),
                match_score=1.0,
                subject_score=1.0,
                predicate_score=1.0,
                object_score=1.0,
                polarity=1.0,
                effective_truth=TruthValue(frequency=0.9, confidence=0.8),
            ),
            PatternMatch(
                pattern=pattern,
                triple=Triple(id="t2", subject="chronic_stress", predicate=Predicate.CAUSES, object="anxiety"),
                match_score=0.8,
                subject_score=0.8,
                predicate_score=1.0,
                object_score=1.0,
                polarity=1.0,
                effective_truth=TruthValue(frequency=0.95, confidence=0.7),
            ),
        ]

        combiner = EvidenceCombiner()
        result = combiner.combine(matches)

        # Multiple supporting matches should increase confidence
        assert result.verdict == VerificationVerdict.SUPPORTED
        assert len(result.matches) == 2


class TestQueryEngineInit:
    """Tests for QueryEngine initialization."""

    def test_init_defaults(self):
        """Test default initialization."""
        entity_store = EntityStore()
        triple_store = TripleStore()
        engine = QueryEngine(entity_store, triple_store)

        assert engine.entity_store is entity_store
        assert engine.triple_store is triple_store
        assert engine.expander is not None
        assert engine.matcher is not None
        assert engine.combiner is not None

    def test_init_custom(self):
        """Test custom initialization."""
        entity_store = EntityStore()
        triple_store = TripleStore()
        engine = QueryEngine(
            entity_store, triple_store,
            min_expansion_score=0.6,
            max_expansions=10,
            match_threshold=0.4,
        )

        assert engine.expander.min_score == 0.6
        assert engine.expander.max_expansions == 10
        assert engine.matcher.match_threshold == 0.4


class TestQueryEngineQuery:
    """Tests for QueryEngine.query() method."""

    @pytest.fixture
    def setup_engine(self):
        """Create engine with test data."""
        entity_store = EntityStore()
        triple_store = TripleStore()

        # Add entities
        entities = [
            Entity(id="e1", name="stress", entity_type="state"),
            Entity(id="e2", name="anxiety", entity_type="emotion"),
            Entity(id="e3", name="chronic_stress", entity_type="state"),
            Entity(id="e4", name="fear", entity_type="emotion"),
            Entity(id="e5", name="exercise", entity_type="activity"),
        ]
        for e in entities:
            entity_store.add(e)

        # Add entity links
        entity_store.add_link(EntityLink(
            source_id="e1", target_id="e3",
            link_type=LinkType.SIMILAR_TO, score=0.85,
        ))
        entity_store.add_link(EntityLink(
            source_id="e2", target_id="e4",
            link_type=LinkType.SIMILAR_TO, score=0.75,
        ))

        # Add triples
        triple_store.add(Triple(
            id="t1", subject="stress",
            predicate=Predicate.CAUSES,
            object="anxiety",
            subject_id="e1", object_id="e2",
        ))
        triple_store.add(Triple(
            id="t2", subject="exercise",
            predicate=Predicate.PREVENTS,
            object="stress",
            subject_id="e5", object_id="e1",
        ))
        triple_store.add(Triple(
            id="t3", subject="chronic_stress",
            predicate=Predicate.CAUSES,
            object="anxiety",
            subject_id="e3", object_id="e2",
        ))

        engine = QueryEngine(
            entity_store, triple_store,
            min_expansion_score=0.5,
            match_threshold=0.4,
        )

        return engine, entity_store, triple_store

    def test_query_exact_match(self, setup_engine):
        """Test query with exact match."""
        engine, _, _ = setup_engine

        result = engine.query("stress", Predicate.CAUSES, "anxiety")

        assert result.verdict == VerificationVerdict.SUPPORTED
        assert len(result.matches) >= 1

    def test_query_via_expansion(self, setup_engine):
        """Test query that matches via entity expansion."""
        engine, _, _ = setup_engine

        # chronic_stress is linked to stress, should find via expansion
        result = engine.query("chronic_stress", Predicate.CAUSES, "anxiety")

        assert result.verdict == VerificationVerdict.SUPPORTED

    def test_query_contradiction(self, setup_engine):
        """Test query that finds contradiction."""
        engine, _, _ = setup_engine

        # exercise PREVENTS stress, so "causes" should be contradicted
        result = engine.query("exercise", Predicate.CAUSES, "stress")

        assert result.verdict == VerificationVerdict.CONTRADICTED

    def test_query_no_match(self, setup_engine):
        """Test query with no matching evidence."""
        engine, _, _ = setup_engine

        result = engine.query("unknown_entity", Predicate.IS_A, "something")

        assert result.verdict == VerificationVerdict.INSUFFICIENT

    def test_query_patterns_searched(self, setup_engine):
        """Test that patterns_searched is populated."""
        engine, _, _ = setup_engine

        result = engine.query("stress", Predicate.CAUSES, "anxiety")

        assert result.patterns_searched > 0


class TestQueryEngineQueryTriple:
    """Tests for QueryEngine.query_triple() method."""

    def test_query_triple(self):
        """Test querying with a QueryTriple object."""
        entity_store = EntityStore()
        triple_store = TripleStore()

        entity_store.add(Entity(id="e1", name="stress"))
        entity_store.add(Entity(id="e2", name="anxiety"))

        triple_store.add(Triple(
            id="t1", subject="stress",
            predicate=Predicate.CAUSES,
            object="anxiety",
            subject_id="e1", object_id="e2",
        ))

        engine = QueryEngine(entity_store, triple_store)

        query = QueryTriple(
            subject_id="e1", subject_name="stress",
            predicate=Predicate.CAUSES,
            object_id="e2", object_name="anxiety",
        )

        result = engine.query_triple(query)

        assert result.verdict == VerificationVerdict.SUPPORTED


class TestQueryEngineVerify:
    """Tests for QueryEngine.verify() method."""

    def test_verify_triple(self):
        """Test verifying an existing Triple."""
        entity_store = EntityStore()
        triple_store = TripleStore()

        entity_store.add(Entity(id="e1", name="stress"))
        entity_store.add(Entity(id="e2", name="anxiety"))

        triple_store.add(Triple(
            id="t1", subject="stress",
            predicate=Predicate.CAUSES,
            object="anxiety",
        ))

        engine = QueryEngine(entity_store, triple_store)

        # Verify a new triple against the store
        query_triple = Triple(
            id="q1", subject="stress",
            predicate=Predicate.CAUSES,
            object="anxiety",
        )

        result = engine.verify(query_triple)

        assert result.verdict == VerificationVerdict.SUPPORTED


class TestQueryEngineIntegration:
    """Integration tests for the full query pipeline."""

    def test_psychology_scenario(self):
        """Test realistic psychology domain scenario."""
        entity_store = EntityStore()
        triple_store = TripleStore()

        # Build knowledge base
        entities = [
            Entity(id="e1", name="chronic_stress", entity_type="condition"),
            Entity(id="e2", name="acute_stress", entity_type="condition"),
            Entity(id="e3", name="cortisol", entity_type="hormone"),
            Entity(id="e4", name="memory_impairment", entity_type="symptom"),
            Entity(id="e5", name="working_memory", entity_type="cognitive"),
            Entity(id="e6", name="exercise", entity_type="activity"),
            Entity(id="e7", name="meditation", entity_type="activity"),
        ]
        for e in entities:
            entity_store.add(e)

        # Entity links
        links = [
            ("e1", "e2", 0.75),  # chronic ~ acute stress
            ("e4", "e5", 0.80),  # memory impairment ~ working memory
            ("e6", "e7", 0.60),  # exercise ~ meditation
        ]
        for src, tgt, score in links:
            entity_store.add_link(EntityLink(
                source_id=src, target_id=tgt,
                link_type=LinkType.SIMILAR_TO, score=score,
            ))

        # Knowledge triples
        triples = [
            Triple(id="t1", subject="chronic_stress", predicate=Predicate.CAUSES,
                   object="memory_impairment", subject_id="e1", object_id="e4"),
            Triple(id="t2", subject="chronic_stress", predicate=Predicate.CAUSES,
                   object="cortisol", subject_id="e1", object_id="e3"),
            Triple(id="t3", subject="exercise", predicate=Predicate.PREVENTS,
                   object="chronic_stress", subject_id="e6", object_id="e1"),
            Triple(id="t4", subject="meditation", predicate=Predicate.PREVENTS,
                   object="acute_stress", subject_id="e7", object_id="e2"),
        ]
        for t in triples:
            triple_store.add(t)

        engine = QueryEngine(
            entity_store, triple_store,
            min_expansion_score=0.5,
            match_threshold=0.4,
        )

        # Test 1: Direct query
        result1 = engine.query("chronic_stress", Predicate.CAUSES, "memory_impairment")
        assert result1.verdict == VerificationVerdict.SUPPORTED

        # Test 2: Query via entity expansion
        # acute_stress ~ chronic_stress, should find evidence
        result2 = engine.query("acute_stress", Predicate.CAUSES, "memory_impairment")
        assert result2.verdict == VerificationVerdict.SUPPORTED
        assert result2.combined_truth.confidence < 1.0  # Reduced by expansion

        # Test 3: Contradiction detection
        # exercise prevents stress, so "causes" should be contradicted
        result3 = engine.query("exercise", Predicate.CAUSES, "chronic_stress")
        assert result3.verdict == VerificationVerdict.CONTRADICTED

        # Test 4: No evidence
        result4 = engine.query("cortisol", Predicate.IS_A, "hormone")
        assert result4.verdict == VerificationVerdict.INSUFFICIENT

    def test_empty_stores(self):
        """Test with empty stores."""
        entity_store = EntityStore()
        triple_store = TripleStore()

        engine = QueryEngine(entity_store, triple_store)

        result = engine.query("anything", Predicate.CAUSES, "something")

        assert result.verdict == VerificationVerdict.INSUFFICIENT
        assert result.combined_truth is None
