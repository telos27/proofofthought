"""Unit tests for QueryExpander."""

import pytest

from z3adapter.ikr.entities import Entity, EntityLink, EntityStore, LinkType
from z3adapter.ikr.entities.query_expander import (
    ExpandedPattern,
    QueryExpander,
    QueryTriple,
    PREDICATE_SIMILARITY,
)
from z3adapter.ikr.triples.schema import Predicate, PREDICATE_OPPOSITES


class TestExpandedPattern:
    """Tests for ExpandedPattern dataclass."""

    def test_create_pattern(self):
        """Test creating an expanded pattern."""
        pattern = ExpandedPattern(
            subject_id="e1",
            subject_name="anxiety",
            predicate=Predicate.CAUSES,
            object_id="e2",
            object_name="stress",
            confidence=0.85,
        )
        assert pattern.subject_id == "e1"
        assert pattern.subject_name == "anxiety"
        assert pattern.predicate == Predicate.CAUSES
        assert pattern.object_id == "e2"
        assert pattern.object_name == "stress"
        assert pattern.confidence == 0.85

    def test_default_scores(self):
        """Test default component scores."""
        pattern = ExpandedPattern(
            subject_id="e1",
            subject_name="anxiety",
            predicate=Predicate.CAUSES,
            object_id="e2",
            object_name="stress",
            confidence=1.0,
        )
        assert pattern.subject_score == 1.0
        assert pattern.object_score == 1.0
        assert pattern.predicate_score == 1.0
        assert pattern.is_original is False

    def test_component_scores(self):
        """Test setting component scores."""
        pattern = ExpandedPattern(
            subject_id="e1",
            subject_name="anxiety",
            predicate=Predicate.CAUSES,
            object_id="e2",
            object_name="stress",
            confidence=0.64,
            subject_score=0.8,
            object_score=0.8,
            predicate_score=1.0,
            is_original=False,
        )
        assert pattern.subject_score == 0.8
        assert pattern.object_score == 0.8
        assert pattern.confidence == 0.64

    def test_original_pattern(self):
        """Test marking pattern as original."""
        pattern = ExpandedPattern(
            subject_id="e1",
            subject_name="anxiety",
            predicate=Predicate.CAUSES,
            object_id="e2",
            object_name="stress",
            confidence=1.0,
            is_original=True,
        )
        assert pattern.is_original is True


class TestQueryTriple:
    """Tests for QueryTriple dataclass."""

    def test_create_query(self):
        """Test creating a query triple."""
        query = QueryTriple(
            subject_id="e1",
            subject_name="stress",
            predicate=Predicate.CAUSES,
            object_id="e2",
            object_name="anxiety",
        )
        assert query.subject_id == "e1"
        assert query.subject_name == "stress"
        assert query.predicate == Predicate.CAUSES
        assert query.object_id == "e2"
        assert query.object_name == "anxiety"


class TestPredicateSimilarity:
    """Tests for predicate similarity constants."""

    def test_causes_similarity(self):
        """Test CAUSES has related predicates."""
        similar = PREDICATE_SIMILARITY[Predicate.CAUSES]
        assert len(similar) > 0
        # RELATED_TO is a weaker form
        assert (Predicate.RELATED_TO, 0.5) in similar

    def test_part_of_has_inverse(self):
        """Test PART_OF and HAS are related."""
        part_of_similar = dict(PREDICATE_SIMILARITY[Predicate.PART_OF])
        has_similar = dict(PREDICATE_SIMILARITY[Predicate.HAS])

        assert Predicate.HAS in part_of_similar
        assert Predicate.PART_OF in has_similar

    def test_related_to_has_no_expansion(self):
        """Test RELATED_TO is most generic."""
        similar = PREDICATE_SIMILARITY[Predicate.RELATED_TO]
        assert len(similar) == 0


class TestQueryExpanderInit:
    """Tests for QueryExpander initialization."""

    def test_init_defaults(self):
        """Test default initialization."""
        store = EntityStore()
        expander = QueryExpander(store)

        assert expander.entity_store is store
        assert expander.min_score == 0.5
        assert expander.max_expansions == 20
        assert expander.expand_predicates is False

    def test_init_custom(self):
        """Test custom initialization."""
        store = EntityStore()
        expander = QueryExpander(
            store,
            min_score=0.7,
            max_expansions=50,
            expand_predicates=True,
        )

        assert expander.min_score == 0.7
        assert expander.max_expansions == 50
        assert expander.expand_predicates is True


class TestQueryExpanderExpand:
    """Tests for QueryExpander.expand() method."""

    @pytest.fixture
    def store_with_entities(self):
        """Create store with entities and links."""
        store = EntityStore()

        # Add entities
        anxiety = Entity(id="e1", name="anxiety", entity_type="emotion")
        stress = Entity(id="e2", name="stress", entity_type="state")
        fear = Entity(id="e3", name="fear", entity_type="emotion")
        memory = Entity(id="e4", name="memory", entity_type="cognitive")
        working_memory = Entity(id="e5", name="working_memory", entity_type="cognitive")
        long_term_memory = Entity(id="e6", name="long_term_memory", entity_type="cognitive")

        for entity in [anxiety, stress, fear, memory, working_memory, long_term_memory]:
            store.add(entity)

        # Add similarity links
        # anxiety ~ stress (0.85)
        store.add_link(EntityLink(
            source_id="e1", target_id="e2",
            link_type=LinkType.SIMILAR_TO, score=0.85
        ))
        # anxiety ~ fear (0.75)
        store.add_link(EntityLink(
            source_id="e1", target_id="e3",
            link_type=LinkType.SIMILAR_TO, score=0.75
        ))
        # memory ~ working_memory (0.80)
        store.add_link(EntityLink(
            source_id="e4", target_id="e5",
            link_type=LinkType.SIMILAR_TO, score=0.80
        ))
        # memory ~ long_term_memory (0.70)
        store.add_link(EntityLink(
            source_id="e4", target_id="e6",
            link_type=LinkType.SIMILAR_TO, score=0.70
        ))

        return store

    def test_expand_single_entity(self):
        """Test expansion with single entity (no links)."""
        store = EntityStore()
        store.add(Entity(id="e1", name="anxiety"))
        store.add(Entity(id="e2", name="memory"))

        expander = QueryExpander(store, min_score=0.5)

        query = QueryTriple(
            subject_id="e1",
            subject_name="anxiety",
            predicate=Predicate.CAUSES,
            object_id="e2",
            object_name="memory",
        )

        patterns = expander.expand(query)

        # Only the original pattern
        assert len(patterns) == 1
        assert patterns[0].is_original is True
        assert patterns[0].confidence == 1.0

    def test_expand_with_subject_links(self, store_with_entities):
        """Test expansion with subject entity links."""
        expander = QueryExpander(store_with_entities, min_score=0.5)

        query = QueryTriple(
            subject_id="e1",
            subject_name="anxiety",
            predicate=Predicate.CAUSES,
            object_id="e4",
            object_name="memory",
        )

        patterns = expander.expand(query)

        # Should have original + 2 subject expansions * object expansions
        subject_names = {p.subject_name for p in patterns}
        assert "anxiety" in subject_names
        assert "stress" in subject_names
        assert "fear" in subject_names

    def test_expand_with_object_links(self, store_with_entities):
        """Test expansion with object entity links."""
        expander = QueryExpander(store_with_entities, min_score=0.5)

        query = QueryTriple(
            subject_id="e1",
            subject_name="anxiety",
            predicate=Predicate.CAUSES,
            object_id="e4",
            object_name="memory",
        )

        patterns = expander.expand(query)

        object_names = {p.object_name for p in patterns}
        assert "memory" in object_names
        assert "working_memory" in object_names
        assert "long_term_memory" in object_names

    def test_expand_cross_product(self, store_with_entities):
        """Test cross-product of subject and object expansions."""
        expander = QueryExpander(store_with_entities, min_score=0.5)

        query = QueryTriple(
            subject_id="e1",
            subject_name="anxiety",
            predicate=Predicate.CAUSES,
            object_id="e4",
            object_name="memory",
        )

        patterns = expander.expand(query)

        # Check cross-product exists
        pattern_pairs = {(p.subject_name, p.object_name) for p in patterns}
        assert ("anxiety", "memory") in pattern_pairs
        assert ("stress", "memory") in pattern_pairs
        assert ("anxiety", "working_memory") in pattern_pairs
        assert ("stress", "working_memory") in pattern_pairs

    def test_expand_confidence_calculation(self, store_with_entities):
        """Test combined confidence is product of scores."""
        expander = QueryExpander(store_with_entities, min_score=0.5)

        query = QueryTriple(
            subject_id="e1",
            subject_name="anxiety",
            predicate=Predicate.CAUSES,
            object_id="e4",
            object_name="memory",
        )

        patterns = expander.expand(query)

        # Find specific pattern
        stress_wm = next(
            p for p in patterns
            if p.subject_name == "stress" and p.object_name == "working_memory"
        )

        # Combined should be subject_score * object_score
        expected = 0.85 * 0.80  # stress score * working_memory score
        assert abs(stress_wm.confidence - expected) < 0.01

    def test_expand_sorted_by_confidence(self, store_with_entities):
        """Test patterns are sorted by confidence descending."""
        expander = QueryExpander(store_with_entities, min_score=0.5)

        query = QueryTriple(
            subject_id="e1",
            subject_name="anxiety",
            predicate=Predicate.CAUSES,
            object_id="e4",
            object_name="memory",
        )

        patterns = expander.expand(query)

        confidences = [p.confidence for p in patterns]
        assert confidences == sorted(confidences, reverse=True)

    def test_expand_max_expansions_limit(self, store_with_entities):
        """Test max_expansions limits results."""
        expander = QueryExpander(store_with_entities, min_score=0.1, max_expansions=3)

        query = QueryTriple(
            subject_id="e1",
            subject_name="anxiety",
            predicate=Predicate.CAUSES,
            object_id="e4",
            object_name="memory",
        )

        patterns = expander.expand(query)

        assert len(patterns) <= 3

    def test_expand_min_score_filter(self, store_with_entities):
        """Test min_score filters low-confidence patterns."""
        expander = QueryExpander(store_with_entities, min_score=0.7)

        query = QueryTriple(
            subject_id="e1",
            subject_name="anxiety",
            predicate=Predicate.CAUSES,
            object_id="e4",
            object_name="memory",
        )

        patterns = expander.expand(query)

        for pattern in patterns:
            assert pattern.confidence >= 0.7

    def test_expand_original_marked(self, store_with_entities):
        """Test original pattern is marked."""
        expander = QueryExpander(store_with_entities, min_score=0.5)

        query = QueryTriple(
            subject_id="e1",
            subject_name="anxiety",
            predicate=Predicate.CAUSES,
            object_id="e4",
            object_name="memory",
        )

        patterns = expander.expand(query)

        originals = [p for p in patterns if p.is_original]
        assert len(originals) == 1
        assert originals[0].subject_name == "anxiety"
        assert originals[0].object_name == "memory"


class TestQueryExpanderPredicates:
    """Tests for predicate expansion."""

    @pytest.fixture
    def store_with_entities(self):
        """Create store with basic entities."""
        store = EntityStore()
        store.add(Entity(id="e1", name="stress"))
        store.add(Entity(id="e2", name="anxiety"))
        return store

    def test_no_predicate_expansion_by_default(self, store_with_entities):
        """Test predicates not expanded by default."""
        expander = QueryExpander(store_with_entities, min_score=0.3)

        query = QueryTriple(
            subject_id="e1",
            subject_name="stress",
            predicate=Predicate.CAUSES,
            object_id="e2",
            object_name="anxiety",
        )

        patterns = expander.expand(query)

        predicates = {p.predicate for p in patterns}
        assert predicates == {Predicate.CAUSES}

    def test_predicate_expansion_enabled(self, store_with_entities):
        """Test predicate expansion when enabled."""
        expander = QueryExpander(
            store_with_entities,
            min_score=0.3,
            expand_predicates=True,
        )

        query = QueryTriple(
            subject_id="e1",
            subject_name="stress",
            predicate=Predicate.CAUSES,
            object_id="e2",
            object_name="anxiety",
        )

        patterns = expander.expand(query)

        predicates = {p.predicate for p in patterns}
        assert Predicate.CAUSES in predicates
        assert Predicate.RELATED_TO in predicates  # From similarity

    def test_predicate_score_affects_confidence(self, store_with_entities):
        """Test predicate score contributes to confidence."""
        expander = QueryExpander(
            store_with_entities,
            min_score=0.3,
            expand_predicates=True,
        )

        query = QueryTriple(
            subject_id="e1",
            subject_name="stress",
            predicate=Predicate.CAUSES,
            object_id="e2",
            object_name="anxiety",
        )

        patterns = expander.expand(query)

        # Find RELATED_TO pattern
        related_to_pattern = next(
            p for p in patterns if p.predicate == Predicate.RELATED_TO
        )

        # Confidence should be lowered by predicate similarity score
        assert related_to_pattern.confidence == 0.5  # RELATED_TO score
        assert related_to_pattern.predicate_score == 0.5


class TestQueryExpanderByName:
    """Tests for QueryExpander.expand_by_name() convenience method."""

    def test_expand_by_name_new_entities(self):
        """Test expand_by_name creates new entities."""
        store = EntityStore()
        expander = QueryExpander(store, min_score=0.5)

        patterns = expander.expand_by_name(
            subject="stress",
            predicate=Predicate.CAUSES,
            obj="anxiety",
        )

        # Should have created entities
        assert store.get_by_name("stress") is not None
        assert store.get_by_name("anxiety") is not None

        # Should return pattern
        assert len(patterns) == 1
        assert patterns[0].subject_name == "stress"
        assert patterns[0].object_name == "anxiety"

    def test_expand_by_name_existing_entities(self):
        """Test expand_by_name uses existing entities."""
        store = EntityStore()
        store.add(Entity(id="e1", name="stress", entity_type="state"))
        store.add(Entity(id="e2", name="anxiety", entity_type="emotion"))

        expander = QueryExpander(store, min_score=0.5)

        patterns = expander.expand_by_name(
            subject="stress",
            predicate=Predicate.CAUSES,
            obj="anxiety",
        )

        # Should use existing entity IDs
        assert patterns[0].subject_id == "e1"
        assert patterns[0].object_id == "e2"

    def test_expand_by_name_with_entity_types(self):
        """Test expand_by_name with entity types."""
        store = EntityStore()
        expander = QueryExpander(store, min_score=0.5)

        patterns = expander.expand_by_name(
            subject="cortisol",
            predicate=Predicate.CAUSES,
            obj="stress_response",
            subject_type="hormone",
            object_type="process",
        )

        cortisol = store.get_by_name("cortisol")
        stress_response = store.get_by_name("stress_response")

        assert cortisol.entity_type == "hormone"
        assert stress_response.entity_type == "process"


class TestQueryExpanderOpposites:
    """Tests for opposite predicate pattern generation."""

    def test_get_opposite_patterns_causes(self):
        """Test generating opposite patterns for CAUSES."""
        store = EntityStore()
        store.add(Entity(id="e1", name="exercise"))
        store.add(Entity(id="e2", name="stress"))

        expander = QueryExpander(store, min_score=0.5)

        query = QueryTriple(
            subject_id="e1",
            subject_name="exercise",
            predicate=Predicate.CAUSES,
            object_id="e2",
            object_name="stress",
        )

        patterns = expander.expand(query)
        opposites = expander.get_opposite_patterns(patterns)

        assert len(opposites) == len(patterns)
        for opp in opposites:
            assert opp.predicate == Predicate.PREVENTS

    def test_get_opposite_patterns_prevents(self):
        """Test generating opposite patterns for PREVENTS."""
        store = EntityStore()
        store.add(Entity(id="e1", name="exercise"))
        store.add(Entity(id="e2", name="stress"))

        expander = QueryExpander(store, min_score=0.5)

        query = QueryTriple(
            subject_id="e1",
            subject_name="exercise",
            predicate=Predicate.PREVENTS,
            object_id="e2",
            object_name="stress",
        )

        patterns = expander.expand(query)
        opposites = expander.get_opposite_patterns(patterns)

        assert len(opposites) == len(patterns)
        for opp in opposites:
            assert opp.predicate == Predicate.CAUSES

    def test_get_opposite_patterns_no_opposite(self):
        """Test predicates without opposites return empty."""
        store = EntityStore()
        store.add(Entity(id="e1", name="cat"))
        store.add(Entity(id="e2", name="mammal"))

        expander = QueryExpander(store, min_score=0.5)

        query = QueryTriple(
            subject_id="e1",
            subject_name="cat",
            predicate=Predicate.IS_A,
            object_id="e2",
            object_name="mammal",
        )

        patterns = expander.expand(query)
        opposites = expander.get_opposite_patterns(patterns)

        assert len(opposites) == 0

    def test_opposite_preserves_confidence(self):
        """Test opposite patterns preserve confidence."""
        store = EntityStore()
        store.add(Entity(id="e1", name="stress"))
        store.add(Entity(id="e2", name="anxiety"))
        store.add_link(EntityLink(
            source_id="e1", target_id="e2",
            link_type=LinkType.SIMILAR_TO, score=0.8
        ))

        expander = QueryExpander(store, min_score=0.5)

        query = QueryTriple(
            subject_id="e1",
            subject_name="stress",
            predicate=Predicate.CAUSES,
            object_id="e2",
            object_name="anxiety",
        )

        patterns = expander.expand(query)
        opposites = expander.get_opposite_patterns(patterns)

        # Confidences should match
        for orig, opp in zip(patterns, opposites):
            assert orig.confidence == opp.confidence


class TestQueryExpanderIntegration:
    """Integration tests for QueryExpander."""

    def test_psychology_domain_scenario(self):
        """Test realistic psychology domain scenario."""
        store = EntityStore()

        # Add psychology entities
        entities = [
            Entity(id="e1", name="chronic_stress", entity_type="condition"),
            Entity(id="e2", name="acute_stress", entity_type="condition"),
            Entity(id="e3", name="anxiety", entity_type="emotion"),
            Entity(id="e4", name="cortisol", entity_type="hormone"),
            Entity(id="e5", name="memory_impairment", entity_type="symptom"),
            Entity(id="e6", name="working_memory", entity_type="cognitive"),
            Entity(id="e7", name="long_term_memory", entity_type="cognitive"),
        ]
        for e in entities:
            store.add(e)

        # Add similarity links
        links = [
            ("e1", "e2", 0.75),  # chronic_stress ~ acute_stress
            ("e1", "e3", 0.65),  # chronic_stress ~ anxiety
            ("e5", "e6", 0.80),  # memory_impairment ~ working_memory
            ("e5", "e7", 0.70),  # memory_impairment ~ long_term_memory
            ("e6", "e7", 0.85),  # working_memory ~ long_term_memory
        ]
        for src, tgt, score in links:
            store.add_link(EntityLink(
                source_id=src, target_id=tgt,
                link_type=LinkType.SIMILAR_TO, score=score
            ))

        expander = QueryExpander(store, min_score=0.5, max_expansions=10)

        # Query: does chronic stress affect memory?
        query = QueryTriple(
            subject_id="e1",
            subject_name="chronic_stress",
            predicate=Predicate.CAUSES,
            object_id="e5",
            object_name="memory_impairment",
        )

        patterns = expander.expand(query)

        # Should expand to related stress types and memory types
        subjects = {p.subject_name for p in patterns}
        objects = {p.object_name for p in patterns}

        assert "chronic_stress" in subjects
        assert "acute_stress" in subjects
        assert "memory_impairment" in objects
        assert "working_memory" in objects

        # Original should be highest confidence
        assert patterns[0].is_original is True
        assert patterns[0].confidence == 1.0

    def test_empty_store(self):
        """Test expansion with empty store."""
        store = EntityStore()
        expander = QueryExpander(store, min_score=0.5)

        # Expand by name will create entities
        patterns = expander.expand_by_name(
            subject="unknown_entity",
            predicate=Predicate.CAUSES,
            obj="another_unknown",
        )

        assert len(patterns) == 1
        assert patterns[0].is_original is True

    def test_bidirectional_links(self):
        """Test with bidirectional similarity links."""
        store = EntityStore()

        store.add(Entity(id="e1", name="stress"))
        store.add(Entity(id="e2", name="anxiety"))

        # Add bidirectional links
        store.add_link(EntityLink(
            source_id="e1", target_id="e2",
            link_type=LinkType.SIMILAR_TO, score=0.8
        ))
        store.add_link(EntityLink(
            source_id="e2", target_id="e1",
            link_type=LinkType.SIMILAR_TO, score=0.8
        ))

        expander = QueryExpander(store, min_score=0.5)

        query = QueryTriple(
            subject_id="e1",
            subject_name="stress",
            predicate=Predicate.CAUSES,
            object_id="e2",
            object_name="anxiety",
        )

        patterns = expander.expand(query)

        # Should have patterns for both directions
        subject_names = {p.subject_name for p in patterns}
        object_names = {p.object_name for p in patterns}

        assert "stress" in subject_names
        assert "anxiety" in subject_names  # From object -> subject link expansion
        assert "anxiety" in object_names
        assert "stress" in object_names  # From subject -> object link expansion
