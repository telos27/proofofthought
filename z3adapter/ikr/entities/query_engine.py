"""QueryEngine: Integrated query pipeline using entity links.

This module provides the full query pipeline that combines:
1. Entity resolution via EntityLinker
2. Query expansion via QueryExpander
3. Pattern matching against TripleStore
4. Evidence combination using NARS revision

The key difference from the existing fuzzy verification is that this
uses pre-computed entity links for expansion rather than on-the-fly
string similarity.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from z3adapter.ikr.entities.query_expander import (
    ExpandedPattern,
    QueryExpander,
    QueryTriple,
)
from z3adapter.ikr.entities.schema import Entity
from z3adapter.ikr.entities.store import EntityStore
from z3adapter.ikr.fuzzy_nars import (
    VerificationVerdict,
    combined_lexical_similarity,
    revise_multiple,
)
from z3adapter.ikr.schema import TruthValue
from z3adapter.ikr.triples.schema import Predicate, Triple, TripleStore, PREDICATE_OPPOSITES


@dataclass
class PatternMatch:
    """A match between an expanded pattern and a stored triple.

    Attributes:
        pattern: The expanded pattern that matched
        triple: The stored triple that was matched
        match_score: Combined match quality [0, 1]
        subject_score: Subject match score
        predicate_score: Predicate match score
        object_score: Object match score
        polarity: 1.0 for supporting, -1.0 for contradicting
        effective_truth: Truth value adjusted for match quality
    """

    pattern: ExpandedPattern
    triple: Triple
    match_score: float
    subject_score: float
    predicate_score: float
    object_score: float
    polarity: float
    effective_truth: TruthValue


@dataclass
class QueryResult:
    """Result of a query against the knowledge base.

    Attributes:
        verdict: Overall verdict (SUPPORTED, CONTRADICTED, INSUFFICIENT)
        combined_truth: Combined truth value from all evidence
        matches: List of pattern matches found
        patterns_searched: Number of patterns searched
        explanation: Human-readable explanation of the result
    """

    verdict: VerificationVerdict
    combined_truth: Optional[TruthValue]
    matches: list[PatternMatch] = field(default_factory=list)
    patterns_searched: int = 0
    explanation: str = ""


class PatternMatcher:
    """Match expanded patterns against a TripleStore.

    Uses entity IDs when available for exact matching, falls back to
    fuzzy string matching when IDs are not available.

    Example:
        store = TripleStore()
        store.add(Triple(id="t1", subject="stress", predicate=Predicate.CAUSES,
                         object="anxiety", subject_id="e1", object_id="e2"))

        matcher = PatternMatcher(store)
        pattern = ExpandedPattern(subject_id="e1", subject_name="stress",
                                  predicate=Predicate.CAUSES,
                                  object_id="e2", object_name="anxiety",
                                  confidence=1.0)
        matches = matcher.match(pattern)
    """

    def __init__(
        self,
        triple_store: TripleStore,
        entity_store: Optional[EntityStore] = None,
        sim_fn: Callable[[str, str], float] = combined_lexical_similarity,
        match_threshold: float = 0.5,
    ):
        """Initialize PatternMatcher.

        Args:
            triple_store: Store of triples to match against
            entity_store: Optional EntityStore for entity lookups
            sim_fn: Similarity function for string matching fallback
            match_threshold: Minimum match score to consider
        """
        self.triple_store = triple_store
        self.entity_store = entity_store
        self.sim_fn = sim_fn
        self.match_threshold = match_threshold

    def match(self, pattern: ExpandedPattern) -> list[PatternMatch]:
        """Find triples matching the given pattern.

        Matching strategy:
        1. Try exact entity ID match first (if IDs available)
        2. Fall back to fuzzy string matching
        3. Check predicate polarity for contradiction detection

        Args:
            pattern: Expanded pattern to match

        Returns:
            List of PatternMatch objects sorted by match score
        """
        matches = []

        for triple in self.triple_store:
            match = self._match_single(pattern, triple)
            if match:
                matches.append(match)

        # Sort by match score descending
        matches.sort(key=lambda m: m.match_score, reverse=True)
        return matches

    def match_all(self, patterns: list[ExpandedPattern]) -> list[PatternMatch]:
        """Find all triples matching any of the patterns.

        Args:
            patterns: List of expanded patterns to match

        Returns:
            List of all PatternMatch objects, deduplicated by triple ID
        """
        seen_triples: set[str] = set()
        all_matches = []

        for pattern in patterns:
            for match in self.match(pattern):
                if match.triple.id not in seen_triples:
                    seen_triples.add(match.triple.id)
                    all_matches.append(match)

        return all_matches

    def _match_single(
        self, pattern: ExpandedPattern, triple: Triple
    ) -> Optional[PatternMatch]:
        """Match a single pattern against a single triple.

        Args:
            pattern: Pattern to match
            triple: Triple to match against

        Returns:
            PatternMatch if match quality >= threshold, else None
        """
        # 1. Match subject
        subject_score = self._match_entity(
            pattern.subject_id, pattern.subject_name,
            triple.subject_id, triple.subject
        )

        # 2. Match object
        object_score = self._match_entity(
            pattern.object_id, pattern.object_name,
            triple.object_id, triple.object
        )

        # 3. Match predicate with polarity detection
        predicate_score, polarity = self._match_predicate(
            pattern.predicate, triple.predicate
        )

        # 4. Compute combined match score
        match_score = subject_score * predicate_score * object_score

        # 5. Apply pattern confidence
        effective_match = match_score * pattern.confidence

        if effective_match < self.match_threshold:
            return None

        # 6. Compute effective truth value
        effective_truth = self._compute_effective_truth(
            triple, match_score, polarity
        )

        return PatternMatch(
            pattern=pattern,
            triple=triple,
            match_score=effective_match,
            subject_score=subject_score,
            predicate_score=predicate_score,
            object_score=object_score,
            polarity=polarity,
            effective_truth=effective_truth,
        )

    def _match_entity(
        self,
        pattern_id: str,
        pattern_name: str,
        triple_id: Optional[str],
        triple_name: str,
    ) -> float:
        """Match an entity from pattern to triple.

        Uses exact ID match when available, fuzzy string match otherwise.
        """
        # Exact ID match
        if pattern_id and triple_id and pattern_id == triple_id:
            return 1.0

        # Check if entities are linked (via EntityStore)
        if self.entity_store and pattern_id and triple_id:
            similar = self.entity_store.get_similar(pattern_id, min_score=0.5)
            for entity, score in similar:
                if entity.id == triple_id:
                    return score

        # Fuzzy string match fallback
        return self.sim_fn(pattern_name, triple_name)

    def _match_predicate(
        self, pattern_pred: Predicate, triple_pred: Predicate
    ) -> tuple[float, float]:
        """Match predicates and detect polarity.

        Returns:
            (score, polarity) where polarity is 1.0 for supporting, -1.0 for opposing
        """
        # Exact match
        if pattern_pred == triple_pred:
            return (1.0, 1.0)

        # Check for opposite predicates
        opposite = PREDICATE_OPPOSITES.get(pattern_pred)
        if opposite == triple_pred:
            return (0.9, -1.0)  # High similarity but opposite meaning

        # Fuzzy string match
        sim = self.sim_fn(pattern_pred.value, triple_pred.value)
        return (sim, 1.0)

    def _compute_effective_truth(
        self, triple: Triple, match_score: float, polarity: float
    ) -> TruthValue:
        """Compute effective truth value adjusted for match quality and polarity."""
        # Get triple's truth value (default if not specified)
        base_truth = triple.truth or TruthValue(frequency=1.0, confidence=0.9)

        # Adjust for negation
        if triple.negated:
            frequency = 1.0 - base_truth.frequency
        else:
            frequency = base_truth.frequency

        # Adjust for polarity (contradiction inverts frequency)
        if polarity < 0:
            frequency = 1.0 - frequency

        # Reduce confidence by match quality
        confidence = base_truth.confidence * match_score
        confidence = max(0.001, min(0.999, confidence))

        return TruthValue(frequency=frequency, confidence=confidence)


class EvidenceCombiner:
    """Combine evidence from multiple matches using NARS revision.

    Pools evidence from all matching triples and computes a combined
    truth value that reflects the aggregate support.
    """

    def __init__(
        self,
        k: float = 1.0,
        support_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
    ):
        """Initialize EvidenceCombiner.

        Args:
            k: NARS evidential horizon constant
            support_threshold: Frequency above which claim is supported
            confidence_threshold: Minimum confidence to make judgment
        """
        self.k = k
        self.support_threshold = support_threshold
        self.confidence_threshold = confidence_threshold

    def combine(self, matches: list[PatternMatch]) -> QueryResult:
        """Combine evidence from matches into a verdict.

        Args:
            matches: List of pattern matches to combine

        Returns:
            QueryResult with verdict and combined truth
        """
        if not matches:
            return QueryResult(
                verdict=VerificationVerdict.INSUFFICIENT,
                combined_truth=None,
                matches=[],
                explanation="No matching triples found",
            )

        # Extract truth values
        truths = [m.effective_truth for m in matches]

        # Combine using NARS revision
        combined = revise_multiple(truths, self.k)

        # Determine verdict
        if combined.confidence < self.confidence_threshold:
            verdict = VerificationVerdict.INSUFFICIENT
            explanation = (
                f"Combined confidence ({combined.confidence:.3f}) "
                f"below threshold ({self.confidence_threshold})"
            )
        elif combined.frequency >= self.support_threshold:
            verdict = VerificationVerdict.SUPPORTED
            explanation = (
                f"Evidence supports claim "
                f"(f={combined.frequency:.3f}, c={combined.confidence:.3f}, "
                f"matches={len(matches)})"
            )
        else:
            verdict = VerificationVerdict.CONTRADICTED
            explanation = (
                f"Evidence contradicts claim "
                f"(f={combined.frequency:.3f}, c={combined.confidence:.3f}, "
                f"matches={len(matches)})"
            )

        return QueryResult(
            verdict=verdict,
            combined_truth=combined,
            matches=matches,
            explanation=explanation,
        )


class QueryEngine:
    """Full query pipeline using entity links.

    Orchestrates the complete query flow:
    1. Resolve query entities via EntityStore
    2. Expand query using pre-computed entity links
    3. Match expanded patterns against TripleStore
    4. Combine evidence using NARS revision
    5. Return verdict with explanation

    Example:
        engine = QueryEngine(entity_store, triple_store)
        result = engine.query("anxiety", Predicate.CAUSES, "memory_problems")

        print(result.verdict)  # SUPPORTED/CONTRADICTED/INSUFFICIENT
        print(result.explanation)
    """

    def __init__(
        self,
        entity_store: EntityStore,
        triple_store: TripleStore,
        min_expansion_score: float = 0.5,
        max_expansions: int = 20,
        match_threshold: float = 0.5,
        expand_predicates: bool = False,
        k: float = 1.0,
        support_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
    ):
        """Initialize QueryEngine.

        Args:
            entity_store: Store for entities and links
            triple_store: Store of triples to query
            min_expansion_score: Minimum score for entity expansion
            max_expansions: Maximum patterns to generate
            match_threshold: Minimum match score to consider
            expand_predicates: Whether to expand predicates
            k: NARS evidential horizon constant
            support_threshold: Frequency above which claim is supported
            confidence_threshold: Minimum confidence to make judgment
        """
        self.entity_store = entity_store
        self.triple_store = triple_store

        # Create components
        self.expander = QueryExpander(
            entity_store=entity_store,
            min_score=min_expansion_score,
            max_expansions=max_expansions,
            expand_predicates=expand_predicates,
        )
        self.matcher = PatternMatcher(
            triple_store=triple_store,
            entity_store=entity_store,
            match_threshold=match_threshold,
        )
        self.combiner = EvidenceCombiner(
            k=k,
            support_threshold=support_threshold,
            confidence_threshold=confidence_threshold,
        )

    def query(
        self,
        subject: str,
        predicate: Predicate,
        obj: str,
        check_contradiction: bool = True,
    ) -> QueryResult:
        """Query the knowledge base.

        Args:
            subject: Subject entity name
            predicate: Query predicate
            obj: Object entity name
            check_contradiction: Whether to also check opposite predicates

        Returns:
            QueryResult with verdict and evidence
        """
        # 1. Expand query using entity links
        patterns = self.expander.expand_by_name(subject, predicate, obj)

        # 2. Optionally add opposite predicate patterns for contradiction detection
        if check_contradiction:
            opposite_patterns = self.expander.get_opposite_patterns(patterns)
            patterns = patterns + opposite_patterns

        # 3. Match patterns against triple store
        matches = self.matcher.match_all(patterns)

        # 4. Combine evidence
        result = self.combiner.combine(matches)
        result.patterns_searched = len(patterns)

        return result

    def query_triple(
        self,
        query: QueryTriple,
        check_contradiction: bool = True,
    ) -> QueryResult:
        """Query using a QueryTriple object.

        Args:
            query: QueryTriple to query
            check_contradiction: Whether to also check opposite predicates

        Returns:
            QueryResult with verdict and evidence
        """
        # 1. Expand query
        patterns = self.expander.expand(query)

        # 2. Optionally add opposite predicate patterns
        if check_contradiction:
            opposite_patterns = self.expander.get_opposite_patterns(patterns)
            patterns = patterns + opposite_patterns

        # 3. Match patterns
        matches = self.matcher.match_all(patterns)

        # 4. Combine evidence
        result = self.combiner.combine(matches)
        result.patterns_searched = len(patterns)

        return result

    def verify(self, triple: Triple) -> QueryResult:
        """Verify an existing Triple against the knowledge base.

        Convenience method that converts a Triple to a query.

        Args:
            triple: Triple to verify

        Returns:
            QueryResult with verdict
        """
        return self.query(
            subject=triple.subject,
            predicate=triple.predicate,
            obj=triple.object,
            check_contradiction=True,
        )
