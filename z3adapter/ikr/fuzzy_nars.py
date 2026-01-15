"""Fuzzy-NARS Unification: Verification with Similarity-Based Matching.

This module combines:
1. NARS (Non-Axiomatic Reasoning System) evidence-based truth values
2. Fuzzy weak unification for similarity-based term matching

The goal is to enable verification of answers against a knowledge base while
avoiding the brittleness of exact symbolic matching.

Key features:
- Fuzzy term matching (lexical + optional embeddings)
- Predicate polarity detection (causes vs prevents)
- NARS evidence pooling via revision
- Verification verdicts (supported/contradicted/insufficient)

Usage:
    from z3adapter.ikr.fuzzy_nars import (
        VerificationTriple,
        verify_triple,
        verify_answer,
        combined_lexical_similarity,
    )

    kb = [
        VerificationTriple("phobia", "is_a", "anxiety_disorder", TruthValue(1.0, 0.9)),
        VerificationTriple("stress", "causes", "cortisol_release", TruthValue(0.95, 0.9)),
    ]

    query = VerificationTriple("phobia", "is_a", "disorder")
    result = verify_triple(query, kb, combined_lexical_similarity)
    print(result.verdict)  # VerificationVerdict.SUPPORTED

References:
- Wang, P. (2013). Non-Axiomatic Logic: A Model of Intelligent Reasoning
- FASILL: Fuzzy Aggregators and Similarity Into a Logic Language
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from z3adapter.ikr.schema import TruthValue


# =============================================================================
# Core Data Structures
# =============================================================================


@dataclass
class VerificationTriple:
    """A triple to be verified: (subject, predicate, object).

    Lightweight structure for verification, independent of database storage.

    Example:
        triple = VerificationTriple("phobia", "is_a", "anxiety_disorder")
        triple_with_truth = VerificationTriple(
            "stress", "causes", "anxiety",
            truth=TruthValue(frequency=0.9, confidence=0.8)
        )
    """

    subject: str
    predicate: str
    obj: str  # 'object' is a Python builtin
    truth: Optional[TruthValue] = None

    def __repr__(self) -> str:
        truth_str = f" {self.truth}" if self.truth else ""
        return f"({self.subject}, {self.predicate}, {self.obj}){truth_str}"


@dataclass
class UnificationResult:
    """Result of fuzzy-NARS unification between two triples."""

    success: bool
    match_quality: float  # Combined similarity in [0,1]
    subject_sim: float  # Subject similarity
    predicate_sim: float  # Predicate similarity
    object_sim: float  # Object similarity
    effective_truth: TruthValue  # Truth adjusted for match quality
    polarity: float = 1.0  # 1.0 for supporting, -1.0 for opposing

    def __repr__(self) -> str:
        return (
            f"UnificationResult(match={self.match_quality:.3f}, "
            f"truth={self.effective_truth}, polarity={self.polarity})"
        )


class VerificationVerdict(Enum):
    """Possible verification outcomes."""

    SUPPORTED = "supported"  # Evidence supports the claim
    CONTRADICTED = "contradicted"  # Evidence contradicts the claim
    INSUFFICIENT = "insufficient"  # Not enough evidence to judge


@dataclass
class VerificationResult:
    """Complete result of verifying a triple against a knowledge base."""

    verdict: VerificationVerdict
    combined_truth: Optional[TruthValue]
    matches: list[UnificationResult] = field(default_factory=list)
    explanation: str = ""

    def __repr__(self) -> str:
        return f"VerificationResult({self.verdict.value}, truth={self.combined_truth})"


# =============================================================================
# Predicate Polarity (for contradiction detection)
# =============================================================================

PREDICATE_OPPOSITES: dict[str, str] = {
    # Causation
    "causes": "prevents",
    "prevents": "causes",
    "leads_to": "prevents",
    "results_in": "prevents",
    # Support/Opposition
    "supports": "contradicts",
    "contradicts": "supports",
    # Magnitude
    "increases": "decreases",
    "decreases": "increases",
    # Enablement
    "enables": "inhibits",
    "inhibits": "enables",
    "promotes": "suppresses",
    "suppresses": "promotes",
    # Requirements
    "requires": "excludes",
    "excludes": "requires",
    # Boolean properties
    "is": "is_not",
    "is_not": "is",
    "has": "lacks",
    "lacks": "has",
}


def get_predicate_polarity(p1: str, p2: str, base_sim: float) -> tuple[float, float]:
    """Get similarity and polarity between two predicates.

    Args:
        p1: First predicate
        p2: Second predicate
        base_sim: Base similarity score from similarity function

    Returns:
        (similarity, polarity) where polarity is 1.0 for supporting, -1.0 for opposing
    """
    # Normalize to lowercase with underscores
    p1_norm = p1.lower().replace(" ", "_")
    p2_norm = p2.lower().replace(" ", "_")

    # Check for known opposites
    if PREDICATE_OPPOSITES.get(p1_norm) == p2_norm:
        return (0.9, -1.0)  # High similarity but opposite meaning
    if PREDICATE_OPPOSITES.get(p2_norm) == p1_norm:
        return (0.9, -1.0)

    # Otherwise use base similarity with positive polarity
    return (base_sim, 1.0)


# =============================================================================
# Similarity Functions
# =============================================================================


def lexical_similarity(term1: str, term2: str) -> float:
    """Normalized Levenshtein similarity.

    Returns a value in [0, 1] where 1.0 means identical.
    Case-insensitive and treats underscores as spaces.

    Args:
        term1: First term
        term2: Second term

    Returns:
        Similarity score in [0, 1]
    """
    # Normalize: lowercase, replace underscores with spaces
    t1 = term1.lower().replace("_", " ").strip()
    t2 = term2.lower().replace("_", " ").strip()

    if t1 == t2:
        return 1.0

    len1, len2 = len(t1), len(t2)
    if len1 == 0 or len2 == 0:
        return 0.0

    # Levenshtein distance via dynamic programming
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if t1[i - 1] == t2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    distance = dp[len1][len2]
    max_len = max(len1, len2)
    return 1.0 - distance / max_len


def jaccard_word_similarity(term1: str, term2: str) -> float:
    """Jaccard similarity based on word overlap.

    Good for multi-word terms like "classical conditioning".

    Args:
        term1: First term
        term2: Second term

    Returns:
        Jaccard similarity in [0, 1]
    """
    t1 = term1.lower().replace("_", " ").strip()
    t2 = term2.lower().replace("_", " ").strip()

    words1 = set(t1.split())
    words2 = set(t2.split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def combined_lexical_similarity(term1: str, term2: str) -> float:
    """Combine Levenshtein and Jaccard for robust lexical matching.

    Uses the maximum of character-level and word-level similarity.

    Args:
        term1: First term
        term2: Second term

    Returns:
        Maximum of Levenshtein and Jaccard similarity
    """
    lev_sim = lexical_similarity(term1, term2)
    jac_sim = jaccard_word_similarity(term1, term2)
    return max(lev_sim, jac_sim)


def make_embedding_similarity(
    get_embedding: Callable[[str], list[float]],
    cache: Optional[dict[str, list[float]]] = None,
) -> Callable[[str, str], float]:
    """Factory for embedding-based similarity function.

    Args:
        get_embedding: Function that returns embedding vector for a term
        cache: Optional dict to cache embeddings

    Returns:
        Similarity function using cosine similarity between embeddings
    """
    _cache = cache if cache is not None else {}

    def embedding_similarity(term1: str, term2: str) -> float:
        """Cosine similarity between term embeddings."""
        # Normalize terms
        t1 = term1.lower().replace("_", " ").strip()
        t2 = term2.lower().replace("_", " ").strip()

        if t1 == t2:
            return 1.0

        # Get embeddings (with caching)
        if t1 not in _cache:
            _cache[t1] = get_embedding(t1)
        if t2 not in _cache:
            _cache[t2] = get_embedding(t2)

        v1, v2 = _cache[t1], _cache[t2]

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    return embedding_similarity


def make_hybrid_similarity(
    get_embedding: Callable[[str], list[float]],
    cache: Optional[dict[str, list[float]]] = None,
    lexical_threshold: float = 0.9,
) -> Callable[[str, str], float]:
    """Factory for hybrid similarity: lexical + semantic.

    Uses lexical similarity when it's high enough, otherwise falls back
    to embedding-based similarity.

    Args:
        get_embedding: Function that returns embedding vector for a term
        cache: Optional dict to cache embeddings
        lexical_threshold: Use lexical if >= this threshold

    Returns:
        Hybrid similarity function
    """
    embed_sim = make_embedding_similarity(get_embedding, cache)

    def hybrid_similarity(term1: str, term2: str) -> float:
        lex_sim = combined_lexical_similarity(term1, term2)

        # If lexically very similar, trust that
        if lex_sim >= lexical_threshold:
            return lex_sim

        # Otherwise, use max of lexical and semantic
        sem_sim = embed_sim(term1, term2)
        return max(lex_sim, sem_sim)

    return hybrid_similarity


# =============================================================================
# Core Algorithms
# =============================================================================


def fuzzy_nars_unify(
    query: VerificationTriple,
    kb_triple: VerificationTriple,
    sim_fn: Callable[[str, str], float],
    threshold: float = 0.5,
    use_polarity: bool = True,
) -> Optional[UnificationResult]:
    """Unify query triple with KB triple using similarity-based matching.

    This implements the core Fuzzy-NARS unification algorithm:
    1. Compute term similarities
    2. Combine similarities using product t-norm
    3. Adjust truth value by match quality
    4. Handle predicate polarity for contradiction detection

    Args:
        query: Triple to verify
        kb_triple: Knowledge base triple
        sim_fn: Term similarity function
        threshold: Minimum match quality to consider a match
        use_polarity: Whether to check predicate polarity for contradictions

    Returns:
        UnificationResult if match quality >= threshold, else None
    """
    from z3adapter.ikr.schema import TruthValue

    # 1. Compute term similarities
    s_subj = sim_fn(query.subject, kb_triple.subject)
    s_obj = sim_fn(query.obj, kb_triple.obj)

    # 2. Handle predicate similarity with polarity
    base_pred_sim = sim_fn(query.predicate, kb_triple.predicate)
    if use_polarity:
        s_pred, polarity = get_predicate_polarity(
            query.predicate, kb_triple.predicate, base_pred_sim
        )
    else:
        s_pred, polarity = base_pred_sim, 1.0

    # 3. Combine similarities (product t-norm)
    match_quality = s_subj * abs(s_pred) * s_obj

    # 4. Check threshold
    if match_quality < threshold:
        return None

    # 5. Get KB triple's truth value (default if not specified)
    kb_truth = kb_triple.truth or TruthValue(frequency=1.0, confidence=0.9)

    # 6. Adjust truth value by match quality
    # Rationale: Uncertain match = less reliable evidence
    # If polarity is negative, we invert the frequency (contradiction)
    effective_frequency = (
        kb_truth.frequency if polarity > 0 else (1.0 - kb_truth.frequency)
    )

    # Confidence is reduced by match quality
    effective_confidence = kb_truth.confidence * match_quality
    # Clamp to valid range
    effective_confidence = max(0.001, min(0.999, effective_confidence))

    effective_truth = TruthValue(
        frequency=effective_frequency,
        confidence=effective_confidence,
    )

    return UnificationResult(
        success=True,
        match_quality=match_quality,
        subject_sim=s_subj,
        predicate_sim=s_pred,
        object_sim=s_obj,
        effective_truth=effective_truth,
        polarity=polarity,
    )


def revise(t1: TruthValue, t2: TruthValue, k: float = 1.0) -> TruthValue:
    """NARS revision rule - combine evidence from two sources.

    Uses evidence pooling: converts truth values to evidence counts,
    pools the evidence, then converts back to truth value.

    The result has:
    - Frequency: weighted average by evidence amount
    - Confidence: higher than either input (more total evidence)

    Args:
        t1: First truth value
        t2: Second truth value
        k: Evidential horizon constant

    Returns:
        Combined truth value
    """
    from z3adapter.ikr.schema import TruthValue

    # Convert confidence to evidence count
    w1_pos, w1_total = t1.to_evidence(k)
    w2_pos, w2_total = t2.to_evidence(k)

    # Handle edge cases
    if w1_total == 0 and w2_total == 0:
        return TruthValue(frequency=0.5, confidence=0.001)
    if w1_total == 0:
        return t2
    if w2_total == 0:
        return t1

    # Handle infinity (very high confidence)
    if w1_total == float("inf") and w2_total == float("inf"):
        # Average frequencies, max confidence
        return TruthValue(
            frequency=(t1.frequency + t2.frequency) / 2,
            confidence=0.999,
        )
    if w1_total == float("inf"):
        return t1
    if w2_total == float("inf"):
        return t2

    # Pool evidence
    w_total = w1_total + w2_total
    w_pos = w1_pos + w2_pos

    # Compute combined truth value
    f_combined = w_pos / w_total if w_total > 0 else 0.5
    c_combined = w_total / (w_total + k)

    # Clamp to valid range
    c_combined = max(0.001, min(0.999, c_combined))

    return TruthValue(frequency=f_combined, confidence=c_combined)


def revise_multiple(truths: list[TruthValue], k: float = 1.0) -> TruthValue:
    """Combine evidence from multiple sources using NARS revision.

    Args:
        truths: List of truth values to combine
        k: Evidential horizon constant

    Returns:
        Combined truth value
    """
    from z3adapter.ikr.schema import TruthValue

    if not truths:
        return TruthValue(frequency=0.5, confidence=0.001)

    result = truths[0]
    for truth in truths[1:]:
        result = revise(result, truth, k)

    return result


# =============================================================================
# Verification Pipeline
# =============================================================================


def verify_triple(
    answer_triple: VerificationTriple,
    kb: list[VerificationTriple],
    sim_fn: Callable[[str, str], float],
    match_threshold: float = 0.5,
    support_threshold: float = 0.5,
    confidence_threshold: float = 0.3,
    k: float = 1.0,
) -> VerificationResult:
    """Verify an answer triple against the knowledge base.

    This is the main verification pipeline:
    1. Find all matching KB triples using fuzzy unification
    2. Combine evidence from all matches using NARS revision
    3. Determine verdict based on combined truth value

    Args:
        answer_triple: Triple to verify
        kb: List of knowledge base triples
        sim_fn: Similarity function for term matching
        match_threshold: Minimum match quality to consider
        support_threshold: Frequency above which claim is supported
        confidence_threshold: Minimum confidence to make a judgment
        k: NARS evidential horizon constant

    Returns:
        VerificationResult with verdict, combined truth, and matches
    """
    # 1. Find all matching KB triples
    matches = []
    for kb_triple in kb:
        result = fuzzy_nars_unify(answer_triple, kb_triple, sim_fn, match_threshold)
        if result:
            matches.append(result)

    # 2. No matches = insufficient evidence
    if not matches:
        return VerificationResult(
            verdict=VerificationVerdict.INSUFFICIENT,
            combined_truth=None,
            matches=[],
            explanation="No matching triples found in knowledge base",
        )

    # 3. Combine evidence from all matches
    truths = [m.effective_truth for m in matches]
    combined = revise_multiple(truths, k)

    # 4. Determine verdict
    if combined.confidence < confidence_threshold:
        verdict = VerificationVerdict.INSUFFICIENT
        explanation = (
            f"Combined confidence ({combined.confidence:.3f}) "
            f"below threshold ({confidence_threshold})"
        )
    elif combined.frequency >= support_threshold:
        verdict = VerificationVerdict.SUPPORTED
        explanation = (
            f"Evidence supports claim "
            f"(f={combined.frequency:.3f}, c={combined.confidence:.3f})"
        )
    else:
        verdict = VerificationVerdict.CONTRADICTED
        explanation = (
            f"Evidence contradicts claim "
            f"(f={combined.frequency:.3f}, c={combined.confidence:.3f})"
        )

    return VerificationResult(
        verdict=verdict,
        combined_truth=combined,
        matches=matches,
        explanation=explanation,
    )


def verify_answer(
    answer_triples: list[VerificationTriple],
    kb: list[VerificationTriple],
    sim_fn: Callable[[str, str], float],
    match_threshold: float = 0.5,
    support_threshold: float = 0.5,
    confidence_threshold: float = 0.3,
    k: float = 1.0,
) -> dict:
    """Verify multiple answer triples against the knowledge base.

    Args:
        answer_triples: List of triples from decomposed answer
        kb: Knowledge base triples
        sim_fn: Similarity function
        match_threshold: Minimum match quality
        support_threshold: Frequency threshold for support
        confidence_threshold: Minimum confidence for judgment
        k: NARS evidential horizon constant

    Returns:
        Dictionary with per-triple results and overall summary
    """
    results = []
    for triple in answer_triples:
        result = verify_triple(
            triple,
            kb,
            sim_fn,
            match_threshold,
            support_threshold,
            confidence_threshold,
            k,
        )
        results.append({"triple": triple, "result": result})

    # Compute summary statistics
    verdicts = [r["result"].verdict for r in results]
    supported = sum(1 for v in verdicts if v == VerificationVerdict.SUPPORTED)
    contradicted = sum(1 for v in verdicts if v == VerificationVerdict.CONTRADICTED)
    insufficient = sum(1 for v in verdicts if v == VerificationVerdict.INSUFFICIENT)

    # Overall verdict logic: any contradiction fails the whole answer
    if contradicted > 0:
        overall = VerificationVerdict.CONTRADICTED
    elif supported > 0 and insufficient == 0:
        overall = VerificationVerdict.SUPPORTED
    elif supported > 0:
        overall = VerificationVerdict.SUPPORTED  # Partial support is still support
    else:
        overall = VerificationVerdict.INSUFFICIENT

    return {
        "triple_results": results,
        "summary": {
            "total": len(results),
            "supported": supported,
            "contradicted": contradicted,
            "insufficient": insufficient,
            "overall_verdict": overall,
        },
    }
