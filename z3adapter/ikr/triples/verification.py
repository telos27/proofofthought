"""Verification bridge between Triple schema and Fuzzy-NARS verification.

This module provides conversion utilities and verification functions that
connect the Triple extraction pipeline with the existing Fuzzy-NARS
verification system.

Key features:
- Convert Triple <-> VerificationTriple
- Verify triples against TripleStore using fuzzy matching
- Support for negated triples
- Predicate polarity detection (causes vs prevents)

Example:
    from z3adapter.ikr.triples import Triple, TripleStore, Predicate
    from z3adapter.ikr.triples.verification import verify_triple_against_store

    # Build knowledge base
    store = TripleStore()
    store.add(Triple(id="t1", subject="stress", predicate=Predicate.CAUSES, object="anxiety"))
    store.add(Triple(id="t2", subject="exercise", predicate=Predicate.PREVENTS, object="anxiety"))

    # Verify a claim
    query = Triple(id="q1", subject="stress", predicate=Predicate.CAUSES, object="anxiety")
    result = verify_triple_against_store(query, store)
    print(result.verdict)  # VerificationVerdict.SUPPORTED

    # Detect contradiction
    query2 = Triple(id="q2", subject="exercise", predicate=Predicate.CAUSES, object="anxiety")
    result2 = verify_triple_against_store(query2, store)
    print(result2.verdict)  # VerificationVerdict.CONTRADICTED
"""

from __future__ import annotations

from typing import Callable, Optional

from z3adapter.ikr.fuzzy_nars import (
    VerificationTriple,
    VerificationResult,
    VerificationVerdict,
    verify_triple,
    combined_lexical_similarity,
)
from z3adapter.ikr.schema import TruthValue
from z3adapter.ikr.triples.schema import Triple, TripleStore


def triple_to_verification(triple: Triple) -> VerificationTriple:
    """Convert a Triple to VerificationTriple for fuzzy-NARS verification.

    Handles:
    - Predicate enum to string conversion
    - Truth value propagation
    - Negation (inverts frequency if negated)

    Args:
        triple: The Triple to convert

    Returns:
        VerificationTriple suitable for fuzzy-NARS verification

    Example:
        triple = Triple(id="t1", subject="stress", predicate=Predicate.CAUSES, object="anxiety")
        vt = triple_to_verification(triple)
        # vt.subject == "stress"
        # vt.predicate == "causes"
        # vt.obj == "anxiety"
    """
    # Get truth value, handling negation
    truth = triple.truth
    if triple.negated and truth is not None:
        # Invert frequency for negated triples
        truth = TruthValue(
            frequency=1.0 - truth.frequency,
            confidence=truth.confidence,
        )
    elif triple.negated and truth is None:
        # Negated with no truth value means "this is false"
        truth = TruthValue(frequency=0.0, confidence=0.9)

    return VerificationTriple(
        subject=triple.subject,
        predicate=triple.predicate.value,  # Enum to string
        obj=triple.object,
        truth=truth,
    )


def verification_to_triple(
    vt: VerificationTriple,
    triple_id: str,
    source: Optional[str] = None,
) -> Triple:
    """Convert a VerificationTriple back to Triple.

    Args:
        vt: The VerificationTriple to convert
        triple_id: ID to assign to the new Triple
        source: Optional source/provenance

    Returns:
        Triple with the given ID

    Raises:
        ValueError: If predicate is not a valid Predicate enum value
    """
    from z3adapter.ikr.triples.schema import Predicate

    # Convert predicate string to enum
    try:
        predicate = Predicate(vt.predicate)
    except ValueError:
        # Fall back to RELATED_TO for unknown predicates
        predicate = Predicate.RELATED_TO

    # Determine if negated based on truth value
    negated = False
    truth = vt.truth
    if truth is not None and truth.frequency < 0.5:
        # Low frequency suggests negation
        negated = True
        truth = TruthValue(
            frequency=1.0 - truth.frequency,
            confidence=truth.confidence,
        )

    return Triple(
        id=triple_id,
        subject=vt.subject,
        predicate=predicate,
        object=vt.obj,
        negated=negated,
        truth=truth,
        source=source,
    )


def store_to_kb(store: TripleStore) -> list[VerificationTriple]:
    """Convert a TripleStore to list of VerificationTriples for verification.

    Skips triples whose subject or object reference other triples (t: prefix),
    as these represent meta-level beliefs that can't be directly verified.

    Args:
        store: The TripleStore to convert

    Returns:
        List of VerificationTriples
    """
    kb = []
    for triple in store:
        # Skip triples with triple references (meta-level)
        if triple.subject_is_triple or triple.object_is_triple:
            continue
        kb.append(triple_to_verification(triple))
    return kb


def verify_triple_against_store(
    query: Triple,
    store: TripleStore,
    sim_fn: Callable[[str, str], float] = combined_lexical_similarity,
    match_threshold: float = 0.5,
    support_threshold: float = 0.5,
    confidence_threshold: float = 0.3,
    k: float = 1.0,
) -> VerificationResult:
    """Verify a triple against the TripleStore using fuzzy-NARS.

    This is the main verification entry point. It:
    1. Converts the query Triple to VerificationTriple
    2. Converts the TripleStore to a list of VerificationTriples
    3. Runs fuzzy-NARS verification
    4. Returns the verification result

    Args:
        query: The Triple to verify
        store: The TripleStore knowledge base
        sim_fn: Similarity function for term matching
        match_threshold: Minimum match quality to consider a match
        support_threshold: Frequency above which claim is supported
        confidence_threshold: Minimum confidence to make a judgment
        k: NARS evidential horizon constant

    Returns:
        VerificationResult with verdict, combined truth, and matches

    Example:
        store = TripleStore()
        store.add(Triple(id="t1", subject="stress", predicate=Predicate.CAUSES, object="anxiety"))

        query = Triple(id="q1", subject="stress", predicate=Predicate.CAUSES, object="worry")
        result = verify_triple_against_store(query, store)
        # "worry" fuzzy-matches "anxiety", so this is SUPPORTED
    """
    # Skip verification for meta-level triples
    if query.subject_is_triple or query.object_is_triple:
        return VerificationResult(
            verdict=VerificationVerdict.INSUFFICIENT,
            combined_truth=None,
            matches=[],
            explanation="Cannot verify meta-level triples (triple references)",
        )

    query_vt = triple_to_verification(query)
    kb = store_to_kb(store)

    return verify_triple(
        query_vt,
        kb,
        sim_fn,
        match_threshold,
        support_threshold,
        confidence_threshold,
        k,
    )


def verify_triples_against_store(
    queries: list[Triple],
    store: TripleStore,
    sim_fn: Callable[[str, str], float] = combined_lexical_similarity,
    match_threshold: float = 0.5,
    support_threshold: float = 0.5,
    confidence_threshold: float = 0.3,
    k: float = 1.0,
) -> dict:
    """Verify multiple triples against the TripleStore.

    Args:
        queries: List of Triples to verify
        store: The TripleStore knowledge base
        sim_fn: Similarity function for term matching
        match_threshold: Minimum match quality to consider a match
        support_threshold: Frequency above which claim is supported
        confidence_threshold: Minimum confidence to make a judgment
        k: NARS evidential horizon constant

    Returns:
        Dictionary with:
        - triple_results: List of {triple, result} pairs
        - summary: Counts and overall verdict
    """
    results = []
    for query in queries:
        result = verify_triple_against_store(
            query,
            store,
            sim_fn,
            match_threshold,
            support_threshold,
            confidence_threshold,
            k,
        )
        results.append({"triple": query, "result": result})

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
