"""NARS-style uncertainty handling for IKR.

This module provides functions for handling uncertain information and
contradictions in IKR using NARS (Non-Axiomatic Reasoning System) semantics.

Key concepts:
- Truth values have (frequency, confidence) pairs
- Revision function combines conflicting evidence
- Facts can be filtered by confidence threshold

Example usage:
    from z3adapter.ikr.uncertainty import revise_conflicting_facts, threshold_filter

    # Combine conflicting facts about the same predicate
    revised_facts = revise_conflicting_facts(facts)

    # Filter out low-confidence facts
    confident_facts = threshold_filter(facts, min_confidence=0.5)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from z3adapter.ikr.schema import Fact, TruthValue

logger = logging.getLogger(__name__)


def revise_conflicting_facts(facts: list[Fact]) -> list[Fact]:
    """Apply NARS revision to conflicting facts about the same predicate.

    Groups facts by (predicate, arguments) and combines those with
    conflicting truth values using the NARS revision function.

    Args:
        facts: List of facts, potentially with conflicts

    Returns:
        List of facts with conflicts resolved via revision
    """
    from z3adapter.ikr.schema import Fact, TruthValue

    # Group facts by (predicate, arguments tuple)
    grouped: dict[tuple, list[Fact]] = defaultdict(list)
    for fact in facts:
        key = (fact.predicate, tuple(fact.arguments))
        grouped[key].append(fact)

    revised: list[Fact] = []
    for key, group in grouped.items():
        if len(group) == 1:
            # No conflict, keep as-is
            revised.append(group[0])
        else:
            # Multiple facts about same predicate - need to revise
            revised_fact = _revise_fact_group(group)
            revised.append(revised_fact)
            logger.debug(
                f"Revised {len(group)} conflicting facts for {key[0]} "
                f"-> f={revised_fact.truth_value.frequency:.3f}, "
                f"c={revised_fact.truth_value.confidence:.3f}"
            )

    return revised


def _revise_fact_group(facts: list[Fact]) -> Fact:
    """Revise a group of facts about the same predicate into one.

    Separates positive and negative evidence, combines each set,
    then performs final revision to get combined truth value.

    Args:
        facts: List of facts about the same (predicate, arguments)

    Returns:
        Single fact with revised truth value
    """
    from z3adapter.ikr.schema import Fact, TruthValue

    # Separate positive and negative evidence
    positive_facts = [f for f in facts if not f.negated]
    negative_facts = [f for f in facts if f.negated]

    # Get or create truth values
    positive_tvs = [
        f.truth_value if f.truth_value else TruthValue(frequency=1.0, confidence=0.9)
        for f in positive_facts
    ]
    negative_tvs = [
        f.truth_value if f.truth_value else TruthValue(frequency=1.0, confidence=0.9)
        for f in negative_facts
    ]

    # Combine positive evidence
    pos_combined = _combine_truth_values(positive_tvs) if positive_tvs else None

    # Combine negative evidence (negate to get "not P" truth values)
    neg_combined = _combine_truth_values(negative_tvs) if negative_tvs else None

    # Final combination
    if pos_combined and neg_combined:
        # Revise positive with negated negative
        final_tv = pos_combined.revise(neg_combined.negate())
    elif pos_combined:
        final_tv = pos_combined
    elif neg_combined:
        # Only negative evidence -> negate it for final truth value
        final_tv = neg_combined.negate()
    else:
        # No evidence (shouldn't happen)
        final_tv = TruthValue(frequency=0.5, confidence=0.1)

    # Determine if final fact is positive or negative
    is_positive = final_tv.frequency >= 0.5

    # Use first fact as template
    base = facts[0]
    return Fact(
        predicate=base.predicate,
        arguments=base.arguments,
        negated=not is_positive,
        value=base.value,
        source="revised",
        justification=f"Revised from {len(facts)} sources",
        truth_value=final_tv,
        epistemic_context=base.epistemic_context,
    )


def _combine_truth_values(tvs: list[TruthValue]) -> TruthValue:
    """Combine multiple truth values via iterative revision.

    Args:
        tvs: List of truth values to combine

    Returns:
        Combined truth value
    """
    if not tvs:
        from z3adapter.ikr.schema import TruthValue
        return TruthValue(frequency=0.5, confidence=0.1)

    result = tvs[0]
    for tv in tvs[1:]:
        result = result.revise(tv)
    return result


def threshold_filter(
    facts: list[Fact],
    min_confidence: float = 0.5,
) -> list[Fact]:
    """Filter facts by confidence threshold.

    Removes facts whose confidence is below the threshold.
    Facts without explicit truth values are kept (assumed classical).

    Args:
        facts: List of facts to filter
        min_confidence: Minimum confidence to keep (default 0.5)

    Returns:
        Filtered list of facts
    """
    return [
        f for f in facts
        if f.truth_value is None or f.truth_value.confidence >= min_confidence
    ]


def expectation_filter(
    facts: list[Fact],
    min_expectation: float = 0.5,
) -> list[Fact]:
    """Filter facts by expected truth value.

    The expectation is: f * c + 0.5 * (1 - c)
    This accounts for both frequency and confidence.

    Args:
        facts: List of facts to filter
        min_expectation: Minimum expectation to keep (default 0.5)

    Returns:
        Filtered list of facts
    """
    return [
        f for f in facts
        if f.truth_value is None or f.truth_value.expectation() >= min_expectation
    ]


def partition_by_certainty(
    facts: list[Fact],
    high_confidence: float = 0.8,
    low_confidence: float = 0.3,
) -> tuple[list[Fact], list[Fact], list[Fact]]:
    """Partition facts into high, medium, and low confidence groups.

    Args:
        facts: List of facts to partition
        high_confidence: Threshold for high confidence (>= this)
        low_confidence: Threshold for low confidence (< this)

    Returns:
        Tuple of (high_confidence_facts, medium_confidence_facts, low_confidence_facts)
    """
    high = []
    medium = []
    low = []

    for f in facts:
        if f.truth_value is None:
            # Classical facts go to high confidence
            high.append(f)
        elif f.truth_value.confidence >= high_confidence:
            high.append(f)
        elif f.truth_value.confidence < low_confidence:
            low.append(f)
        else:
            medium.append(f)

    return high, medium, low


def compute_fact_weight(fact: Fact) -> int:
    """Compute integer weight for a fact (for soft constraints).

    Weight is frequency * confidence * 1000, scaled to integer.

    Args:
        fact: Fact to compute weight for

    Returns:
        Integer weight (0-1000)
    """
    if fact.truth_value is None:
        # Classical facts get maximum weight
        return 1000

    tv = fact.truth_value
    weight = int(tv.frequency * tv.confidence * 1000)
    return max(1, min(1000, weight))  # Clamp to [1, 1000]


def detect_conflicts(facts: list[Fact]) -> list[tuple[Fact, Fact]]:
    """Detect pairs of directly conflicting facts.

    Two facts conflict if they have the same (predicate, arguments)
    but opposite negation or very different frequency values.

    Args:
        facts: List of facts to check

    Returns:
        List of (fact1, fact2) pairs that conflict
    """
    conflicts = []
    grouped: dict[tuple, list[Fact]] = defaultdict(list)

    for fact in facts:
        key = (fact.predicate, tuple(fact.arguments))
        grouped[key].append(fact)

    for key, group in grouped.items():
        if len(group) < 2:
            continue

        # Check all pairs
        for i, f1 in enumerate(group):
            for f2 in group[i + 1:]:
                if _facts_conflict(f1, f2):
                    conflicts.append((f1, f2))

    return conflicts


def _facts_conflict(f1: Fact, f2: Fact) -> bool:
    """Check if two facts about the same predicate conflict."""
    # Different negation = direct conflict
    if f1.negated != f2.negated:
        return True

    # Both have truth values with very different frequencies
    if f1.truth_value and f2.truth_value:
        freq_diff = abs(f1.truth_value.frequency - f2.truth_value.frequency)
        # Consider > 0.5 difference as conflict
        if freq_diff > 0.5:
            return True

    return False
