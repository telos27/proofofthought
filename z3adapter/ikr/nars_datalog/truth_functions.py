"""NARS truth functions for Datalog inference.

This module implements the NARS truth value functions used during
Datalog inference to propagate uncertainty through rule applications.

Key functions:
- conjunction: Combine truth values for AND in rule body
- deduction: Derive conclusion truth from premise and rule
- negation: Invert frequency for NOT

Re-exports revise and revise_multiple from fuzzy_nars for evidence combination.
"""

from __future__ import annotations

from z3adapter.ikr.schema import TruthValue
from z3adapter.ikr.fuzzy_nars import revise, revise_multiple

__all__ = [
    "conjunction",
    "deduction",
    "negation",
    "revise",
    "revise_multiple",
    "DEFAULT_RULE_TRUTH",
]

# Default truth value for rules (certain rules)
DEFAULT_RULE_TRUTH = TruthValue(frequency=1.0, confidence=0.9)


def conjunction(truths: list[TruthValue]) -> TruthValue:
    """NARS intersection: combine truth values for conjunction (AND).

    Used when a rule body has multiple atoms: body1 AND body2 AND ...

    Formula (extensional intersection):
        f = f1 * f2 * ... * fn
        c = c1 * c2 * ... * cn (clamped to valid range)

    Args:
        truths: Truth values of conjuncts

    Returns:
        Combined truth value representing "all are true"
    """
    if not truths:
        # Empty conjunction is vacuously true with high confidence
        return TruthValue(frequency=1.0, confidence=0.9)

    f_result = 1.0
    c_result = 1.0

    for tv in truths:
        f_result *= tv.frequency
        c_result *= tv.confidence

    # Clamp confidence to valid range (0, 1)
    c_result = max(0.001, min(0.999, c_result))

    return TruthValue(frequency=f_result, confidence=c_result)


def deduction(premise_tv: TruthValue, rule_tv: TruthValue | None = None) -> TruthValue:
    """NARS deduction: derive conclusion from premise and rule.

    If we have premise A with truth (f1, c1) and rule A=>B with truth (f2, c2),
    derive B with truth computed by deduction formula.

    Formula:
        f = f1 * f2
        c = f1 * f2 * c1 * c2 (confidence degrades through inference)

    Args:
        premise_tv: Truth value of the premise (matched body atoms)
        rule_tv: Truth value of the rule itself (default: certain rule)

    Returns:
        Truth value for the derived conclusion
    """
    if rule_tv is None:
        rule_tv = DEFAULT_RULE_TRUTH

    f1, c1 = premise_tv.frequency, premise_tv.confidence
    f2, c2 = rule_tv.frequency, rule_tv.confidence

    f_result = f1 * f2

    # Confidence reduction through inference chain
    # Higher premise frequency means more confident deduction
    c_result = f1 * f2 * c1 * c2

    # Clamp to valid range
    c_result = max(0.001, min(0.999, c_result))

    return TruthValue(frequency=f_result, confidence=c_result)


def negation(tv: TruthValue) -> TruthValue:
    """NARS negation: invert frequency, preserve confidence.

    Uses the existing TruthValue.negate() method.

    Args:
        tv: Truth value to negate

    Returns:
        Negated truth value with frequency = 1 - f, confidence unchanged
    """
    return tv.negate()
