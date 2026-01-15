"""Pluggable truth function strategies for NARS-Datalog.

This module provides different truth value computation strategies to address
confidence degradation in long inference chains:

- current: Original formula (c = f1*f2*c1*c2) - aggressive degradation
- opennars: OpenNARS-style (c = c1*c2) - frequency-independent confidence
- floor: Confidence floor variant - ensures minimum confidence
- evidence: Evidence-based using evidence pooling

Usage:
    from z3adapter.ikr.nars_datalog.truth_strategies import get_strategy

    strategy = get_strategy("opennars")
    result = strategy.deduction(premise_tv, rule_tv)

    # Or with the engine:
    engine = NARSDatalogEngine(truth_formula="opennars")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from z3adapter.ikr.schema import TruthValue

__all__ = [
    "TruthStrategy",
    "CurrentStrategy",
    "OpenNARSStrategy",
    "FloorStrategy",
    "EvidenceStrategy",
    "TruthFormulaName",
    "get_strategy",
    "STRATEGIES",
]

TruthFormulaName = Literal["current", "opennars", "floor", "evidence"]


class TruthStrategy(ABC):
    """Abstract base for truth computation strategies."""

    @abstractmethod
    def deduction(self, premise_tv: TruthValue, rule_tv: TruthValue) -> TruthValue:
        """Compute deduction truth value.

        Args:
            premise_tv: Truth value of the premise (matched body atoms)
            rule_tv: Truth value of the rule itself

        Returns:
            Truth value for the derived conclusion
        """
        pass

    @abstractmethod
    def conjunction(self, truths: list[TruthValue]) -> TruthValue:
        """Compute conjunction truth value.

        Args:
            truths: Truth values of conjuncts

        Returns:
            Combined truth value representing "all are true"
        """
        pass


class CurrentStrategy(TruthStrategy):
    """Original NARS formula with aggressive confidence degradation.

    Deduction: f = f1 * f2, c = f1 * f2 * c1 * c2
    Conjunction: f = product(fi), c = product(ci)

    This matches the standard NARS deduction formula but degrades
    confidence quickly through inference chains.
    """

    def deduction(self, premise_tv: TruthValue, rule_tv: TruthValue) -> TruthValue:
        f1, c1 = premise_tv.frequency, premise_tv.confidence
        f2, c2 = rule_tv.frequency, rule_tv.confidence

        f_result = f1 * f2
        c_result = f1 * f2 * c1 * c2

        c_result = max(0.001, min(0.999, c_result))
        return TruthValue(frequency=f_result, confidence=c_result)

    def conjunction(self, truths: list[TruthValue]) -> TruthValue:
        if not truths:
            return TruthValue(frequency=1.0, confidence=0.9)

        f_result = 1.0
        c_result = 1.0

        for tv in truths:
            f_result *= tv.frequency
            c_result *= tv.confidence

        c_result = max(0.001, min(0.999, c_result))
        return TruthValue(frequency=f_result, confidence=c_result)


class OpenNARSStrategy(TruthStrategy):
    """OpenNARS-style formula where confidence is independent of frequency.

    Deduction: f = f1 * f2, c = c1 * c2
    Conjunction: f = product(fi), c = product(ci)

    This preserves confidence better through inference chains because
    low frequency doesn't reduce confidence.
    """

    def deduction(self, premise_tv: TruthValue, rule_tv: TruthValue) -> TruthValue:
        f1, c1 = premise_tv.frequency, premise_tv.confidence
        f2, c2 = rule_tv.frequency, rule_tv.confidence

        f_result = f1 * f2
        c_result = c1 * c2  # Frequency doesn't affect confidence

        c_result = max(0.001, min(0.999, c_result))
        return TruthValue(frequency=f_result, confidence=c_result)

    def conjunction(self, truths: list[TruthValue]) -> TruthValue:
        if not truths:
            return TruthValue(frequency=1.0, confidence=0.9)

        f_result = 1.0
        c_result = 1.0

        for tv in truths:
            f_result *= tv.frequency
            c_result *= tv.confidence

        c_result = max(0.001, min(0.999, c_result))
        return TruthValue(frequency=f_result, confidence=c_result)


class FloorStrategy(TruthStrategy):
    """Confidence floor variant that ensures minimum confidence survives.

    Deduction: f = f1 * f2, c = max(floor, c1 * c2)
    Conjunction: f = product(fi), c = max(floor, product(ci))

    This prevents confidence from collapsing to zero in long chains.

    Args:
        floor: Minimum confidence value (default: 0.1)
    """

    def __init__(self, floor: float = 0.1) -> None:
        self.floor = floor

    def deduction(self, premise_tv: TruthValue, rule_tv: TruthValue) -> TruthValue:
        f1, c1 = premise_tv.frequency, premise_tv.confidence
        f2, c2 = rule_tv.frequency, rule_tv.confidence

        f_result = f1 * f2
        c_result = max(self.floor, c1 * c2)

        c_result = max(0.001, min(0.999, c_result))
        return TruthValue(frequency=f_result, confidence=c_result)

    def conjunction(self, truths: list[TruthValue]) -> TruthValue:
        if not truths:
            return TruthValue(frequency=1.0, confidence=0.9)

        f_result = 1.0
        c_result = 1.0

        for tv in truths:
            f_result *= tv.frequency
            c_result *= tv.confidence

        c_result = max(self.floor, c_result)
        c_result = max(0.001, min(0.999, c_result))
        return TruthValue(frequency=f_result, confidence=c_result)


class EvidenceStrategy(TruthStrategy):
    """Evidence-based strategy using evidence pooling.

    Converts truth values to evidence counts, pools them, then converts back.
    This uses the same evidence model as NARS revision.

    Deduction: Confidence based on minimum evidence (weakest link principle)
    Conjunction: Confidence based on combined evidence

    Args:
        k: Evidential horizon constant (default: 1.0)
    """

    def __init__(self, k: float = 1.0) -> None:
        self.k = k

    def deduction(self, premise_tv: TruthValue, rule_tv: TruthValue) -> TruthValue:
        f1, c1 = premise_tv.frequency, premise_tv.confidence
        f2, c2 = rule_tv.frequency, rule_tv.confidence

        f_result = f1 * f2

        # Convert to evidence counts
        w1_pos, w1_total = premise_tv.to_evidence(self.k)
        w2_pos, w2_total = rule_tv.to_evidence(self.k)

        # Use minimum evidence (weakest link)
        if w1_total == float("inf") or w2_total == float("inf"):
            # High confidence case
            c_result = min(c1, c2)
        else:
            w_total = min(w1_total, w2_total)
            c_result = w_total / (w_total + self.k) if w_total > 0 else 0.001

        c_result = max(0.001, min(0.999, c_result))
        return TruthValue(frequency=f_result, confidence=c_result)

    def conjunction(self, truths: list[TruthValue]) -> TruthValue:
        if not truths:
            return TruthValue(frequency=1.0, confidence=0.9)

        f_result = 1.0
        min_w_total = float("inf")

        for tv in truths:
            f_result *= tv.frequency
            w_pos, w_total = tv.to_evidence(self.k)
            if w_total < min_w_total:
                min_w_total = w_total

        # Convert back from evidence
        if min_w_total == float("inf"):
            c_result = 0.999
        elif min_w_total == 0:
            c_result = 0.001
        else:
            c_result = min_w_total / (min_w_total + self.k)

        c_result = max(0.001, min(0.999, c_result))
        return TruthValue(frequency=f_result, confidence=c_result)


# Pre-built strategy instances
STRATEGIES: dict[str, TruthStrategy] = {
    "current": CurrentStrategy(),
    "opennars": OpenNARSStrategy(),
    "floor": FloorStrategy(),
    "evidence": EvidenceStrategy(),
}


def get_strategy(name: TruthFormulaName = "current") -> TruthStrategy:
    """Get truth strategy by name.

    Args:
        name: Strategy name ("current", "opennars", "floor", "evidence")

    Returns:
        TruthStrategy instance

    Raises:
        KeyError: If strategy name is unknown
    """
    if name not in STRATEGIES:
        raise KeyError(f"Unknown truth strategy: {name}. Valid: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]
