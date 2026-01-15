"""NARS-Datalog inference engine with semi-naive evaluation.

This module implements the main inference engine that combines Datalog's
bottom-up evaluation with NARS truth value propagation. Key features:

- Semi-naive evaluation for efficiency (only use new facts each iteration)
- NARS truth values propagate through rule applications
- Multiple derivations of same fact combine via NARS revision
- Stratified negation support
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from z3adapter.ikr.schema import TruthValue

from .fact_store import FactStore, GroundAtom, StoredFact
from .rule import InternalRule, compile_rules
from .unification import RuleAtom, Bindings, is_variable, unify_atom_with_fact
from .truth_functions import revise_multiple
from .truth_strategies import TruthStrategy, TruthFormulaName, get_strategy

if TYPE_CHECKING:
    from z3adapter.ikr.schema import IKR, Query

__all__ = [
    "NARSDatalogEngine",
    "InferenceResult",
    "from_ikr",
]

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result of running inference on a query.

    Attributes:
        query_atom: The atom that was queried
        found: Whether the query was derivable
        truth_value: Truth value if found
        iterations: Number of fixpoint iterations
        facts_derived: Total facts in the store after inference
        explanation: Human-readable explanation
    """

    query_atom: GroundAtom
    found: bool
    truth_value: TruthValue | None
    iterations: int
    facts_derived: int
    explanation: str = ""


class NARSDatalogEngine:
    """Datalog engine with NARS truth value propagation.

    Implements semi-naive evaluation where each derived fact carries
    a truth value computed from premise truth values using NARS formulas.

    The engine supports:
    - Loading facts and rules from IKR
    - Semi-naive fixpoint evaluation
    - NARS truth propagation through conjunctions and deductions
    - Evidence combination via revision when same fact derived multiple ways
    - Stratified negation (negation-as-failure)
    """

    def __init__(
        self,
        max_iterations: int = 100,
        min_confidence: float = 0.01,
        truth_formula: TruthFormulaName = "current",
    ) -> None:
        """Initialize the engine.

        Args:
            max_iterations: Maximum fixpoint iterations before stopping
            min_confidence: Minimum confidence to keep a derived fact
            truth_formula: Truth function strategy ("current", "opennars", "floor", "evidence")
        """
        self.max_iterations = max_iterations
        self.min_confidence = min_confidence
        self._truth_strategy: TruthStrategy = get_strategy(truth_formula)

        self._fact_store = FactStore()
        self._rules: list[InternalRule] = []
        self._strata: list[list[int]] = []
        self._iterations_run = 0

    @property
    def fact_store(self) -> FactStore:
        """Access the fact store."""
        return self._fact_store

    @property
    def rules(self) -> list[InternalRule]:
        """Access the compiled rules."""
        return self._rules

    def load_ikr(self, ikr: "IKR") -> None:
        """Load facts and rules from IKR.

        Handles epistemic contexts: facts with epistemic_context are stored
        as agent-specific beliefs, while facts without are objective facts.

        Args:
            ikr: The IKR to load
        """
        self._fact_store.clear()
        self._rules.clear()

        # Track agents from epistemic config
        self._agents: set[str] = set()
        if ikr.epistemic_config:
            self._agents = set(ikr.epistemic_config.agents)

        # Load base facts
        for fact in ikr.facts:
            # Extract agent from epistemic context (MVP: only top-level agent)
            agent = None
            if fact.epistemic_context:
                agent = fact.epistemic_context.agent
                self._agents.add(agent)

            atom = GroundAtom(
                predicate=fact.predicate,
                arguments=tuple(fact.arguments),
                negated=fact.negated,
                agent=agent,
            )

            # Get truth value (default to certain if not specified)
            tv = fact.truth_value or TruthValue(frequency=1.0, confidence=0.9)

            # Negated facts have inverted frequency semantically
            # but we store them as-is with the negated flag
            self._fact_store.add(atom, tv, source="base")

        # Compile rules
        self._rules = compile_rules(ikr.rules)

        # Compute stratification
        self._compute_strata()

        logger.debug(
            f"Loaded {self._fact_store.size()} facts and {len(self._rules)} rules "
            f"({len(self._agents)} agents)"
        )

    def run(self) -> int:
        """Run inference to fixpoint.

        Evaluates rules stratum by stratum, each to fixpoint.

        Returns:
            Total number of iterations across all strata
        """
        total_iterations = 0

        for stratum_idx, rule_indices in enumerate(self._strata):
            if rule_indices:
                logger.debug(f"Evaluating stratum {stratum_idx} with {len(rule_indices)} rules")
                iterations = self._evaluate_stratum(rule_indices)
                total_iterations += iterations

        self._iterations_run = total_iterations
        return total_iterations

    def query(self, q: "Query", agent: str | None = None) -> InferenceResult:
        """Query for a specific atom after inference.

        Runs inference if not already run, then looks up the query.
        Supports epistemic queries where agent specifies whose beliefs to query.

        Args:
            q: The query from IKR
            agent: Agent perspective for epistemic queries:
                   - None: Query objective facts only
                   - "*": Query all facts (objective + all agents)
                   - "alice": Query from Alice's perspective (objective + Alice's beliefs)

        Returns:
            InferenceResult with truth value if found
        """
        # Run inference if needed
        if self._iterations_run == 0:
            self.run()

        # For agent-specific queries, check both objective and agent-specific facts
        query_agent = None if agent == "*" else agent

        atom = GroundAtom(
            predicate=q.predicate,
            arguments=tuple(q.arguments),
            negated=q.negated,
            agent=query_agent,
        )

        # First try exact match with the agent
        stored = self._fact_store.get(atom)

        # If querying from an agent's perspective and no agent-specific fact,
        # fall back to objective fact
        if stored is None and agent is not None and agent != "*":
            objective_atom = atom.with_agent(None)
            stored = self._fact_store.get(objective_atom)

        if stored:
            return InferenceResult(
                query_atom=atom,
                found=True,
                truth_value=stored.truth_value,
                iterations=self._iterations_run,
                facts_derived=self._fact_store.size(),
                explanation=f"Derived with {stored.derivation_count} derivation(s), "
                f"source: {stored.source}"
                + (f", agent: {stored.atom.agent}" if stored.atom.agent else ""),
            )
        else:
            # Check if the positive version exists for negated queries
            if q.negated:
                positive_atom = atom.positive()
                positive_stored = self._fact_store.get(positive_atom)

                # Also check objective positive if querying from agent perspective
                if positive_stored is None and agent is not None and agent != "*":
                    objective_positive = positive_atom.with_agent(None)
                    positive_stored = self._fact_store.get(objective_positive)

                if positive_stored is None:
                    # Negation as failure: not(P) is true if P is not derivable
                    return InferenceResult(
                        query_atom=atom,
                        found=True,
                        truth_value=TruthValue(frequency=1.0, confidence=0.9),
                        iterations=self._iterations_run,
                        facts_derived=self._fact_store.size(),
                        explanation="Negation-as-failure: positive atom not derivable",
                    )

            return InferenceResult(
                query_atom=atom,
                found=False,
                truth_value=None,
                iterations=self._iterations_run,
                facts_derived=self._fact_store.size(),
                explanation="Query not derivable (closed-world assumption: False)",
            )

    def _evaluate_stratum(self, rule_indices: list[int]) -> int:
        """Evaluate rules in a stratum to fixpoint.

        Uses semi-naive evaluation: only newly derived facts are used
        in subsequent iterations to avoid redundant derivations.

        Args:
            rule_indices: Indices of rules in this stratum

        Returns:
            Number of iterations to reach fixpoint
        """
        rules = [self._rules[i] for i in rule_indices]

        # Initial delta: all current facts (for first iteration)
        delta: set[GroundAtom] = set(self._fact_store.all_atoms())

        for iteration in range(self.max_iterations):
            # Collect new derivations for this iteration
            new_derivations: dict[GroundAtom, list[TruthValue]] = {}

            for rule in rules:
                derivations = self._evaluate_rule(rule, delta)

                for atom, tv in derivations:
                    if tv.confidence >= self.min_confidence:
                        if atom not in new_derivations:
                            new_derivations[atom] = []
                        new_derivations[atom].append(tv)

            # Add new facts (with revision for duplicates)
            truly_new: set[GroundAtom] = set()

            for atom, tvs in new_derivations.items():
                # Combine multiple derivations via revision
                combined_tv = revise_multiple(tvs) if len(tvs) > 1 else tvs[0]

                is_new = self._fact_store.add(atom, combined_tv, source="derived")
                if is_new:
                    truly_new.add(atom)

            if not truly_new:
                # Fixpoint reached
                logger.debug(
                    f"Fixpoint reached in {iteration + 1} iterations, "
                    f"{self._fact_store.size()} facts"
                )
                return iteration + 1

            # Update delta to only new facts for next iteration
            delta = truly_new

        logger.warning(f"Max iterations ({self.max_iterations}) reached")
        return self.max_iterations

    def _evaluate_rule(
        self,
        rule: InternalRule,
        delta: set[GroundAtom],
    ) -> list[tuple[GroundAtom, TruthValue]]:
        """Evaluate a single rule, finding all derivations.

        For semi-naive evaluation, at least one body atom must match
        a fact from the delta set.

        Args:
            rule: The rule to evaluate
            delta: Set of newly derived facts

        Returns:
            List of (derived_atom, truth_value) pairs
        """
        if not rule.body:
            # Rule with empty body (unconditional) - shouldn't normally happen
            return []

        results: list[tuple[GroundAtom, TruthValue]] = []

        # Semi-naive: try each body position as the delta atom
        for delta_idx in range(len(rule.body)):
            derivations = self._evaluate_rule_with_delta_at(
                rule, delta_idx, delta
            )
            results.extend(derivations)

        return results

    def _evaluate_rule_with_delta_at(
        self,
        rule: InternalRule,
        delta_idx: int,
        delta: set[GroundAtom],
    ) -> list[tuple[GroundAtom, TruthValue]]:
        """Evaluate rule with delta atom at specific body position.

        Args:
            rule: The rule to evaluate
            delta_idx: Index of the body atom that must match delta
            delta: Set of delta facts

        Returns:
            List of derivations
        """
        results: list[tuple[GroundAtom, TruthValue]] = []
        delta_atom = rule.body[delta_idx]

        # For negated rule atoms, match against negated facts (explicit negatives)
        if delta_atom.negated:
            # Match against explicitly negated facts in delta
            positive_atom = RuleAtom(
                predicate=delta_atom.predicate,
                arguments=delta_atom.arguments,
                negated=False,
            )
            for stored_fact in self._fact_store.get_by_predicate(delta_atom.predicate):
                # Must be negated fact and in delta
                if not stored_fact.atom.negated:
                    continue
                if stored_fact.atom not in delta:
                    continue

                bindings = unify_atom_with_fact(
                    positive_atom, stored_fact.atom.positive(), {}
                )
                if bindings is None:
                    continue

                body_truths = [stored_fact.truth_value]
                remaining = rule.body[:delta_idx] + rule.body[delta_idx + 1:]
                all_solutions = self._match_body(remaining, bindings, rule.variables)

                for final_bindings, more_truths in all_solutions:
                    body_truths_full = body_truths + more_truths
                    body_tv = self._truth_strategy.conjunction(body_truths_full)
                    derived_tv = self._truth_strategy.deduction(body_tv, rule.rule_truth)

                    head_args = tuple(
                        final_bindings.get(arg, arg) for arg in rule.head.arguments
                    )
                    head_atom = GroundAtom(
                        predicate=rule.head.predicate,
                        arguments=head_args,
                        negated=rule.head.negated,
                    )
                    results.append((head_atom, derived_tv))

            return results

        # Match positive delta atom against positive facts
        for stored_fact in self._fact_store.get_by_predicate(delta_atom.predicate):
            # Must be in delta for semi-naive
            if stored_fact.atom not in delta:
                continue

            # Skip negated facts for positive atoms
            if stored_fact.atom.negated:
                continue

            bindings = unify_atom_with_fact(delta_atom, stored_fact.atom, {})
            if bindings is None:
                continue

            # Start truth values with the delta atom's truth
            body_truths = [stored_fact.truth_value]

            # Match remaining body atoms
            remaining = rule.body[:delta_idx] + rule.body[delta_idx + 1:]
            all_solutions = self._match_body(remaining, bindings, rule.variables)

            for final_bindings, more_truths in all_solutions:
                body_truths_full = body_truths + more_truths

                # Compute derived truth: conjunction of body + deduction
                body_tv = self._truth_strategy.conjunction(body_truths_full)
                derived_tv = self._truth_strategy.deduction(body_tv, rule.rule_truth)

                # Build head atom with bindings
                head_args = tuple(
                    final_bindings.get(arg, arg) for arg in rule.head.arguments
                )
                head_atom = GroundAtom(
                    predicate=rule.head.predicate,
                    arguments=head_args,
                    negated=rule.head.negated,
                )

                results.append((head_atom, derived_tv))

        return results

    def _match_body(
        self,
        body: list[RuleAtom],
        bindings: Bindings,
        variables: set[str],
    ) -> list[tuple[Bindings, list[TruthValue]]]:
        """Match remaining body atoms, collecting all solutions.

        Recursively matches body atoms, extending bindings and
        collecting truth values.

        Args:
            body: Remaining body atoms to match
            bindings: Current variable bindings
            variables: Set of variable names in the rule

        Returns:
            List of (final_bindings, truth_values) for each solution
        """
        if not body:
            return [(bindings, [])]

        atom = body[0]
        rest = body[1:]
        results: list[tuple[Bindings, list[TruthValue]]] = []

        # Handle negated atoms
        if atom.negated:
            # First, try to match explicitly negated facts
            # e.g., NOT contains_meat(F) matches fact contains_meat(plant_burger, negated=True)
            found_explicit_negative = False
            positive_atom = RuleAtom(
                predicate=atom.predicate,
                arguments=atom.arguments,
                negated=False,
            )

            for stored_fact in self._fact_store.get_by_predicate(atom.predicate):
                if stored_fact.atom.negated:
                    # This is an explicitly negated fact - try to match
                    new_bindings = unify_atom_with_fact(
                        positive_atom, stored_fact.atom.positive(), bindings
                    )
                    if new_bindings is not None:
                        found_explicit_negative = True
                        for rest_bindings, rest_truths in self._match_body(
                            rest, new_bindings, variables
                        ):
                            results.append(
                                (rest_bindings, [stored_fact.truth_value] + rest_truths)
                            )

            # If no explicit negatives found, use negation-as-failure
            if not found_explicit_negative:
                # Check if any positive match exists
                has_positive = False
                for stored_fact in self._fact_store.get_by_predicate(atom.predicate):
                    if stored_fact.atom.negated:
                        continue
                    if unify_atom_with_fact(positive_atom, stored_fact.atom, bindings):
                        has_positive = True
                        break

                if not has_positive:
                    # Negation succeeds via NAF
                    neg_tv = TruthValue(frequency=1.0, confidence=0.9)
                    for rest_bindings, rest_truths in self._match_body(
                        rest, bindings, variables
                    ):
                        results.append((rest_bindings, [neg_tv] + rest_truths))

        else:
            # Positive atom - find all matches
            for stored_fact in self._fact_store.get_by_predicate(atom.predicate):
                if stored_fact.atom.negated:
                    continue

                new_bindings = unify_atom_with_fact(atom, stored_fact.atom, bindings)
                if new_bindings is not None:
                    for rest_bindings, rest_truths in self._match_body(
                        rest, new_bindings, variables
                    ):
                        results.append(
                            (rest_bindings, [stored_fact.truth_value] + rest_truths)
                        )

        return results

    def _compute_strata(self) -> None:
        """Compute stratification for negation.

        Simple two-stratum approach:
        - Stratum 0: Rules without negation in body
        - Stratum 1: Rules with negation in body

        This ensures positive facts are derived before negation-as-failure
        is applied.
        """
        non_negated: list[int] = []
        with_negation: list[int] = []

        for i, rule in enumerate(self._rules):
            if rule.has_negation:
                with_negation.append(i)
            else:
                non_negated.append(i)

        self._strata = []
        if non_negated:
            self._strata.append(non_negated)
        if with_negation:
            self._strata.append(with_negation)

        logger.debug(
            f"Computed {len(self._strata)} strata: "
            f"{[len(s) for s in self._strata]} rules each"
        )


def from_ikr(
    ikr: "IKR",
    truth_formula: TruthFormulaName = "current",
) -> NARSDatalogEngine:
    """Convenience function to create engine from IKR.

    Args:
        ikr: The IKR to load
        truth_formula: Truth function strategy ("current", "opennars", "floor", "evidence")

    Returns:
        Initialized NARSDatalogEngine with facts and rules loaded
    """
    engine = NARSDatalogEngine(truth_formula=truth_formula)
    engine.load_ikr(ikr)
    return engine
