"""Internal rule representation for Datalog inference.

This module converts IKR rules to an internal representation optimized
for evaluation. IKR rules are implications (antecedent => consequent)
which become Datalog Horn clauses (consequent :- antecedent).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from z3adapter.ikr.schema import TruthValue
from .unification import RuleAtom, is_variable

if TYPE_CHECKING:
    from z3adapter.ikr.schema import Rule as IKRRule, RuleCondition

__all__ = [
    "InternalRule",
    "compile_rules",
]

logger = logging.getLogger(__name__)


@dataclass
class InternalRule:
    """Internal representation of a Datalog rule.

    Represents: head :- body_atom1, body_atom2, ...

    Compiled from IKR Rule where:
    - antecedent becomes body (conjunction of atoms)
    - consequent becomes head (single atom)

    Attributes:
        name: Optional rule name for debugging
        head: The rule head (conclusion)
        body: List of body atoms (premises, conjunction)
        variables: All variables appearing in the rule
        has_negation: Whether any body atom is negated
        rule_truth: Truth value of the rule itself
    """

    name: str | None
    head: RuleAtom
    body: list[RuleAtom]
    variables: set[str]
    has_negation: bool = False
    rule_truth: TruthValue = field(
        default_factory=lambda: TruthValue(frequency=1.0, confidence=0.9)
    )

    def __str__(self) -> str:
        head_str = str(self.head)
        if self.body:
            body_str = ", ".join(str(atom) for atom in self.body)
            return f"{head_str} :- {body_str}."
        return f"{head_str}."

    @classmethod
    def from_ikr_rule(cls, ikr_rule: "IKRRule") -> "InternalRule | None":
        """Convert IKR Rule to internal representation.

        Args:
            ikr_rule: The IKR rule to convert

        Returns:
            InternalRule if conversion succeeds, None if rule cannot
            be converted (e.g., constraint rules without implication)
        """
        if not ikr_rule.is_implication():
            logger.debug(f"Skipping non-implication rule: {ikr_rule.name}")
            return None

        # Extract head from consequent
        head = _condition_to_atom(ikr_rule.consequent)
        if head is None:
            logger.warning(
                f"Cannot convert consequent to atom: {ikr_rule.consequent}"
            )
            return None

        # Extract body from antecedent
        body = _condition_to_atoms(ikr_rule.antecedent)

        # Collect variables from quantified_vars
        variables = {var.name for var in ikr_rule.quantified_vars}

        # Also collect any uppercase arguments as variables
        for atom in [head] + body:
            for arg in atom.arguments:
                if is_variable(arg):
                    variables.add(arg)

        # Check for negation in body
        has_negation = any(atom.negated for atom in body)

        return cls(
            name=ikr_rule.name,
            head=head,
            body=body,
            variables=variables,
            has_negation=has_negation,
            rule_truth=TruthValue(frequency=1.0, confidence=0.9),
        )


def _condition_to_atom(cond: "RuleCondition | None") -> RuleAtom | None:
    """Convert a simple RuleCondition to RuleAtom.

    Args:
        cond: The condition to convert

    Returns:
        RuleAtom if condition is simple, None otherwise
    """
    if cond is None:
        return None

    if not cond.is_simple():
        logger.warning(f"Cannot convert compound condition to atom: {cond}")
        return None

    return RuleAtom(
        predicate=cond.predicate,
        arguments=tuple(cond.arguments),
        negated=cond.negated,
    )


def _condition_to_atoms(cond: "RuleCondition | None") -> list[RuleAtom]:
    """Convert a RuleCondition (possibly conjunction) to list of atoms.

    Handles conjunctions by flattening them. Disjunctions are not
    fully supported - only the first disjunct is used.

    Args:
        cond: The condition to convert

    Returns:
        List of RuleAtom objects
    """
    if cond is None:
        return []

    # Handle conjunction (AND)
    if cond.and_:
        result = []
        for sub in cond.and_:
            result.extend(_condition_to_atoms(sub))
        return result

    # Handle disjunction (OR) - use first disjunct only
    if cond.or_:
        logger.warning(
            f"Disjunction not fully supported, using first disjunct: {cond}"
        )
        if cond.or_:
            return _condition_to_atoms(cond.or_[0])
        return []

    # Simple atom
    if cond.is_simple():
        atom = _condition_to_atom(cond)
        return [atom] if atom else []

    return []


def compile_rules(ikr_rules: list["IKRRule"]) -> list[InternalRule]:
    """Compile a list of IKR rules to internal representation.

    Args:
        ikr_rules: List of IKR rules

    Returns:
        List of successfully compiled InternalRule objects
    """
    result = []
    for rule in ikr_rules:
        internal = InternalRule.from_ikr_rule(rule)
        if internal:
            result.append(internal)
    return result
