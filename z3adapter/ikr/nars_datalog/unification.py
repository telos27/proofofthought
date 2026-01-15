"""Variable binding and unification for Datalog inference.

This module handles variable substitution and matching for rule evaluation.
Datalog has restricted unification (no function symbols), so simple
dictionary-based bindings are sufficient.

Variable convention: Uppercase first letter (e.g., X, Y, Person)
Constants: Lowercase first letter or quoted strings
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, TYPE_CHECKING

from z3adapter.ikr.schema import TruthValue

if TYPE_CHECKING:
    from .fact_store import FactStore, GroundAtom, StoredFact

__all__ = [
    "Bindings",
    "RuleAtom",
    "is_variable",
    "unify_atom_with_fact",
    "find_all_bindings",
]

# Type alias for variable bindings: variable name -> constant value
Bindings = dict[str, str]


def is_variable(arg: str) -> bool:
    """Check if an argument is a variable.

    Variables are identified by uppercase first letter.

    Args:
        arg: The argument string

    Returns:
        True if this is a variable
    """
    return bool(arg) and arg[0].isupper()


@dataclass
class RuleAtom:
    """An atom that may contain variables.

    Used in rule heads and bodies where variables represent
    universally quantified positions.

    Attributes:
        predicate: The relation/predicate name
        arguments: Tuple of arguments (may include variables)
        negated: Whether this is a negated atom
    """

    predicate: str
    arguments: tuple[str, ...]
    negated: bool = False

    def __str__(self) -> str:
        neg = "!" if self.negated else ""
        args = ", ".join(self.arguments)
        return f"{neg}{self.predicate}({args})"

    def is_ground(self, bindings: Bindings) -> bool:
        """Check if all variables are bound.

        Args:
            bindings: Current variable bindings

        Returns:
            True if all variables have values in bindings
        """
        return all(
            arg in bindings or not is_variable(arg)
            for arg in self.arguments
        )

    def substitute(self, bindings: Bindings) -> tuple[str, ...]:
        """Apply bindings to get ground arguments.

        Args:
            bindings: Variable bindings to apply

        Returns:
            Tuple with variables replaced by bound values
        """
        return tuple(bindings.get(arg, arg) for arg in self.arguments)

    def get_variables(self) -> set[str]:
        """Get all variables in this atom.

        Returns:
            Set of variable names
        """
        return {arg for arg in self.arguments if is_variable(arg)}


def unify_atom_with_fact(
    rule_atom: RuleAtom,
    fact_atom: "GroundAtom",
    current_bindings: Bindings,
) -> Bindings | None:
    """Try to unify a rule atom with a ground fact.

    Extends current bindings if unification succeeds. Returns None
    if unification fails (predicate mismatch, arity mismatch, or
    binding conflict).

    Args:
        rule_atom: Atom from rule body (may have variables)
        fact_atom: Ground atom from fact store
        current_bindings: Existing variable bindings

    Returns:
        Extended bindings if unification succeeds, None otherwise
    """
    # Predicate must match exactly
    if rule_atom.predicate != fact_atom.predicate:
        return None

    # Arity must match
    if len(rule_atom.arguments) != len(fact_atom.arguments):
        return None

    # Note: negation handled separately in rule evaluation
    # We match predicates regardless of negation here

    # Try to extend bindings
    new_bindings = dict(current_bindings)

    for rule_arg, fact_arg in zip(rule_atom.arguments, fact_atom.arguments):
        if is_variable(rule_arg):
            if rule_arg in new_bindings:
                # Already bound - must match
                if new_bindings[rule_arg] != fact_arg:
                    return None
            else:
                # New binding
                new_bindings[rule_arg] = fact_arg
        else:
            # Constant - must match exactly
            if rule_arg != fact_arg:
                return None

    return new_bindings


def find_all_bindings(
    rule_atom: RuleAtom,
    fact_store: "FactStore",
    current_bindings: Bindings,
) -> Iterator[tuple[Bindings, TruthValue]]:
    """Find all ways to match a rule atom against the fact store.

    Iterates through facts with matching predicate and tries to
    unify each one, yielding extended bindings and truth values.

    Args:
        rule_atom: Atom from rule body
        fact_store: Current fact store
        current_bindings: Existing bindings to extend

    Yields:
        (extended_bindings, fact_truth_value) pairs
    """
    for stored_fact in fact_store.get_by_predicate(rule_atom.predicate):
        # Skip if negation doesn't match what we're looking for
        if stored_fact.atom.negated != rule_atom.negated:
            continue

        bindings = unify_atom_with_fact(
            rule_atom, stored_fact.atom, current_bindings
        )
        if bindings is not None:
            yield bindings, stored_fact.truth_value
