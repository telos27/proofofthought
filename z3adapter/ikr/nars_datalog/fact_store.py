"""Indexed fact storage with NARS truth values.

This module provides efficient storage for ground atoms with associated
truth values, supporting lookup by predicate and revision when the same
fact is derived multiple times.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator

from z3adapter.ikr.schema import TruthValue

__all__ = [
    "GroundAtom",
    "StoredFact",
    "FactStore",
]


@dataclass(frozen=True)
class GroundAtom:
    """A ground (variable-free) atom.

    Hashable for use as dictionary key. Represents a specific
    predicate applied to specific constants.

    Attributes:
        predicate: The relation/predicate name
        arguments: Tuple of constant arguments (tuple for hashability)
        negated: Whether this is a negated atom
    """

    predicate: str
    arguments: tuple[str, ...]
    negated: bool = False

    def __str__(self) -> str:
        neg = "NOT " if self.negated else ""
        args = ", ".join(self.arguments)
        return f"{neg}{self.predicate}({args})"

    def __repr__(self) -> str:
        return self.__str__()

    def positive(self) -> GroundAtom:
        """Return the positive version of this atom."""
        if not self.negated:
            return self
        return GroundAtom(
            predicate=self.predicate,
            arguments=self.arguments,
            negated=False,
        )


@dataclass
class StoredFact:
    """A fact with its truth value and derivation metadata.

    Attributes:
        atom: The ground atom
        truth_value: NARS truth value (frequency, confidence)
        source: Where this fact came from ("base", "derived", "revised")
        derivation_count: Number of times this fact was derived
    """

    atom: GroundAtom
    truth_value: TruthValue
    source: str = "base"
    derivation_count: int = 1

    def __str__(self) -> str:
        return f"{self.atom} : <{self.truth_value.frequency:.3f}, {self.truth_value.confidence:.3f}>"


class FactStore:
    """Indexed storage for facts with NARS truth values.

    Supports efficient lookup by:
    - Exact atom (predicate + all arguments)
    - Predicate only (for rule matching)
    - Predicate + first argument (for join optimization)

    When a fact is added that already exists, truth values are
    combined using NARS revision (evidence pooling).
    """

    def __init__(self) -> None:
        """Initialize empty fact store."""
        # Primary storage: atom -> StoredFact
        self._facts: dict[GroundAtom, StoredFact] = {}

        # Index by predicate for rule body matching
        self._by_predicate: dict[str, set[GroundAtom]] = defaultdict(set)

        # Index by (predicate, first_arg) for join optimization
        self._by_pred_arg0: dict[tuple[str, str], set[GroundAtom]] = defaultdict(set)

    def add(
        self,
        atom: GroundAtom,
        truth_value: TruthValue,
        source: str = "base",
    ) -> bool:
        """Add a fact or revise existing fact's truth value.

        If the fact already exists, revises the truth value using
        NARS revision (evidence pooling). This ensures that multiple
        derivations of the same fact increase confidence.

        Args:
            atom: The ground atom to add
            truth_value: Truth value for this derivation
            source: Source of the fact ("base", "derived")

        Returns:
            True if this is a new fact, False if revised existing
        """
        if atom in self._facts:
            # Fact exists - revise truth value
            existing = self._facts[atom]
            revised_tv = existing.truth_value.revise(truth_value)
            self._facts[atom] = StoredFact(
                atom=atom,
                truth_value=revised_tv,
                source="revised",
                derivation_count=existing.derivation_count + 1,
            )
            return False
        else:
            # New fact - add and index
            self._facts[atom] = StoredFact(
                atom=atom,
                truth_value=truth_value,
                source=source,
                derivation_count=1,
            )
            self._index_fact(atom)
            return True

    def get(self, atom: GroundAtom) -> StoredFact | None:
        """Get a stored fact by exact atom.

        Args:
            atom: The ground atom to look up

        Returns:
            StoredFact if found, None otherwise
        """
        return self._facts.get(atom)

    def get_truth(self, atom: GroundAtom) -> TruthValue | None:
        """Get just the truth value for an atom.

        Args:
            atom: The ground atom to look up

        Returns:
            TruthValue if found, None otherwise
        """
        stored = self._facts.get(atom)
        return stored.truth_value if stored else None

    def get_by_predicate(self, predicate: str) -> Iterator[StoredFact]:
        """Get all facts with a given predicate.

        Args:
            predicate: The predicate name

        Yields:
            StoredFact objects for matching facts
        """
        for atom in self._by_predicate.get(predicate, set()):
            yield self._facts[atom]

    def get_by_predicate_arg0(
        self, predicate: str, arg0: str
    ) -> Iterator[StoredFact]:
        """Get facts with given predicate and first argument.

        Useful for join optimization when first argument is bound.

        Args:
            predicate: The predicate name
            arg0: The first argument value

        Yields:
            StoredFact objects for matching facts
        """
        key = (predicate, arg0)
        for atom in self._by_pred_arg0.get(key, set()):
            yield self._facts[atom]

    def contains(self, atom: GroundAtom) -> bool:
        """Check if a fact exists.

        Args:
            atom: The ground atom to check

        Returns:
            True if the fact exists
        """
        return atom in self._facts

    def all_facts(self) -> Iterator[StoredFact]:
        """Iterate over all stored facts.

        Yields:
            All StoredFact objects
        """
        return iter(self._facts.values())

    def all_atoms(self) -> Iterator[GroundAtom]:
        """Iterate over all atom keys.

        Yields:
            All GroundAtom keys
        """
        return iter(self._facts.keys())

    def size(self) -> int:
        """Get number of facts stored.

        Returns:
            Number of unique facts
        """
        return len(self._facts)

    def clear(self) -> None:
        """Remove all facts."""
        self._facts.clear()
        self._by_predicate.clear()
        self._by_pred_arg0.clear()

    def _index_fact(self, atom: GroundAtom) -> None:
        """Add atom to indices.

        Args:
            atom: The atom to index
        """
        self._by_predicate[atom.predicate].add(atom)
        if atom.arguments:
            key = (atom.predicate, atom.arguments[0])
            self._by_pred_arg0[key].add(atom)
