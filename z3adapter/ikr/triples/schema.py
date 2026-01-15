"""Triple schema for knowledge extraction.

Follows Wikidata's philosophy: predicates are fixed schema, entities emerge from content.
Uses 7 generic predicates covering most semantic relationships.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from z3adapter.ikr.schema import TruthValue


class Predicate(str, Enum):
    """Fixed predicate vocabulary (Wikidata-inspired).

    Seven generic predicates covering most semantic relationships:
    - Taxonomy: is_a
    - Structure: part_of
    - Attributes: has
    - Causation: causes, prevents
    - Epistemic: believes
    - Catch-all: related_to
    """

    IS_A = "is_a"  # Taxonomy: X is a type of Y
    PART_OF = "part_of"  # Structure: X is part of Y
    HAS = "has"  # Attributes: X has property Y
    CAUSES = "causes"  # Causation: X causes Y
    PREVENTS = "prevents"  # Negative causation: X prevents Y
    BELIEVES = "believes"  # Epistemic: X believes Y
    RELATED_TO = "related_to"  # Catch-all


# Predicate opposites for contradiction detection
PREDICATE_OPPOSITES: dict[Predicate, Predicate] = {
    Predicate.CAUSES: Predicate.PREVENTS,
    Predicate.PREVENTS: Predicate.CAUSES,
}


@dataclass
class Triple:
    """A semantic triple with optional reification support.

    Subject and object can reference:
    - Entities: bare strings like "stress", "memory"
    - Other triples: prefixed with "t:" like "t:t1"

    Example:
        # Simple fact
        Triple(id="t1", subject="stress", predicate=Predicate.CAUSES, object="anxiety")

        # Nested belief (Alice believes t1)
        Triple(id="t2", subject="alice", predicate=Predicate.BELIEVES, object="t:t1")

        # Negated fact
        Triple(id="t3", subject="exercise", predicate=Predicate.CAUSES, object="stress", negated=True)
    """

    id: str  # Unique ID (e.g., "t1", "t2")
    subject: str  # Entity or triple reference (t:xxx)
    predicate: Predicate  # One of 7 predicates
    object: str  # Entity or triple reference (t:xxx)

    # Negation
    negated: bool = False  # "X does NOT predicate Y"

    # Uncertainty (NARS)
    truth: Optional[TruthValue] = None

    # Provenance
    source: Optional[str] = None  # "Zimbardo 2017 p.42"
    surface_form: Optional[str] = None  # Original text

    @staticmethod
    def is_triple_reference(ref: str) -> bool:
        """Check if reference points to another triple."""
        return ref.startswith("t:")

    @staticmethod
    def get_triple_id(ref: str) -> Optional[str]:
        """Extract triple ID from reference."""
        if Triple.is_triple_reference(ref):
            return ref[2:]
        return None

    @property
    def subject_is_triple(self) -> bool:
        """Check if subject references another triple."""
        return self.is_triple_reference(self.subject)

    @property
    def object_is_triple(self) -> bool:
        """Check if object references another triple."""
        return self.is_triple_reference(self.object)

    def __hash__(self) -> int:
        """Hash based on ID for set operations."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on ID."""
        if not isinstance(other, Triple):
            return False
        return self.id == other.id


@dataclass
class TripleStore:
    """In-memory store for triples with indexing.

    Provides fast lookup by subject, predicate, or object via indexes.
    Supports triple references for reification (multi-level beliefs).

    Example:
        store = TripleStore()
        store.add(Triple(id="t1", subject="cat", predicate=Predicate.IS_A, object="mammal"))
        store.add(Triple(id="t2", subject="mammal", predicate=Predicate.IS_A, object="animal"))

        # Query by subject
        cat_facts = store.query(subject="cat")

        # Query by predicate
        taxonomies = store.query(predicate=Predicate.IS_A)

        # Resolve triple reference
        resolved = store.resolve("t:t1")  # Returns Triple with id="t1"
    """

    triples: dict[str, Triple] = field(default_factory=dict)

    # Indexes for fast lookup
    by_subject: dict[str, list[str]] = field(default_factory=dict)
    by_predicate: dict[Predicate, list[str]] = field(default_factory=dict)
    by_object: dict[str, list[str]] = field(default_factory=dict)

    def add(self, triple: Triple) -> None:
        """Add triple and update indexes.

        If a triple with the same ID exists, it will be replaced.
        """
        # Remove from indexes if replacing
        if triple.id in self.triples:
            self._remove_from_indexes(self.triples[triple.id])

        self.triples[triple.id] = triple

        # Update indexes
        self.by_subject.setdefault(triple.subject, []).append(triple.id)
        self.by_predicate.setdefault(triple.predicate, []).append(triple.id)
        self.by_object.setdefault(triple.object, []).append(triple.id)

    def _remove_from_indexes(self, triple: Triple) -> None:
        """Remove triple from all indexes."""
        if triple.subject in self.by_subject:
            self.by_subject[triple.subject] = [
                tid for tid in self.by_subject[triple.subject] if tid != triple.id
            ]
        if triple.predicate in self.by_predicate:
            self.by_predicate[triple.predicate] = [
                tid for tid in self.by_predicate[triple.predicate] if tid != triple.id
            ]
        if triple.object in self.by_object:
            self.by_object[triple.object] = [
                tid for tid in self.by_object[triple.object] if tid != triple.id
            ]

    def get(self, triple_id: str) -> Optional[Triple]:
        """Get triple by ID."""
        return self.triples.get(triple_id)

    def remove(self, triple_id: str) -> Optional[Triple]:
        """Remove triple by ID and return it."""
        triple = self.triples.pop(triple_id, None)
        if triple:
            self._remove_from_indexes(triple)
        return triple

    def resolve(self, ref: str) -> Triple | str:
        """Resolve reference to triple or return entity string.

        Args:
            ref: Entity string or triple reference (t:xxx)

        Returns:
            Triple if ref is a valid triple reference, else the original string
        """
        if ref.startswith("t:"):
            triple_id = ref[2:]
            return self.triples.get(triple_id, ref)
        return ref

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[Predicate] = None,
        obj: Optional[str] = None,
    ) -> list[Triple]:
        """Query triples by subject/predicate/object pattern.

        Args:
            subject: Filter by subject (exact match)
            predicate: Filter by predicate
            obj: Filter by object (exact match)

        Returns:
            List of matching triples
        """
        candidates = set(self.triples.keys())

        if subject is not None:
            candidates &= set(self.by_subject.get(subject, []))
        if predicate is not None:
            candidates &= set(self.by_predicate.get(predicate, []))
        if obj is not None:
            candidates &= set(self.by_object.get(obj, []))

        return [self.triples[tid] for tid in candidates]

    def __len__(self) -> int:
        """Return number of triples in store."""
        return len(self.triples)

    def __contains__(self, triple_id: str) -> bool:
        """Check if triple ID exists in store."""
        return triple_id in self.triples

    def __iter__(self):
        """Iterate over all triples."""
        return iter(self.triples.values())
