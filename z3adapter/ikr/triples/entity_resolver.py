"""Entity resolution for fuzzy entity matching in triple extraction.

This module provides entity resolution capabilities for normalizing surface
forms to canonical entities using fuzzy matching.

Key features:
- Pluggable similarity function (default: combined_lexical_similarity)
- Configurable match threshold
- Surface form tracking (remembers aliases)
- Normalization (lowercase, underscores)

Example:
    from z3adapter.ikr.triples import EntityResolver, EntityMatch

    resolver = EntityResolver(threshold=0.8)
    resolver.add_entity("working_memory", ["WM", "short-term memory"])

    match = resolver.resolve("short term memory")
    print(match.canonical)  # "working_memory"
    print(match.similarity)  # 1.0 (exact match after normalization)

References:
    - Uses similarity functions from z3adapter.ikr.fuzzy_nars
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class EntityMatch:
    """Result of entity resolution.

    Attributes:
        canonical: The canonical entity name
        surface_form: The original surface form that was resolved
        similarity: Match confidence score in [0, 1]
        is_new: Whether this entity was newly added (no existing match found)
    """

    canonical: str
    surface_form: str
    similarity: float
    is_new: bool = False

    def __repr__(self) -> str:
        new_str = " (new)" if self.is_new else ""
        return f"EntityMatch('{self.canonical}' <- '{self.surface_form}', sim={self.similarity:.3f}{new_str})"


@dataclass
class EntityResolver:
    """Resolve surface forms to canonical entities using fuzzy matching.

    This class maintains a registry of known entities and their surface forms,
    and can resolve new surface forms to existing entities based on similarity.

    Attributes:
        threshold: Minimum similarity score to consider a match (default: 0.8)
        entities: Dict mapping canonical names to sets of known surface forms

    Example:
        resolver = EntityResolver(threshold=0.8)
        resolver.add_entity("anxiety_disorder", ["anxiety", "anxious disorder"])

        # Exact match (normalized)
        m1 = resolver.resolve("Anxiety Disorder")
        assert m1.canonical == "anxiety_disorder"
        assert m1.similarity == 1.0

        # Fuzzy match
        m2 = resolver.resolve("anxiety disorders")  # plural
        assert m2.canonical == "anxiety_disorder"
        assert m2.similarity > 0.8

        # New entity (below threshold)
        m3 = resolver.resolve("depression")
        assert m3.canonical == "depression"
        assert m3.is_new == True
    """

    threshold: float = 0.8
    entities: dict[str, set[str]] = field(default_factory=dict)
    _similarity_fn: Optional[Callable[[str, str], float]] = field(
        default=None, repr=False
    )

    def __post_init__(self) -> None:
        """Initialize similarity function if not provided."""
        if self._similarity_fn is None:
            # Import here to avoid circular dependencies
            from z3adapter.ikr.fuzzy_nars import combined_lexical_similarity

            self._similarity_fn = combined_lexical_similarity

    @property
    def similarity_fn(self) -> Callable[[str, str], float]:
        """Get the similarity function."""
        if self._similarity_fn is None:
            from z3adapter.ikr.fuzzy_nars import combined_lexical_similarity

            self._similarity_fn = combined_lexical_similarity
        return self._similarity_fn

    @similarity_fn.setter
    def similarity_fn(self, fn: Callable[[str, str], float]) -> None:
        """Set the similarity function."""
        self._similarity_fn = fn

    def add_entity(
        self, canonical: str, surface_forms: Optional[list[str]] = None
    ) -> None:
        """Register an entity with optional surface forms.

        Args:
            canonical: The canonical entity name (will be normalized)
            surface_forms: Optional list of known aliases/surface forms
        """
        normalized = self._normalize(canonical)
        if normalized not in self.entities:
            self.entities[normalized] = set()

        # Add the canonical form itself as a surface form
        self.entities[normalized].add(normalized)

        if surface_forms:
            for form in surface_forms:
                norm_form = self._normalize(form)
                self.entities[normalized].add(norm_form)

    def add_surface_form(self, canonical: str, surface_form: str) -> None:
        """Add a surface form to an existing entity.

        Args:
            canonical: The canonical entity name
            surface_form: The surface form to add

        Raises:
            KeyError: If canonical entity doesn't exist
        """
        normalized_canonical = self._normalize(canonical)
        if normalized_canonical not in self.entities:
            raise KeyError(f"Entity '{canonical}' not found. Use add_entity() first.")
        self.entities[normalized_canonical].add(self._normalize(surface_form))

    def resolve(self, surface_form: str, auto_add: bool = True) -> EntityMatch:
        """Find the canonical entity for a surface form.

        Resolution strategy:
        1. Exact match (after normalization) - returns immediately
        2. Fuzzy match against all canonical names and their surface forms
        3. If best match >= threshold, use that entity
        4. Otherwise, create new entity (if auto_add=True)

        Args:
            surface_form: The surface form to resolve
            auto_add: If True, automatically add unmatched terms as new entities

        Returns:
            EntityMatch with canonical name, original surface form, and similarity
        """
        normalized = self._normalize(surface_form)

        # 1. Exact match on canonical name
        if normalized in self.entities:
            return EntityMatch(
                canonical=normalized,
                surface_form=surface_form,
                similarity=1.0,
                is_new=False,
            )

        # 2. Exact match in surface forms
        for canonical, forms in self.entities.items():
            if normalized in forms:
                return EntityMatch(
                    canonical=canonical,
                    surface_form=surface_form,
                    similarity=1.0,
                    is_new=False,
                )

        # 3. Fuzzy match
        best_match: Optional[str] = None
        best_score = 0.0

        for canonical, forms in self.entities.items():
            # Check canonical name
            score = self.similarity_fn(normalized, canonical)
            if score > best_score:
                best_match, best_score = canonical, score

            # Check all known surface forms
            for form in forms:
                score = self.similarity_fn(normalized, form)
                if score > best_score:
                    best_match, best_score = canonical, score

        # 4. Above threshold - use existing entity
        if best_match is not None and best_score >= self.threshold:
            # Learn this surface form for future exact matches
            self.entities[best_match].add(normalized)
            return EntityMatch(
                canonical=best_match,
                surface_form=surface_form,
                similarity=best_score,
                is_new=False,
            )

        # 5. Below threshold - new entity
        if auto_add:
            self.add_entity(normalized)
        return EntityMatch(
            canonical=normalized,
            surface_form=surface_form,
            similarity=1.0,
            is_new=True,
        )

    def resolve_or_none(self, surface_form: str) -> Optional[EntityMatch]:
        """Resolve surface form without auto-adding new entities.

        Returns None if no match found above threshold.

        Args:
            surface_form: The surface form to resolve

        Returns:
            EntityMatch if found, None otherwise
        """
        result = self.resolve(surface_form, auto_add=False)
        if result.is_new:
            return None
        return result

    def get_surface_forms(self, canonical: str) -> set[str]:
        """Get all known surface forms for an entity.

        Args:
            canonical: The canonical entity name

        Returns:
            Set of surface forms (empty if entity not found)
        """
        normalized = self._normalize(canonical)
        return self.entities.get(normalized, set()).copy()

    def get_all_entities(self) -> list[str]:
        """Get all canonical entity names.

        Returns:
            List of canonical entity names
        """
        return list(self.entities.keys())

    def merge_entities(self, keep: str, merge: str) -> int:
        """Merge two entities, keeping one as canonical.

        All surface forms from 'merge' are added to 'keep', then 'merge' is removed.

        Args:
            keep: Entity to keep (will be normalized)
            merge: Entity to merge into 'keep' (will be removed)

        Returns:
            Number of surface forms added to 'keep'

        Raises:
            KeyError: If either entity doesn't exist
        """
        keep_norm = self._normalize(keep)
        merge_norm = self._normalize(merge)

        if keep_norm not in self.entities:
            raise KeyError(f"Entity '{keep}' not found")
        if merge_norm not in self.entities:
            raise KeyError(f"Entity '{merge}' not found")
        if keep_norm == merge_norm:
            return 0

        # Move all surface forms from merge to keep
        forms_to_add = self.entities[merge_norm]
        forms_added = len(forms_to_add - self.entities[keep_norm])
        self.entities[keep_norm].update(forms_to_add)

        # Remove merged entity
        del self.entities[merge_norm]

        return forms_added

    def clear(self) -> None:
        """Remove all entities."""
        self.entities.clear()

    def __len__(self) -> int:
        """Return number of canonical entities."""
        return len(self.entities)

    def __contains__(self, entity: str) -> bool:
        """Check if canonical entity exists."""
        return self._normalize(entity) in self.entities

    def _normalize(self, text: str) -> str:
        """Normalize entity name.

        Normalization rules:
        - Lowercase
        - Strip whitespace
        - Replace spaces and hyphens with underscores
        - Collapse multiple underscores

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        result = text.lower().strip()
        result = result.replace(" ", "_").replace("-", "_")
        # Collapse multiple underscores
        while "__" in result:
            result = result.replace("__", "_")
        return result
