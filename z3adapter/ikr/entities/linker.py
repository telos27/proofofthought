"""EntityLinker: Multi-level entity resolution with linking.

This module provides entity resolution using multiple strategies:
1. Exact match (O(1) hash lookup)
2. Surface form lookup (O(1) learned mappings)
3. Vector search (O(log n) ANN via FAISS)

Key features:
- Creates new entities only when necessary
- Generates similarity links for related entities
- Learns surface forms from successful resolutions

Example:
    from z3adapter.ikr.entities import EntityStore, EntityLinker, VectorIndex

    store = EntityStore("knowledge.db")
    index = VectorIndex(dimension=1536)

    def embed_fn(text: str) -> list[float]:
        return openai.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding

    linker = EntityLinker(
        entity_store=store,
        vector_index=index,
        embed_fn=embed_fn,
    )

    # Link a mention to an entity
    result = linker.link("working memory")

    if result.is_new:
        # New entity was created
        store.add(result.entity)
        for link in result.links:
            store.add_link(link)
    else:
        # Existing entity was found
        print(f"Resolved to: {result.entity.name}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from z3adapter.ikr.entities.schema import Entity, EntityLink, LinkType
from z3adapter.ikr.entities.store import EntityStore
from z3adapter.ikr.entities.vector_index import VectorIndex


@dataclass
class LinkResult:
    """Result of entity linking operation.

    Attributes:
        entity: The resolved or newly created entity
        is_new: Whether a new entity was created
        links: Similarity links to create (empty if existing entity found)
        resolution_method: How the entity was resolved
            - "exact": Exact name match
            - "surface_form": Surface form lookup
            - "vector_high": Vector search with high confidence (identity)
            - "vector_link": Vector search with moderate confidence (new entity + links)
            - "new": No match found, created new entity
        score: Similarity score (1.0 for exact matches, 0.0 for new entities)
    """

    entity: Entity
    is_new: bool
    links: list[EntityLink] = field(default_factory=list)
    resolution_method: str = "new"
    score: float = 0.0


class EntityLinker:
    """Multi-level entity resolution with linking.

    This class resolves text mentions to entities using a cascading strategy:
    1. Exact match (O(1)): Look up by normalized canonical name
    2. Surface form lookup (O(1)): Check learned surface form mappings
    3. Vector search (O(log n)): Find semantically similar entities via FAISS

    Based on the vector search score:
    - High confidence (≥identity_threshold): Use existing entity, learn surface form
    - Moderate confidence (≥link_threshold): Create new entity with similarity links
    - Low confidence: Create new entity without links

    Attributes:
        entity_store: Storage backend for entities and surface forms
        vector_index: FAISS index for semantic search
        embed_fn: Function to compute embeddings for text
        link_threshold: Minimum score to create similarity links (default: 0.5)
        identity_threshold: Minimum score to treat as same entity (default: 0.9)
        learn_surface_forms: Whether to learn surface forms from resolutions (default: True)
        max_links: Maximum number of similarity links to create (default: 10)

    Example:
        linker = EntityLinker(
            entity_store=store,
            vector_index=index,
            embed_fn=embed_fn,
            link_threshold=0.5,
            identity_threshold=0.9,
        )

        result = linker.link("anxiety")

        if result.is_new:
            # Persist new entity and links
            store.add(result.entity)
            for link in result.links:
                store.add_link(link)
    """

    def __init__(
        self,
        entity_store: EntityStore,
        vector_index: Optional[VectorIndex] = None,
        embed_fn: Optional[Callable[[str], list[float]]] = None,
        link_threshold: float = 0.5,
        identity_threshold: float = 0.9,
        learn_surface_forms: bool = True,
        max_links: int = 10,
    ) -> None:
        """Initialize EntityLinker.

        Args:
            entity_store: Storage backend for entities and surface forms
            vector_index: FAISS index for semantic search. If None, only exact
                and surface form resolution is available.
            embed_fn: Function to compute embeddings. Required if vector_index is provided.
            link_threshold: Minimum score to create similarity links (default: 0.5)
            identity_threshold: Minimum score to treat as same entity (default: 0.9)
            learn_surface_forms: Whether to learn surface forms from resolutions (default: True)
            max_links: Maximum number of similarity links per entity (default: 10)

        Raises:
            ValueError: If vector_index is provided but embed_fn is not.
        """
        if vector_index is not None and embed_fn is None:
            raise ValueError("embed_fn is required when vector_index is provided")

        self.entity_store = entity_store
        self.vector_index = vector_index
        self.embed_fn = embed_fn
        self.link_threshold = link_threshold
        self.identity_threshold = identity_threshold
        self.learn_surface_forms = learn_surface_forms
        self.max_links = max_links

    def link(
        self,
        mention: str,
        entity_type: Optional[str] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
    ) -> LinkResult:
        """Link a text mention to an entity.

        Resolution order:
        1. Exact match by canonical name (O(1))
        2. Surface form lookup (O(1))
        3. Vector similarity search (O(log n))

        Args:
            mention: Text mention to resolve (e.g., "working memory", "WM")
            entity_type: Optional type for new entities (e.g., "concept", "disorder")
            source: Optional source for new entities (e.g., "Psychology 101")
            context: Optional context for embedding (not used in current implementation)

        Returns:
            LinkResult with the resolved or new entity, links to create, and metadata.
        """
        original_mention = mention
        normalized = Entity._normalize_name(mention)

        # 1. Exact match by canonical name
        entity = self.entity_store.get_by_name(normalized)
        if entity:
            return LinkResult(
                entity=entity,
                is_new=False,
                links=[],
                resolution_method="exact",
                score=1.0,
            )

        # 2. Surface form lookup
        lookup = self.entity_store.lookup_surface_form(normalized)
        if lookup:
            entity_id, score = lookup
            entity = self.entity_store.get(entity_id)
            if entity:
                return LinkResult(
                    entity=entity,
                    is_new=False,
                    links=[],
                    resolution_method="surface_form",
                    score=score,
                )

        # 3. Vector similarity search
        if self.vector_index is not None and self.embed_fn is not None:
            return self._resolve_via_vector(
                mention=original_mention,
                normalized=normalized,
                entity_type=entity_type,
                source=source,
            )

        # No vector index available - create new entity
        return self._create_new_entity(
            normalized=normalized,
            entity_type=entity_type,
            source=source,
            method="new",
        )

    def _resolve_via_vector(
        self,
        mention: str,
        normalized: str,
        entity_type: Optional[str],
        source: Optional[str],
    ) -> LinkResult:
        """Resolve mention via vector similarity search.

        Args:
            mention: Original text mention (for embedding)
            normalized: Normalized name for the entity
            entity_type: Optional type for new entities
            source: Optional source for new entities

        Returns:
            LinkResult based on vector search results.
        """
        # Compute embedding for the mention
        embedding = self.embed_fn(mention)

        # Search for similar entities
        candidates = self.vector_index.search(embedding, k=self.max_links + 1)

        if not candidates:
            # No candidates found - create new entity
            return self._create_new_entity(
                normalized=normalized,
                entity_type=entity_type,
                source=source,
                method="new",
            )

        best_id, best_score = candidates[0]

        # High confidence: treat as same entity
        if best_score >= self.identity_threshold:
            entity = self.entity_store.get(best_id)
            if entity:
                # Learn surface form for future O(1) lookup
                if self.learn_surface_forms:
                    self.entity_store.add_surface_form(
                        form=normalized,
                        entity_id=entity.id,
                        score=best_score,
                        source="embedding",
                    )
                return LinkResult(
                    entity=entity,
                    is_new=False,
                    links=[],
                    resolution_method="vector_high",
                    score=best_score,
                )

        # Moderate confidence: create new entity with similarity links
        if best_score >= self.link_threshold:
            entity = Entity(
                name=normalized,
                entity_type=entity_type,
                source=source,
            )

            # Create similarity links to candidates above threshold
            links = []
            for candidate_id, score in candidates:
                if score >= self.link_threshold and len(links) < self.max_links:
                    links.append(
                        EntityLink(
                            source_id=entity.id,
                            target_id=candidate_id,
                            link_type=LinkType.SIMILAR_TO,
                            score=score,
                            method="embedding",
                        )
                    )

            return LinkResult(
                entity=entity,
                is_new=True,
                links=links,
                resolution_method="vector_link",
                score=best_score,
            )

        # Low confidence: create new entity without links
        return self._create_new_entity(
            normalized=normalized,
            entity_type=entity_type,
            source=source,
            method="new",
        )

    def _create_new_entity(
        self,
        normalized: str,
        entity_type: Optional[str],
        source: Optional[str],
        method: str,
    ) -> LinkResult:
        """Create a new entity with no links.

        Args:
            normalized: Normalized name for the entity
            entity_type: Optional type for the entity
            source: Optional source for the entity
            method: Resolution method string

        Returns:
            LinkResult with new entity.
        """
        entity = Entity(
            name=normalized,
            entity_type=entity_type,
            source=source,
        )
        return LinkResult(
            entity=entity,
            is_new=True,
            links=[],
            resolution_method=method,
            score=0.0,
        )

    def link_and_store(
        self,
        mention: str,
        entity_type: Optional[str] = None,
        source: Optional[str] = None,
        context: Optional[str] = None,
    ) -> LinkResult:
        """Link a mention and automatically persist to storage.

        Convenience method that combines link() with persistence.
        New entities and links are automatically added to the store.

        Args:
            mention: Text mention to resolve
            entity_type: Optional type for new entities
            source: Optional source for new entities
            context: Optional context for embedding

        Returns:
            LinkResult with the resolved or newly created entity.
        """
        result = self.link(
            mention=mention,
            entity_type=entity_type,
            source=source,
            context=context,
        )

        if result.is_new:
            # Persist new entity
            self.entity_store.add(result.entity)

            # Persist similarity links
            for link in result.links:
                self.entity_store.add_link(link)

            # If we have vector index and embedding function, add to index
            if self.vector_index is not None and self.embed_fn is not None:
                embedding = self.embed_fn(mention)
                self.vector_index.add(result.entity.id, embedding)

                # Store embedding in entity store
                self.entity_store.save_embedding(
                    entity_id=result.entity.id,
                    embedding=embedding,
                    model="embedding",  # Generic model name
                )

        return result

    def link_batch(
        self,
        mentions: list[str],
        entity_type: Optional[str] = None,
        source: Optional[str] = None,
    ) -> list[LinkResult]:
        """Link multiple mentions (for efficiency with batched embeddings).

        This is a simple implementation that calls link() for each mention.
        For production use with many mentions, consider implementing batched
        embedding calls.

        Args:
            mentions: List of text mentions to resolve
            entity_type: Optional type for new entities
            source: Optional source for new entities

        Returns:
            List of LinkResults, one per mention.
        """
        return [
            self.link(mention=m, entity_type=entity_type, source=source)
            for m in mentions
        ]

    @property
    def has_vector_search(self) -> bool:
        """Check if vector search is available."""
        return self.vector_index is not None and self.embed_fn is not None
