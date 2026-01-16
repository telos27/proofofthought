"""PrecomputationPipeline: Batch pre-computation of embeddings and links.

This module provides batch operations for:
1. Computing embeddings for entities missing them
2. Rebuilding the FAISS vector index
3. Pre-computing similarity links between entities

These operations are designed for offline/batch processing, not real-time use.

Example:
    from z3adapter.ikr.entities import EntityStore, VectorIndex, PrecomputationPipeline

    store = EntityStore("knowledge.db")
    index = VectorIndex(dimension=1536)

    def batch_embed(texts: list[str]) -> list[list[float]]:
        response = openai.embeddings.create(input=texts, model="text-embedding-3-small")
        return [d.embedding for d in response.data]

    pipeline = PrecomputationPipeline(
        entity_store=store,
        vector_index=index,
        batch_embed_fn=batch_embed,
        model_name="text-embedding-3-small",
    )

    # Compute missing embeddings
    n_computed = pipeline.compute_embeddings(batch_size=100)

    # Rebuild vector index
    pipeline.rebuild_vector_index()

    # Pre-compute similarity links
    n_links = pipeline.compute_links(k=50, min_score=0.5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from z3adapter.ikr.entities.schema import Entity, EntityLink, LinkType
from z3adapter.ikr.entities.store import EntityStore
from z3adapter.ikr.entities.vector_index import VectorIndex


@dataclass
class PrecomputationStats:
    """Statistics from a pre-computation run.

    Attributes:
        entities_processed: Number of entities processed
        embeddings_computed: Number of new embeddings computed
        links_created: Number of similarity links created
        index_size: Number of vectors in the rebuilt index
    """

    entities_processed: int = 0
    embeddings_computed: int = 0
    links_created: int = 0
    index_size: int = 0


def _chunks(items: list, size: int) -> Iterator[list]:
    """Split a list into chunks of the given size."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


class PrecomputationPipeline:
    """Batch pre-computation of embeddings and similarity links.

    This class provides batch operations for building and maintaining
    the entity knowledge graph:

    1. **compute_embeddings**: Compute embeddings for entities that don't have them
    2. **rebuild_vector_index**: Rebuild the FAISS index from stored embeddings
    3. **compute_links**: Pre-compute similarity links between entities

    These operations are designed for offline/batch use. For real-time
    entity resolution, use EntityLinker instead.

    Attributes:
        entity_store: Storage backend for entities and embeddings
        vector_index: FAISS index for similarity search
        batch_embed_fn: Function to compute embeddings for a batch of texts
        model_name: Name of the embedding model (for metadata)
        progress_callback: Optional callback for progress updates

    Example:
        pipeline = PrecomputationPipeline(
            entity_store=store,
            vector_index=index,
            batch_embed_fn=batch_embed,
        )

        # Full pipeline run
        stats = pipeline.run_full_pipeline(
            embedding_batch_size=100,
            link_k=50,
            link_min_score=0.5,
        )
        print(f"Computed {stats.embeddings_computed} embeddings")
        print(f"Created {stats.links_created} links")
    """

    def __init__(
        self,
        entity_store: EntityStore,
        vector_index: VectorIndex,
        batch_embed_fn: Callable[[list[str]], list[list[float]]],
        model_name: str = "embedding",
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """Initialize PrecomputationPipeline.

        Args:
            entity_store: Storage backend for entities and embeddings
            vector_index: FAISS index for similarity search
            batch_embed_fn: Function that takes a list of texts and returns
                a list of embedding vectors. Should handle batching internally
                if needed for API rate limits.
            model_name: Name of the embedding model (stored as metadata)
            progress_callback: Optional callback for progress updates.
                Called with (operation_name, current, total).
        """
        self.entity_store = entity_store
        self.vector_index = vector_index
        self.batch_embed_fn = batch_embed_fn
        self.model_name = model_name
        self.progress_callback = progress_callback

    def _report_progress(self, operation: str, current: int, total: int) -> None:
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(operation, current, total)

    def compute_embeddings(self, batch_size: int = 100) -> int:
        """Compute embeddings for entities that don't have them.

        Finds all entities without embeddings and computes them in batches.
        Embeddings are saved to the entity store.

        Args:
            batch_size: Number of entities to embed per batch. Adjust based
                on your embedding API's limits.

        Returns:
            Number of embeddings computed.
        """
        missing = self.entity_store.get_entities_without_embeddings()

        if not missing:
            return 0

        total = len(missing)
        computed = 0

        for batch in _chunks(missing, batch_size):
            # Convert entity names to natural text
            texts = [self._entity_to_text(e) for e in batch]

            # Compute embeddings
            embeddings = self.batch_embed_fn(texts)

            # Save embeddings
            for entity, embedding in zip(batch, embeddings):
                self.entity_store.save_embedding(
                    entity_id=entity.id,
                    embedding=embedding,
                    model=self.model_name,
                )
                computed += 1

            self._report_progress("compute_embeddings", computed, total)

        return computed

    def rebuild_vector_index(self, index_path: Optional[str] = None) -> int:
        """Rebuild the FAISS vector index from stored embeddings.

        Loads all embeddings from the entity store and builds a new
        FAISS index. Optionally saves the index to disk.

        Args:
            index_path: Optional path to save the index. If provided,
                the index is saved to {index_path}.index and {index_path}.idmap.

        Returns:
            Number of vectors in the rebuilt index.
        """
        embeddings = self.entity_store.get_all_embeddings()

        if not embeddings:
            return 0

        self._report_progress("rebuild_index", 0, 1)

        # Build index
        self.vector_index.build(embeddings)

        # Save if path provided
        if index_path:
            self.vector_index.save(index_path)

        self._report_progress("rebuild_index", 1, 1)

        return len(embeddings)

    def compute_links(
        self,
        k: int = 50,
        min_score: float = 0.5,
        clear_existing: bool = False,
    ) -> int:
        """Pre-compute similarity links between entities.

        For each entity, finds the top-k most similar entities and creates
        similarity links for those above the minimum score threshold.

        Args:
            k: Number of nearest neighbors to find per entity.
            min_score: Minimum similarity score to create a link.
            clear_existing: If True, clears existing similarity links first.

        Returns:
            Number of links created.
        """
        embeddings = self.entity_store.get_all_embeddings()

        if not embeddings:
            return 0

        total = len(embeddings)
        links_created = 0
        processed = 0

        # Clear existing links if requested
        if clear_existing:
            self._clear_similarity_links()

        for entity_id, embedding in embeddings.items():
            # Search for similar entities (+1 to account for self)
            candidates = self.vector_index.search(embedding, k=k + 1)

            for target_id, score in candidates:
                # Skip self
                if target_id == entity_id:
                    continue

                # Skip below threshold
                if score < min_score:
                    continue

                # Create link
                link = EntityLink(
                    source_id=entity_id,
                    target_id=target_id,
                    link_type=LinkType.SIMILAR_TO,
                    score=score,
                    method="embedding",
                )
                self.entity_store.add_link(link)
                links_created += 1

            processed += 1
            self._report_progress("compute_links", processed, total)

        return links_created

    def run_full_pipeline(
        self,
        embedding_batch_size: int = 100,
        link_k: int = 50,
        link_min_score: float = 0.5,
        index_path: Optional[str] = None,
        clear_existing_links: bool = True,
    ) -> PrecomputationStats:
        """Run the full pre-computation pipeline.

        Executes all three stages in order:
        1. Compute missing embeddings
        2. Rebuild vector index
        3. Compute similarity links

        Args:
            embedding_batch_size: Batch size for embedding computation.
            link_k: Number of nearest neighbors for link computation.
            link_min_score: Minimum score for similarity links.
            index_path: Optional path to save the vector index.
            clear_existing_links: Whether to clear existing similarity links.

        Returns:
            PrecomputationStats with counts for each operation.
        """
        stats = PrecomputationStats()

        # Stage 1: Compute embeddings
        stats.embeddings_computed = self.compute_embeddings(
            batch_size=embedding_batch_size
        )
        stats.entities_processed = self.entity_store.count()

        # Stage 2: Rebuild index
        stats.index_size = self.rebuild_vector_index(index_path=index_path)

        # Stage 3: Compute links
        stats.links_created = self.compute_links(
            k=link_k,
            min_score=link_min_score,
            clear_existing=clear_existing_links,
        )

        return stats

    def _entity_to_text(self, entity: Entity) -> str:
        """Convert entity to text for embedding.

        Uses the entity name with underscores replaced by spaces.
        If description is available, appends it for richer context.

        Args:
            entity: Entity to convert.

        Returns:
            Text representation for embedding.
        """
        text = entity.name.replace("_", " ")
        if entity.description:
            text = f"{text}: {entity.description}"
        return text

    def _clear_similarity_links(self) -> None:
        """Clear all existing similarity links from the store."""
        conn = self.entity_store._get_conn()
        conn.execute(
            "DELETE FROM entity_links WHERE link_type = ?",
            (LinkType.SIMILAR_TO.value,),
        )
        conn.commit()

    def get_entities_needing_embeddings(self) -> list[Entity]:
        """Get list of entities that need embeddings computed.

        Returns:
            List of entities without embeddings.
        """
        return self.entity_store.get_entities_without_embeddings()

    def get_embedding_coverage(self) -> tuple[int, int]:
        """Get embedding coverage statistics.

        Returns:
            Tuple of (entities_with_embeddings, total_entities).
        """
        total = self.entity_store.count()
        missing = len(self.entity_store.get_entities_without_embeddings())
        return (total - missing, total)
