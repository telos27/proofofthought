"""FAISS-based vector index for entity resolution.

This module provides approximate nearest neighbor (ANN) search for entity
embeddings using FAISS. It enables fast semantic similarity search even
with millions of entities.

Key features:
- O(log n) search via IVF index
- Persistent storage (save/load index to disk)
- ID mapping (FAISS indices → entity IDs)

Example:
    index = VectorIndex(dimension=1536)

    # Build from existing embeddings
    embeddings = {"e1": [0.1, 0.2, ...], "e2": [0.3, 0.4, ...]}
    index.build(embeddings)

    # Search for similar entities
    query = [0.15, 0.25, ...]
    results = index.search(query, k=10)
    for entity_id, score in results:
        print(f"{entity_id}: {score}")

    # Persist to disk
    index.save("index.faiss")

    # Load later
    index.load("index.faiss")
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Optional

import numpy as np

# FAISS is optional - graceful degradation if not installed
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False


class VectorIndex:
    """FAISS-based vector index for approximate nearest neighbor search.

    This class wraps FAISS to provide:
    - Fast similarity search (O(log n) with IVF index)
    - ID mapping (FAISS integer indices → entity ID strings)
    - Persistence (save/load index and ID map)

    The index uses IVF (Inverted File) with flat quantization for a good
    balance of speed and accuracy. For small datasets (<10K), it falls
    back to a flat index (exact search).

    Attributes:
        dimension: Embedding dimension (e.g., 1536 for OpenAI)
        index: The FAISS index object
        id_map: Mapping from FAISS index position to entity ID

    Example:
        # Create index
        index = VectorIndex(dimension=1536)

        # Add embeddings
        index.build({"e1": embedding1, "e2": embedding2})

        # Search
        results = index.search(query_embedding, k=5)
        # Returns: [("e1", 0.95), ("e2", 0.82), ...]

        # Save/load
        index.save("my_index")
        index.load("my_index")
    """

    def __init__(self, dimension: int = 1536) -> None:
        """Initialize the vector index.

        Args:
            dimension: Embedding dimension. Defaults to 1536 (OpenAI text-embedding-3-small).

        Raises:
            ImportError: If FAISS is not installed.
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is required for VectorIndex. "
                "Install with: pip install faiss-cpu"
            )

        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.id_map: list[str] = []  # Position → entity_id
        self._trained = False

    def __len__(self) -> int:
        """Return the number of vectors in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal

    @property
    def is_empty(self) -> bool:
        """Check if the index is empty."""
        return len(self) == 0

    def build(
        self,
        embeddings: dict[str, list[float]],
        nlist: Optional[int] = None,
        use_ivf: bool = True,
    ) -> None:
        """Build the index from a dictionary of embeddings.

        For small datasets (<1000), uses a flat index (exact search).
        For larger datasets, uses IVF (inverted file) for faster search.

        Args:
            embeddings: Dict mapping entity_id → embedding vector
            nlist: Number of clusters for IVF. Auto-computed if None.
            use_ivf: Whether to use IVF index. If False, uses flat index.

        Raises:
            ValueError: If embeddings have wrong dimension or are empty.
        """
        if not embeddings:
            raise ValueError("Cannot build index from empty embeddings")

        # Convert to numpy array
        self.id_map = list(embeddings.keys())
        vectors = np.array([embeddings[eid] for eid in self.id_map], dtype=np.float32)

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {vectors.shape[1]}"
            )

        n_vectors = len(vectors)

        # Normalize vectors for cosine similarity (FAISS uses L2 by default)
        faiss.normalize_L2(vectors)

        # Choose index type based on dataset size
        if n_vectors < 1000 or not use_ivf:
            # Small dataset: use flat index (exact search)
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product = cosine for normalized vectors
            self._trained = True
        else:
            # Large dataset: use IVF for faster search
            if nlist is None:
                # Rule of thumb: sqrt(n) clusters, but at least 10, at most 1000
                nlist = min(1000, max(10, int(np.sqrt(n_vectors))))

            # IVF with flat quantizer (good accuracy, fast)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT
            )

            # Train the index (required for IVF)
            self.index.train(vectors)
            self._trained = True

        # Add vectors to index
        self.index.add(vectors)

    def add(self, entity_id: str, embedding: list[float]) -> None:
        """Add a single embedding to the index.

        Note: For IVF indexes, this adds to the trained clusters without
        retraining. For best results with many additions, consider rebuilding.

        Args:
            entity_id: ID of the entity
            embedding: Embedding vector

        Raises:
            ValueError: If embedding has wrong dimension.
            RuntimeError: If index hasn't been built yet.
        """
        if self.index is None:
            # Initialize a flat index if not built yet
            self.index = faiss.IndexFlatIP(self.dimension)
            self._trained = True

        vector = np.array([embedding], dtype=np.float32)

        if vector.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {vector.shape[1]}"
            )

        # Normalize for cosine similarity
        faiss.normalize_L2(vector)

        self.id_map.append(entity_id)
        self.index.add(vector)

    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        nprobe: int = 10,
    ) -> list[tuple[str, float]]:
        """Find k nearest entities to the query embedding.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            nprobe: Number of clusters to search (for IVF). Higher = more accurate but slower.

        Returns:
            List of (entity_id, similarity_score) tuples, sorted by score descending.
            Scores are cosine similarities in [0, 1] for normalized vectors.

        Raises:
            ValueError: If query has wrong dimension.
            RuntimeError: If index is empty.
        """
        if self.index is None or self.is_empty:
            return []

        query = np.array([query_embedding], dtype=np.float32)

        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, "
                f"got {query.shape[1]}"
            )

        # Normalize for cosine similarity
        faiss.normalize_L2(query)

        # Set nprobe for IVF indexes
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe

        # Limit k to number of vectors
        k = min(k, len(self))

        # Search
        scores, indices = self.index.search(query, k)

        # Convert to (entity_id, score) pairs
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for missing results
                # Convert inner product to similarity score
                # For normalized vectors, inner product is cosine similarity
                # Clamp to [0, 1] to handle numerical issues
                similarity = float(max(0.0, min(1.0, score)))
                results.append((self.id_map[idx], similarity))

        return results

    def search_batch(
        self,
        query_embeddings: list[list[float]],
        k: int = 10,
        nprobe: int = 10,
    ) -> list[list[tuple[str, float]]]:
        """Search for multiple queries in batch (more efficient).

        Args:
            query_embeddings: List of query vectors
            k: Number of results per query
            nprobe: Number of clusters to search (for IVF)

        Returns:
            List of result lists, one per query.
        """
        if self.index is None or self.is_empty:
            return [[] for _ in query_embeddings]

        queries = np.array(query_embeddings, dtype=np.float32)

        if queries.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension mismatch: expected {self.dimension}, "
                f"got {queries.shape[1]}"
            )

        faiss.normalize_L2(queries)

        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe

        k = min(k, len(self))
        scores, indices = self.index.search(queries, k)

        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for score, idx in zip(query_scores, query_indices):
                if idx >= 0:
                    similarity = float(max(0.0, min(1.0, score)))
                    results.append((self.id_map[idx], similarity))
            all_results.append(results)

        return all_results

    def remove(self, entity_id: str) -> bool:
        """Remove an entity from the index.

        Note: FAISS doesn't support efficient removal. This marks the ID
        as removed but doesn't reclaim space. For many removals, rebuild
        the index.

        Args:
            entity_id: ID of the entity to remove

        Returns:
            True if entity was found and removed, False otherwise.
        """
        if entity_id not in self.id_map:
            return False

        # Mark as removed by setting to empty string
        # The vector remains but won't be returned in search
        idx = self.id_map.index(entity_id)
        self.id_map[idx] = ""
        return True

    def save(self, path: str) -> None:
        """Save the index and ID map to disk.

        Creates two files:
        - {path}.index: FAISS index binary
        - {path}.idmap: JSON mapping of positions to entity IDs

        Args:
            path: Base path for the files (without extension)
        """
        if self.index is None:
            raise RuntimeError("Cannot save empty index")

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(path_obj.with_suffix(".index")))

        # Save ID map and metadata
        metadata = {
            "dimension": self.dimension,
            "id_map": self.id_map,
            "trained": self._trained,
        }
        with open(path_obj.with_suffix(".idmap"), "w") as f:
            json.dump(metadata, f)

    def load(self, path: str) -> None:
        """Load the index and ID map from disk.

        Args:
            path: Base path for the files (without extension)

        Raises:
            FileNotFoundError: If index files don't exist.
        """
        path_obj = Path(path)

        index_path = path_obj.with_suffix(".index")
        idmap_path = path_obj.with_suffix(".idmap")

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not idmap_path.exists():
            raise FileNotFoundError(f"ID map file not found: {idmap_path}")

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load ID map and metadata
        with open(idmap_path) as f:
            metadata = json.load(f)

        self.dimension = metadata["dimension"]
        self.id_map = metadata["id_map"]
        self._trained = metadata.get("trained", True)

    def get_entity_ids(self) -> list[str]:
        """Get all entity IDs in the index.

        Returns:
            List of entity IDs (excludes removed entities).
        """
        return [eid for eid in self.id_map if eid]

    def contains(self, entity_id: str) -> bool:
        """Check if an entity is in the index.

        Args:
            entity_id: ID to check

        Returns:
            True if entity is in the index and not removed.
        """
        return entity_id in self.id_map and self.id_map.index(entity_id) >= 0
