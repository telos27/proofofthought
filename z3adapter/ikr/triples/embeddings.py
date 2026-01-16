"""Embedding backends for semantic similarity in entity resolution.

This module provides embedding infrastructure for semantic matching:
- Abstract EmbeddingBackend interface
- OpenAI embedding backend (text-embedding-3-small)
- Mock embedding backend for testing (deterministic, no API calls)
- Factory functions to create similarity functions from embeddings

Usage:
    from z3adapter.ikr.triples.embeddings import (
        OpenAIEmbedding,
        MockEmbedding,
        make_embedding_similarity,
    )

    # For production (requires OPENAI_API_KEY)
    backend = OpenAIEmbedding()
    sim_fn = make_embedding_similarity(backend)
    score = sim_fn("anxiety", "worry")  # ~0.85

    # For testing (no API calls)
    backend = MockEmbedding()
    sim_fn = make_embedding_similarity(backend)
    score = sim_fn("stress", "stress")  # 1.0

References:
    - Ported from pysem/psychology-knowledge/retrieval/embedding_manager.py
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

# Try to import numpy, fall back to pure Python if not available
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class EmbeddingResult:
    """Result of embedding a text."""

    embedding: list[float]
    model: str
    tokens_used: int = 0


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @abstractmethod
    def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with embedding vector
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed multiple texts.

        Default implementation calls embed() for each text.
        Subclasses can override for more efficient batch processing.

        Args:
            texts: List of texts to embed

        Returns:
            List of EmbeddingResult
        """
        return [self.embed(text) for text in texts]


class OpenAIEmbedding(EmbeddingBackend):
    """OpenAI embedding backend.

    Uses OpenAI's text-embedding-3-small model by default.
    Requires OPENAI_API_KEY environment variable or explicit client.

    Example:
        backend = OpenAIEmbedding()
        result = backend.embed("cognitive behavioral therapy")
        print(f"Dimension: {len(result.embedding)}")  # 1536
    """

    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        client: Optional[object] = None,
    ):
        """Initialize OpenAI embedding backend.

        Args:
            model: OpenAI embedding model name
            client: Optional OpenAI client instance
        """
        self._model = model
        self._dimension = self.MODEL_DIMENSIONS.get(model, 1536)
        self._client = client

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI()
            except ImportError:
                raise ImportError(
                    "openai package required for OpenAI embeddings. "
                    "Install with: pip install openai"
                )
        return self._client

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text using OpenAI API."""
        client = self._get_client()
        response = client.embeddings.create(model=self._model, input=text)

        embedding = list(response.data[0].embedding)
        tokens = response.usage.total_tokens

        return EmbeddingResult(
            embedding=embedding,
            model=self._model,
            tokens_used=tokens,
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed multiple texts in a single API call."""
        if not texts:
            return []

        client = self._get_client()
        response = client.embeddings.create(model=self._model, input=texts)

        results = []
        tokens_per_text = response.usage.total_tokens // len(texts)

        for data in response.data:
            results.append(
                EmbeddingResult(
                    embedding=list(data.embedding),
                    model=self._model,
                    tokens_used=tokens_per_text,
                )
            )

        return results


class MockEmbedding(EmbeddingBackend):
    """Mock embedding backend for testing.

    Generates deterministic embeddings based on text hash.
    Similar texts get similar embeddings through character-level features.

    Example:
        backend = MockEmbedding()
        result = backend.embed("test")
        print(f"Dimension: {len(result.embedding)}")  # 64 by default
    """

    def __init__(self, dimension: int = 64):
        """Initialize mock embedding backend.

        Args:
            dimension: Embedding dimension (default: 64 for fast tests)
        """
        self._dimension = dimension
        self._model = "mock-embedding"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> EmbeddingResult:
        """Generate deterministic embedding from text.

        Uses character-level features to create embeddings where
        similar texts have similar embeddings.
        """
        normalized = text.lower().replace("_", " ").strip()

        # Create embedding from character n-gram features
        embedding = [0.0] * self._dimension

        # Use character trigrams as features
        padded = f"  {normalized}  "
        for i in range(len(padded) - 2):
            trigram = padded[i : i + 3]
            # Hash trigram to position
            pos = hash(trigram) % self._dimension
            # Add contribution (use sine for smoother distribution)
            embedding[pos] += math.sin(hash(trigram) / 1000.0) * 0.5 + 0.5

        # Normalize to unit length
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return EmbeddingResult(
            embedding=embedding,
            model=self._model,
            tokens_used=len(normalized.split()),
        )


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Cosine similarity in [-1, 1], typically [0, 1] for embeddings
    """
    if HAS_NUMPY:
        a1, a2 = np.array(v1), np.array(v2)
        dot = np.dot(a1, a2)
        norm1, norm2 = np.linalg.norm(a1), np.linalg.norm(a2)
    else:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(x * x for x in v1))
        norm2 = math.sqrt(sum(x * x for x in v2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot / (norm1 * norm2))


@dataclass
class EmbeddingCache:
    """Cache for embeddings to avoid recomputation.

    Attributes:
        embeddings: Dict mapping normalized text to embedding vector
        hits: Number of cache hits
        misses: Number of cache misses
    """

    embeddings: dict[str, list[float]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, text: str) -> Optional[list[float]]:
        """Get embedding from cache."""
        key = text.lower().replace("_", " ").strip()
        if key in self.embeddings:
            self.hits += 1
            return self.embeddings[key]
        self.misses += 1
        return None

    def put(self, text: str, embedding: list[float]) -> None:
        """Store embedding in cache."""
        key = text.lower().replace("_", " ").strip()
        self.embeddings[key] = embedding

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self.embeddings.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


def make_embedding_similarity(
    backend: EmbeddingBackend,
    cache: Optional[EmbeddingCache] = None,
) -> Callable[[str, str], float]:
    """Factory for embedding-based similarity function.

    Creates a similarity function that uses embeddings for semantic matching.
    The returned function computes cosine similarity between text embeddings.

    Args:
        backend: Embedding backend to use
        cache: Optional cache for embeddings (created if not provided)

    Returns:
        Similarity function: (str, str) -> float in [0, 1]

    Example:
        backend = OpenAIEmbedding()
        sim_fn = make_embedding_similarity(backend)

        score = sim_fn("anxiety", "worry")
        print(f"Similarity: {score:.3f}")  # ~0.85
    """
    _cache = cache if cache is not None else EmbeddingCache()

    def get_embedding(text: str) -> list[float]:
        """Get embedding, using cache if available."""
        cached = _cache.get(text)
        if cached is not None:
            return cached

        result = backend.embed(text)
        _cache.put(text, result.embedding)
        return result.embedding

    def embedding_similarity(term1: str, term2: str) -> float:
        """Compute cosine similarity between term embeddings."""
        # Normalize terms
        t1 = term1.lower().replace("_", " ").strip()
        t2 = term2.lower().replace("_", " ").strip()

        # Fast path for identical terms
        if t1 == t2:
            return 1.0

        # Get embeddings and compute similarity
        v1 = get_embedding(t1)
        v2 = get_embedding(t2)

        return max(0.0, cosine_similarity(v1, v2))

    return embedding_similarity


def make_hybrid_similarity(
    backend: EmbeddingBackend,
    cache: Optional[EmbeddingCache] = None,
    lexical_weight: float = 0.3,
) -> Callable[[str, str], float]:
    """Factory for hybrid similarity: lexical + semantic.

    Combines lexical similarity (fast, handles typos/variants) with
    embedding similarity (semantic understanding).

    Args:
        backend: Embedding backend to use
        cache: Optional cache for embeddings
        lexical_weight: Weight for lexical component (0-1), rest is embedding

    Returns:
        Similarity function: (str, str) -> float in [0, 1]

    Example:
        backend = OpenAIEmbedding()
        sim_fn = make_hybrid_similarity(backend, lexical_weight=0.3)

        # Combines lexical (0.3) + embedding (0.7)
        score = sim_fn("working_memory", "WM")
    """
    from z3adapter.ikr.fuzzy_nars import combined_lexical_similarity

    embed_sim = make_embedding_similarity(backend, cache)

    def hybrid_similarity(term1: str, term2: str) -> float:
        """Compute weighted combination of lexical and embedding similarity."""
        lex_sim = combined_lexical_similarity(term1, term2)
        emb_sim = embed_sim(term1, term2)

        return lexical_weight * lex_sim + (1 - lexical_weight) * emb_sim

    return hybrid_similarity
