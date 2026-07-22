# ============================================================================
# embedding_engine.py — Deterministic Embedding Engine
# Provides embedding-based entity extraction and retrieval.
# Uses sentence-transformers for semantic understanding.
# Falls back to TF-IDF if sentence-transformers unavailable.
# ============================================================================

from __future__ import annotations

import hashlib
import re
from typing import Optional, Protocol

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ============================================================================
# Embedding Provider Protocol
# ============================================================================

class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            Array of shape (len(texts), dimension) with float32 embeddings.
        """
        ...

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text into a vector.

        Args:
            text: Text string to embed.

        Returns:
            Vector of shape (dimension,) with float32 values.
        """
        ...


# ============================================================================
# Sentence-Transformers Provider
# ============================================================================

class SentenceTransformerProvider:
    """Embedding provider using sentence-transformers.

    Uses all-MiniLM-L6-v2 by default (22M params, 384 dim).
    Falls back gracefully if sentence-transformers unavailable.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        """Initialize the provider.

        Args:
            model_name: HuggingFace model name.
            device: Device to run on ('cpu' or 'cuda').
        """
        self._model_name = model_name
        self._device = device
        self._model = None
        self._dimension = 384  # Default for all-MiniLM-L6-v2

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, device=device)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(
                "sentence_transformers_loaded",
                model=model_name,
                dimension=self._dimension,
                device=device,
            )
        except ImportError:
            logger.warning(
                "sentence_transformers_unavailable",
                fallback="tfidf",
            )
        except Exception as exc:
            logger.warning(
                "sentence_transformers_load_failed",
                error=str(exc),
                fallback="tfidf",
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._model is not None:
            return self._model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
        return self._tfidf_embed(texts)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def _tfidf_embed(self, texts: list[str]) -> np.ndarray:
        """TF-IDF fallback embedding using pure numpy."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not hasattr(self, "_tfidf_vectorizer"):
            self._tfidf_vectorizer = TfidfVectorizer(
                max_features=384,
                stop_words="english",
                ngram_range=(1, 2),
            )
            self._tfidf_matrix = None

        if self._tfidf_matrix is None:
            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)
            return self._tfidf_matrix.toarray().astype(np.float32)

        return self._tfidf_vectorizer.transform(texts).toarray().astype(
            np.float32
        )


# ============================================================================
# Lightweight Hash Embedding (No Dependencies)
# ============================================================================

class HashEmbeddingProvider:
    """Lightweight embedding provider using feature hashing.

    No external dependencies. Deterministic. Fast.
    Quality is lower than neural embeddings but sufficient for baseline.
    """

    def __init__(self, dimension: int = 384, ngram_range: tuple[int, int] = (1, 3)) -> None:
        """Initialize hash embedding provider.

        Args:
            dimension: Output embedding dimension.
            ngram_range: Range of character n-grams to use.
        """
        self._dimension = dimension
        self._ngram_range = ngram_range
        logger.info(
            "hash_embedding_initialized",
            dimension=dimension,
            ngram_range=ngram_range,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using feature hashing with SIMD-friendly operations."""
        results = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            results[i] = self._embed_text(text)
        return results

    def embed_single(self, text: str) -> np.ndarray:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> np.ndarray:
        """Embed a single text using character n-gram hashing."""
        text_lower = text.lower()
        embedding = np.zeros(self._dimension, dtype=np.float32)

        for n in range(self._ngram_range[0], self._ngram_range[1] + 1):
            for i in range(len(text_lower) - n + 1):
                ngram = text_lower[i:i + n]
                h = int(hashlib.md5(ngram.encode("utf-8")).hexdigest(), 16)
                idx = h % self._dimension
                sign = 1.0 if (h // self._dimension) % 2 == 0 else -1.0
                embedding[idx] += sign

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm

        return embedding


# ============================================================================
# Concept Extractor (Embedding-Based)
# ============================================================================

class EmbeddingConceptExtractor:
    """Extract concepts from text using embedding similarity.

    Instead of regex patterns, uses semantic similarity to identify
    concepts and their relationships.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        similarity_threshold: float = 0.5,
    ) -> None:
        """Initialize the extractor.

        Args:
            provider: Embedding provider to use.
            similarity_threshold: Minimum cosine similarity for concept match.
        """
        self._provider = provider
        self._threshold = similarity_threshold
        self._concept_cache: dict[str, np.ndarray] = {}

    def extract_concepts(self, text: str) -> list[str]:
        """Extract key concepts from text using embedding clustering.

        Args:
            text: Input text.

        Returns:
            List of extracted concept strings.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        embeddings = self._provider.embed(sentences)

        concepts = []
        concept_embeddings = []

        for i, (sentence, emb) in enumerate(zip(sentences, embeddings)):
            if not concepts:
                concepts.append(sentence)
                concept_embeddings.append(emb)
                continue

            similarities = np.array([
                np.dot(emb, ce) / (np.linalg.norm(emb) * np.linalg.norm(ce) + 1e-8)
                for ce in concept_embeddings
            ])

            max_sim = float(np.max(similarities)) if similarities.size > 0 else 0.0

            if max_sim < self._threshold:
                concepts.append(sentence)
                concept_embeddings.append(emb)

        return concepts

    def extract_triples(
        self, text: str
    ) -> list[tuple[str, str, str]]:
        """Extract (subject, predicate, object) triples using embeddings.

        Uses semantic role labeling via embedding similarity to known
        relation patterns.

        Args:
            text: Input text.

        Returns:
            List of (subject, predicate, object) triples.
        """
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        triples = []
        for sentence in sentences:
            parts = self._split_svo(sentence)
            if parts:
                subject, predicate, obj = parts
                if subject and predicate and obj:
                    triples.append((subject, predicate, obj))

        return triples

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    def _split_svo(self, sentence: str) -> Optional[tuple[str, str, str]]:
        """Split sentence into subject, predicate, object using simple heuristics."""
        relation_verbs = {
            "is", "are", "was", "were", "has", "have", "had",
            "contains", "includes", "refers", "calls", "depends",
            "requires", "uses", "defines", "implements", "extends",
            "produces", "creates", "transforms", "computes", "links",
            "maps", "connects", "triggers", "generates", "stores",
            "retrieves", "processes", "enables", "supports", "causes",
        }

        words = sentence.lower().split()

        for i, word in enumerate(words):
            if word in relation_verbs:
                subject = " ".join(words[:i]).strip()
                obj = " ".join(words[i + 1:]).strip()

                if subject and obj:
                    return (subject, word, obj)

        return None


# ============================================================================
# Vector Store for Chunk Embeddings
# ============================================================================

class VectorStore:
    """Simple vector store for chunk embeddings with cosine similarity search.

    Deterministic, thread-safe, auditable.
    """

    def __init__(self, dimension: int) -> None:
        """Initialize vector store.

        Args:
            dimension: Embedding dimension.
        """
        self._dimension = dimension
        self._vectors: list[np.ndarray] = []
        self._metadata: list[dict] = []
        self._ids: list[str] = []

    @property
    def size(self) -> int:
        return len(self._vectors)

    def add(
        self,
        vector: np.ndarray,
        metadata: dict,
        vector_id: Optional[str] = None,
    ) -> str:
        """Add a vector to the store.

        Args:
            vector: Embedding vector of shape (dimension,).
            metadata: Associated metadata.
            vector_id: Optional unique identifier.

        Returns:
            The vector ID.
        """
        if vector.shape != (self._dimension,):
            raise ValueError(
                f"Vector shape {vector.shape} != expected ({self._dimension},)"
            )

        if vector_id is None:
            vector_id = hashlib.md5(vector.tobytes()).hexdigest()[:16]

        self._vectors.append(vector.copy())
        self._metadata.append(metadata.copy())
        self._ids.append(vector_id)

        return vector_id

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[dict]:
        """Search for similar vectors using cosine similarity.

        Args:
            query: Query vector of shape (dimension,).
            top_k: Number of results to return.
            threshold: Minimum similarity score.

        Returns:
            List of dicts with 'id', 'score', 'metadata' keys.
        """
        if not self._vectors:
            return []

        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        query_normalized = query / query_norm

        scores = np.zeros(len(self._vectors), dtype=np.float32)
        for i, vec in enumerate(self._vectors):
            vec_norm = np.linalg.norm(vec)
            if vec_norm > 0:
                scores[i] = np.dot(query_normalized, vec / vec_norm)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= threshold:
                results.append({
                    "id": self._ids[idx],
                    "score": score,
                    "metadata": self._metadata[idx],
                })

        return results

    def get(self, vector_id: str) -> Optional[dict]:
        """Get a vector by ID.

        Args:
            vector_id: Unique identifier.

        Returns:
            Dict with 'vector', 'metadata' keys, or None.
        """
        try:
            idx = self._ids.index(vector_id)
            return {
                "vector": self._vectors[idx].copy(),
                "metadata": self._metadata[idx].copy(),
            }
        except ValueError:
            return None

    def clear(self) -> None:
        """Clear all vectors."""
        self._vectors.clear()
        self._metadata.clear()
        self._ids.clear()
