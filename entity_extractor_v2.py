# ============================================================================
# entity_extractor_v2.py — Embedding-Based Entity Extractor
# Replaces regex-based extraction with semantic understanding.
# Maintains backward compatibility with HeuristicExtractor interface.
# ============================================================================

from __future__ import annotations

import re
from typing import Optional, Protocol

import numpy as np
import structlog

from embedding_engine import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    VectorStore,
)

logger = structlog.get_logger(__name__)


# ============================================================================
# Entity Extractor Protocol
# ============================================================================

class EntityExtractorV2(Protocol):
    """Protocol for entity extractors v2."""

    def extract_triples(
        self, text: str
    ) -> list[tuple[str, str, str]]:
        """Extract (subject, predicate, object) triples from text."""
        ...

    def extract_concepts(self, text: str) -> list[str]:
        """Extract key concepts from text."""
        ...


# ============================================================================
# Embedding-Based Entity Extractor
# ============================================================================

class EmbeddingEntityExtractor:
    """Entity extractor using embedding similarity.

    Extracts concepts and relationships by:
    1. Splitting text into semantic units (sentences/chunks)
    2. Embedding each unit
    3. Clustering similar units into concepts
    4. Extracting relationships via semantic role patterns

    Deterministic. No external LLM calls.
    """

    # Common relation patterns (semantic roles)
    RELATION_PATTERNS = [
        (r"(\w+(?:\s+\w+){0,3})\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+){0,3})", "is_a"),
        (r"(\w+(?:\s+\w+){0,3})\s+(?:has|have|had)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+){0,3})", "has"),
        (r"(\w+(?:\s+\w+){0,3})\s+(?:uses?|used|using)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+){0,3})", "uses"),
        (r"(\w+(?:\s+\w+){0,3})\s+(?:contains?|contained|containing)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+){0,3})", "contains"),
        (r"(\w+(?:\s+\w+){0,3})\s+(?:depends?|depended|depending)\s+(?:on|upon)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+){0,3})", "depends_on"),
        (r"(\w+(?:\s+\w+){0,3})\s+(?:requires?|required|requiring)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+){0,3})", "requires"),
        (r"(\w+(?:\s+\w+){0,3})\s+(?:produces?|produced|producing)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+){0,3})", "produces"),
        (r"(\w+(?:\s+\w+){0,3})\s+(?:creates?|created|creating)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+){0,3})", "creates"),
        (r"(\w+(?:\s+\w+){0,3})\s+(?:enables?|enabled|enabling)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+){0,3})", "enables"),
        (r"(\w+(?:\s+\w+){0,3})\s+(?:supports?|supported|supporting)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+){0,3})", "supports"),
        (r"(\w+(?:\s+\w+){0,3})\s+(?:causes?|caused|causing)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+){0,3})", "causes"),
    ]

    def __init__(
        self,
        provider: Optional[EmbeddingProvider] = None,
        min_concept_length: int = 3,
        similarity_threshold: float = 0.6,
        max_sentence_words: int = 50,
    ) -> None:
        """Initialize the embedding entity extractor.

        Args:
            provider: Embedding provider. If None, uses HashEmbeddingProvider.
            min_concept_length: Minimum concept string length.
            similarity_threshold: Threshold for concept deduplication.
            max_sentence_words: Maximum words per sentence to process.
        """
        if provider is None:
            provider = HashEmbeddingProvider(dimension=384)
        self._provider = provider
        self._min_concept_length = min_concept_length
        self._similarity_threshold = similarity_threshold
        self._max_sentence_words = max_sentence_words
        self._concept_store = VectorStore(provider.dimension)

        logger.info(
            "embedding_entity_extractor_initialized",
            dimension=provider.dimension,
            threshold=similarity_threshold,
        )

    def extract_triples(
        self, text: str
    ) -> list[tuple[str, str, str]]:
        """Extract (subject, predicate, object) triples using embeddings.

        Process:
        1. Split text into sentences
        2. Apply relation patterns to extract SVO triples
        3. Embed each triple's components
        4. Deduplicate using embedding similarity

        Args:
            text: Input text.

        Returns:
            List of (subject, predicate, object) triples.
        """
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        triples = []
        seen = set()

        for sentence in sentences:
            if len(sentence.split()) > self._max_sentence_words:
                continue

            for pattern, relation in self.RELATION_PATTERNS:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    subject = self._normalize(match.group(1))
                    obj = self._normalize(match.group(2))

                    if not self._is_valid_concept(subject):
                        continue
                    if not self._is_valid_concept(obj):
                        continue

                    triple = (subject, relation, obj)
                    triple_key = (subject.lower(), relation, obj.lower())

                    if triple_key not in seen:
                        if not self._is_duplicate_triple(triple):
                            seen.add(triple_key)
                            triples.append(triple)
                            self._register_triple(triple)

        logger.debug(
            "embedding_extract_triples",
            text_length=len(text),
            n_triples=len(triples),
        )

        return triples

    def extract_concepts(self, text: str) -> list[str]:
        """Extract key concepts from text using embedding clustering.

        Args:
            text: Input text.

        Returns:
            List of deduplicated concept strings.
        """
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        embeddings = self._provider.embed(sentences)

        concepts = []
        concept_embeddings = []

        for sentence, emb in zip(sentences, embeddings):
            if not self._is_valid_concept(sentence):
                continue

            if not concepts:
                concepts.append(sentence)
                concept_embeddings.append(emb)
                continue

            similarities = np.array([
                float(np.dot(emb, ce) / (np.linalg.norm(emb) * np.linalg.norm(ce) + 1e-8))
                for ce in concept_embeddings
            ])

            max_sim = float(np.max(similarities)) if similarities.size > 0 else 0.0

            if max_sim < self._similarity_threshold:
                concepts.append(sentence)
                concept_embeddings.append(emb)

        logger.debug(
            "embedding_extract_concepts",
            text_length=len(text),
            n_concepts=len(concepts),
        )

        return concepts

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _normalize(self, phrase: str) -> str:
        """Normalize a phrase."""
        return phrase.strip().lower().strip(".,;:!?\"'()[]{}")

    def _is_valid_concept(self, phrase: str) -> bool:
        """Check if phrase is a valid concept."""
        if not phrase or len(phrase) < self._min_concept_length:
            return False

        words = phrase.split()
        meaningful = [
            w for w in words
            if len(w) >= 2 and w.isalpha()
        ]

        return len(meaningful) >= 1

    def _is_duplicate_triple(self, triple: tuple[str, str, str]) -> bool:
        """Check if triple is duplicate using embedding similarity."""
        if self._concept_store.size == 0:
            return False

        triple_text = f"{triple[0]} {triple[1]} {triple[2]}"
        triple_emb = self._provider.embed_single(triple_text)

        results = self._concept_store.search(triple_emb, top_k=1, threshold=0.95)

        return len(results) > 0

    def _register_triple(self, triple: tuple[str, str, str]) -> None:
        """Register a triple in the concept store."""
        triple_text = f"{triple[0]} {triple[1]} {triple[2]}"
        triple_emb = self._provider.embed_single(triple_text)

        self._concept_store.add(
            vector=triple_emb,
            metadata={
                "subject": triple[0],
                "predicate": triple[1],
                "object": triple[2],
            },
        )
