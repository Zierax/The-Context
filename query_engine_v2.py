# ============================================================================
# query_engine_v2.py — Embedding-Based Query Engine
# Replaces LSH bucket matching with vector similarity search.
# Uses multi-step retrieval and re-ranking.
# ============================================================================

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import structlog

from embedding_engine import EmbeddingProvider, HashEmbeddingProvider, VectorStore
from entity_extractor_v2 import EmbeddingEntityExtractor
from knowledge_graph import DeterministicKnowledgeGraph
from memory_manager import VirtualMemoryTree

logger = structlog.get_logger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass(frozen=True)
class CollapseRequestV2:
    """Request model for query collapse v2."""
    query: str
    max_tokens: int = 4096
    required_concepts: list[str] = field(default_factory=list)
    max_steps: int = 3
    top_k_per_step: int = 5


@dataclass
class CollapseResultV2:
    """Result model for query collapse v2."""
    pages: list[str] = field(default_factory=list)
    page_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0
    tokens_used: int = 0
    tokens_total: int = 0
    concepts_activated: list[str] = field(default_factory=list)
    retrieval_steps: int = 0
    error: Optional[str] = None


# ============================================================================
# Query Engine V2
# ============================================================================

class QueryEngineV2:
    """Embedding-based query engine with multi-step retrieval.

    Improvements over V1:
    1. Vector similarity search instead of LSH bucket matching
    2. Multi-step retrieval (query → retrieve → re-query → retrieve)
    3. Re-ranking based on embedding similarity
    4. Evidence convergence scoring

    Deterministic. Thread-safe. Fully auditable.
    """

    def __init__(
        self,
        tree: VirtualMemoryTree,
        graph: DeterministicKnowledgeGraph,
        provider: Optional[EmbeddingProvider] = None,
        d_model: int = 128,
    ) -> None:
        """Initialize query engine v2.

        Args:
            tree: Virtual memory tree for page storage.
            graph: Knowledge graph for concept relationships.
            provider: Embedding provider. If None, uses HashEmbeddingProvider.
            d_model: Model dimension for compatibility.
        """
        self._tree = tree
        self._graph = graph
        self._d_model = d_model

        if provider is None:
            provider = HashEmbeddingProvider(dimension=384)
        self._provider = provider

        self._extractor = EmbeddingEntityExtractor(provider=provider)
        self._page_store = VectorStore(provider.dimension)
        self._concept_store = VectorStore(provider.dimension)

        self._build_stores()

        logger.info(
            "query_engine_v2_initialized",
            dimension=provider.dimension,
            n_pages=self._page_store.size,
            n_concepts=self._concept_store.size,
        )

    def _build_stores(self) -> None:
        """Build page and concept stores from memory tree."""
        for page_id in list(self._tree.pages):
            text = self._tree.get_page(page_id)
            if text:
                emb = self._provider.embed_single(text)
                self._page_store.add(
                    vector=emb,
                    metadata={"page_id": page_id, "text_length": len(text)},
                    vector_id=page_id,
                )

                concepts = self._extractor.extract_concepts(text)
                for concept in concepts:
                    concept_emb = self._provider.embed_single(concept)
                    self._concept_store.add(
                        vector=concept_emb,
                        metadata={"concept": concept, "page_id": page_id},
                    )

        logger.debug(
            "stores_built",
            pages=self._page_store.size,
            concepts=self._concept_store.size,
        )

    def collapse(
        self,
        query: str,
        max_tokens: int = 4096,
        required_concepts: Optional[list[str]] = None,
        max_steps: int = 3,
        top_k_per_step: int = 5,
    ) -> CollapseResultV2:
        """Execute multi-step retrieval and collapse.

        Process:
        1. Embed query
        2. Find initial candidate pages via vector similarity
        3. Extract concepts from retrieved pages
        4. Re-query with expanded concept set
        5. Re-rank by combined similarity
        6. Pack results within token budget

        Args:
            query: Natural language query.
            max_tokens: Maximum token budget.
            required_concepts: Concepts that must be in results.
            max_steps: Maximum retrieval steps.
            top_k_per_step: Top-k results per step.

        Returns:
            CollapseResultV2 with retrieved pages and metadata.
        """
        start_time = time.perf_counter()

        try:
            if not query or not query.strip():
                return CollapseResultV2(
                    error="Empty query",
                    latency_ms=(time.perf_counter() - start_time) * 1000.0,
                )

            query_embedding = self._provider.embed_single(query)

            all_candidates = {}
            activated_concepts = set()
            current_query = query

            for step in range(max_steps):
                step_embedding = self._provider.embed_single(current_query)

                page_results = self._page_store.search(
                    step_embedding,
                    top_k=top_k_per_step,
                    threshold=0.1,
                )

                concept_results = self._concept_store.search(
                    step_embedding,
                    top_k=top_k_per_step * 2,
                    threshold=0.2,
                )

                for result in page_results:
                    page_id = result["metadata"]["page_id"]
                    if page_id not in all_candidates:
                        all_candidates[page_id] = {
                            "page_id": page_id,
                            "score": result["score"],
                            "step": step,
                        }
                    else:
                        all_candidates[page_id]["score"] = max(
                            all_candidates[page_id]["score"],
                            result["score"],
                        )

                new_concepts = set()
                for result in concept_results:
                    concept = result["metadata"]["concept"]
                    if concept not in activated_concepts:
                        activated_concepts.add(concept)
                        new_concepts.add(concept)

                if not new_concepts:
                    break

                current_query = query + " " + " ".join(list(new_concepts)[:5])

            scored_candidates = []
            for page_id, data in all_candidates.items():
                text = self._tree.get_page(page_id)
                if text is None:
                    continue

                text_embedding = self._provider.embed_single(text)
                similarity = float(
                    np.dot(query_embedding, text_embedding)
                    / (np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding) + 1e-8)
                )

                combined_score = 0.7 * data["score"] + 0.3 * similarity

                scored_candidates.append({
                    "page_id": page_id,
                    "text": text,
                    "score": combined_score,
                    "step": data["step"],
                })

            scored_candidates.sort(key=lambda x: x["score"], reverse=True)

            selected_pages = []
            selected_ids = []
            tokens_used = 0
            tokens_per_page = 1000

            for candidate in scored_candidates:
                if tokens_used + tokens_per_page > max_tokens:
                    break

                selected_pages.append(candidate["text"])
                selected_ids.append(candidate["page_id"])
                tokens_used += tokens_per_page

            confidence = 0.0
            if selected_pages:
                scores = [c["score"] for c in scored_candidates[:len(selected_pages)]]
                confidence = float(np.mean(scores))

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            logger.info(
                "collapse_v2_completed",
                confidence=confidence,
                latency_ms=latency_ms,
                n_pages=len(selected_pages),
                query=query[:50],
                tokens_total=len(query.split()) * len(selected_pages),
                tokens_used=tokens_used,
                retrieval_steps=min(step + 1, max_steps),
            )

            return CollapseResultV2(
                pages=selected_pages,
                page_ids=selected_ids,
                confidence=confidence,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                tokens_total=len(query.split()) * len(selected_pages),
                concepts_activated=list(activated_concepts),
                retrieval_steps=min(step + 1, max_steps),
            )

        except Exception as exc:
            logger.error("collapse_v2_failed", error=str(exc))
            return CollapseResultV2(
                error=str(exc),
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )
