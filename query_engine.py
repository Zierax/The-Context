# ============================================================================
# query_engine.py — Hierarchical Beacon Query Engine
# ORCHESTRATION LAYER: Query pipeline execution
# Executes Q -> LSH bucket -> concept diffusion -> B3->B2->B1 expansion
# -> submodular packing -> CollapseResult assembly
# ============================================================================

import time
from typing import Optional

import numpy as np
import structlog
from pydantic import BaseModel, Field

import scipy.sparse as sp

from math_engine import (
    SeededLSH,
    sinusoidal_encode,
    submodular_pack,
    estimate_token_count,
    fokker_planck_step,
)
from knowledge_graph import DeterministicKnowledgeGraph
from memory_manager import VirtualMemoryTree

logger = structlog.get_logger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pydantic Models (shared between query_engine and mcp_server)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CollapseRequest(BaseModel):
    """Request schema for the collapse_quantum_memory tool."""

    query: str
    max_tokens: int = Field(default=4096, ge=1, le=128000)
    temperature: float = Field(default=0.0)  # ignored; determinism enforced
    required_concepts: list[str] = Field(default_factory=list)
    session_id: str = Field(default="default")


class CollapseResult(BaseModel):
    """Response schema from the collapse_quantum_memory tool."""

    pages: list[str] = Field(default_factory=list)
    beacon_ids: list[str] = Field(default_factory=list)
    concepts_activated: list[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    tokens_used: int = Field(default=0, ge=0)
    tokens_total: int = Field(default=0, ge=0)
    compression_ratio: float = Field(default=0.0, ge=0.0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    error: str | None = Field(default=None)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# QueryEngine: Orchestration Layer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class QueryEngine:
    """Hierarchical beacon query engine.

    Executes the full query→result pipeline deterministically:
    1. Embed query → q
    2. Hash q → bucket via seeded LSH
    3. Get concepts in bucket from knowledge graph
    4. Compute proximity = exp(-||q - v_i||²)
    5. Diffuse via Fokker-Planck
    6. Rank beacon regions by active mass
    7. Expand top-k B3 → B2 → B1 (with B2/B1 fallback)
    8. Submodular pack into max_tokens
    9. Assemble CollapseResult with telemetry
    """

    def __init__(
        self,
        tree: VirtualMemoryTree,
        graph: DeterministicKnowledgeGraph,
        lsh: SeededLSH,
        d_model: int = 512,
    ) -> None:
        """Initialise the QueryEngine with core components.

        Args:
            tree: VirtualMemoryTree managing page storage and beacon hierarchy.
            graph: DeterministicKnowledgeGraph with concept adjacency.
            lsh: SeededLSH for deterministic Voronoi partitioning.
            d_model: Dimensionality of the embedding space (default 512).
                      Must match lsh.d.
        """
        self.tree = tree
        self.graph = graph
        self.lsh = lsh
        self.d_model = d_model

        # Caches for concept embeddings and LSH buckets
        # These are lazily built on first query and rebuilt when graph changes
        self._cached_concepts: list[str] | None = None
        self._cached_embeddings: np.ndarray | None = None
        self._cached_buckets: list[tuple[int, ...]] | None = None

        # Track graph state for cache invalidation (O(1) check)
        self._last_concept_count: int = -1

        if self.lsh.d != d_model:
            logger.warning(
                "d_model_mismatch",
                lsh_d=lsh.d,
                d_model=d_model,
            )

        logger.debug("QueryEngine initialized", d_model=d_model)

    def _ensure_caches(self) -> None:
        """Build or rebuild concept embedding and LSH bucket caches.

        Uses concept count as a fast O(1) invalidation check.
        Caches are rebuilt only when the number of concepts changes.
        """
        current_count = len(self.graph.node_to_idx)
        if self._cached_concepts is not None and self._last_concept_count == current_count:
            return  # Cache is still valid

        concepts = list(self.graph.node_to_idx.keys())
        if not concepts:
            self._cached_concepts = []
            self._cached_embeddings = np.zeros((0, self.d_model), dtype=np.float64)
            self._cached_buckets = []
            self._last_concept_count = 0
            return

        logger.debug(
            "rebuilding_concept_caches",
            n_concepts=len(concepts),
        )

        # Compute embeddings for all concepts once
        embeddings = sinusoidal_encode(concepts, d_model=self.d_model)

        # Hash all concept embeddings through LSH once
        buckets = self.lsh.hash_batch(embeddings)

        self._cached_concepts = concepts
        self._cached_embeddings = embeddings
        self._cached_buckets = buckets
        self._last_concept_count = current_count

    def collapse(
        self,
        query: str,
        max_tokens: int = 4096,
        required_concepts: list[str] | None = None,
    ) -> CollapseResult:
        """Execute the full collapse pipeline for a query.

        Args:
            query: Natural language query string.
            max_tokens: Maximum token budget for returned pages (default 4096).
            required_concepts: Optional list of concepts that MUST be in results.

        Returns:
            CollapseResult containing ordered pages, metadata, and telemetry.

        Raises:
            This method does not raise; all errors are captured in CollapseResult.error.
        """
        start_time = time.perf_counter()

        try:
            # Validate input
            if not query or not query.strip():
                return CollapseResult(
                    error="Empty query",
                    latency_ms=(time.perf_counter() - start_time) * 1000.0,
                    concepts_activated=[],
                )

            req_concepts = required_concepts or []

            # STEP 1: Embed query into semantic space
            try:
                query_tokens = query.lower().split()
                if not query_tokens:
                    return CollapseResult(
                        error="Query produced no tokens",
                        latency_ms=(time.perf_counter() - start_time) * 1000.0,
                    )
                # Use query tokens as "concepts" for sinusoidal encoding
                query_embedding = sinusoidal_encode(query_tokens, d_model=self.d_model)
                q = np.mean(query_embedding, axis=0).astype(np.float64)
            except Exception as exc:
                logger.error("query_embedding_failed", query=query, error=str(exc))
                return CollapseResult(
                    error=f"Query embedding failed: {exc}",
                    latency_ms=(time.perf_counter() - start_time) * 1000.0,
                )

            # STEP 2: Hash q into LSH bucket
            try:
                bucket = self.lsh.hash_vector(q)
            except Exception as exc:
                logger.error("lsh_hash_failed", query=query, error=str(exc))
                return CollapseResult(
                    error=f"LSH hashing failed: {exc}",
                    latency_ms=(time.perf_counter() - start_time) * 1000.0,
                )

            # STEP 3: Identify candidate concepts (uses caches)
            self._ensure_caches()
            if not self._cached_concepts:
                return CollapseResult(
                    error="Knowledge graph is empty — no concepts indexed",
                    latency_ms=(time.perf_counter() - start_time) * 1000.0,
                )

            all_concepts = self._cached_concepts
            concept_embeddings = self._cached_embeddings
            concept_buckets = self._cached_buckets

            # STEP 4: Compute concept proximity via LSH bucket filtering
            # Use exact bucket match (all m dimensions must agree).
            # This maps to the correct Voronoi cell — any-match is too loose
            # and defeats the purpose of LSH partitioning.
            candidate_indices: list[int] = []
            for i, cb in enumerate(concept_buckets):
                if cb == bucket:
                    candidate_indices.append(i)

            if not candidate_indices:
                # Fallback: compute exact cosine proximity for all concepts
                # rather than blindly returning everything.
                logger.warning(
                    "no_concepts_in_bucket",
                    bucket=bucket,
                    n_concepts=len(all_concepts),
                )
                diffs_all = concept_embeddings - q[np.newaxis, :]
                sq_dists_all = np.sum(diffs_all ** 2, axis=1)
                # Take concepts within 2σ of the nearest concept distance
                nearest_dist = np.min(sq_dists_all) if sq_dists_all.size > 0 else 0.0
                threshold = nearest_dist + 2.0 * max(np.std(sq_dists_all), 1.0)
                candidate_indices = list(np.where(sq_dists_all <= threshold)[0])
                if not candidate_indices:
                    candidate_indices = list(range(len(all_concepts)))

            # Compute proximity scores (vectorized)
            candidate_names = [all_concepts[i] for i in candidate_indices]
            candidate_embs = concept_embeddings[candidate_indices]
            diffs = candidate_embs - q[np.newaxis, :]
            sq_dists = np.sum(diffs ** 2, axis=1)
            proximities = np.exp(-sq_dists)

            # STEP 5: Diffuse via Fokker-Planck
            try:
                # Reset rho to uniform before each query (stateless context engineering)
                n_nodes = len(self.graph.node_to_idx)
                if n_nodes > 0:
                    self.graph.rho = np.ones(n_nodes, dtype=np.float64) / n_nodes

                # Build activation vector from top-10 proximal concepts
                top_prox_indices = np.argsort(proximities)[-10:]
                activated_concepts = [candidate_names[i] for i in top_prox_indices]

                # Also activate concepts that share exact tokens with the query
                query_tokens_set = set(query_tokens)
                for concept_name in all_concepts:
                    concept_tokens = set(concept_name.lower().split())
                    overlap = query_tokens_set & concept_tokens
                    # If >= 2 query words appear in the concept, activate it
                    if len(overlap) >= 2 and concept_name not in activated_concepts:
                        activated_concepts.append(concept_name)
                    # Also activate if a query word exactly matches a concept word (4+ chars)
                    elif any(w in concept_tokens for w in query_tokens if len(w) >= 4) and concept_name not in activated_concepts:
                        activated_concepts.append(concept_name)

                # Also add required concepts
                for rc in req_concepts:
                    if rc not in activated_concepts and rc in self.graph.node_to_idx:
                        activated_concepts.append(rc)

                diffused_rho = self.graph.concept_diffusion(
                    activated_concepts, steps=3
                )
            except RuntimeError as exc:
                # Laplacian not built yet — return empty result
                return CollapseResult(
                    error=f"Diffusion failed: {exc}",
                    latency_ms=(time.perf_counter() - start_time) * 1000.0,
                )

            # STEP 6: Rank regions by active mass
            # Try B3 → B2 → B1, fall back to lower levels if hierarchy incomplete
            beacon_to_concepts = self.graph.beacon_to_concepts
            candidate_pages: list[dict] = []
            all_pages_set: set[str] = set()

            def _expand_beacon_to_pages(b1_id: str) -> list[dict]:
                """Expand a single B1 beacon to candidate pages."""
                pages_list = []
                for page_id in self.tree.b1_to_pages.get(b1_id, []):
                    if page_id in all_pages_set:
                        continue
                    all_pages_set.add(page_id)
                    text = self.tree.get_page(page_id)
                    if text is None:
                        continue
                    concept_coverage: dict[str, float] = {}
                    for concept in beacon_to_concepts.get(b1_id, []):
                        if concept in self.graph.node_to_idx:
                            idx = self.graph.node_to_idx[concept]
                            concept_coverage[concept] = float(diffused_rho[idx])
                    token_count = estimate_token_count(text)
                    strength = float(
                        np.mean(diffused_rho) if diffused_rho.size > 0 else 0.0
                    )
                    pages_list.append({
                        "id": page_id,
                        "text": text,
                        "token_count": token_count,
                        "concept_coverage": concept_coverage,
                        "strength": strength,
                    })
                return pages_list

            def _rank_beacon_by_concepts(b_id: str) -> float:
                """Compute active mass for a beacon via beacon_to_concepts."""
                mass = 0.0
                for b1_id in self.tree.b2_to_b1_list.get(b_id, []):
                    for concept in beacon_to_concepts.get(b1_id, []):
                        if concept in self.graph.node_to_idx:
                            idx = self.graph.node_to_idx[concept]
                            mass += diffused_rho[idx]
                return mass

            b3_ids = self.tree.get_all_b3_ids()
            if b3_ids:
                # Full B3 hierarchy available
                b3_scores: list[tuple[str, float]] = []
                for b3_id in b3_ids:
                    active_mass = _rank_beacon_by_concepts(b3_id)
                    b3_scores.append((b3_id, active_mass))
                b3_scores.sort(key=lambda x: x[1], reverse=True)
                top_b3 = b3_scores[:5]

                for b3_id, _score in top_b3:
                    b2_ids = self.tree.b3_to_b2_list.get(b3_id, [])
                    for b2_id in b2_ids:
                        b1_ids = self.tree.b2_to_b1_list.get(b2_id, [])
                        for b1_id in b1_ids:
                            candidate_pages.extend(_expand_beacon_to_pages(b1_id))
            else:
                # Fallback: rank B2 beacons directly
                b2_ids = list(self.tree.beacon_b2.keys())
                if b2_ids:
                    b2_scores = [(b2_id, _rank_beacon_by_concepts(b2_id)) for b2_id in b2_ids]
                    b2_scores.sort(key=lambda x: x[1], reverse=True)
                    for b2_id, _score in b2_scores[:5]:
                        b1_ids = self.tree.b2_to_b1_list.get(b2_id, [])
                        for b1_id in b1_ids:
                            candidate_pages.extend(_expand_beacon_to_pages(b1_id))
                else:
                    # Last resort: expand all B1 beacons
                    for b1_id in list(self.tree.beacon_b1.keys()):
                        candidate_pages.extend(_expand_beacon_to_pages(b1_id))

            # Include required concepts pages
            if req_concepts:
                for rc in req_concepts:
                    if rc in self.graph.beacon_map:
                        for beacon_id in self.graph.beacon_map[rc]:
                            pages = self.tree.get_pages_for_b1(beacon_id)
                            for page_id in pages:
                                if page_id not in all_pages_set:
                                    all_pages_set.add(page_id)
                                    text = self.tree.get_page(page_id)
                                    if text is None:
                                        continue
                                    token_count = estimate_token_count(text)
                                    candidate_pages.append({
                                        "id": page_id,
                                        "text": text,
                                        "token_count": token_count,
                                        "concept_coverage": {rc: 1.0},
                                        "strength": 1.0,
                                    })

            if not candidate_pages:
                return CollapseResult(
                    error="No candidate pages found",
                    latency_ms=(time.perf_counter() - start_time) * 1000.0,
                    concepts_activated=activated_concepts,
                )

            # STEP 8: Rank candidates by query-specific activation density,
            # then submodular pack into token budget
            tokens_total = sum(c["token_count"] for c in candidate_pages)
            # Sort by activation density (sum of concept_rho / token_count)
            # This ensures the most query-relevant page is prioritized
            for c in candidate_pages:
                c["_activation_density"] = sum(c["concept_coverage"].values()) / max(c["token_count"], 1)
            candidate_pages.sort(key=lambda c: c["_activation_density"], reverse=True)
            selected_ids = submodular_pack(candidate_pages, max_tokens)

            # Build result
            selected_pages: list[str] = []
            selected_beacons: list[str] = []
            tokens_used = 0
            total_coverage = 0.0

            for sid in selected_ids:
                c = next(c for c in candidate_pages if c["id"] == sid)
                selected_pages.append(c["text"])
                tokens_used += c["token_count"]
                total_coverage += sum(c["concept_coverage"].values())
                # Get beacon IDs for this page
                b1_id = self.tree.get_beacon_for_page(sid)
                if b1_id and b1_id not in selected_beacons:
                    selected_beacons.append(b1_id)

            # Compute confidence score
            confidence = min(1.0, total_coverage / max(tokens_used, 1) * 100.0)

            # Update memory strength (Hebbian reinforcement) for retrieved pages
            for sid in selected_ids:
                self.tree.update_page_score(sid, 0.5)

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            result = CollapseResult(
                pages=selected_pages,
                beacon_ids=selected_beacons,
                concepts_activated=activated_concepts,
                confidence_score=round(confidence, 4),
                tokens_used=tokens_used,
                tokens_total=tokens_total,
                compression_ratio=(
                    round(tokens_total / max(tokens_used, 1), 2)
                    if tokens_used > 0
                    else 0.0
                ),
                latency_ms=round(elapsed_ms, 2),
            )

            logger.info(
                "collapse_completed",
                query=query[:50],
                n_pages=len(selected_pages),
                tokens_used=tokens_used,
                tokens_total=tokens_total,
                confidence=confidence,
                latency_ms=elapsed_ms,
            )
            return result

        except Exception as exc:
            logger.exception(
                "collapse_failed",
                query=query[:50],
                error=str(exc),
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return CollapseResult(
                error=f"Collapse pipeline failed: {exc}",
                latency_ms=round(elapsed_ms, 2),
                concepts_activated=[],
            )
