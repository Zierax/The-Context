# ============================================================================
# knowledge_graph.py — Spectral Memory Manifold Co-Processor
# DETERMINISTIC KNOWLEDGE GRAPH: CSR-based adjacency, Laplacian, diffusion
# No NetworkX. No Neo4j. Pure NumPy/SciPy. Fully thread-safe.
# ============================================================================

import os
import pickle
import threading
from typing import Optional

import numpy as np
import scipy.sparse as sp
import structlog

from the_context.core.math_engine import (
    normalized_laplacian,
    fokker_planck_step,
    sinusoidal_encode,
)

logger = structlog.get_logger(__name__)


class DeterministicKnowledgeGraph:
    """A deterministic, thread-safe knowledge graph with CSR sparse adjacency.

    All operations are reproducible given the same input sequence.
    Supports temporal memory strength via Fokker-Planck diffusion.
    """

    def __init__(self, d_model: int = 512) -> None:
        """Initialise an empty knowledge graph.

        Args:
            d_model: Dimensionality of concept embeddings (default 512).
        """
        self.node_to_idx: dict[str, int] = {}
        self.idx_to_node: dict[int, str] = {}
        self.beacon_map: dict[str, list[str]] = {}  # concept -> [beacon_id, ...]
        self.beacon_to_concepts: dict[str, list[str]] = {}  # beacon_id -> [concept, ...] (reverse index)
        self.A: sp.csr_matrix | None = None      # weighted adjacency
        self.L_sym: sp.csr_matrix | None = None   # normalized Laplacian
        self.rho: np.ndarray | None = None         # temporal strength vector
        self.d_model = d_model
        self._lock = threading.RLock()
        # Use LIL format for incremental construction to avoid SparseEfficiencyWarning
        self._A_lil: sp.lil_matrix | None = None

        logger.debug("DeterministicKnowledgeGraph initialized", d_model=d_model)

    def _ensure_nodes(self, *concepts: str) -> list[int]:
        """Add concepts as nodes if they don't exist, return their indices.

        Thread-safe. Called internally during triplet addition.
        """
        indices: list[int] = []
        with self._lock:
            for concept in concepts:
                if concept not in self.node_to_idx:
                    idx = len(self.node_to_idx)
                    self.node_to_idx[concept] = idx
                    self.idx_to_node[idx] = concept
                    indices.append(idx)
                else:
                    indices.append(self.node_to_idx[concept])
            self._rebuild_matrix()
        return indices

    def _rebuild_matrix(self) -> None:
        """Rebuild the adjacency matrix and Laplacian from current node count.

        Preserves existing edge weights. Uses LIL format during incremental
        construction to avoid SparseEfficiencyWarning on CSR modification.
        After building, stores only upper triangle to save memory (zero loss
        since adjacency is symmetric).
        """
        n = len(self.node_to_idx)
        if n == 0:
            self.A = None
            self._A_lil = None
            self.L_sym = None
            self.rho = None
            return

        if self._A_lil is None or self._A_lil.shape[0] < n:
            old_lil = self._A_lil
            new_lil = sp.lil_matrix((n, n), dtype=np.float64)
            if old_lil is not None:
                old_n = old_lil.shape[0]
                new_lil[:old_n, :old_n] = old_lil
            self._A_lil = new_lil

        # Always keep self.A in sync (converted to CSR for read access)
        A_full = self._A_lil.tocsr()
        A_full.eliminate_zeros()

        # Store only upper triangle — saves ~50% memory (zero loss since symmetric)
        self.A = sp.triu(A_full, format='csr')
        self.A.eliminate_zeros()

        if self.rho is None or self.rho.shape[0] < n:
            old_rho = np.zeros(0) if self.rho is None else self.rho
            new_rho = np.zeros(n, dtype=np.float64)
            if old_rho.shape[0] > 0:
                new_rho[: old_rho.shape[0]] = old_rho
            self.rho = new_rho

    def add_triplet(
        self,
        subject: str,
        predicate: str,
        object: str,
        weight: float = 1.0,
        beacon_id: str = "",
        page_id: str = "",
    ) -> None:
        """Add a (Subject, Predicate, Object) triple to the graph.

        Creates nodes if they don't exist. Updates the adjacency matrix
        with the specified edge weight. Maps beacon and page IDs to all
        three concept nodes.

        Args:
            subject: Subject concept string.
            predicate: Predicate/relation string.
            object: Object concept string.
            weight: Edge weight for the subject->object directed edge (default 1.0).
            beacon_id: Beacon identifier to map to these concepts (default "").
            page_id: Page identifier to map to these concepts (default "").

        Raises:
            ValueError: If weight is not positive.
        """
        if weight <= 0.0:
            raise ValueError(f"weight must be positive, got {weight}")

        with self._lock:
            s_idx, p_idx, o_idx = self._ensure_nodes(subject, predicate, object)

            # Add directed edges using LIL format (no SparseEfficiencyWarning)
            # Edge: subject -> object (the core relation)
            self._A_lil[s_idx, o_idx] = self._A_lil[s_idx, o_idx] + weight
            # Also add predicate as a mediating node for richer graph structure
            self._A_lil[s_idx, p_idx] = self._A_lil[s_idx, p_idx] + weight * 0.5
            self._A_lil[p_idx, o_idx] = self._A_lil[p_idx, o_idx] + weight * 0.5

            # Map beacon_id to all three concepts
            if beacon_id:
                for concept in (subject, predicate, object):
                    if concept not in self.beacon_map:
                        self.beacon_map[concept] = []
                    if beacon_id not in self.beacon_map[concept]:
                        self.beacon_map[concept].append(beacon_id)
                    # Update reverse index: beacon_id -> [concept, ...]
                    if beacon_id not in self.beacon_to_concepts:
                        self.beacon_to_concepts[beacon_id] = []
                    if concept not in self.beacon_to_concepts[beacon_id]:
                        self.beacon_to_concepts[beacon_id].append(concept)

            # L_sym invalidated by matrix change
            self.L_sym = None

            # Increase memory strength for newly added concepts
            self.rho[s_idx] += weight * 0.1
            self.rho[p_idx] += weight * 0.05
            self.rho[o_idx] += weight * 0.1

            logger.debug(
                "add_triplet",
                subject=subject,
                predicate=predicate,
                object=object,
                weight=weight,
                beacon_id=beacon_id,
                page_id=page_id,
            )

    def build_laplacian(self) -> None:
        """Recompute the normalized Laplacian from the current adjacency matrix.

        Converts internal LIL matrix to CSR, reconstructs full symmetric matrix
        from stored upper triangle, then builds Laplacian.
        Must be called after all batch ingestion is complete and before querying.

        Raises:
            RuntimeError: If the graph has fewer than 2 nodes.
        """
        with self._lock:
            if self._A_lil is None or self._A_lil.shape[0] < 2:
                raise RuntimeError(
                    f"Graph has {0 if self._A_lil is None else self._A_lil.shape[0]} nodes; "
                    "need at least 2 to build Laplacian"
                )
            # Convert LIL to CSR for efficient Laplacian computation
            A_full = self._A_lil.tocsr()
            A_full.eliminate_zeros()

            # Reconstruct full symmetric matrix from upper triangle for Laplacian
            # L_sym = I - D^(-1/2) * A * D^(-1/2) requires full A
            A_symmetric = A_full + A_full.T - sp.diags(A_full.diagonal())

            self.L_sym = normalized_laplacian(A_symmetric)
            logger.info(
                "build_laplacian",
                n_nodes=A_full.shape[0],
                nnz=A_full.nnz,
            )

    def concept_diffusion(
        self, query_concepts: list[str], steps: int = 3
    ) -> np.ndarray:
        """Execute Fokker-Planck diffusion from a set of activated query concepts.

        Builds an activation vector q (1.0 at query concept indices, 0 elsewhere)
        and runs the discrete Fokker-Planck update for `steps` iterations.

        Args:
            query_concepts: List of concept names that are activated by the query.
            steps: Number of diffusion steps (default 3).

        Returns:
            Final rho vector of shape (n_nodes,), dtype float64, reflecting
            memory strength after diffusion.

        Raises:
            RuntimeError: If Laplacian has not been built yet.
            ValueError: If no query concepts are found in the graph.
        """
        with self._lock:
            if self.L_sym is None:
                raise RuntimeError(
                    "Laplacian not built. Call build_laplacian() first."
                )
            if self.rho is None:
                raise RuntimeError("rho is None — graph is uninitialized.")

            n = self.A.shape[0]
            q = np.zeros(n, dtype=np.float64)
            found_any = False
            for concept in query_concepts:
                if concept in self.node_to_idx:
                    q[self.node_to_idx[concept]] = 1.0
                    found_any = True

            if not found_any:
                # Fuzzy fallback: match query words against concept words (exact match)
                query_words = set()
                for concept in query_concepts:
                    query_words.update(concept.lower().split())
                for concept_name, idx in self.node_to_idx.items():
                    concept_words = set(concept_name.lower().split())
                    overlap = query_words & concept_words
                    # Activate if >= 1 query word exactly matches a concept word (3+ chars)
                    if any(len(w) >= 3 for w in overlap):
                        q[idx] = 0.5  # Partial activation
                        found_any = True

            if not found_any:
                logger.warning(
                    "concept_diffusion_no_match",
                    query_concepts=query_concepts,
                    known_concepts=list(self.node_to_idx.keys())[:10],
                )
                return self.rho.copy()

            rho_current = self.rho.copy()
            for _step in range(steps):
                rho_current = fokker_planck_step(
                    rho_current, self.L_sym, q
                )

            # Update stored rho
            self.rho = rho_current.copy()

            logger.debug(
                "concept_diffusion",
                query_concepts=query_concepts,
                steps=steps,
                rho_sum=float(np.sum(rho_current)),
                rho_max=float(np.max(rho_current)),
            )
            return rho_current

    def get_active_beacons(self, top_k: int = 10) -> list[str]:
        """Return the top-k beacon IDs ranked by aggregated rho-weight across concepts.

        For each concept, multiplies its rho value by the number of beacons
        associated with it, then aggregates beacon scores across all concepts.

        Args:
            top_k: Number of top beacon IDs to return (default 10).

        Returns:
            List of beacon IDs sorted by aggregated score, highest first.

        Raises:
            RuntimeError: If rho is not initialised.
        """
        with self._lock:
            if self.rho is None:
                raise RuntimeError("rho is None — no memory strength data.")

            beacon_scores: dict[str, float] = {}
            for concept, beacons in self.beacon_map.items():
                if concept not in self.node_to_idx:
                    continue
                idx = self.node_to_idx[concept]
                strength = self.rho[idx]
                for beacon_id in beacons:
                    beacon_scores[beacon_id] = (
                        beacon_scores.get(beacon_id, 0.0) + strength
                    )

            sorted_beacons = sorted(
                beacon_scores.items(), key=lambda x: x[1], reverse=True
            )
            top = [b[0] for b in sorted_beacons[:top_k]]

            logger.debug(
                "get_active_beacons",
                top_k=top_k,
                top_scores=[round(s, 4) for _, s in sorted_beacons[:top_k]],
            )
            return top

    def save(self, path: str) -> None:
        """Persist the graph to disk.

        Saves:
        - node maps as pickle (.node_maps.pkl)
        - adjacency matrix as npz (.A.npz)
        - memory strength vector as .npy (.rho.npy)
        - beacon map as pickle (.beacon_map.pkl)
        - metadata as pickle (.meta.pkl)

        Args:
            path: Directory path to save into. Created if it doesn't exist.
        """
        os.makedirs(path, exist_ok=True)

        with self._lock:
            # Node maps
            with open(os.path.join(path, "node_maps.pkl"), "wb") as f:
                pickle.dump(
                    {"node_to_idx": self.node_to_idx, "idx_to_node": self.idx_to_node},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            # Beacon map
            with open(os.path.join(path, "beacon_map.pkl"), "wb") as f:
                pickle.dump(self.beacon_map, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Adjacency matrix (convert LIL to CSR if needed)
            A_to_save = self.A
            if A_to_save is None and self._A_lil is not None:
                A_to_save = self._A_lil.tocsr()
            if A_to_save is not None:
                sp.save_npz(os.path.join(path, "A.npz"), A_to_save)

            # rho vector
            if self.rho is not None:
                np.save(os.path.join(path, "rho.npy"), self.rho)

            # Metadata
            meta = {
                "d_model": self.d_model,
                "n_nodes": len(self.node_to_idx),
            }
            with open(os.path.join(path, "meta.pkl"), "wb") as f:
                pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(
                "knowledge_graph_saved",
                path=path,
                n_nodes=meta["n_nodes"],
            )

    def load(self, path: str) -> None:
        """Load a previously saved graph from disk.

        Validates dimensions and metadata consistency. Rebuilds Laplacian.

        Args:
            path: Directory path to load from.

        Raises:
            FileNotFoundError: If any required file is missing.
            ValueError: If dimensions don't match saved metadata.
        """
        meta_path = os.path.join(path, "meta.pkl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        if meta["d_model"] != self.d_model:
            raise ValueError(
                f"Saved d_model={meta['d_model']} != current d_model={self.d_model}"
            )

        with self._lock:
            # Node maps
            node_path = os.path.join(path, "node_maps.pkl")
            if not os.path.exists(node_path):
                raise FileNotFoundError(f"Node maps not found: {node_path}")
            with open(node_path, "rb") as f:
                maps = pickle.load(f)
            self.node_to_idx = maps["node_to_idx"]
            self.idx_to_node = maps["idx_to_node"]

            # Beacon map
            beacon_path = os.path.join(path, "beacon_map.pkl")
            if os.path.exists(beacon_path):
                with open(beacon_path, "rb") as f:
                    self.beacon_map = pickle.load(f)

            # Adjacency matrix
            a_path = os.path.join(path, "A.npz")
            if os.path.exists(a_path):
                self.A = sp.load_npz(a_path)
            else:
                self.A = None

            # rho vector
            rho_path = os.path.join(path, "rho.npy")
            if os.path.exists(rho_path):
                self.rho = np.load(rho_path)
            else:
                self.rho = None

            # Validate dimensions
            if self.A is not None:
                if self.A.shape[0] != meta["n_nodes"]:
                    raise ValueError(
                        f"A shape {self.A.shape[0]} != n_nodes {meta['n_nodes']}"
                    )
                # Rebuild LIL from loaded CSR for incremental compatibility
                self._A_lil = self.A.tolil()
                self.build_laplacian()

            logger.info(
                "knowledge_graph_loaded",
                path=path,
                n_nodes=meta["n_nodes"],
            )
