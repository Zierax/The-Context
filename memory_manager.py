# ============================================================================
# memory_manager.py — Spectral Memory Manifold Co-Processor
# VIRTUAL MEMORY TREE: Page allocation, beacon hierarchy, LRU eviction
# Disk-backed persistence. Thread-safe. Deterministic eviction.
# ============================================================================

import os
import threading
from collections import OrderedDict
from typing import Iterator, Optional

import numpy as np
import scipy.sparse as sp
import structlog

from math_engine import (
    compute_gaussian_patch,
    spectral_signature,
    normalized_laplacian,
    sinusoidal_encode,
    estimate_token_count,
)

logger = structlog.get_logger(__name__)


def _generate_beacon_id(prefix: str, index: int) -> str:
    """Generate a deterministic beacon ID string.

    Args:
        prefix: Beacon level prefix (e.g., 'b1', 'b2', 'b3').
        index: Monotonic index for uniqueness.

    Returns:
        Beacon ID string like 'b1_0000042'.
    """
    return f"{prefix}_{index:07d}"


class VirtualMemoryTree:
    """Hierarchical virtual memory with B1→B2→B3 beacon compression.

    Manages the three-tier beacon hierarchy:
    - B1: Individual page embeddings (1,000 tokens each)
    - B2: Gaussian patches over 10 B1 beacons (10,000 tokens)
    - B3: Spectral signatures over 10 B2 beacons (100,000 tokens)

    LRU eviction is backed by wave interference (rho * kappa) scores.
    Disk persistence for evicted pages.
    """

    def __init__(
        self,
        page_size: int = 1000,
        cache_size: int = 10,
        persist_dir: str = ".context",
    ) -> None:
        """Initialise the virtual memory tree.

        Args:
            page_size: Tokens per B1 page (default 1000).
            cache_size: Max B1 pages in working memory before LRU eviction (default 100).
            persist_dir: Directory for disk-backed page storage (default '.context').
        """
        self.page_size = page_size
        self.cache_size = cache_size

        # Persistence directory
        self.persist_dir = persist_dir
        self.pages_dir = os.path.join(persist_dir, "pages")
        os.makedirs(self.pages_dir, exist_ok=True)

        # B1 level: individual pages
        self.pages: OrderedDict[str, str] = OrderedDict()  # page_id -> text
        self.beacon_b1: dict[str, np.ndarray] = {}         # beacon_id -> embedding
        self.page_to_beacon: dict[str, str] = {}            # page_id -> b1_beacon_id

        # B2 level: Gaussian patches over 10 B1 beacons
        self.beacon_b2: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # beacon_id -> (mu, Sigma_inv_diag)
        # mu: float16 shape (d,) — mean of B1 embeddings
        # Sigma_inv_diag: float64 shape (d,) — diagonal of precision matrix
        # (full matrix not needed — never used in query path)

        # B3 level: spectral signatures over 10 B2 beacons
        self.beacon_b3: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        # beacon_id -> (eigenvalues, eigenvectors)

        # B1 -> B2 mapping
        self.b1_to_b2: dict[str, str] = {}  # b1_beacon_id -> b2_beacon_id
        # B2 -> B3 mapping
        self.b2_to_b3: dict[str, str] = {}  # b2_beacon_id -> b3_beacon_id

        # Reverse index maps for O(1) parent→children lookups (query hot path)
        self.b2_to_b1_list: dict[str, list[str]] = {}  # b2_id -> [b1_id, ...]
        self.b3_to_b2_list: dict[str, list[str]] = {}  # b3_id -> [b2_id, ...]
        self.b1_to_pages: dict[str, list[str]] = {}    # b1_id -> [page_id, ...]

        # LRU scores: page_id -> rho * kappa score
        self.lru_scores: dict[str, float] = {}

        # Monotonic counters for beacon ID generation
        self._b1_counter = 0
        self._b2_counter = 0
        self._b3_counter = 0

        # Batching for B2 compression
        self._pending_b1_for_b2: list[str] = []  # B1 beacon IDs waiting for B2

        # Batching for B3 compression
        self._pending_b2_for_b3: list[str] = []  # B2 beacon IDs waiting for B3

        self._lock = threading.RLock()

        logger.debug(
            "VirtualMemoryTree initialized",
            page_size=page_size,
            cache_size=cache_size,
            persist_dir=persist_dir,
        )

    def ingest_stream(self, token_stream: Iterator[str]) -> Iterator[str]:
        """Consume a token stream, chunk into pages, yield page IDs.

        Streams tokens without materialising the entire corpus in RAM.
        Automatically triggers B2 and B3 compression when batches are full.

        Args:
            token_stream: Iterator yielding individual tokens (strings).

        Yields:
            Page IDs (strings) for each created page.
        """
        buffer: list[str] = []
        token_count = 0

        for token in token_stream:
            buffer.append(token)
            token_count += 1

            if token_count >= self.page_size:
                page_id = self._flush_buffer(buffer)
                yield page_id
                buffer = []
                token_count = 0

        # Flush remaining tokens
        if buffer:
            page_id = self._flush_buffer(buffer)
            yield page_id

        # Final flush: compress any remaining pending beacons into B3
        self._flush_pending_beacons()

    def _flush_buffer(self, tokens: list[str]) -> str:
        """Flush a token buffer into a B1 page with embedding.

        Args:
            tokens: List of tokens to form a page.

        Returns:
            Page ID string.
        """
        with self._lock:
            text = " ".join(tokens)
            page_id = f"page_{len(self.pages) + len(os.listdir(self.pages_dir)):07d}"

            # Compute embedding via sinusoidal encoding of page content
            # Use page text segments as pseudo-concepts for embedding
            concepts = self._extract_embedding_concepts(text)
            if concepts:
                embedding = sinusoidal_encode(concepts, d_model=512, dtype=np.float16)
                # Mean pool over concepts to get page-level embedding
                page_embedding = np.mean(embedding, axis=0).astype(np.float16)
            else:
                page_embedding = np.zeros(512, dtype=np.float16)

            # Allocate B1 beacon
            beacon_id = self.allocate_b1(page_id, text, page_embedding)

            logger.debug(
                "page_flushed",
                page_id=page_id,
                beacon_id=beacon_id,
                n_tokens=len(tokens),
            )
            return page_id

    def _extract_embedding_concepts(self, text: str) -> list[str]:
        """Extract meaningful embedding concepts from page text.

        Splits text into unique lowercase words, filters short/noise words.

        Args:
            text: Page text content.

        Returns:
            List of concept strings (max 50 to limit computation).
        """
        words = text.lower().split()
        concepts = list(
            OrderedDict.fromkeys(
                w.strip(".,;:!?\"'()[]{}") for w in words if len(w) >= 3
            )
        )
        return concepts[:50]

    def allocate_b1(self, page_id: str, text: str, embedding: np.ndarray) -> str:
        """Store a B1 beacon. Evict lowest-score page if cache is full.

        Args:
            page_id: Page identifier string.
            text: Full page text content.
            embedding: Page embedding vector of shape (d_model,).

        Returns:
            B1 beacon ID string.
        """
        with self._lock:
            self._b1_counter += 1
            beacon_id = _generate_beacon_id("b1", self._b1_counter)

            # Evict if cache full
            if len(self.pages) >= self.cache_size:
                self._evict_lowest_score()

            self.pages[page_id] = text
            self.beacon_b1[beacon_id] = embedding
            self.page_to_beacon[page_id] = beacon_id
            # Update reverse map: b1_id -> [page_id, ...]
            if beacon_id not in self.b1_to_pages:
                self.b1_to_pages[beacon_id] = []
            self.b1_to_pages[beacon_id].append(page_id)
            self.lru_scores[page_id] = 0.0

            # Queue for B2 compression
            self._pending_b1_for_b2.append(beacon_id)
            if len(self._pending_b1_for_b2) >= 10:
                self._compress_pending_b2()

            return beacon_id

    def _evict_lowest_score(self) -> None:
        """Evict the page with the lowest LRU score.

        Writes evicted page to disk. Removes from working memory.
        """
        if not self.pages:
            return

        # Find page with lowest LRU score
        worst_page = min(self.lru_scores, key=self.lru_scores.get)
        self.evict_page(worst_page)

    def compress_b2(self, b1_beacons: list[str]) -> str:
        """Group B1 beacons into a B2 Gaussian patch.

        Args:
            b1_beacons: List of exactly 10 B1 beacon IDs.

        Returns:
            B2 beacon ID string.

        Raises:
            ValueError: If not exactly 10 B1 beacons provided.
        """
        if len(b1_beacons) != 10:
            raise ValueError(
                f"Need exactly 10 B1 beacons for B2 compression, got {len(b1_beacons)}"
            )

        with self._lock:
            self._b2_counter += 1
            b2_id = _generate_beacon_id("b2", self._b2_counter)

            # Gather embeddings for all B1 beacons
            vectors_list: list[np.ndarray] = []
            valid_beacons: list[str] = []
            for bid in b1_beacons:
                if bid in self.beacon_b1:
                    vectors_list.append(self.beacon_b1[bid])
                    valid_beacons.append(bid)

            if len(vectors_list) < 2:
                logger.warning(
                    "compress_b2_insufficient_vectors",
                    b2_id=b2_id,
                    n_vectors=len(vectors_list),
                )
                # Return a zero-patch placeholder
                mu = np.zeros(self.beacon_b1[next(iter(self.beacon_b1))].shape[0] if self.beacon_b1 else 512, dtype=np.float16)
                Sigma_inv_diag = np.ones(mu.shape[0], dtype=np.float64)
                self.beacon_b2[b2_id] = (mu, Sigma_inv_diag)
                return b2_id

            vectors = np.stack(vectors_list, axis=0)
            mu, Sigma_inv_diag = compute_gaussian_patch(vectors, store_diagonal_only=True)
            # Store mu as float16 to save memory (saves 512*6 = 3KB per B2 beacon)
            self.beacon_b2[b2_id] = (mu.astype(np.float16), Sigma_inv_diag)

            # Map B1 -> B2 and update reverse index
            self.b2_to_b1_list[b2_id] = list(valid_beacons)
            for bid in valid_beacons:
                self.b1_to_b2[bid] = b2_id

            # Queue for B3 compression
            self._pending_b2_for_b3.append(b2_id)
            if len(self._pending_b2_for_b3) >= 10:
                self._compress_pending_b3()

            logger.debug(
                "compress_b2",
                b2_id=b2_id,
                n_vectors=len(vectors_list),
            )
            return b2_id

    def _compress_pending_b2(self) -> None:
        """Compress all pending B1 beacons into B2 beacons in batch."""
        while len(self._pending_b1_for_b2) >= 10:
            batch = self._pending_b1_for_b2[:10]
            self._pending_b1_for_b2 = self._pending_b1_for_b2[10:]
            self.compress_b2(batch)

    def compress_b3(self, b2_beacons: list[str]) -> str:
        """Group B2 beacons into a B3 spectral signature.

        Args:
            b2_beacons: List of exactly 10 B2 beacon IDs.

        Returns:
            B3 beacon ID string.

        Raises:
            ValueError: If not exactly 10 B2 beacons provided.
        """
        if len(b2_beacons) != 10:
            raise ValueError(
                f"Need exactly 10 B2 beacons for B3 compression, got {len(b2_beacons)}"
            )

        with self._lock:
            self._b3_counter += 1
            b3_id = _generate_beacon_id("b3", self._b3_counter)

            # Build adjacency from B2 patch similarities
            n = len(b2_beacons)
            vectors_list: list[np.ndarray] = []
            valid_beacons: list[str] = []
            for bid in b2_beacons:
                if bid in self.beacon_b2:
                    mu, _ = self.beacon_b2[bid]
                    # Convert float16 back to float64 for spectral computation
                    vectors_list.append(mu.astype(np.float64))
                    valid_beacons.append(bid)

            if len(vectors_list) < 2:
                logger.warning(
                    "compress_b3_insufficient_vectors",
                    b3_id=b3_id,
                    n_vectors=len(vectors_list),
                )
                # Store dummy signature
                dummy_ev = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                dummy_evec = np.eye(3, dtype=np.float64)
                self.beacon_b3[b3_id] = (dummy_ev, dummy_evec)
                return b3_id

            # Compute pairwise similarity adjacency
            vecs = np.stack(vectors_list, axis=0)
            sim = vecs @ vecs.T
            sim = np.maximum(sim, 0.0)  # Ensure non-negative
            np.fill_diagonal(sim, 0.0)

            # Build graph Laplacian on this small (n x n) graph
            A_small = sim / (np.max(sim) + 1e-10)
            A_sparse = sp.csr_matrix(A_small.astype(np.float64))

            # Compute spectral signature
            # Use k=5 instead of k=10 to save memory (50% reduction in B3 storage)
            # Still captures the dominant spectral modes for accurate queries
            k = min(5, n - 1)
            eigenvalues, eigenvectors = spectral_signature(A_sparse, k=k)

            self.beacon_b3[b3_id] = (eigenvalues, eigenvectors)

            # Map B2 -> B3 and update reverse index
            self.b3_to_b2_list[b3_id] = list(valid_beacons)
            for bid in valid_beacons:
                self.b2_to_b3[bid] = b3_id

            logger.debug(
                "compress_b3",
                b3_id=b3_id,
                k=k,
                eigenvalues=list(eigenvalues),
            )
            return b3_id

    def _flush_pending_beacons(self) -> None:
        """Flush remaining pending beacons after ingestion completes.

        Compresses remaining B1→B2 and B2→B3. For partial batches (< 10),
        compresses with what's available — partial spectral signatures
        are still better than no hierarchy at all.
        """
        # Flush B1 → B2 (no lock — compress_b2 acquires its own)
        while len(self._pending_b1_for_b2) >= 10:
            batch = self._pending_b1_for_b2[:10]
            self._pending_b1_for_b2 = self._pending_b1_for_b2[10:]
            self.compress_b2(batch)
        if self._pending_b1_for_b2:
            batch = list(self._pending_b1_for_b2)
            self._pending_b1_for_b2 = []
            if len(batch) >= 2:
                self.compress_b2_partial(batch)

        # Flush B2 → B3
        while len(self._pending_b2_for_b3) >= 10:
            batch = self._pending_b2_for_b3[:10]
            self._pending_b2_for_b3 = self._pending_b2_for_b3[10:]
            self.compress_b3(batch)
        if self._pending_b2_for_b3:
            batch = list(self._pending_b2_for_b3)
            self._pending_b2_for_b3 = []
            if len(batch) >= 2:
                self.compress_b3_partial(batch)

    def compress_b2_partial(self, b1_beacons: list[str]) -> str:
        """Compress a partial batch of B1 beacons (< 10) into a B2 patch.

        Same as compress_b2 but accepts any number of beacons >= 2.
        """
        if len(b1_beacons) < 2:
            raise ValueError(f"Need at least 2 B1 beacons, got {len(b1_beacons)}")

        with self._lock:
            self._b2_counter += 1
            b2_id = _generate_beacon_id("b2", self._b2_counter)

            vectors_list: list[np.ndarray] = []
            valid_beacons: list[str] = []
            for bid in b1_beacons:
                if bid in self.beacon_b1:
                    vectors_list.append(self.beacon_b1[bid])
                    valid_beacons.append(bid)

            if len(vectors_list) < 2:
                dummy_mu = np.zeros(self._d_model, dtype=np.float16)
                dummy_sigma = np.eye(self._d_model, dtype=np.float16)
                self.beacon_b2[b2_id] = (dummy_mu, dummy_sigma)
                self.b2_to_b1_list[b2_id] = valid_beacons
                return b2_id

            vecs = np.stack(vectors_list, axis=0)
            mu = np.mean(vecs, axis=0)
            centered = vecs - mu[np.newaxis, :]
            Sigma = (centered.T @ centered) / len(vecs)
            Sigma_inv = np.linalg.pinv(Sigma.astype(np.float64)).astype(np.float16)

            self.beacon_b2[b2_id] = (mu.astype(np.float16), Sigma_inv)
            self.b2_to_b1_list[b2_id] = valid_beacons
            for bid in valid_beacons:
                self.b1_to_b2[bid] = b2_id
            self._pending_b2_for_b3.append(b2_id)
            return b2_id

    def compress_b3_partial(self, b2_beacons: list[str]) -> str:
        """Compress a partial batch of B2 beacons (< 10) into a B3 signature.

        Same as compress_b3 but accepts any number of beacons >= 2.
        """
        if len(b2_beacons) < 2:
            raise ValueError(f"Need at least 2 B2 beacons, got {len(b2_beacons)}")

        with self._lock:
            self._b3_counter += 1
            b3_id = _generate_beacon_id("b3", self._b3_counter)

            vectors_list: list[np.ndarray] = []
            valid_beacons: list[str] = []
            for bid in b2_beacons:
                if bid in self.beacon_b2:
                    mu, _ = self.beacon_b2[bid]
                    vectors_list.append(mu.astype(np.float64))
                    valid_beacons.append(bid)

            if len(vectors_list) < 2:
                k = min(3, len(vectors_list))
                dummy_ev = np.zeros(max(k, 1), dtype=np.float64)
                dummy_evec = np.eye(max(k, 1), dtype=np.float64)
                self.beacon_b3[b3_id] = (dummy_ev, dummy_evec)
                self.b3_to_b2_list[b3_id] = valid_beacons
                for bid in valid_beacons:
                    self.b2_to_b3[bid] = b3_id
                return b3_id

            vecs = np.stack(vectors_list, axis=0)
            sim = vecs @ vecs.T
            sim = np.maximum(sim, 0.0)
            np.fill_diagonal(sim, 0.0)

            A_small = sim / (np.max(sim) + 1e-10)
            A_sparse = sp.csr_matrix(A_small.astype(np.float64))

            k = min(5, len(vectors_list) - 1)
            eigenvalues, eigenvectors = spectral_signature(A_sparse, k=max(k, 1))

            self.beacon_b3[b3_id] = (eigenvalues, eigenvectors)
            self.b3_to_b2_list[b3_id] = valid_beacons
            for bid in valid_beacons:
                self.b2_to_b3[bid] = b3_id
            return b3_id

    def _compress_pending_b3(self) -> None:
        """Compress all pending B2 beacons into B3 beacons in batch."""
        while len(self._pending_b2_for_b3) >= 10:
            batch = self._pending_b2_for_b3[:10]
            self._pending_b2_for_b3 = self._pending_b2_for_b3[10:]
            self.compress_b3(batch)

    def get_page(self, page_id: str) -> str | None:
        """Retrieve a page text by ID. Update LRU score.

        Moves retrieved page to the end of the OrderedDict (most recently used).

        Args:
            page_id: Page identifier string.

        Returns:
            Page text string, or None if not in working memory or on disk.
        """
        with self._lock:
            if page_id in self.pages:
                # Move to end (most recently used)
                text = self.pages.pop(page_id)
                self.pages[page_id] = text
                # Update LRU score (higher = more recently used)
                self.lru_scores[page_id] = self.lru_scores.get(page_id, 0.0) + 1.0
                return text

            # Try loading from disk
            disk_text = self.load_page_from_disk(page_id)
            if disk_text is not None:
                # Restore to cache if space available
                if len(self.pages) < self.cache_size:
                    self.pages[page_id] = disk_text
                    self.lru_scores[page_id] = 1.0
                return disk_text

            return None

    def update_page_score(self, page_id: str, score_delta: float) -> None:
        """Update the LRU score for a page by adding a delta.

        Args:
            page_id: Page identifier string.
            score_delta: Amount to add to the current score.
        """
        with self._lock:
            if page_id in self.lru_scores:
                self.lru_scores[page_id] = self.lru_scores.get(page_id, 0.0) + score_delta

    def evict_page(self, page_id: str) -> None:
        """Write a page to disk and remove from working memory.

        Args:
            page_id: Page identifier string to evict.
        """
        with self._lock:
            if page_id not in self.pages:
                return

            text = self.pages[page_id]
            page_path = os.path.join(self.pages_dir, f"{page_id}.txt")
            try:
                with open(page_path, "w", encoding="utf-8") as f:
                    f.write(text)
            except OSError as exc:
                logger.error(
                    "evict_page_write_failed",
                    page_id=page_id,
                    error=str(exc),
                )
                raise

            # Clean up working memory
            del self.pages[page_id]
            if page_id in self.lru_scores:
                del self.lru_scores[page_id]

            # NOTE: page_to_beacon and b1_to_pages are intentionally PRESERVED.
            # The query pipeline (collapse → expand_b3_to_pages → b1_to_pages)
            # needs these mappings to discover page IDs, then calls get_page()
            # which loads from disk. Only the page text is evicted.

            logger.debug(
                "evict_page",
                page_id=page_id,
                text_length=len(text),
            )

    def load_page_from_disk(self, page_id: str) -> str | None:
        """Read a page from disk storage.

        Args:
            page_id: Page identifier string.

        Returns:
            Page text string, or None if the file doesn't exist.
        """
        page_path = os.path.join(self.pages_dir, f"{page_id}.txt")
        try:
            if not os.path.exists(page_path):
                return None
            with open(page_path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError as exc:
            logger.error(
                "load_page_from_disk_failed",
                page_id=page_id,
                error=str(exc),
            )
            return None

    def get_b1_beacon(self, beacon_id: str) -> np.ndarray | None:
        """Retrieve a B1 beacon embedding by ID.

        Args:
            beacon_id: B1 beacon identifier string.

        Returns:
            Embedding vector of shape (d_model,), or None if not found.
        """
        return self.beacon_b1.get(beacon_id)

    def get_b2_beacon(self, beacon_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Retrieve a B2 Gaussian patch by ID.

        Args:
            beacon_id: B2 beacon identifier string.

        Returns:
            Tuple of (mu, Sigma_inv_diag) where:
                mu: float16 array shape (d,) — mean of B1 embeddings.
                Sigma_inv_diag: float64 array shape (d,) — diagonal of precision matrix.
            Returns None if not found.
        """
        return self.beacon_b2.get(beacon_id)

    def get_b3_beacon(self, beacon_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Retrieve a B3 spectral signature by ID.

        Args:
            beacon_id: B3 beacon identifier string.

        Returns:
            Tuple of (eigenvalues, eigenvectors), or None if not found.
        """
        return self.beacon_b3.get(beacon_id)

    def get_beacon_for_page(self, page_id: str) -> str | None:
        """Get the B1 beacon ID associated with a page.

        Args:
            page_id: Page identifier string.

        Returns:
            B1 beacon ID string, or None if not mapped.
        """
        return self.page_to_beacon.get(page_id)

    def get_pages_for_b1(self, b1_id: str) -> list[str]:
        """Get all page IDs mapped to a B1 beacon.

        Args:
            b1_id: B1 beacon identifier string.

        Returns:
            List of page IDs.
        """
        return [
            pid for pid, bid in self.page_to_beacon.items() if bid == b1_id
        ]

    def get_b2_for_b1(self, b1_id: str) -> str | None:
        """Get the B2 beacon containing a given B1 beacon.

        Args:
            b1_id: B1 beacon identifier string.

        Returns:
            B2 beacon ID, or None if not yet compressed.
        """
        return self.b1_to_b2.get(b1_id)

    def get_b3_for_b2(self, b2_id: str) -> str | None:
        """Get the B3 beacon containing a given B2 beacon.

        Args:
            b2_id: B2 beacon identifier string.

        Returns:
            B3 beacon ID, or None if not yet compressed.
        """
        return self.b2_to_b3.get(b2_id)

    def get_all_b1_ids(self) -> list[str]:
        """Get all B1 beacon IDs currently in memory.

        Returns:
            List of B1 beacon ID strings.
        """
        return list(self.beacon_b1.keys())

    def get_all_b2_ids(self) -> list[str]:
        """Get all B2 beacon IDs currently in memory.

        Returns:
            List of B2 beacon ID strings.
        """
        return list(self.beacon_b2.keys())

    def get_all_b3_ids(self) -> list[str]:
        """Get all B3 beacon IDs currently in memory.

        Returns:
            List of B3 beacon ID strings.
        """
        return list(self.beacon_b3.keys())

    def save_state(self, path: str) -> None:
        """Persist the entire memory tree state to disk.

        Args:
            path: Directory path to save into.
        """
        import pickle

        os.makedirs(path, exist_ok=True)

        with self._lock:
            state = {
                "beacon_b1_ids": list(self.beacon_b1.keys()),
                "beacon_b2_ids": list(self.beacon_b2.keys()),
                "beacon_b3_ids": list(self.beacon_b3.keys()),
                "page_to_beacon": self.page_to_beacon,
                "b1_to_b2": self.b1_to_b2,
                "b2_to_b3": self.b2_to_b3,
                "lru_scores": self.lru_scores,
                "b1_counter": self._b1_counter,
                "b2_counter": self._b2_counter,
                "b3_counter": self._b3_counter,
            }
            with open(os.path.join(path, "memory_state.pkl"), "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save B1 embeddings
            if self.beacon_b1:
                b1_ids = list(self.beacon_b1.keys())
                b1_embs = np.stack([self.beacon_b1[bid] for bid in b1_ids], axis=0)
                np.savez(os.path.join(path, "beacon_b1.npz"), ids=b1_ids, embeddings=b1_embs)

            logger.info("memory_tree_saved", path=path)

    def get_cache_fill(self) -> float:
        """Get the fraction of cache currently in use.

        Returns:
            Float between 0.0 and 1.0.
        """
        return len(self.pages) / max(self.cache_size, 1)

    def get_total_pages(self) -> int:
        """Get the total number of pages managed (memory + disk).

        Returns:
            Total page count.
        """
        memory_pages = len(self.pages)
        disk_pages = 0
        if os.path.exists(self.pages_dir):
            disk_pages = len(
                [f for f in os.listdir(self.pages_dir) if f.endswith(".txt")]
            )
        return memory_pages + disk_pages
