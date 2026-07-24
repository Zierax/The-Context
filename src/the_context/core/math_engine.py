# ============================================================================
# math_engine.py — Spectral Memory Manifold Co-Processor
# MATHEMATICAL ENGINE: All vector/matrix operations
# Pure NumPy/SciPy. Zero external ML APIs. Fully deterministic.
# ============================================================================

import hashlib
import math
from typing import Iterator

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import structlog

logger = structlog.get_logger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3.1 Sinusoidal Concept Encoder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def sinusoidal_encode(concepts: list[str], d_model: int = 512, dtype: np.dtype = np.float32) -> np.ndarray:
    """Deterministic static embedding for raw concept strings.

    Uses a fixed random Fourier feature map:
        z(x) = [cos(w_1 * h(x) + b_1), ..., cos(w_d * h(x) + b_d)]

    where w_j ~ N(0, 4.0), b_j ~ Uniform(0, 2π), and h(x) is a deterministic
    MD5-based hash of the concept string mapped to a float in [0, 2^31).

    Seeds: w_seed=42, b_seed=43. NEVER change.

    Args:
        concepts: List of concept strings to embed.
        d_model: Dimensionality of the output embedding space (default 512).
        dtype: Output data type (default float32 for memory efficiency).

    Returns:
        ndarray of shape (n_concepts, d_model), dtype as specified.

    Raises:
        ValueError: If concepts is empty.
    """
    if not concepts:
        raise ValueError("concepts list must not be empty")

    n = len(concepts)
    # Fixed random seeds for deterministic Fourier features
    rng_w = np.random.RandomState(42)
    rng_b = np.random.RandomState(43)

    w = rng_w.randn(d_model).astype(np.float64) * 2.0  # sigma = 2.0
    b = rng_b.uniform(0.0, 2.0 * math.pi, d_model).astype(np.float64)

    # Deterministic string-to-scalar hash for each concept
    hashes = np.zeros(n, dtype=np.float64)
    for i, concept in enumerate(concepts):
        raw = hashlib.md5(concept.encode("utf-8")).hexdigest()
        hash_int = int(raw, 16) % (2**31)
        hashes[i] = float(hash_int)

    # Vectorized: (n, d_model) = cos( (n,1) * (1,d_model) + (1,d_model) )
    embeddings = np.cos(
        hashes[:, np.newaxis] * w[np.newaxis, :] + b[np.newaxis, :]
    ).astype(dtype)

    logger.debug(
        "sinusoidal_encode",
        n_concepts=n,
        d_model=d_model,
        embeddings_shape=list(embeddings.shape),
        dtype=str(dtype),
    )
    return embeddings


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3.2 Seeded LSH Voronoi Partition
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SeededLSH:
    """Deterministic Locality-Sensitive Hashing with fixed random seeds.

    Partitions the manifold into Voronoi cells using seeded hash functions.
    Same query vector always maps to the same bucket. Bit-for-bit reproducible.
    """

    def __init__(self, d: int, w: float = 10.0, m: int = 8, seed: int = 42) -> None:
        """Initialise LSH with fixed random projections.

        Args:
            d: Dimensionality of input vectors.
            w: Bucket width for quantization.
            m: Number of hash functions (each produces one bucket coordinate).
            seed: Fixed random seed for reproducibility.
        """
        self.d = d
        self.w = w
        self.m = m
        self.seed = seed

        rng = np.random.RandomState(seed)
        # m hash vectors of dimension d: shape (m, d)
        self.a = rng.randn(m, d).astype(np.float64)
        # m bias terms: shape (m,)
        self.b = rng.uniform(0.0, w, m).astype(np.float64)

        logger.debug("SeededLSH initialized", d=d, w=w, m=m, seed=seed)

    def hash_vector(self, x: np.ndarray) -> tuple[int, ...]:
        """Hash a single query vector into an m-dimensional bucket index.

        Args:
            x: Vector of shape (d,), dtype float64.

        Returns:
            Tuple of m integer bucket coordinates.

        Raises:
            ValueError: If x shape does not match self.d.
        """
        if x.shape != (self.d,):
            raise ValueError(
                f"Expected vector shape ({self.d},), got {x.shape}"
            )
        h = np.floor((self.a @ x + self.b) / self.w).astype(int)
        return tuple(h.tolist())

    def hash_batch(self, X: np.ndarray) -> list[tuple[int, ...]]:
        """Hash a batch of vectors into bucket indices (vectorized).

        Args:
            X: Array of shape (n_vectors, d), dtype float64.

        Returns:
            List of n_vectors tuples, each of m integer bucket coordinates.

        Raises:
            ValueError: If X shape does not match (n, d).
        """
        if X.ndim != 2 or X.shape[1] != self.d:
            raise ValueError(
                f"Expected shape (n, {self.d}), got {X.shape}"
            )
        # X @ a.T: (n, m) dot products
        H = np.floor((X @ self.a.T + self.b) / self.w).astype(int)
        return [tuple(h.tolist()) for h in H]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3.2b LSH Deduplication
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def lsh_deduplicate(
    ids: list[str],
    vectors: np.ndarray,
    lsh: SeededLSH | None = None,
    similarity_threshold: float = 0.95,
    d: int = 512,
    m: int = 4,
    w: float = 10.0,
    seed: int = 42,
) -> list[tuple[str, str, float]]:
    """Detect near-duplicate pairs using LSH bucket collision.

    Two vectors are considered near-duplicates if they land in the same
    LSH bucket (all m hash functions agree). This is a very conservative
    threshold — only nearly identical content will collide.

    For finer-grained deduplication, use cosine similarity after bucketing.

    Args:
        ids: List of identifiers for each vector.
        vectors: Array of shape (n, d), the embeddings to deduplicate.
        lsh: Pre-initialized SeededLSH instance. If None, creates a new one.
        similarity_threshold: Cosine similarity threshold for duplicate detection.
            Default 0.95 (very near-duplicates only).
        d: Dimensionality of vectors (used only if lsh is None).
        m: Number of hash functions (used only if lsh is None).
        w: Hash width (used only if lsh is None).
        seed: Random seed (used only if lsh is None).

    Returns:
        List of (id_a, id_b, similarity) tuples for each detected duplicate pair.
        Each pair appears once (a < b by index).
    """
    if len(ids) != vectors.shape[0]:
        raise ValueError(f"ids length {len(ids)} != vectors shape {vectors.shape[0]}")

    if lsh is None:
        lsh = SeededLSH(d=d, m=m, w=w, seed=seed)

    n = len(ids)
    if n < 2:
        return []

    # Hash all vectors into buckets
    bucket_hashes = lsh.hash_batch(vectors.astype(np.float64))

    # Group indices by bucket hash
    buckets: dict[tuple[int, ...], list[int]] = {}
    for i, h in enumerate(bucket_hashes):
        buckets.setdefault(h, []).append(i)

    # Within each bucket, check cosine similarity for candidates
    duplicates: list[tuple[str, str, float]] = []
    for bucket_indices in buckets.values():
        if len(bucket_indices) < 2:
            continue
        # Check all pairs within this bucket
        for ii in range(len(bucket_indices)):
            for jj in range(ii + 1, len(bucket_indices)):
                i, j = bucket_indices[ii], bucket_indices[jj]
                # Cosine similarity
                vi = vectors[i].astype(np.float64)
                vj = vectors[j].astype(np.float64)
                norm_i = np.linalg.norm(vi)
                norm_j = np.linalg.norm(vj)
                if norm_i < 1e-10 or norm_j < 1e-10:
                    continue
                sim = float(np.dot(vi, vj) / (norm_i * norm_j))
                if sim >= similarity_threshold:
                    duplicates.append((ids[i], ids[j], sim))

    logger.debug(
        "lsh_deduplicate",
        n_vectors=n,
        n_buckets=len(buckets),
        n_duplicates=len(duplicates),
    )
    return duplicates


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Random Projection (Johnson-Lindenstrauss Lemma)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def random_projection_compress(
    vectors: np.ndarray,
    target_dim: int = 64,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Compress high-dimensional embeddings via random projection.

    Uses the Johnson-Lindenstrauss lemma: for n points in R^d, a random
    projection R^{d→k} where k = O(ε^{-2} log n) preserves pairwise
    distances within factor (1 ± ε) with high probability.

    For 256K tokens with ε=0.1: k ≈ 64 dimensions suffices.
    Compression: d/k = 512/64 = 8x

    Args:
        vectors: Input vectors of shape (n, d).
        target_dim: Target dimension k (default 64).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (compressed, projection_matrix) where:
            compressed: ndarray shape (n, k), the compressed vectors.
            projection_matrix: ndarray shape (d, k), the random projection matrix.
                Can be reused for new vectors without recomputing.
    """
    n, d = vectors.shape
    if target_dim >= d:
        return vectors.copy(), np.eye(d, target_dim)

    # Fixed random projection matrix (Gaussian)
    rng = np.random.RandomState(seed)
    R = rng.randn(d, target_dim).astype(np.float64) / np.sqrt(target_dim)

    # Project: (n, d) @ (d, k) = (n, k)
    compressed = (vectors.astype(np.float64) @ R).astype(np.float16)

    logger.debug(
        "random_projection_compress",
        input_shape=[n, d],
        output_shape=[n, target_dim],
        compression_ratio=d / target_dim,
    )
    return compressed, R


def random_projection_decompress(
    compressed: np.ndarray,
    projection_matrix: np.ndarray,
) -> np.ndarray:
    """Approximately reconstruct vectors from random projection.

    This is a lossy reconstruction. The reconstruction error depends on
    the target dimension and the number of original points.

    Args:
        compressed: Compressed vectors of shape (n, k).
        projection_matrix: Projection matrix of shape (d, k).

    Returns:
        Approximate reconstruction of shape (n, d).
    """
    # Pseudo-inverse reconstruction
    R_pinv = np.linalg.pinv(projection_matrix)
    return (compressed @ R_pinv).astype(np.float64)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3.3 Graph Laplacian Operations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def normalized_laplacian(A: sp.csr_matrix) -> sp.csr_matrix:
    """Compute the symmetric normalized graph Laplacian.

    L_sym = I - D^(-1/2) * A * D^(-1/2)

    A must be a symmetric, non-negative weighted adjacency matrix in CSR format.

    Args:
        A: Weighted adjacency matrix (CSR, symmetric, non-negative).

    Returns:
        L_sym as a sparse CSR matrix of the same shape as A.

    Raises:
        ValueError: If A is not square, not CSR, or has negative entries.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency must be square, got {A.shape}")
    if not isinstance(A, sp.csr_matrix):
        raise TypeError("A must be a scipy.sparse.csr_matrix")
    if A.data.size > 0 and A.data.min() < 0:
        raise ValueError("Adjacency matrix has negative entries")

    n = A.shape[0]
    row_sums = np.array(A.sum(axis=1)).flatten().astype(np.float64)
    # Compute D^(-1/2) without triggering divide-by-zero on isolated nodes.
    # Use a mask to only invert rows with positive degree.
    d_inv_sqrt = np.zeros(n, dtype=np.float64)
    positive_mask = row_sums > 0
    d_inv_sqrt[positive_mask] = 1.0 / np.sqrt(row_sums[positive_mask])
    D_inv_sqrt = sp.diags(d_inv_sqrt, format="csr")
    L_sym = sp.eye(n, format="csr") - D_inv_sqrt @ A @ D_inv_sqrt
    L_sym.eliminate_zeros()

    logger.debug("normalized_laplacian", n=n, nnz=L_sym.nnz)
    return L_sym


def spectral_signature(
    L_sym: sp.csr_matrix, k: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the top-k eigenvalues and eigenvectors of the Laplacian.

    Uses scipy.sparse.linalg.eigsh for efficient sparse eigensolving.
    Returns eigenvalues sorted ascending.

    Args:
        L_sym: Symmetric normalized Laplacian (CSR).
        k: Number of eigenvalues/eigenvectors to compute (default 10).

    Returns:
        Tuple of (eigenvalues, eigenvectors) where:
            eigenvalues: ndarray shape (k,), sorted ascending.
            eigenvectors: ndarray shape (n, k), corresponding eigenvectors.

    Raises:
        ValueError: If k >= n (too many eigenvalues requested for matrix size)
            or if eigsh does not converge.
    """
    n = L_sym.shape[0]
    if k >= n:
        raise ValueError(
            f"k={k} must be less than matrix dimension n={n}"
        )

    # Use which='SM' (smallest magnitude) for Laplacian eigenvalues.
    # For symmetric positive semi-definite matrices like graph Laplacians,
    # the smallest eigenvalues are real and non-negative.
    # Set maxiter proportional to k and use tight tolerance for convergence.
    max_iter = max(500, 20 * k)

    try:
        eigenvalues, eigenvectors = eigsh(
            L_sym, k=k, which="SM",
            maxiter=max_iter, tol=1e-8,
        )
    except Exception as exc:
        logger.error("spectral_signature_failed", k=k, n=n, error=str(exc))
        raise ValueError(
            f"eigsh did not converge for matrix of shape ({n},{n}) with k={k}"
        ) from exc

    # eigsh with which='SM' returns eigenvalues in ascending order
    logger.debug("spectral_signature", k=k, n=n, eigenvalues=list(eigenvalues))
    return eigenvalues, eigenvectors


def reconstruct_from_spectral(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    n_nodes: int,
    sigma: float = 1.0,
    sparsity_threshold: float = 1e-6,
) -> sp.csr_matrix:
    """Reconstruct an adjacency matrix from a spectral signature.

    Uses spectral graph drawing:
        X = diag(λ)^(-1/2) · U^T
    Then:
        A_ij = exp(-||x_i - x_j||² / 2σ²)  for i != j
        A_ii = 0

    Args:
        eigenvalues: ndarray shape (k,), sorted ascending.
        eigenvectors: ndarray shape (n_nodes, k).
        n_nodes: Total number of nodes in original graph.
        sigma: RBF kernel width (default 1.0).
        sparsity_threshold: Values below this are set to zero (default 1e-6).

    Returns:
        Reconstructed adjacency as sparse CSR matrix of shape (n_nodes, n_nodes).

    Raises:
        ValueError: If eigenvalue/eigenvector dimensions are inconsistent.
    """
    k = len(eigenvalues)
    if eigenvectors.shape != (n_nodes, k):
        raise ValueError(
            f"eigenvectors shape {eigenvectors.shape} != ({n_nodes}, {k})"
        )

    # X = diag(λ)^(-1/2) · U^T   → shape (k, n_nodes)
    # Use masked computation to avoid sqrt of negative/zero eigenvalues.
    lambda_inv_sqrt = np.zeros(k, dtype=np.float64)
    positive_mask = eigenvalues > 1e-10
    lambda_inv_sqrt[positive_mask] = 1.0 / np.sqrt(eigenvalues[positive_mask])
    X = (eigenvectors * lambda_inv_sqrt[np.newaxis, :]).T  # (k, n_nodes)

    # Squared pairwise distances
    # (k, 1, n_nodes) - (k, n_nodes, 1) broadcasting for efficiency
    X_3d = X[:, :, np.newaxis]  # (k, n_nodes, 1)
    sq_diffs = (X_3d - X_3d.transpose(0, 2, 1)) ** 2  # (k, n_nodes, n_nodes)
    sq_dists = np.sum(sq_diffs, axis=0)  # (n_nodes, n_nodes)

    A_dense = np.exp(-sq_dists / (2.0 * sigma * sigma), dtype=np.float64)
    np.fill_diagonal(A_dense, 0.0)
    A_dense[A_dense < sparsity_threshold] = 0.0

    A_recon = sp.csr_matrix(A_dense)
    A_recon.eliminate_zeros()

    # Compute reconstruction error wrt identity (theoretical bound)
    reconstruction_error = math.sqrt(
        np.sum((A_dense - np.eye(n_nodes)) ** 2)
    ) / (n_nodes * n_nodes)

    logger.debug(
        "reconstruct_from_spectral",
        n_nodes=n_nodes,
        k=k,
        nnz=A_recon.nnz,
        reconstruction_error=reconstruction_error,
    )
    return A_recon


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3.4 Fokker-Planck Diffusion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def fokker_planck_step(
    rho: np.ndarray,
    L: sp.csr_matrix,
    q: np.ndarray,
    alpha: float = 0.1,
    beta: float = 0.5,
    gamma: float = 0.01,
) -> np.ndarray:
    """One discrete step of Fokker-Planck diffusion on the graph.

    ρ_{t+1} = ρ_t + α·(L·ρ_t) + β·(q ⊙ ρ_t) - γ·ρ_t

    where:
        L    = normalized Laplacian (CSR)
        q    = query activation vector (1 at activated nodes, 0 elsewhere)
        α    = diffusion coefficient (default 0.1)
        β    = Hebbian reinforcement (default 0.5)
        γ    = decay rate (default 0.01)

    Args:
        rho: Current memory strength vector, shape (n_nodes,), dtype float64.
        L: Normalized graph Laplacian (CSR), shape (n_nodes, n_nodes).
        q: Query activation vector, shape (n_nodes,), 1.0 at activated positions.
        alpha: Diffusion coefficient (default 0.1).
        beta: Hebbian reinforcement coefficient (default 0.5).
        gamma: Exponential decay rate (default 0.01).

    Returns:
        Updated rho vector of shape (n_nodes,), dtype float64.

    Raises:
        ValueError: If rho or q have wrong shape or contain NaN.
    """
    n = L.shape[0]
    if rho.shape != (n,):
        raise ValueError(
            f"rho shape {rho.shape} != ({n},)"
        )
    if q.shape != (n,):
        raise ValueError(
            f"q shape {q.shape} != ({n},)"
        )
    if np.any(np.isnan(rho)):
        raise ValueError("rho contains NaN values")
    if np.any(np.isnan(q)):
        raise ValueError("q contains NaN values")

    # ρ_{t+1} = ρ_t + α·(L·ρ_t) + β·(q⊙ρ_t) - γ·ρ_t
    diffusion = L @ rho  # sparse mat-vec, O(nnz)
    hebbian = q * rho
    rho_next = rho + alpha * diffusion + beta * hebbian - gamma * rho

    # Clamp to non-negative to prevent pathological states
    rho_next = np.maximum(rho_next, 0.0)

    # Normalize to prevent unbounded Hebbian growth.
    # Without this, the multiplicative β·(q⊙ρ) term causes ρ to grow
    # exponentially across queries (reaching 1e+102 after ~100 queries).
    # Normalization keeps ρ as a valid probability distribution over nodes.
    rho_sum = np.sum(rho_next)
    if rho_sum > 0.0:
        rho_next = rho_next / rho_sum

    return rho_next


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3.5 Submodular Context Packing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def submodular_pack(
    candidates: list[dict], budget: int
) -> list[str]:
    """Select a subset of candidates maximizing submodular coverage under a token budget.

    Uses the greedy algorithm which achieves a (1 - 1/e) approximation guarantee.
    At each step, selects the candidate with the highest marginal gain per token.

    Optimised with incremental coverage computation: instead of recomputing
    f(S) from scratch for every candidate at each step, maintains a running
    coverage map and computes per-candidate marginal gains incrementally.

    Args:
        candidates: List of dicts, each with keys:
            'id' (str), 'text' (str), 'token_count' (int),
            'concept_coverage' (dict[str, float]), 'strength' (float).
        budget: Maximum total token count for the selected subset.

    Returns:
        Ordered list of candidate IDs (highest priority first).

    Raises:
        ValueError: If budget < 1 or candidates is empty.
    """
    if budget < 1:
        raise ValueError(f"budget must be >= 1, got {budget}")
    if not candidates:
        return []

    selected_ids: set[str] = set()
    selected: list[str] = []
    remaining_budget = budget

    # Pre-filter candidates that fit in the budget
    valid = [c for c in candidates if c["token_count"] <= budget]
    if not valid:
        # No candidate fits — return the one with highest concept coverage density
        # (concepts per token), not just the smallest
        best = max(
            candidates,
            key=lambda c: sum(c["concept_coverage"].values()) / max(c["token_count"], 1),
        )
        return [best["id"]]

    # Precompute per-candidate gains: for each concept, the score contribution
    # Running coverage map: concept -> max weighted score seen so far
    running_coverage: dict[str, float] = {}

    while remaining_budget > 0:
        best_gain = -1.0
        best_candidate: dict | None = None

        for c in valid:
            if c["id"] in selected_ids:
                continue
            if c["token_count"] > remaining_budget:
                continue

            # Compute marginal gain incrementally without recomputing f(S)
            marginal = 0.0
            strength = c["strength"]
            for concept, score in c["concept_coverage"].items():
                weighted = score * strength
                current_best = running_coverage.get(concept, 0.0)
                if weighted > current_best:
                    marginal += weighted - current_best

            gain_per_token = marginal / max(c["token_count"], 1)
            if gain_per_token > best_gain:
                best_gain = gain_per_token
                best_candidate = c

        if best_candidate is None:
            # Fallback: select remaining candidates by raw strength (fill budget)
            remaining = [
                c for c in valid
                if c["id"] not in selected_ids and c["token_count"] <= remaining_budget
            ]
            remaining.sort(key=lambda c: c["strength"], reverse=True)
            for c in remaining:
                if c["token_count"] <= remaining_budget:
                    selected.append(c["id"])
                    selected_ids.add(c["id"])
                    remaining_budget -= c["token_count"]
                    if remaining_budget <= 0:
                        break
            break

        # Update running coverage with selected candidate
        strength = best_candidate["strength"]
        for concept, score in best_candidate["concept_coverage"].items():
            weighted = score * strength
            if weighted > running_coverage.get(concept, 0.0):
                running_coverage[concept] = weighted

        selected.append(best_candidate["id"])
        selected_ids.add(best_candidate["id"])
        remaining_budget -= best_candidate["token_count"]

    logger.debug(
        "submodular_pack",
        n_candidates=len(candidates),
        budget=budget,
        n_selected=len(selected),
        remaining_budget=remaining_budget,
    )
    return selected


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3.6 Gaussian Patch Operations (B2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def compute_gaussian_patch(
    vectors: np.ndarray,
    store_diagonal_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a Gaussian patch (B2 beacon) from a neighborhood of B1 embeddings.

    The patch is represented by its mean and precision matrix (inverse covariance).

    Args:
        vectors: Array of shape (n_vectors, d) — B1 embeddings in a neighborhood.
            Must have at least 2 vectors.
        store_diagonal_only: If True, return only the diagonal of Sigma_inv as a 1D
            array of shape (d,) instead of the full (d, d) matrix. This saves
            significant memory (~2 MB per B2 beacon for d=512) with zero loss
            since Sigma_inv is never used in the query path. Default False for
            backward compatibility.

    Returns:
        Tuple of (mu, Sigma_inv) where:
            mu: ndarray shape (d,), the mean of the neighborhood.
            Sigma_inv: ndarray shape (d, d) if store_diagonal_only=False,
                or (d,) if store_diagonal_only=True — pseudo-inverse of covariance.

    Raises:
        ValueError: If fewer than 2 vectors are provided.
    """
    n, d = vectors.shape
    if n < 2:
        raise ValueError(
            f"Need at least 2 vectors for a Gaussian patch, got {n}"
        )

    mu = np.mean(vectors, axis=0).astype(np.float64)
    centered = vectors - mu
    cov = (centered.T @ centered) / (n - 1)

    # Compute precision matrix (inverse covariance) using SVD-based pseudo-inverse.
    # For high-dimensional data where n < d (underdetermined case), the covariance
    # matrix is singular and pinv is the correct choice. np.linalg.pinv uses LAPACK
    # gesvd which handles singular matrices correctly.
    Sigma_inv = np.linalg.pinv(cov).astype(np.float64)

    if store_diagonal_only:
        # Return only diagonal — saves d*(d-1)*8 bytes per B2 beacon
        # This is safe because Sigma_inv is never used in the query path
        Sigma_inv_diag = np.diag(Sigma_inv).astype(np.float64)
        logger.debug(
            "compute_gaussian_patch",
            n_vectors=n,
            d=d,
            mu_norm=np.linalg.norm(mu),
            store_diagonal_only=True,
        )
        return mu, Sigma_inv_diag

    logger.debug(
        "compute_gaussian_patch",
        n_vectors=n,
        d=d,
        mu_norm=np.linalg.norm(mu),
    )
    return mu, Sigma_inv


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Spectral Compression via SVD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def svd_compress(
    matrix: np.ndarray,
    rank: int | None = None,
    compression_ratio: float | None = None,
    explained_variance_threshold: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compress a matrix via truncated SVD (low-rank approximation).

    Given matrix A of shape (m, n), computes A ≈ U_k @ diag(S_k) @ V_k^T
    where k << min(m, n). This achieves compression by storing only the top-k
    singular values and vectors.

    Compression ratio: (m * n) / (m * k + k + n * k) ≈ min(m, n) / (m + n + 1)
    For square n×n matrix with rank k: n² / (2nk + k) ≈ n / (2k)

    Args:
        matrix: Input matrix of shape (m, n).
        rank: Target rank k. Must satisfy 1 <= k <= min(m, n).
            Mutually exclusive with compression_ratio.
        compression_ratio: Target compression ratio (must be >= 1.0).
            If provided, rank is computed as floor(min(m, n) / compression_ratio / 2).
            Mutually exclusive with rank.
        explained_variance_threshold: If rank is None and compression_ratio is None,
            choose k such that cumulative explained variance >= this threshold.
            Default 0.95 (keep 95% of variance).

    Returns:
        Tuple of (U_k, S_k, V_kT, actual_compression_ratio) where:
            U_k: ndarray shape (m, k), left singular vectors.
            S_k: ndarray shape (k,), singular values.
            V_kT: ndarray shape (k, n), right singular vectors transposed.
            actual_compression_ratio: float, actual compression achieved.

    Raises:
        ValueError: If matrix is empty, rank is out of bounds, or both rank
            and compression_ratio are provided.
    """
    m, n = matrix.shape
    if m == 0 or n == 0:
        raise ValueError("Matrix must be non-empty")

    k_max = min(m, n)

    if rank is not None and compression_ratio is not None:
        raise ValueError("Provide either rank or compression_ratio, not both")

    if compression_ratio is not None:
        if compression_ratio < 1.0:
            raise ValueError("compression_ratio must be >= 1.0")
        k = max(1, int(k_max / compression_ratio / 2))
    elif rank is not None:
        if not (1 <= rank <= k_max):
            raise ValueError(f"rank must be in [1, {k_max}], got {rank}")
        k = rank
    else:
        # Auto-select rank based on explained variance
        # Use full SVD for small matrices, randomized for large
        if k_max <= 500:
            _, S_full, _ = np.linalg.svd(matrix.astype(np.float64), full_matrices=False)
        else:
            from scipy.sparse.linalg import svds
            # svds returns largest singular values
            S_full = svds(matrix.astype(np.float64), k=min(k_max - 1, 100))[1]
            S_full = np.sort(S_full)[::-1]

        # Cumulative explained variance
        total_var = np.sum(S_full ** 2)
        if total_var > 0:
            cumvar = np.cumsum(S_full ** 2) / total_var
            k = int(np.searchsorted(cumvar, explained_variance_threshold) + 1)
            k = max(1, min(k, k_max))
        else:
            k = 1

    # Compute truncated SVD
    if k_max <= 500:
        U_k, S_k, V_kT = np.linalg.svd(
            matrix.astype(np.float64), full_matrices=False
        )
        U_k = U_k[:, :k]
        S_k = S_k[:k]
        V_kT = V_kT[:k, :]
    else:
        from scipy.sparse.linalg import svds
        # svds returns k smallest, we need k largest
        k_compute = min(k, k_max - 1)
        U_k, S_k, V_kT = svds(matrix.astype(np.float64), k=k_compute)
        # Sort by descending singular value
        sort_idx = np.argsort(S_k)[::-1]
        U_k = U_k[:, sort_idx[:k]]
        S_k = S_k[sort_idx[:k]]
        V_kT = V_kT[sort_idx[:k], :]

    # Compute actual compression ratio
    original_size = m * n
    compressed_size = m * k + k + k * n  # U_k + S_k + V_kT
    actual_ratio = original_size / max(compressed_size, 1)

    logger.debug(
        "svd_compress",
        input_shape=[m, n],
        rank=k,
        explained_variance=float(np.sum(S_k ** 2) / max(np.sum(np.linalg.svd(matrix.astype(np.float64), full_matrices=False)[1] ** 2), 1e-10)),
        compression_ratio=actual_ratio,
    )

    return U_k, S_k, V_kT, actual_ratio


def svd_decompress(
    U_k: np.ndarray,
    S_k: np.ndarray,
    V_kT: np.ndarray,
) -> np.ndarray:
    """Reconstruct matrix from truncated SVD components.

    Args:
        U_k: Left singular vectors, shape (m, k).
        S_k: Singular values, shape (k,).
        V_kT: Right singular vectors transposed, shape (k, n).

    Returns:
        Reconstructed matrix of shape (m, n).
    """
    return U_k @ np.diag(S_k) @ V_kT


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Utility: approximate token count for a text string
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def estimate_token_count(text: str) -> int:
    """Estimate the number of tokens in a text string.

    For space-separated token lists (as produced by ingest), word count
    is the most accurate estimate. For free-form text, BPE splitting
    adds ~30% overhead.

    Args:
        text: Input text string.

    Returns:
        Estimated token count (minimum 1).
    """
    if not text:
        return 1
    words = text.split()
    word_count = len(words)
    # Space-separated token lists: word count IS the token count
    # Free-form text: add ~30% for BPE subword splitting
    return max(1, int(word_count * 1.1))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Utility: generate a stream of tokens from a text corpus iterator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def tokenize(text: str) -> Iterator[str]:
    """Split text into whitespace-delimited tokens, yielding one at a time.

    Args:
        text: Input text string.

    Yields:
        Individual tokens (strings).
    """
    for token in text.split():
        yield token
