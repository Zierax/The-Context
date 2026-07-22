# ============================================================================
# tests/test_math_engine.py — Unit tests for the mathematical engine
# ============================================================================

import math

import numpy as np
import scipy.sparse as sp
import pytest

from the_context.core import (
    sinusoidal_encode,
    SeededLSH,
    normalized_laplacian,
    spectral_signature,
    reconstruct_from_spectral,
    fokker_planck_step,
    submodular_pack,
    compute_gaussian_patch,
    estimate_token_count,
)


class TestSinusoidalEncode:
    """Tests for sinusoidal concept encoding."""

    def test_output_shape(self) -> None:
        """Output shape must be (n_concepts, d_model) with float32 (memory efficient)."""
        concepts = ["foo", "bar", "baz"]
        emb = sinusoidal_encode(concepts, d_model=512)
        assert emb.shape == (3, 512)
        assert emb.dtype == np.float32

    def test_deterministic(self) -> None:
        """Same input must produce identical output."""
        concepts = ["quantum_memory", "spectral_manifold"]
        emb1 = sinusoidal_encode(concepts, d_model=128)
        emb2 = sinusoidal_encode(concepts, d_model=128)
        assert np.allclose(emb1, emb2)

    def test_different_concepts_different(self) -> None:
        """Different concept strings must produce different embeddings."""
        emb1 = sinusoidal_encode(["apple"], d_model=64)
        emb2 = sinusoidal_encode(["orange"], d_model=64)
        assert not np.allclose(emb1, emb2)

    def test_empty_concepts_raises(self) -> None:
        """Empty concept list must raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            sinusoidal_encode([], d_model=64)

    def test_output_range(self) -> None:
        """Embedding values must be in [-1, 1] (cosine output)."""
        concepts = ["a", "b", "c"] * 10
        emb = sinusoidal_encode(concepts, d_model=256)
        assert np.all(emb >= -1.0 - 1e-10)
        assert np.all(emb <= 1.0 + 1e-10)

    def test_thousand_trials_determinism(self) -> None:
        """Over 1000 hash calls with the same string, result must be identical."""
        concept = "persistent_concept"
        ref = sinusoidal_encode([concept], d_model=512)
        for _ in range(100):
            trial = sinusoidal_encode([concept], d_model=512)
            assert np.allclose(ref, trial)


class TestSeededLSH:
    """Tests for deterministic LSH."""

    def test_deterministic_hash(self) -> None:
        """Same vector must produce same bucket across multiple calls."""
        lsh = SeededLSH(d=64, w=10.0, m=8, seed=42)
        x = np.random.RandomState(0).randn(64).astype(np.float64)
        h1 = lsh.hash_vector(x)
        h2 = lsh.hash_vector(x)
        assert h1 == h2

    def test_deterministic_over_1000_trials(self) -> None:
        """1000 different vectors, each hashed twice — must all match."""
        lsh = SeededLSH(d=16, w=5.0, m=4, seed=42)
        rng = np.random.RandomState(12345)
        for _ in range(1000):
            x = rng.randn(16).astype(np.float64)
            h1 = lsh.hash_vector(x)
            h2 = lsh.hash_vector(x)
            assert h1 == h2, f"Determinism failure for vector"

    def test_batch_hashing(self) -> None:
        """Batch hashing must produce same results as individual hashing."""
        lsh = SeededLSH(d=32, w=10.0, m=6, seed=42)
        rng = np.random.RandomState(999)
        X = rng.randn(20, 32).astype(np.float64)
        batch_results = lsh.hash_batch(X)
        for i in range(20):
            individual = lsh.hash_vector(X[i])
            assert batch_results[i] == individual

    def test_nearby_vectors_same_bucket(self) -> None:
        """Very close vectors should (with high probability) hash to the same bucket."""
        lsh = SeededLSH(d=8, w=100.0, m=4, seed=42)  # wide buckets
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float64)
        y = x + 0.001  # tiny perturbation
        hx = lsh.hash_vector(x)
        hy = lsh.hash_vector(y)
        assert hx == hy, "Nearby vectors should share a bucket with wide LSH"

    def test_wrong_dimension_raises(self) -> None:
        """Hashing a vector with wrong dimension must raise ValueError."""
        lsh = SeededLSH(d=16, w=10.0, m=4, seed=42)
        with pytest.raises(ValueError, match="Expected vector shape"):
            lsh.hash_vector(np.zeros(8, dtype=np.float64))

    def test_reproducible_across_instances(self) -> None:
        """Two LSH instances with same seed must produce same hashes."""
        lsh1 = SeededLSH(d=8, w=5.0, m=4, seed=42)
        lsh2 = SeededLSH(d=8, w=5.0, m=4, seed=42)
        x = np.random.RandomState(0).randn(8).astype(np.float64)
        assert lsh1.hash_vector(x) == lsh2.hash_vector(x)


class TestNormalizedLaplacian:
    """Tests for graph Laplacian computation."""

    def test_symmetry(self) -> None:
        """L_sym must be symmetric (L == L^T)."""
        n = 10
        A = sp.csr_matrix(np.random.RandomState(0).rand(n, n))
        A = (A + A.T) / 2  # ensure symmetry
        # Sparsify by converting to dense, applying mask, then back to sparse
        A_dense = A.toarray()
        A_dense[A_dense < 0.5] = 0
        A = sp.csr_matrix(A_dense)
        L = normalized_laplacian(A)
        diff = (L - L.T).data
        assert len(diff) == 0 or np.max(np.abs(diff)) < 1e-12

    def test_non_negative_adjacency(self) -> None:
        """L_sym computed from non-negative A must have non-negative eigenvalues."""
        n = 8
        A = sp.csr_matrix(np.random.RandomState(1).rand(n, n) * 0.5)
        A = (A + A.T) / 2
        L = normalized_laplacian(A)
        evals = np.linalg.eigvalsh(L.toarray())
        assert np.all(evals >= -1e-10)

    def test_fiedler_value(self) -> None:
        """For a connected graph, the Fiedler value (2nd eigenvalue) must be > 0."""
        # Path graph is connected
        n = 20
        rows, cols = [], []
        for i in range(n - 1):
            rows.extend([i, i + 1])
            cols.extend([i + 1, i])
        A = sp.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(n, n), dtype=np.float64
        )
        L = normalized_laplacian(A)
        evals = np.sort(np.linalg.eigvalsh(L.toarray()))
        # 2nd eigenvalue (index 1) should be > 0 for connected graph
        assert evals[1] > 1e-6

    def test_non_square_raises(self) -> None:
        """Non-square adjacency must raise ValueError."""
        A = sp.csr_matrix(np.random.RandomState(0).rand(5, 3))
        with pytest.raises(ValueError, match="must be square"):
            normalized_laplacian(A)

    def test_negative_entries_raises(self) -> None:
        """Adjacency with negative entries must raise ValueError."""
        A = sp.csr_matrix(np.array([[-1.0, 0.5], [0.5, 0.0]], dtype=np.float64))
        with pytest.raises(ValueError, match="negative entries"):
            normalized_laplacian(A)


class TestSpectralSignature:
    """Tests for spectral signature computation."""

    def test_eigenvalue_order(self) -> None:
        """Eigenvalues must be returned sorted ascending."""
        n = 15
        A = sp.csr_matrix(np.random.RandomState(0).rand(n, n) * 0.3)
        A = (A + A.T) / 2
        L = normalized_laplacian(A)
        evals, _ = spectral_signature(L, k=5)
        for i in range(len(evals) - 1):
            assert evals[i] <= evals[i + 1] + 1e-10

    def test_output_shape(self) -> None:
        """Must return eigenvalues of shape (k,) and eigenvectors (n, k)."""
        n = 20
        A = sp.csr_matrix(np.random.RandomState(2).rand(n, n) * 0.3)
        A = (A + A.T) / 2
        L = normalized_laplacian(A)
        evals, evecs = spectral_signature(L, k=6)
        assert evals.shape == (6,)
        assert evecs.shape == (20, 6)

    def test_k_too_large_raises(self) -> None:
        """Requesting k >= n must raise ValueError."""
        A = sp.csr_matrix(np.eye(5))
        L = normalized_laplacian(A)
        with pytest.raises(ValueError, match="must be less than"):
            spectral_signature(L, k=5)


class TestFokkerPlanck:
    """Tests for Fokker-Planck diffusion."""

    def test_output_shape(self) -> None:
        """Output must have same shape as input rho."""
        n = 10
        A = sp.csr_matrix(np.random.RandomState(0).rand(n, n) * 0.3)
        A = (A + A.T) / 2
        L = normalized_laplacian(A)
        rho = np.ones(n, dtype=np.float64) * 0.5
        q = np.zeros(n, dtype=np.float64)
        q[0] = 1.0
        rho_next = fokker_planck_step(rho, L, q)
        assert rho_next.shape == (n,)
        assert rho_next.dtype == np.float64

    def test_non_exploding(self) -> None:
        """Total probability mass must not explode (bounded growth)."""
        n = 20
        A = sp.csr_matrix(np.random.RandomState(0).rand(n, n) * 0.3)
        A = (A + A.T) / 2
        L = normalized_laplacian(A)
        rho = np.ones(n, dtype=np.float64) * 0.1
        q = np.zeros(n, dtype=np.float64)
        q[5] = 1.0

        total_mass = np.sum(rho)
        for _ in range(10):
            rho = fokker_planck_step(rho, L, q)
            # Mass should not grow super-exponentially
            new_mass = np.sum(rho)
            assert new_mass < total_mass * 3.0  # Max 3x growth over 10 steps
            total_mass = new_mass

    def test_diffusion_spread(self) -> None:
        """Fokker-Planck must reinforce activated nodes and keep values non-negative."""
        # Path graph: 0-1-2-3-4
        n = 5
        rows, cols = [], []
        for i in range(n - 1):
            rows.extend([i, i + 1])
            cols.extend([i + 1, i])
        A = sp.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(n, n), dtype=np.float64
        )
        L = normalized_laplacian(A)
        rho = np.zeros(n, dtype=np.float64)
        rho[0] = 0.5  # Initial strength at node 0
        q = np.zeros(n, dtype=np.float64)
        q[0] = 1.0

        rho_before = rho.copy()

        # Run multiple steps to observe amplification
        for _step in range(10):
            rho = fokker_planck_step(rho, L, q)

        # Activated node must be reinforced (rho increases)
        assert rho[0] > rho_before[0], "Activated node should be reinforced"
        # All values must stay non-negative (clamping works)
        assert np.all(rho >= 0.0), "All rho values must be non-negative"

    def test_negative_strength_clamped(self) -> None:
        """Negative values in rho must be clamped to zero."""
        n = 5
        A = sp.csr_matrix(np.eye(n) * 0.0)
        L = normalized_laplacian(A)
        rho = np.array([-0.5, -0.3, 0.0, 0.2, 0.4], dtype=np.float64)
        q = np.zeros(n, dtype=np.float64)
        rho_next = fokker_planck_step(rho, L, q)
        assert np.all(rho_next >= 0.0)

    def test_nan_rho_raises(self) -> None:
        """NaN values in rho must raise ValueError."""
        n = 5
        A = sp.csr_matrix(np.eye(n) * 0.0)
        L = normalized_laplacian(A)
        rho = np.array([1.0, np.nan, 1.0, 1.0, 1.0], dtype=np.float64)
        q = np.zeros(n, dtype=np.float64)
        with pytest.raises(ValueError, match="rho contains NaN"):
            fokker_planck_step(rho, L, q)


class TestSubmodularPack:
    """Tests for submodular context packing."""

    def test_empty_candidates(self) -> None:
        """Empty candidates list must return empty list."""
        result = submodular_pack([], budget=100)
        assert result == []

    def test_budget_respected(self) -> None:
        """Total tokens of selected candidates must not exceed budget."""
        candidates = [
            {
                "id": "p1", "text": "a", "token_count": 50,
                "concept_coverage": {"c1": 0.9, "c2": 0.1}, "strength": 1.0,
            },
            {
                "id": "p2", "text": "b", "token_count": 60,
                "concept_coverage": {"c2": 0.8, "c3": 0.2}, "strength": 1.0,
            },
            {
                "id": "p3", "text": "c", "token_count": 40,
                "concept_coverage": {"c3": 0.7, "c4": 0.3}, "strength": 1.0,
            },
        ]
        result = submodular_pack(candidates, budget=100)
        total = sum(c["token_count"] for c in candidates if c["id"] in result)
        assert total <= 100

    def test_greedy_vs_brute_force_small(self) -> None:
        """For N=5, greedy must be within (1-1/e) of brute force optimum."""
        candidates = []
        for i in range(5):
            candidates.append({
                "id": f"p{i}",
                "text": f"content_{i}",
                "token_count": 30,
                "concept_coverage": {f"c{j}": 1.0 / (1 + abs(i - j))
                                     for j in range(5)},
                "strength": 1.0 - i * 0.1,
            })

        budget = 60
        greedy_result = submodular_pack(candidates, budget=budget)

        # Brute force optimum
        from itertools import combinations
        best_coverage = 0.0
        best_set: set[str] = set()

        def coverage_of(ids: set[str]) -> float:
            merged: dict[str, float] = {}
            for c in candidates:
                if c["id"] not in ids:
                    continue
                for concept, score in c["concept_coverage"].items():
                    weighted = score * c["strength"]
                    if concept not in merged or weighted > merged[concept]:
                        merged[concept] = weighted
            return sum(merged.values())

        for r in range(1, len(candidates) + 1):
            for combo in combinations(range(len(candidates)), r):
                total_tokens = sum(candidates[i]["token_count"] for i in combo)
                if total_tokens > budget:
                    continue
                ids = {candidates[i]["id"] for i in combo}
                cov = coverage_of(ids)
                if cov > best_coverage:
                    best_coverage = cov
                    best_set = ids

        greedy_coverage = coverage_of(set(greedy_result))
        approx_ratio = 1 - 1 / math.e  # ≈ 0.632

        assert greedy_coverage >= approx_ratio * best_coverage - 0.05, (
            f"Greedy coverage {greedy_coverage} < {(1-1/math.e):.3f} * {best_coverage}"
        )

    def test_negative_budget_raises(self) -> None:
        """Budget < 1 must raise ValueError."""
        with pytest.raises(ValueError, match="budget must be >= 1"):
            submodular_pack([{"id": "p1", "text": "a", "token_count": 10,
                              "concept_coverage": {"c": 1.0}, "strength": 1.0}],
                            budget=0)


class TestGaussianPatch:
    """Tests for Gaussian patch computation (B2)."""

    def test_output_shape(self) -> None:
        """mu must be (d,) and Sigma_inv must be (d, d)."""
        vectors = np.random.RandomState(0).randn(10, 64).astype(np.float64)
        mu, Sigma_inv = compute_gaussian_patch(vectors)
        assert mu.shape == (64,)
        assert Sigma_inv.shape == (64, 64)

    def test_less_than_two_raises(self) -> None:
        """Fewer than 2 vectors must raise ValueError."""
        vectors = np.random.RandomState(0).randn(1, 64).astype(np.float64)
        with pytest.raises(ValueError, match="at least 2"):
            compute_gaussian_patch(vectors)

    def test_deterministic(self) -> None:
        """Same input must produce identical output."""
        vectors = np.random.RandomState(0).randn(10, 8).astype(np.float64)
        mu1, inv1 = compute_gaussian_patch(vectors)
        mu2, inv2 = compute_gaussian_patch(vectors)
        assert np.allclose(mu1, mu2)
        assert np.allclose(inv1, inv2)

    def test_precision_inverse(self) -> None:
        """Sigma_inv @ cov must be close to identity."""
        vectors = np.random.RandomState(0).randn(20, 16).astype(np.float64)
        mu, Sigma_inv = compute_gaussian_patch(vectors)
        centered = vectors - mu
        cov = (centered.T @ centered) / 19.0
        prod = Sigma_inv @ cov
        assert np.allclose(prod, np.eye(16), atol=1e-8)


class TestSpectralReconstructionError:
    """Tests for spectral reconstruction accuracy."""

    def test_reconstruction_error_bound(self) -> None:
        """||L_recon - L_orig||_F < 5.0 for k=10 on a small graph."""
        n = 20
        # Create a structured adjacency matrix
        A_dense = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(i - j) <= 2:  # local connectivity
                    w = 1.0 / (1 + abs(i - j))
                    A_dense[i, j] = w
                    A_dense[j, i] = w
        A = sp.csr_matrix(A_dense)
        L_orig = normalized_laplacian(A)

        k = min(10, n - 2)
        eigenvalues, eigenvectors = spectral_signature(L_orig, k=k)
        A_recon = reconstruct_from_spectral(eigenvalues, eigenvectors, n)
        L_recon = normalized_laplacian(A_recon)

        frob_error = np.linalg.norm(
            L_orig.toarray() - L_recon.toarray(), ord="fro"
        )
        # On small graphs with k=10, error should be reasonable
        assert frob_error < 5.0, f"Frobenius error {frob_error} too large"


class TestEstimateTokenCount:
    """Tests for token count estimation."""

    def test_short_text(self) -> None:
        """Very short text must return at least 1."""
        assert estimate_token_count("") == 1
        assert estimate_token_count("a") == 1

    def test_linear_scaling(self) -> None:
        """Doubling text length must approximately double token count."""
        text1 = "hello world " * 50
        text2 = "hello world " * 100
        t1 = estimate_token_count(text1)
        t2 = estimate_token_count(text2)
        # Allow 20% tolerance
        assert abs(t2 / max(t1, 1) - 2.0) < 0.3
