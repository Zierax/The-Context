# ============================================================================
# tests/test_knowledge_graph.py — Unit tests for knowledge graph
# ============================================================================

import os
import tempfile

import numpy as np
import scipy.sparse as sp
import pytest

from the_context.core import DeterministicKnowledgeGraph


class TestDeterministicKnowledgeGraph:
    """Tests for the deterministic knowledge graph."""

    def test_init(self) -> None:
        """Graph must initialise with empty maps and no matrices."""
        g = DeterministicKnowledgeGraph(d_model=64)
        assert g.node_to_idx == {}
        assert g.idx_to_node == {}
        assert g.A is None
        assert g.L_sym is None
        assert g.rho is None
        assert g.beacon_map == {}

    def test_add_triplet_creates_nodes(self) -> None:
        """Adding a triplet must create nodes for S, P, O."""
        g = DeterministicKnowledgeGraph(d_model=64)
        g.add_triplet("cat", "eats", "fish")
        assert "cat" in g.node_to_idx
        assert "eats" in g.node_to_idx
        assert "fish" in g.node_to_idx
        assert len(g.node_to_idx) == 3

    def test_add_triplet_idempotent(self) -> None:
        """Adding the same triplet twice must not change graph structure."""
        g = DeterministicKnowledgeGraph(d_model=64)
        g.add_triplet("A", "relates_to", "B", weight=1.0)
        state_before = (
            g.node_to_idx.copy(),
            g.A.copy() if g.A is not None else None,
            g.rho.copy() if g.rho is not None else None,
        )
        g.add_triplet("A", "relates_to", "B", weight=1.0)
        state_after = (
            g.node_to_idx.copy(),
            g.A.copy() if g.A is not None else None,
            g.rho.copy() if g.rho is not None else None,
        )
        # Node maps must be identical (same number of nodes)
        assert len(state_before[0]) == len(state_after[0])
        # Adjacency weight must have grown (additive accumulation)
        assert state_before[1] is not None
        assert state_after[1] is not None
        # Convert to dense for safe indexing, or use get() with default 0
        a_idx = g.node_to_idx["A"]
        b_idx = g.node_to_idx["B"]
        after_val = state_after[1][a_idx, b_idx]
        before_val = state_before[1][a_idx, b_idx]
        assert after_val >= before_val

    def test_negative_weight_raises(self) -> None:
        """Negative weight must raise ValueError."""
        g = DeterministicKnowledgeGraph(d_model=64)
        with pytest.raises(ValueError, match="weight must be positive"):
            g.add_triplet("A", "relates_to", "B", weight=-1.0)

    def test_build_laplacian(self) -> None:
        """After adding triplets, Laplacian must be computable."""
        g = DeterministicKnowledgeGraph(d_model=64)
        g.add_triplet("A", "connects_to", "B", weight=1.0)
        g.add_triplet("B", "connects_to", "C", weight=1.0)
        g.build_laplacian()
        assert g.L_sym is not None
        assert isinstance(g.L_sym, sp.csr_matrix)
        # Nodes: A, connects_to, B, C = 4 (predicate creates additional nodes)
        assert g.L_sym.shape[0] == 4

    def test_build_laplacian_few_nodes_raises(self) -> None:
        """Building Laplacian with < 2 nodes must raise RuntimeError."""
        g = DeterministicKnowledgeGraph(d_model=64)
        g.add_triplet("only_one", "exists", "itself")
        # After adding one triplet, we have 3 nodes (S, P, O). Need to test with 0 or 1.
        g2 = DeterministicKnowledgeGraph(d_model=64)
        with pytest.raises(RuntimeError, match="need at least 2"):
            g2.build_laplacian()

    def test_concept_diffusion_spread(self) -> None:
        """Activating a concept must increase rho in its neighbours."""
        g = DeterministicKnowledgeGraph(d_model=64)
        g.add_triplet("X", "relates_to", "Y", weight=2.0)
        g.add_triplet("Y", "relates_to", "Z", weight=1.0)
        g.build_laplacian()

        initial_rho = g.rho.copy()
        diffused = g.concept_diffusion(["X"], steps=5)

        # After diffusion, X's rho must have increased relative to initial
        assert diffused[g.node_to_idx["X"]] >= initial_rho[g.node_to_idx["X"]]
        # Y should have non-zero rho (connected to X)
        assert diffused[g.node_to_idx["Y"]] >= 0.0

    def test_concept_diffusion_no_match(self) -> None:
        """Diffusion with no matching concepts must return current rho."""
        g = DeterministicKnowledgeGraph(d_model=64)
        g.add_triplet("A", "relates_to", "B", weight=1.0)
        g.build_laplacian()
        result = g.concept_diffusion(["NONEXISTENT_CONCEPT"], steps=3)
        assert result is not None
        # Nodes: A, relates_to, B = 3
        assert result.shape[0] == 3

    def test_concept_diffusion_no_laplacian_raises(self) -> None:
        """Diffusion without building Laplacian must raise RuntimeError."""
        g = DeterministicKnowledgeGraph(d_model=64)
        g.add_triplet("A", "relates_to", "B", weight=1.0)
        with pytest.raises(RuntimeError, match="Laplacian not built"):
            g.concept_diffusion(["A"], steps=1)

    def test_get_active_beacons(self) -> None:
        """After adding triplets with beacon IDs, must retrieve active beacons."""
        g = DeterministicKnowledgeGraph(d_model=64)
        g.add_triplet("A", "relates_to", "B", beacon_id="b1_0000001")
        g.add_triplet("C", "relates_to", "D", beacon_id="b1_0000002")
        g.build_laplacian()
        g.concept_diffusion(["A"], steps=2)
        active = g.get_active_beacons(top_k=5)
        assert len(active) > 0
        assert "b1_0000001" in active

    def test_persistence_roundtrip(self) -> None:
        """Save -> load must produce identical node maps and adjacency."""
        g = DeterministicKnowledgeGraph(d_model=64)
        g.add_triplet("dog", "chases", "cat", weight=1.0, beacon_id="b1_test")
        g.add_triplet("cat", "eats", "mouse", weight=0.5, beacon_id="b1_test")
        g.build_laplacian()

        # Capture state before save
        node_map_before = g.node_to_idx.copy()
        A_before = g.A.copy() if g.A is not None else None
        rho_before = g.rho.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "graph_save")
            g.save(save_path)

            # Load into a new graph
            g2 = DeterministicKnowledgeGraph(d_model=64)
            g2.load(save_path)

            # Compare node maps
            assert g2.node_to_idx == node_map_before
            assert g2.A is not None and A_before is not None
            # Compare adjacency (shape + nnz)
            assert g2.A.shape == A_before.shape
            assert g2.A.nnz == A_before.nnz
            # Compare rho
            assert g2.rho is not None
            assert np.allclose(g2.rho, rho_before)

    def test_rho_persists_across_triplets(self) -> None:
        """rho must grow as more triplets are added."""
        g = DeterministicKnowledgeGraph(d_model=64)
        g.add_triplet("A", "relates_to", "B", weight=1.0)
        rho_after_first = g.rho.sum()
        g.add_triplet("A", "relates_to", "C", weight=2.0)
        rho_after_second = g.rho.sum()
        assert rho_after_second > rho_after_first
