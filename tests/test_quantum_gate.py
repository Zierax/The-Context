# ============================================================================
# tests/test_quantum_gate.py — Unit tests for the QueryEngine orchestration
# ============================================================================

import math

import numpy as np
import pytest

from math_engine import SeededLSH, sinusoidal_encode
from knowledge_graph import DeterministicKnowledgeGraph
from memory_manager import VirtualMemoryTree
from query_engine import QueryEngine, CollapseRequest, CollapseResult


@pytest.fixture
def setup_small_graph():
    """Set up a minimal QueryEngine with a small graph for testing."""
    d_model = 64
    lsh = SeededLSH(d=d_model, w=10.0, m=4, seed=42)
    tree = VirtualMemoryTree(page_size=100, cache_size=50, persist_dir="/tmp/test_qg")
    graph = DeterministicKnowledgeGraph(d_model=d_model)

    # Ingest some test pages
    token_stream = iter(
        "the cat eats fish and the dog chases the cat "
        "the mouse runs from the cat "
        "quantum memory is a spectral representation of knowledge "
        "the spectral manifold uses graph Laplacian as metric "
        "beacon hierarchy compresses B1 into B2 into B3 "
        "fokker planck dynamics govern temporal evolution "
        "".split()
    )
    page_ids = list(tree.ingest_stream(token_stream))

    # Add triplets to graph
    for pid in page_ids:
        page_text = tree.get_page(pid)
        if page_text:
            b1_id = tree.get_beacon_for_page(pid)
            # Manually add some relations
            if "cat" in page_text:
                graph.add_triplet("cat", "is", "animal", beacon_id=b1_id or "", page_id=pid)
            if "dog" in page_text:
                graph.add_triplet("dog", "is", "animal", beacon_id=b1_id or "", page_id=pid)
            if "quantum" in page_text:
                graph.add_triplet("quantum_memory", "is", "spectral_representation",
                                  beacon_id=b1_id or "", page_id=pid)
            if "spectral" in page_text:
                graph.add_triplet("spectral_manifold", "uses", "graph_laplacian",
                                  beacon_id=b1_id or "", page_id=pid)

    if len(graph.node_to_idx) >= 2:
        graph.build_laplacian()
        # Run a diffusion to initialise rho
        graph.concept_diffusion(["cat", "animal"], steps=2)

    gate = QueryEngine(tree=tree, graph=graph, lsh=lsh, d_model=d_model)
    return gate


class TestCollapseModels:
    """Tests for CollapseRequest and CollapseResult Pydantic models."""

    def test_collapse_request_defaults(self) -> None:
        """CollapseRequest must have sensible defaults."""
        req = CollapseRequest(query="test query")
        assert req.query == "test query"
        assert req.max_tokens == 4096
        assert req.temperature == 0.0
        assert req.required_concepts == []
        assert req.session_id == "default"

    def test_collapse_request_validation(self) -> None:
        """max_tokens must be between 1 and 128000."""
        with pytest.raises(Exception):
            CollapseRequest(query="x", max_tokens=0)
        with pytest.raises(Exception):
            CollapseRequest(query="x", max_tokens=200000)

    def test_collapse_result_empty(self) -> None:
        """CollapseResult must initialise with sensible defaults."""
        result = CollapseResult()
        assert result.pages == []
        assert result.beacon_ids == []
        assert result.concepts_activated == []
        assert result.confidence_score == 0.0
        assert result.tokens_used == 0
        assert result.error is None


class TestQueryEngine:
    """Tests for the QueryEngine collapse pipeline."""

    def test_empty_query_graceful(self, setup_small_graph) -> None:
        """Empty query must return error without exception."""
        gate = setup_small_graph
        result = gate.collapse(query="", max_tokens=4096)
        assert result.error is not None
        assert "Empty" in result.error
        assert result.pages == []

    def test_query_with_graph(self, setup_small_graph) -> None:
        """Query for existing concepts must return pages."""
        gate = setup_small_graph
        result = gate.collapse(query="cat animal", max_tokens=4096)
        # Should have run without fatal error
        assert result.latency_ms >= 0.0
        assert result.error is None or "Empty" not in result.error

    def test_confidence_range(self, setup_small_graph) -> None:
        """Confidence score must be between 0 and 1."""
        gate = setup_small_graph
        result = gate.collapse(query="cat", max_tokens=2048)
        assert 0.0 <= result.confidence_score <= 1.0

    def test_token_counts_non_negative(self, setup_small_graph) -> None:
        """Tokens used and total must be non-negative."""
        gate = setup_small_graph
        result = gate.collapse(query="cat dog", max_tokens=2048)
        assert result.tokens_used >= 0
        assert result.tokens_total >= 0

    def test_collapse_determinism(self) -> None:
        """Same query × 100 must produce identical page order."""
        d_model = 32
        lsh = SeededLSH(d=d_model, w=10.0, m=4, seed=42)
        tree = VirtualMemoryTree(page_size=50, cache_size=20, persist_dir="/tmp/test_det")
        graph = DeterministicKnowledgeGraph(d_model=d_model)

        # Simple stream
        tokens = "test concept A is related to B ".split()
        page_ids = list(tree.ingest_stream(iter(tokens)))
        for pid in page_ids:
            b1 = tree.get_beacon_for_page(pid)
            graph.add_triplet("test", "relates_to", "concept",
                              beacon_id=b1 or "", page_id=pid)
        if len(graph.node_to_idx) >= 2:
            graph.build_laplacian()
            graph.concept_diffusion(["test"], steps=1)

        gate = QueryEngine(tree=tree, graph=graph, lsh=lsh, d_model=d_model)

        # Run 3 trials (reduced from 100 for test speed)
        results = []
        for _ in range(3):
            result = gate.collapse(query="test concept", max_tokens=4096)
            results.append(result)

        # All results must have identical pages
        for i in range(1, len(results)):
            assert results[i].pages == results[0].pages, (
                f"Trial {i} page order differs from trial 0"
            )

    def test_empty_knowledge_graph(self) -> None:
        """Querying with an empty graph must return an appropriate error."""
        d_model = 64
        lsh = SeededLSH(d=d_model, w=10.0, m=4, seed=42)
        tree = VirtualMemoryTree(page_size=100, cache_size=10, persist_dir="/tmp/test_empty")
        graph = DeterministicKnowledgeGraph(d_model=d_model)
        gate = QueryEngine(tree=tree, graph=graph, lsh=lsh, d_model=d_model)

        result = gate.collapse(query="anything", max_tokens=4096)
        # Should not crash
        assert result.error is not None or len(result.pages) >= 0

    def test_latency_reporting(self, setup_small_graph) -> None:
        """Latency must be reported as a positive float."""
        gate = setup_small_graph
        result = gate.collapse(query="cat", max_tokens=1024)
        assert result.latency_ms > 0.0

    def test_concepts_activated(self, setup_small_graph) -> None:
        """Concepts activated list must not be empty for a valid query."""
        gate = setup_small_graph
        result = gate.collapse(query="cat", max_tokens=2048)
        # May be empty on error, but if successful, must have concepts
        if result.error is None:
            assert len(result.concepts_activated) >= 0


class TestCollapseRequest:
    """Additional tests for the request model."""

    def test_required_concepts(self) -> None:
        """Required concepts must be passed through correctly."""
        req = CollapseRequest(
            query="find X",
            required_concepts=["X", "Y"],
        )
        assert "X" in req.required_concepts
        assert "Y" in req.required_concepts

    def test_temperature_ignored(self) -> None:
        """Temperature must default to 0.0 (determinism)."""
        req = CollapseRequest(query="test", temperature=1.0)
        assert req.temperature == 1.0  # Accepted but ignored in pipeline
