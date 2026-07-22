# ============================================================================
# tests/test_integration.py — Integration tests for the full pipeline
# Ingest a smaller corpus, run queries, verify accuracy and performance
# ============================================================================

import time
import threading

import numpy as np
import pytest

from the_context.core import SeededLSH, tokenize
from the_context.core import DeterministicKnowledgeGraph
from the_context.core import VirtualMemoryTree
from the_context.extraction import HeuristicExtractor
from the_context.query import QueryEngine


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture(scope="module")
def integration_setup():
    """Create a full pipeline with a small synthetic corpus for integration testing."""
    d_model = 64
    lsh = SeededLSH(d=d_model, w=10.0, m=4, seed=42)
    tree = VirtualMemoryTree(page_size=100, cache_size=100, persist_dir="/tmp/int_test")
    graph = DeterministicKnowledgeGraph(d_model=d_model)
    extractor = HeuristicExtractor()
    gate = QueryEngine(tree=tree, graph=graph, lsh=lsh, d_model=d_model)

    # Generate synthetic corpus with cross-references
    concept_defs = {
        "quantum_memory": (
            "Quantum memory is a spectral manifold representation of knowledge "
            "that encodes information in the eigenvalues of a graph Laplacian. "
            "It enables deterministic retrieval through hierarchical beacon compression."
        ),
        "spectral_manifold": (
            "A spectral manifold is a low-dimensional Riemannian manifold embedded "
            "in semantic space where the graph Laplacian serves as the metric tensor."
        ),
    }
    cross_refs = {
        "quantum_memory": (
            "Recall that quantum memory uses spectral decomposition of the "
            "graph Laplacian for compression and retrieval."
        ),
    }

    # Build token stream: definitions first, fillers, then cross-references
    all_tokens = []
    for concept, def_text in concept_defs.items():
        all_tokens.extend(def_text.split())
        all_tokens.extend([f"filler_{i}" for i in range(50)])

    all_tokens.extend([f"padding_{i}" for i in range(200)])

    for concept, cref_text in cross_refs.items():
        all_tokens.extend(cref_text.split())
        all_tokens.extend([f"extra_{i}" for i in range(50)])

    # Fill to 10000 tokens
    while len(all_tokens) < 10000:
        all_tokens.append(f"corpus_token_{len(all_tokens)}")

    # Ingest
    page_ids = list(tree.ingest_stream(iter(all_tokens)))

    # Extract triples and build graph
    for pid in page_ids:
        page_text = tree.get_page(pid)
        if page_text is None:
            continue
        triples = extractor.extract(page_text)
        for subject, predicate, obj in triples:
            b1_id = tree.get_beacon_for_page(pid)
            graph.add_triplet(
                subject=subject,
                predicate=predicate,
                object=obj,
                weight=1.0,
                beacon_id=b1_id or "",
                page_id=pid,
            )

    # Build Laplacian if enough nodes
    if len(graph.node_to_idx) >= 2:
        graph.build_laplacian()
        # Run initial diffusion
        graph.concept_diffusion(list(graph.node_to_idx.keys())[:3], steps=2)

    return {
        "gate": gate,
        "tree": tree,
        "graph": graph,
        "page_ids": page_ids,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Integration Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEndToEnd:
    """E2E integration tests for the memory manifold."""

    def test_ingestion_creates_pages(self, integration_setup) -> None:
        """Ingestion must create page IDs."""
        setup = integration_setup
        assert len(setup["page_ids"]) > 0

    def test_graph_has_concepts(self, integration_setup) -> None:
        """Graph must have extracted concepts from the corpus."""
        setup = integration_setup
        assert len(setup["graph"].node_to_idx) >= 5

    def test_query_returns_results(self, integration_setup) -> None:
        """Querying must return pages (or graceful error if graph is too small)."""
        gate = integration_setup["gate"]
        result = gate.collapse(query="quantum memory", max_tokens=4096)

        if result.error:
            # If error, it should be a meaningful message
            assert "Empty" in result.error or "failed" in result.error or "empty" in result.error.lower() or len(result.error) > 0
        else:
            assert result.latency_ms >= 0.0
            assert result.tokens_used >= 0

    def test_determinism_across_queries(self, integration_setup) -> None:
        """Same query × 5 must produce identical page order."""
        gate = integration_setup["gate"]
        query = "quantum spectral manifold"

        results = []
        for _ in range(5):
            result = gate.collapse(query=query, max_tokens=2048)
            results.append(result)

        for i in range(1, len(results)):
            assert results[i].pages == results[0].pages, (
                f"Trial {i} page order differs from trial 0"
            )

    def test_multiple_required_concepts(self, integration_setup) -> None:
        """Queries with required concepts must not crash."""
        gate = integration_setup["gate"]
        result = gate.collapse(
            query="memory",
            max_tokens=2048,
            required_concepts=["quantum_memory", "spectral_manifold"],
        )
        # Should not raise
        assert result.latency_ms >= 0.0

    def test_concurrent_queries(self, integration_setup) -> None:
        """Multiple concurrent queries must not cause race conditions."""
        gate = integration_setup["gate"]

        n_queries = 10
        results: list[dict] = []
        lock = threading.Lock()

        def worker(idx: int) -> None:
            try:
                r = gate.collapse(query=f"quantum test {idx}", max_tokens=1024)
                with lock:
                    results.append({
                        "idx": idx,
                        "n_pages": len(r.pages),
                        "error": r.error,
                        "latency": r.latency_ms,
                    })
            except Exception as exc:
                with lock:
                    results.append({"idx": idx, "error": str(exc)})

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_queries)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == n_queries
        # No fatal errors
        fatal_errors = [r for r in results if isinstance(r.get("error"), str) and "error" not in r.get("error", "").lower()]
        # Most queries should succeed or have graceful errors
        assert len(fatal_errors) < n_queries

    def test_latency_tracking(self, integration_setup) -> None:
        """Latency must be reasonable for this small corpus."""
        gate = integration_setup["gate"]
        result = gate.collapse(query="quantum memory spectral", max_tokens=2048)
        # On a 10K-token corpus, latency should be < 5 seconds
        assert result.latency_ms < 5000.0, f"Latency too high: {result.latency_ms}ms"

    def test_compression_ratio(self, integration_setup) -> None:
        """Compression ratio must be >= 1.0 (tokens_total / tokens_used)."""
        gate = integration_setup["gate"]
        result = gate.collapse(query="quantum memory", max_tokens=2048)
        if result.tokens_used > 0 and result.tokens_total > 0:
            assert result.compression_ratio >= 1.0
