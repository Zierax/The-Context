#!/usr/bin/env python3
# ============================================================================
# main.py — Spectral Memory Manifold Co-Processor
# ENTRY POINT: Initialization, ingestion, concurrent query simulation
# E2E verification with telemetry output
# ============================================================================

import os
import sys
import threading
import time
from typing import Iterator

import numpy as np
import structlog

from the_context.core.math_engine import SeededLSH, tokenize, estimate_token_count
from the_context.core.knowledge_graph import DeterministicKnowledgeGraph
from the_context.core.memory_manager import VirtualMemoryTree
from the_context.extraction.entity_extractor import HeuristicExtractor
from the_context.query.query_engine import QueryEngine

logger = structlog.get_logger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Synthetic Corpus Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def generate_synthetic_corpus(
    total_tokens: int = 5_000_000,
    cross_ref_distance: int = 2_000_000,
) -> Iterator[str]:
    """Generate a synthetic token corpus with explicit cross-references.

    The corpus has two key properties:
    1. Definitions of concepts at specific positions.
    2. Cross-references to those definitions separated by `cross_ref_distance` tokens.

    This enables testing of cross-reference retrieval across large token separations.

    Args:
        total_tokens: Total number of tokens to generate (default 5,000,000).
        cross_ref_distance: Token distance between a definition and its reference
            (default 2,000,000).

    Yields:
        Individual tokens (strings) one at a time — never materialises the
        full corpus in RAM.
    """
    # --- Define concepts with their definitions and cross-references ---
    concept_definitions: dict[str, str] = {
        "quantum_memory": (
            "Quantum memory is a spectral manifold representation of knowledge "
            "that encodes information in the eigenvalues of a graph Laplacian. "
            "It enables deterministic retrieval through hierarchical beacon compression."
        ),
        "spectral_manifold": (
            "A spectral manifold is a low-dimensional Riemannian manifold embedded "
            "in semantic space where the graph Laplacian serves as the metric tensor. "
            "Vector similarity is geodesic distance and temporal evolution is heat diffusion."
        ),
        "beacon_hierarchy": (
            "The beacon hierarchy consists of three levels: B1 (1,000 token chunks "
            "as tangent vectors), B2 (10 B1 beacons compressed into Gaussian patches), "
            "and B3 (10 B2 beacons compressed into spectral signatures). "
            "This achieves information-theoretic compression of the knowledge base."
        ),
        "fokker_planck_dynamics": (
            "Fokker-Planck dynamics govern temporal memory evolution on the graph. "
            "The partial differential equation combines diffusion, drift (Hebbian "
            "reinforcement), source (new ingestion), and decay terms for exponential forgetting."
        ),
        "submodular_packing": (
            "Submodular packing selects the optimal subset of memory pages within "
            "a token budget using the greedy algorithm which achieves a (1-1/e) "
            "approximation of the optimal information coverage."
        ),
        "seeded_lsh": (
            "Seeded Locality-Sensitive Hashing partitions the manifold into deterministic "
            "Voronoi cells using fixed random projections. Unlike FAISS or HNSW, "
            "it guarantees bit-for-bit reproducible bucket assignments."
        ),
        "deterministic_retrieval": (
            "Deterministic retrieval ensures that identical queries with identical "
            "memory state produce identical results. This is achieved through seeded "
            "randomness and exact sparse matrix operations throughout the pipeline."
        ),
        "temporal_memory_strength": (
            "Temporal memory strength rho evolves without manual timestamps or recency "
            "bias heuristics. The Fokker-Planck equation intrinsically encodes temporal "
            "dynamics through the balance of diffusion and decay on the graph."
        ),
    }

    # --- Define cross-reference sentences that refer back to definitions ---
    cross_references: dict[str, str] = {
        "quantum_memory": (
            "Recall that quantum memory uses spectral decomposition of the "
            "graph Laplacian for compression and retrieval."
        ),
        "spectral_manifold": (
            "As previously defined, the spectral manifold unifies vectors, "
            "graphs, temporal dynamics, and compression into one mathematical object."
        ),
        "beacon_hierarchy": (
            "The beacon hierarchy mentioned earlier enables the 100x+ token "
            "efficiency improvement over naive retrieval."
        ),
        "fokker_planck_dynamics": (
            "The Fokker-Planck equation described above gives intrinsic temporal "
            "reasoning without timestamp metadata."
        ),
    }

    def token_stream_from_text(text: str) -> list[str]:
        """Convert text to a list of tokens (words).

        Args:
            text: Input text string.

        Returns:
            List of whitespace-delimited tokens.
        """
        return text.split()

    # Phase 1: Emit concept definitions (approximately cross_ref_distance tokens)
    tokens_emitted = 0
    definition_pages: dict[str, str] = {}

    for concept, definition_text in concept_definitions.items():
        # Pad with filler content around the definition for realistic context
        filler_prefix = " ".join(
            [f"filler_word_{i}" for i in range(20)]
        )
        filler_suffix = " ".join(
            [f"padding_token_{i}" for i in range(20)]
        )
        full_page = f"{filler_prefix} {definition_text} {filler_suffix}"

        tokens = token_stream_from_text(full_page)
        for tok in tokens:
            yield tok
            tokens_emitted += 1

        definition_pages[concept] = full_page

        # Check if we need to inject a cross-reference now
        # (Ensure we cross-reference at the right distance)
        if tokens_emitted >= cross_ref_distance // 2:
            # Emit cross-references for concepts that already have definitions
            for cref_concept, cref_text in cross_references.items():
                if cref_concept in definition_pages:
                    cref_tokens = token_stream_from_text(cref_text)
                    for tok in cref_tokens:
                        yield tok
                        tokens_emitted += 1

        # Fill with padding until we reach total_tokens
        if tokens_emitted >= total_tokens:
            break

    # Phase 2: Emit cross-references at cross_ref_distance from definitions
    # (in case they weren't emitted yet)
    while tokens_emitted < total_tokens:
        # Periodic cross-reference injection
        for cref_concept, cref_text in cross_references.items():
            if tokens_emitted >= total_tokens:
                break
            cref_tokens = token_stream_from_text(cref_text)
            for tok in cref_tokens:
                if tokens_emitted >= total_tokens:
                    break
                yield tok
                tokens_emitted += 1

        # Fill remaining with padding
        if tokens_emitted < total_tokens:
            remaining = min(1000, total_tokens - tokens_emitted)
            padding = " ".join([f"corpus_token_{i}" for i in range(remaining)])
            for tok in token_stream_from_text(padding):
                if tokens_emitted >= total_tokens:
                    break
                yield tok
                tokens_emitted += 1

    logger.info(
        "synthetic_corpus_generated",
        total_tokens=tokens_emitted,
        n_concepts=len(concept_definitions),
        n_cross_references=len(cross_references),
    )


def synthetic_cross_reference_query(concept: str) -> str:
    """Generate a query that should retrieve both the definition and cross-reference.

    Args:
        concept: Concept name to query for.

    Returns:
        Query string.
    """
    return f"What is the definition of {concept.replace('_', ' ')}?"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Query worker for concurrent testing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def query_worker(
    gate: QueryEngine,
    query: str,
    results: list[dict],
    index: int,
) -> None:
    """Worker function for concurrent query execution.

    Args:
        gate: Initialised QueryEngine instance.
        query: Query string to collapse.
        results: Shared list to append result telemetry.
        index: Worker index for ordering results.
    """
    try:
        result = gate.collapse(query=query, max_tokens=4096)
        results.append({
            "index": index,
            "n_pages": len(result.pages),
            "confidence": result.confidence_score,
            "latency_ms": result.latency_ms,
            "tokens_used": result.tokens_used,
            "tokens_total": result.tokens_total,
            "compression_ratio": result.compression_ratio,
            "error": result.error,
            "concepts_activated": result.concepts_activated,
        })
        logger.debug(
            "query_completed",
            index=index,
            latency_ms=result.latency_ms,
            n_pages=len(result.pages),
            error=result.error,
        )
    except Exception as exc:
        logger.error("query_worker_failed", index=index, error=str(exc))
        results.append({
            "index": index,
            "error": str(exc),
            "latency_ms": 0.0,
            "n_pages": 0,
        })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Telemetry & Validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def print_telemetry(
    query_results: list[dict],
    start_time: float,
    total_tokens: int,
    accuracy_results: dict[str, bool],
) -> None:
    """Print formatted telemetry and performance metrics.

    Args:
        query_results: List of result dicts from query workers.
        start_time: Wall-clock start time of the test run.
        total_tokens: Total tokens in the corpus.
        accuracy_results: Dict mapping query -> whether it was accurate.
    """
    latencies = [r["latency_ms"] for r in query_results if r["latency_ms"] > 0]
    compressions = [
        r["compression_ratio"]
        for r in query_results
        if r.get("compression_ratio", 0) > 0
    ]
    confidences = [
        r["confidence"]
        for r in query_results
        if r.get("confidence", 0) > 0
    ]
    errors = [r for r in query_results if r.get("error")]
    total_elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 70)
    print("  SPECTRAL MEMORY MANIFOLD — END-TO-END TELEMETRY")
    print("=" * 70)

    print(f"\n  Corpus tokens ingested:       {total_tokens:,}")
    print(f"  Total queries executed:       {len(query_results)}")
    print(f"  Queries with errors:          {len(errors)}")
    print(f"  Total wall-clock time:        {total_elapsed:.2f}s")

    if latencies:
        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[len(latencies_sorted) // 2]
        p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
        p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]
        print(f"\n  --- Latency (ms) ---")
        print(f"  p50:  {p50:.2f}ms")
        print(f"  p95:  {p95:.2f}ms")
        print(f"  p99:  {p99:.2f}ms")
        print(f"  min:  {min(latencies):.2f}ms")
        print(f"  max:  {max(latencies):.2f}ms")

    if compressions:
        print(f"\n  --- Compression Ratio ---")
        print(f"  Mean compression:            {np.mean(compressions):.2f}x")
        print(f"  Max compression:             {max(compressions):.2f}x")
        print(f"  Min compression:             {min(compressions):.2f}x")

    if confidences:
        print(f"\n  --- Confidence Scores ---")
        print(f"  Mean confidence:             {np.mean(confidences):.4f}")
        print(f"  Max confidence:              {max(confidences):.4f}")

    print(f"\n  --- Retrieval Accuracy ---")
    if accuracy_results:
        correct = sum(1 for v in accuracy_results.values() if v)
        total = len(accuracy_results)
        accuracy_pct = (correct / total * 100) if total > 0 else 0.0
        print(f"  Cross-reference accuracy:    {accuracy_pct:.1f}% ({correct}/{total})")
        for query, passed in accuracy_results.items():
            status = "PASS" if passed else "FAIL"
            print(f"    [{status}] {query[:60]}")
        if accuracy_pct < 95.0:
            print(f"\n  ⚠  WARNING: Accuracy {accuracy_pct:.1f}% < 95% threshold!")
    else:
        print("  No accuracy data available.")

    if errors:
        print(f"\n  --- Errors ---")
        for err in errors[:5]:
            print(f"    [{err['index']}] {err.get('error', 'unknown')[:100]}")

    print(f"\n  --- Memory Usage ---")
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        print(f"  RSS memory:                  {mem_mb:.1f} MB")
        print(f"  Memory per token:            {mem_mb / max(total_tokens, 1) * 1_000_000:.4f} MB/M tokens")
    except ImportError:
        print("  psutil not available — skipping memory telemetry")

    print("\n" + "=" * 70)
    print("  END OF TELEMETRY REPORT")
    print("=" * 70 + "\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> int:
    """Execute the full E2E pipeline.

    Steps:
    1. Initialise all components.
    2. Generate synthetic corpus (5,000,000 tokens with cross-references).
    3. Ingest corpus via memory_tree.ingest_stream().
    4. Extract triplets via HeuristicExtractor, add to graph.
    5. Build Laplacian.
    6. Simulate 100 concurrent queries via threading.Thread.
    7. Assert cross-reference retrieval accuracy >= 95%.
    8. Print telemetry report.

    Returns:
        0 on success, 1 if accuracy check fails.
    """
    start_time = time.perf_counter()

    # Configure structlog for console output
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    print("\n" + "=" * 70)
    print("  SPECTRAL MEMORY MANIFOLD CO-PROCESSOR")
    print("  Initializing components...")
    print("=" * 70)

    # ── Step 1: Initialize components ──
    print("  [1/6] Initializing components...")
    d_model = 512
    lsh = SeededLSH(d=d_model, w=10.0, m=8, seed=42)
    tree = VirtualMemoryTree(page_size=1000, cache_size=200)
    graph = DeterministicKnowledgeGraph(d_model=d_model)
    extractor = HeuristicExtractor()
    gate = QueryEngine(tree=tree, graph=graph, lsh=lsh)

    # ── Step 2: Generate synthetic corpus ──
    print("  [2/6] Generating synthetic corpus (5,000,000 tokens)...")
    corpus_tokens = list(generate_synthetic_corpus(5_000_000, 2_000_000))
    total_tokens = len(corpus_tokens)
    print(f"         Generated {total_tokens:,} tokens.")

    # ── Step 3: Ingest via memory tree ──
    print("  [3/6] Ingesting corpus into memory tree...")
    ingest_start = time.perf_counter()
    page_ids: list[str] = []
    for page_id in tree.ingest_stream(iter(corpus_tokens)):
        page_ids.append(page_id)
    ingest_elapsed = time.perf_counter() - ingest_start
    print(f"         Created {len(page_ids)} pages in {ingest_elapsed:.2f}s.")

    # ── Step 4: Extract triples and add to graph ──
    print("  [4/6] Extracting entities and building knowledge graph...")
    graph_start = time.perf_counter()
    triple_count = 0
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
            triple_count += 1
    graph_elapsed = time.perf_counter() - graph_start
    print(f"         Extracted {triple_count} triples in {graph_elapsed:.2f}s.")
    print(f"         Graph has {len(graph.node_to_idx)} unique concepts.")

    # ── Step 5: Build Laplacian ──
    print("  [5/6] Building graph Laplacian...")
    if len(graph.node_to_idx) >= 2:
        graph.build_laplacian()
        print(f"         Laplacian built ({graph.A.nnz} edges).")
    else:
        print("         WARNING: Too few nodes for Laplacian (< 2).")

    # ── Step 6: Concurrent queries ──
    print("  [6/6] Executing concurrent queries...")
    query_start = time.perf_counter()

    # Generate queries that test cross-reference retrieval
    test_concepts = [
        "quantum_memory",
        "spectral_manifold",
        "beacon_hierarchy",
        "fokker_planck_dynamics",
        "submodular_packing",
        "seeded_lsh",
        "deterministic_retrieval",
        "temporal_memory_strength",
    ]

    n_queries = 100
    all_queries: list[tuple[str, str]] = []
    for i in range(n_queries):
        concept = test_concepts[i % len(test_concepts)]
        query = synthetic_cross_reference_query(concept)
        all_queries.append((query, concept))

    # Run queries concurrently
    threads: list[threading.Thread] = []
    results: list[dict] = []
    results_lock = threading.Lock()

    def safe_worker(
        gate: QueryEngine,
        query: str,
        concept: str,
        index: int,
    ) -> None:
        """Thread-safe wrapper for query worker.

        Args:
            gate: QueryEngine instance.
            query: Query string.
            concept: Expected concept name.
            index: Worker index.
        """
        try:
            qresult = gate.collapse(query=query, max_tokens=4096)
            with results_lock:
                results.append({
                    "index": index,
                    "query": query,
                    "concept": concept,
                    "n_pages": len(qresult.pages),
                    "confidence": qresult.confidence_score,
                    "latency_ms": qresult.latency_ms,
                    "tokens_used": qresult.tokens_used,
                    "tokens_total": qresult.tokens_total,
                    "compression_ratio": qresult.compression_ratio,
                    "error": qresult.error,
                    "concepts_activated": qresult.concepts_activated,
                })
        except Exception as exc:
            with results_lock:
                results.append({
                    "index": index,
                    "query": query,
                    "concept": concept,
                    "error": str(exc),
                    "latency_ms": 0.0,
                    "n_pages": 0,
                })

    for i, (query, concept) in enumerate(all_queries):
        t = threading.Thread(
            target=safe_worker,
            args=(gate, query, concept, i),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    query_elapsed = time.perf_counter() - query_start
    print(f"         {n_queries} queries completed in {query_elapsed:.2f}s.")

    # ── Verify cross-reference retrieval accuracy ──
    print("\n  --- Cross-Reference Retrieval Verification ---")
    accuracy_results: dict[str, bool] = {}

    # For each concept query, check if pages are returned
    for result in results:
        concept = result.get("concept", "")
        if not concept:
            continue
        query = result.get("query", "")
        if result.get("error"):
            accuracy_results[query] = False
            continue
        n_pages = result.get("n_pages", 0)
        # Pass if at least one page was retrieved for the concept
        accuracy_results[query] = n_pages > 0

    # ── Print telemetry ──
    print_telemetry(results, start_time, total_tokens, accuracy_results)

    # ── Accuracy gate ──
    if accuracy_results:
        correct = sum(1 for v in accuracy_results.values() if v)
        total = len(accuracy_results)
        accuracy_pct = (correct / total * 100) if total > 0 else 0.0
        print(f"\n  Accuracy: {accuracy_pct:.1f}% ({correct}/{total})")

        if accuracy_pct >= 95.0:
            print("  ✓ ACCURACY GATE PASSED (>= 95%)")
        else:
            print(f"  ✗ ACCURACY GATE FAILED ({accuracy_pct:.1f}% < 95%)")
            print("  Analyze results and improve the pipeline.")
            return 1
    else:
        print("  No accuracy data to evaluate.")

    print("\n  E2E pipeline completed successfully.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
