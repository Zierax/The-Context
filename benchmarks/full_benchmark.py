#!/usr/bin/env python3
"""Full benchmark suite: 1M to 500M tokens.

Tests:
1. Memory overhead per component at each scale
2. Ingestion throughput (tokens/sec)
3. Query latency (p50, p95, p99)
4. Compression ratio
5. Projection accuracy for 25M and 500M targets
"""

import gc
import os
import sys
import tempfile
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_engine import (
    SeededLSH,
    sinusoidal_encode,
    compute_gaussian_patch,
    estimate_token_count,
)
from knowledge_graph import DeterministicKnowledgeGraph
from memory_manager import VirtualMemoryTree
from entity_extractor import HeuristicExtractor
from query_engine import QueryEngine


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def generate_corpus(n_tokens, seed=42):
    """Generate a realistic corpus with repeated concepts."""
    rng = np.random.RandomState(seed)
    corpus = []
    
    # Define core concepts that repeat
    concepts = [
        "quantum memory spectral manifold representation",
        "beacon hierarchy compression B1 B2 B3",
        "Fokker-Planck diffusion temporal dynamics",
        "submodular packing context optimization",
        "seeded LSH Voronoi partition hashing",
        "deterministic retrieval reproducible results",
        "knowledge graph adjacency Laplacian",
        "spectral signature eigenvalue decomposition",
    ]
    
    # Interleave concepts with filler tokens
    concept_idx = 0
    for i in range(n_tokens):
        if i % 500 == 0:
            # Insert a concept sentence
            corpus.extend(concepts[concept_idx % len(concepts)].split())
            concept_idx += 1
        elif i % 100 == 0:
            # Insert a cross-reference
            corpus.extend("recall that the spectral manifold uses".split())
        else:
            corpus.append(f"token_{i}")
    
    return corpus[:n_tokens]


def measure_memory(tree, graph, lsh):
    """Measure memory usage of all components."""
    d_model = 512
    
    # Pages text
    pages_text = sum(len(text.encode('utf-8')) for text in tree.pages.values())
    
    # B1 embeddings
    b1_count = len(tree.beacon_b1)
    b1_memory = b1_count * d_model * 2  # float16
    
    # B2 patches (diagonal + float16 mu)
    b2_count = len(tree.beacon_b2)
    b2_memory = b2_count * (d_model * 2 + d_model * 8)  # mu (float16) + Sigma_inv_diag (float64)
    
    # B3 signatures (k=5)
    b3_count = len(tree.beacon_b3)
    b3_memory = b3_count * (5 * 8 + 10 * 5 * 8)  # eigenvalues + eigenvectors
    
    # Graph (upper triangle)
    if graph.A is not None:
        adj_memory = graph.A.data.nbytes + graph.A.indices.nbytes + graph.A.indptr.nbytes
    else:
        adj_memory = 0
    
    # Laplacian
    if graph.L_sym is not None:
        lap_memory = graph.L_sym.data.nbytes + graph.L_sym.indices.nbytes + graph.L_sym.indptr.nbytes
    else:
        lap_memory = 0
    
    # LSH
    lsh_memory = lsh.a.nbytes + lsh.b.nbytes
    
    total = pages_text + b1_memory + b2_memory + b3_memory + adj_memory + lap_memory + lsh_memory
    
    return {
        'pages_text': pages_text,
        'b1_embeddings': b1_memory,
        'b2_patches': b2_memory,
        'b3_signatures': b3_memory,
        'graph_adj': adj_memory,
        'graph_lap': lap_memory,
        'lsh': lsh_memory,
        'total': total,
    }


def run_benchmark(n_tokens, d_model=512, cache_size=10):
    """Run full benchmark for a given corpus size."""
    print(f"\n  Corpus: {n_tokens:,} tokens ({n_tokens // 1000:,}K)")
    
    # Generate corpus
    t0 = time.perf_counter()
    corpus = generate_corpus(n_tokens)
    gen_time = (time.perf_counter() - t0) * 1000
    print(f"  Corpus generation: {gen_time:.2f} ms")
    
    # Initialize components
    persist_dir = tempfile.mkdtemp(prefix=f"bench_{n_tokens}_")
    lsh = SeededLSH(d=d_model, w=10.0, m=8, seed=42)
    tree = VirtualMemoryTree(page_size=1000, cache_size=cache_size, persist_dir=persist_dir)
    graph = DeterministicKnowledgeGraph(d_model=d_model)
    extractor = HeuristicExtractor()
    
    # Measure ingestion
    tracemalloc.start()
    t0 = time.perf_counter()
    page_ids = list(tree.ingest_stream(iter(corpus)))
    ingest_time = (time.perf_counter() - t0) * 1000
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    n_pages = len(page_ids)
    tokens_per_sec = n_tokens / (ingest_time / 1000) if ingest_time > 0 else 0
    print(f"  Ingestion: {ingest_time:.2f} ms ({tokens_per_sec:,.0f} tokens/sec)")
    print(f"  Pages created: {n_pages:,}")
    print(f"  Peak memory during ingestion: {peak / 1024 / 1024:.2f} MB")
    
    # Extract triples and build graph
    t0 = time.perf_counter()
    triple_count = 0
    for pid in page_ids:
        page_text = tree.get_page(pid)
        if page_text is None:
            continue
        triples = extractor.extract(page_text)
        for subject, predicate, obj in triples:
            b1_id = tree.get_beacon_for_page(pid)
            graph.add_triplet(subject, predicate, obj, beacon_id=b1_id or "")
            triple_count += 1
    graph_time = (time.perf_counter() - t0) * 1000
    print(f"  Graph construction: {graph_time:.2f} ms ({triple_count:,} triples)")
    
    # Build Laplacian
    if len(graph.node_to_idx) >= 2:
        t0 = time.perf_counter()
        graph.build_laplacian()
        lap_time = (time.perf_counter() - t0) * 1000
        print(f"  Laplacian built: {lap_time:.2f} ms")
    
    # Measure memory
    mem = measure_memory(tree, graph, lsh)
    print(f"\n  --- Memory Breakdown ---")
    print(f"  Pages text:     {mem['pages_text'] / 1024 / 1024:>8.2f} MB ({mem['pages_text'] / mem['total'] * 100:.1f}%)")
    print(f"  B1 embeddings:  {mem['b1_embeddings'] / 1024 / 1024:>8.2f} MB ({mem['b1_embeddings'] / mem['total'] * 100:.1f}%)")
    print(f"  B2 patches:     {mem['b2_patches'] / 1024 / 1024:>8.2f} MB ({mem['b2_patches'] / mem['total'] * 100:.1f}%)")
    print(f"  B3 signatures:  {mem['b3_signatures'] / 1024 / 1024:>8.2f} MB ({mem['b3_signatures'] / mem['total'] * 100:.1f}%)")
    print(f"  Graph adj:      {mem['graph_adj'] / 1024 / 1024:>8.2f} MB ({mem['graph_adj'] / mem['total'] * 100:.1f}%)")
    print(f"  Graph lap:      {mem['graph_lap'] / 1024 / 1024:>8.2f} MB ({mem['graph_lap'] / mem['total'] * 100:.1f}%)")
    print(f"  LSH:            {mem['lsh'] / 1024 / 1024:>8.2f} MB ({mem['lsh'] / mem['total'] * 100:.1f}%)")
    print(f"  ---")
    print(f"  Total memory:   {mem['total'] / 1024 / 1024:>8.2f} MB")
    print(f"  Bytes/token:    {mem['total'] / n_tokens:.4f}")
    
    # Query latency
    gate = QueryEngine(tree=tree, graph=graph, lsh=lsh, d_model=d_model)
    
    # Warmup
    for i in range(5):
        gate.collapse(query=f"quantum memory spectral {i}", max_tokens=2048)
    
    # Benchmark queries
    latencies = []
    n_queries = min(100, max(10, n_tokens // 10000))
    t0 = time.perf_counter()
    for i in range(n_queries):
        q = f"quantum memory spectral manifold {i % 10}"
        result = gate.collapse(query=q, max_tokens=2048)
        latencies.append(result.latency_ms)
    query_time = (time.perf_counter() - t0) * 1000
    
    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]
    
    print(f"\n  --- Query Latency ({n_queries} queries) ---")
    print(f"  p50: {p50:.2f} ms")
    print(f"  p95: {p95:.2f} ms")
    print(f"  p99: {p99:.2f} ms")
    print(f"  avg: {np.mean(latencies):.2f} ms")
    
    # Compression ratio
    text_bytes = sum(len(text.encode('utf-8')) for text in corpus)
    compression_ratio = text_bytes / mem['total'] if mem['total'] > 0 else 0
    
    print(f"\n  --- Compression ---")
    print(f"  Text size: {text_bytes / 1024 / 1024:.2f} MB")
    print(f"  Memory: {mem['total'] / 1024 / 1024:.2f} MB")
    print(f"  Ratio: {compression_ratio:.1f}x")
    
    # Cleanup
    import shutil
    try:
        shutil.rmtree(persist_dir, ignore_errors=True)
    except:
        pass
    
    return {
        'n_tokens': n_tokens,
        'n_pages': n_pages,
        'triple_count': triple_count,
        'ingest_time_ms': ingest_time,
        'tokens_per_sec': tokens_per_sec,
        'graph_time_ms': graph_time,
        'memory_bytes': mem['total'],
        'bytes_per_token': mem['total'] / n_tokens,
        'compression_ratio': compression_ratio,
        'p50_ms': p50,
        'p95_ms': p95,
        'p99_ms': p99,
        'components': mem,
    }


def main():
    separator("FULL BENCHMARK SUITE: 1M to 500M Tokens")
    print("  Configuration:")
    print("  - page_size: 1000 tokens")
    print("  - cache_size: 10 pages")
    print("  - d_model: 512")
    print("  - B2: diagonal covariance + float16 mu")
    print("  - B3: k=5 eigenvalues")
    print("  - Graph: upper triangle storage")
    
    # Test sizes
    test_sizes = [
        1_000_000,      # 1M
        5_000_000,      # 5M
        10_000_000,     # 10M
        25_000_000,     # 25M
        50_000_000,     # 50M
        100_000_000,    # 100M
        250_000_000,    # 250M
        500_000_000,    # 500M
    ]
    
    results = []
    
    for n_tokens in test_sizes:
        try:
            result = run_benchmark(n_tokens)
            results.append(result)
        except Exception as e:
            print(f"\n  ERROR at {n_tokens:,} tokens: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Summary table
    separator("SUMMARY TABLE")
    print(f"\n  {'Tokens':>12} {'Memory':>10} {'B/Tok':>8} {'Ratio':>8} {'Ingest':>10} {'Tok/s':>12} {'p50':>8} {'p95':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*12} {'-'*8} {'-'*8}")
    
    for r in results:
        print(f"  {r['n_tokens']:>12,} {r['memory_bytes']/1024/1024:>9.2f}M {r['bytes_per_token']:>7.4f} {r['compression_ratio']:>7.1f}x {r['ingest_time_ms']:>9.0f}ms {r['tokens_per_sec']:>11,.0f} {r['p50_ms']:>7.1f}ms {r['p95_ms']:>7.1f}ms")
    
    # Projections
    separator("PROJECTIONS")
    
    if len(results) >= 2:
        # Linear regression for memory
        tokens = np.array([r['n_tokens'] for r in results])
        memory = np.array([r['memory_bytes'] for r in results])
        
        # Fit: memory = a * tokens + b
        coeffs = np.polyfit(tokens, memory, 1)
        a, b = coeffs
        
        print(f"\n  Memory model: {a:.4f} * tokens + {b:.0f} bytes")
        print(f"  (a = bytes per additional token, b = fixed overhead)")
        
        # Project to 25M and 500M
        for target in [25_000_000, 500_000_000]:
            projected_memory = a * target + b
            projected_bpt = projected_memory / target
            print(f"\n  {target:,} tokens projection:")
            print(f"    Memory: {projected_memory / 1024 / 1024:.2f} MB")
            print(f"    Bytes/token: {projected_bpt:.4f}")
    
    # Component scaling
    separator("COMPONENT SCALING")
    print(f"\n  {'Tokens':>12} {'B1':>10} {'B2':>10} {'B3':>10} {'Graph':>10} {'Pages':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for r in results:
        c = r['components']
        print(f"  {r['n_tokens']:>12,} {c['b1_embeddings']/1024/1024:>9.2f}M {c['b2_patches']/1024/1024:>9.2f}M {c['b3_signatures']/1024/1024:>9.2f}M {(c['graph_adj']+c['graph_lap'])/1024/1024:>9.2f}M {c['pages_text']/1024/1024:>9.2f}M")
    
    separator("BENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
