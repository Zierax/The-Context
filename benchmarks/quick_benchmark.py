#!/usr/bin/env python3
"""Quick benchmark: 1M to 50M tokens (quiet mode)."""

import gc
import os
import sys
import tempfile
import time
import warnings

import numpy as np

# Suppress all logging
import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(40))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_engine import SeededLSH, compute_gaussian_patch
from knowledge_graph import DeterministicKnowledgeGraph
from memory_manager import VirtualMemoryTree
from entity_extractor import HeuristicExtractor
from query_engine import QueryEngine


def generate_corpus(n_tokens, seed=42):
    """Generate a realistic corpus with repeated concepts."""
    rng = np.random.RandomState(seed)
    corpus = []
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
    concept_idx = 0
    for i in range(n_tokens):
        if i % 500 == 0:
            corpus.extend(concepts[concept_idx % len(concepts)].split())
            concept_idx += 1
        else:
            corpus.append(f"token_{i}")
    return corpus[:n_tokens]


def measure_memory(tree, graph, lsh):
    """Measure memory usage of all components."""
    d_model = 512
    pages_text = sum(len(text.encode('utf-8')) for text in tree.pages.values())
    b1_count = len(tree.beacon_b1)
    b1_memory = b1_count * d_model * 2
    b2_count = len(tree.beacon_b2)
    b2_memory = b2_count * (d_model * 2 + d_model * 8)
    b3_count = len(tree.beacon_b3)
    b3_memory = b3_count * (5 * 8 + 10 * 5 * 8)
    if graph.A is not None:
        adj_memory = graph.A.data.nbytes + graph.A.indices.nbytes + graph.A.indptr.nbytes
    else:
        adj_memory = 0
    if graph.L_sym is not None:
        lap_memory = graph.L_sym.data.nbytes + graph.L_sym.indices.nbytes + graph.L_sym.indptr.nbytes
    else:
        lap_memory = 0
    lsh_memory = lsh.a.nbytes + lsh.b.nbytes
    total = pages_text + b1_memory + b2_memory + b3_memory + adj_memory + lap_memory + lsh_memory
    return {
        'pages_text': pages_text, 'b1_embeddings': b1_memory, 'b2_patches': b2_memory,
        'b3_signatures': b3_memory, 'graph_adj': adj_memory, 'graph_lap': lap_memory,
        'lsh': lsh_memory, 'total': total,
    }


def run_benchmark(n_tokens, d_model=512, cache_size=10):
    """Run full benchmark for a given corpus size."""
    corpus = generate_corpus(n_tokens)
    
    persist_dir = tempfile.mkdtemp(prefix=f"bench_{n_tokens}_")
    lsh = SeededLSH(d=d_model, w=10.0, m=8, seed=42)
    tree = VirtualMemoryTree(page_size=1000, cache_size=cache_size, persist_dir=persist_dir)
    graph = DeterministicKnowledgeGraph(d_model=d_model)
    extractor = HeuristicExtractor()
    
    # Ingestion
    t0 = time.perf_counter()
    page_ids = list(tree.ingest_stream(iter(corpus)))
    ingest_time = (time.perf_counter() - t0) * 1000
    n_pages = len(page_ids)
    tokens_per_sec = n_tokens / (ingest_time / 1000) if ingest_time > 0 else 0
    
    # Graph
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
    
    # Laplacian
    if len(graph.node_to_idx) >= 2:
        graph.build_laplacian()
    
    # Memory
    mem = measure_memory(tree, graph, lsh)
    
    # Query latency
    gate = QueryEngine(tree=tree, graph=graph, lsh=lsh, d_model=d_model)
    for i in range(3):
        gate.collapse(query=f"test {i}", max_tokens=1024)
    
    latencies = []
    n_queries = min(50, max(10, n_tokens // 50000))
    for i in range(n_queries):
        q = f"quantum memory spectral manifold {i % 10}"
        result = gate.collapse(query=q, max_tokens=2048)
        latencies.append(result.latency_ms)
    
    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    
    # Compression
    text_bytes = sum(len(text.encode('utf-8')) for text in corpus)
    compression_ratio = text_bytes / mem['total'] if mem['total'] > 0 else 0
    
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
        'memory_bytes': mem['total'],
        'bytes_per_token': mem['total'] / n_tokens,
        'compression_ratio': compression_ratio,
        'p50_ms': p50,
        'p95_ms': p95,
        'components': mem,
    }


def main():
    print("=" * 80)
    print("  QUICK BENCHMARK: 1M to 50M Tokens")
    print("=" * 80)
    print("  Config: page_size=1000, cache_size=10, d_model=512")
    print("  Optimizations: B2 diagonal, float16 mu, graph upper triangle, B3 k=5")
    
    test_sizes = [
        1_000_000,      # 1M
        5_000_000,      # 5M
        10_000_000,     # 10M
        25_000_000,     # 25M
        50_000_000,     # 50M
    ]
    
    results = []
    
    for n_tokens in test_sizes:
        print(f"\n  Running {n_tokens:,} tokens...")
        try:
            result = run_benchmark(n_tokens)
            results.append(result)
            print(f"    Done: {result['memory_bytes']/1024/1024:.2f} MB, {result['bytes_per_token']:.4f} B/tok, {result['compression_ratio']:.1f}x, p50={result['p50_ms']:.1f}ms")
        except Exception as e:
            print(f"    ERROR: {e}")
            break
    
    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY TABLE")
    print("=" * 80)
    print(f"\n  {'Tokens':>12} {'Memory':>10} {'B/Tok':>8} {'Ratio':>8} {'Ingest':>10} {'Tok/s':>12} {'p50':>8} {'p95':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*12} {'-'*8} {'-'*8}")
    
    for r in results:
        print(f"  {r['n_tokens']:>12,} {r['memory_bytes']/1024/1024:>9.2f}M {r['bytes_per_token']:>7.4f} {r['compression_ratio']:>7.1f}x {r['ingest_time_ms']:>9.0f}ms {r['tokens_per_sec']:>11,.0f} {r['p50_ms']:>7.1f}ms {r['p95_ms']:>7.1f}ms")
    
    # Projections
    print("\n" + "=" * 80)
    print("  PROJECTIONS")
    print("=" * 80)
    
    if len(results) >= 2:
        tokens = np.array([r['n_tokens'] for r in results])
        memory = np.array([r['memory_bytes'] for r in results])
        coeffs = np.polyfit(tokens, memory, 1)
        a, b = coeffs
        
        print(f"\n  Memory model: {a:.6f} * tokens + {b:.0f} bytes")
        print(f"  (a = {a:.2f} bytes per additional token)")
        
        for target in [25_000_000, 100_000_000, 250_000_000, 500_000_000]:
            projected_memory = a * target + b
            projected_bpt = projected_memory / target
            print(f"\n  {target:,} tokens projection:")
            print(f"    Memory: {projected_memory / 1024 / 1024:.2f} MB")
            print(f"    Bytes/token: {projected_bpt:.4f}")
    
    # Component scaling
    print("\n" + "=" * 80)
    print("  COMPONENT SCALING")
    print("=" * 80)
    print(f"\n  {'Tokens':>12} {'B1':>10} {'B2':>10} {'B3':>10} {'Graph':>10} {'Pages':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    for r in results:
        c = r['components']
        print(f"  {r['n_tokens']:>12,} {c['b1_embeddings']/1024/1024:>9.2f}M {c['b2_patches']/1024/1024:>9.2f}M {c['b3_signatures']/1024/1024:>9.2f}M {(c['graph_adj']+c['graph_lap'])/1024/1024:>9.2f}M {c['pages_text']/1024/1024:>9.2f}M")
    
    print("\n" + "=" * 80)
    print("  BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
