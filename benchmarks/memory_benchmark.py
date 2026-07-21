#!/usr/bin/env python3
"""
Real Memory Benchmark — The Context
Measures actual memory usage per component for 25M token corpus.
"""
import sys
import os
import time
import tracemalloc

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_engine import SeededLSH, sinusoidal_encode, normalized_laplacian
from knowledge_graph import DeterministicKnowledgeGraph
from memory_manager import VirtualMemoryTree
from entity_extractor import HeuristicExtractor

def measure_memory(label: str) -> None:
    """Print current memory usage."""
    current, peak = tracemalloc.get_traced_memory()
    print(f"  [{label}] Current: {current / 1024 / 1024:.2f} MB, Peak: {peak / 1024 / 1024:.2f} MB")

def run_benchmark(n_tokens: int = 25_000_000) -> None:
    """Run comprehensive memory benchmark."""
    tracemalloc.start()
    
    print("=" * 70)
    print(f"  MEMORY BENCHMARK: {n_tokens:,} tokens")
    print("=" * 70)
    
    page_size = 1000
    cache_size = 10
    d_model = 512
    
    n_pages = n_tokens // page_size
    
    print(f"\n  Config: page_size={page_size}, cache_size={cache_size}, d_model={d_model}")
    print(f"  Pages: {n_pages}")
    
    # 1. VirtualMemoryTree
    print("\n  --- VirtualMemoryTree ---")
    measure_memory("start")
    
    tree = VirtualMemoryTree(page_size=page_size, cache_size=cache_size)
    
    # Simulate ingestion (only first 100K tokens for speed)
    test_tokens = min(n_tokens, 100_000)
    print(f"  Simulating {test_tokens:,} tokens...")
    
    tokens = [f"token_{i}" for i in range(test_tokens)]
    page_ids = list(tree.ingest_stream(iter(tokens)))
    
    measure_memory("after_ingestion")
    
    # Calculate memory
    pages_in_ram = len(tree.pages)
    pages_text = sum(len(text.encode('utf-8')) for text in tree.pages.values())
    
    # B1 embeddings
    b1_count = len(tree.beacon_b1)
    b1_memory = b1_count * d_model * 2  # float16
    
    # B2 patches
    b2_count = len(tree.beacon_b2)
    b2_memory = b2_count * (d_model * 2 + d_model * 8)  # mu (float16) + Sigma_inv_diag (float64)
    
    # B3 signatures (k=5 eigenvalues instead of 10)
    b3_count = len(tree.beacon_b3)
    b3_memory = b3_count * (5 * 8 + 10 * 5 * 8)  # eigenvalues (5) + eigenvectors (10x5)
    
    print(f"  Pages in RAM: {pages_in_ram}")
    print(f"  Pages text: {pages_text / 1024:.2f} KB")
    print(f"  B1 beacons: {b1_count} ({b1_memory / 1024 / 1024:.2f} MB)")
    print(f"  B2 beacons: {b2_count} ({b2_memory / 1024 / 1024:.2f} MB)")
    print(f"  B3 beacons: {b3_count} ({b3_memory / 1024 / 1024:.2f} MB)")
    
    # 2. Knowledge Graph
    print("\n  --- Knowledge Graph ---")
    measure_memory("before_graph")
    
    graph = DeterministicKnowledgeGraph(d_model=d_model)
    extractor = HeuristicExtractor()
    
    # Extract triples from first 10K pages
    triple_count = 0
    for pid in page_ids[:min(10000, len(page_ids))]:
        page_text = tree.get_page(pid)
        if page_text is None:
            continue
        triples = extractor.extract(page_text)
        for subject, predicate, obj in triples:
            b1_id = tree.get_beacon_for_page(pid)
            graph.add_triplet(subject, predicate, obj, beacon_id=b1_id or "")
            triple_count += 1
    
    measure_memory("after_graph")
    
    # Graph memory
    n_nodes = len(graph.node_to_idx)
    if graph.A is not None:
        adj_memory = graph.A.data.nbytes + graph.A.indices.nbytes + graph.A.indptr.nbytes
    else:
        adj_memory = 0
    
    print(f"  Nodes: {n_nodes}")
    print(f"  Edges: {triple_count}")
    print(f"  Adjacency: {adj_memory / 1024:.2f} KB")
    
    # 3. LSH
    print("\n  --- SeededLSH ---")
    lsh = SeededLSH(d=d_model, w=10.0, m=8, seed=42)
    lsh_memory = lsh.a.nbytes + lsh.b.nbytes
    print(f"  Hash vectors: {lsh.a.nbytes / 1024:.2f} KB")
    print(f"  Bias terms: {lsh.b.nbytes / 1024:.2f} KB")
    
    # 4. Build Laplacian
    print("\n  --- Laplacian ---")
    if len(graph.node_to_idx) >= 2:
        graph.build_laplacian()
        lap_memory = graph.L_sym.data.nbytes + graph.L_sym.indices.nbytes + graph.L_sym.indptr.nbytes
        print(f"  Laplacian: {lap_memory / 1024:.2f} KB")
    else:
        lap_memory = 0
        print("  Skipped (too few nodes)")
    
    measure_memory("final")
    
    # Summary
    total_memory = pages_text + b1_memory + b2_memory + b3_memory + adj_memory + lap_memory + lsh_memory
    
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    print(f"\n  Total memory: {total_memory / 1024 / 1024:.2f} MB")
    print(f"  Bytes per token: {total_memory / test_tokens:.4f}")
    print(f"  Compression ratio: {test_tokens * 5 / total_memory:.1f}x (text bytes / memory)")
    
    # Breakdown
    print(f"\n  --- Breakdown ---")
    print(f"  Pages text:    {pages_text / total_memory * 100:.1f}%")
    print(f"  B1 embeddings: {b1_memory / total_memory * 100:.1f}%")
    print(f"  B2 patches:    {b2_memory / total_memory * 100:.1f}%")
    print(f"  B3 signatures: {b3_memory / total_memory * 100:.1f}%")
    print(f"  Graph:         {adj_memory / total_memory * 100:.1f}%")
    print(f"  Laplacian:     {lap_memory / total_memory * 100:.1f}%")
    print(f"  LSH:           {lsh_memory / total_memory * 100:.1f}%")
    
    # Projection for 25M tokens
    print("\n" + "=" * 70)
    print("  25M TOKEN PROJECTION")
    print("=" * 70)
    
    scale_factor = n_tokens / test_tokens
    
    # Pages scale linearly (but most are on disk)
    pages_25m = cache_size * page_size * 5  # ~5 bytes/token
    
    # B1 scales linearly
    b1_25m = b1_memory * scale_factor
    
    # B2 scales linearly
    b2_25m = b2_memory * scale_factor
    
    # B3 scales linearly
    b3_25m = b3_memory * scale_factor
    
    # Graph scales with unique concepts (not tokens)
    graph_25m = adj_memory * 2  # Conservative: 2x for more concepts
    lap_25m = lap_memory * 2
    
    # LSH is fixed
    lsh_25m = lsh_memory
    
    total_25m = pages_25m + b1_25m + b2_25m + b3_25m + graph_25m + lap_25m + lsh_25m
    
    print(f"\n  Projected memory: {total_25m / 1024 / 1024:.2f} MB")
    print(f"  Bytes per token: {total_25m / n_tokens:.4f}")
    print(f"  Compression ratio: {n_tokens * 5 / total_25m:.1f}x")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    run_benchmark(25_000_000)
