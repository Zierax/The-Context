#!/usr/bin/env python3
"""Analyze memory usage per component to find optimization targets."""
import sys
import os
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from math_engine import SeededLSH, sinusoidal_encode, normalized_laplacian
from knowledge_graph import DeterministicKnowledgeGraph
from memory_manager import VirtualMemoryTree
from entity_extractor import HeuristicExtractor

def analyze_memory():
    print("=" * 70)
    print("  MEMORY USAGE ANALYSIS (REALISTIC)")
    print("=" * 70)
    
    # Simulate a 1M token corpus
    n_tokens = 1_000_000
    page_size = 1000
    n_pages = n_tokens // page_size
    d_model = 512
    
    # Realistic cache size - only keep 10 pages in RAM (rest on disk)
    cache_size = 10
    
    print(f"\n  Corpus: {n_tokens:,} tokens, {n_pages} pages, d_model={d_model}")
    print(f"  Cache size: {cache_size} pages (rest on disk)")
    
    # 1. VirtualMemoryTree
    print("\n  --- VirtualMemoryTree ---")
    tree = VirtualMemoryTree(page_size=page_size, cache_size=cache_size)
    
    # Simulate ingestion
    tokens = [f"token_{i}" for i in range(n_tokens)]
    page_ids = list(tree.ingest_stream(iter(tokens)))
    
    # Calculate memory - ONLY pages in RAM
    pages_in_ram = len(tree.pages)
    pages_memory = sum(len(text.encode('utf-8')) for text in tree.pages.values())
    print(f"  Total pages created: {n_pages}")
    print(f"  Pages in RAM: {pages_in_ram} (cache_size={cache_size})")
    print(f"  Pages on disk: {n_pages - pages_in_ram}")
    print(f"  Pages memory (RAM only): {pages_memory / 1024:.2f} KB")
    print(f"  Bytes per token (pages): {pages_memory / n_tokens:.4f}")
    
    # 2. Knowledge Graph
    print("\n  --- Knowledge Graph ---")
    graph = DeterministicKnowledgeGraph(d_model=d_model)
    
    # Simulate triplets
    n_concepts = 1000
    for i in range(n_concepts):
        graph.add_triplet(f"subject_{i}", "relates_to", f"object_{i}")
    
    graph.build_laplacian()
    
    # Calculate memory
    adj_memory = graph.A.data.nbytes + graph.A.indices.nbytes + graph.A.indptr.nbytes
    lap_memory = graph.L_sym.data.nbytes + graph.L_sym.indices.nbytes + graph.L_sym.indptr.nbytes
    rho_memory = graph.rho.nbytes if graph.rho is not None else 0
    
    print(f"  Nodes: {len(graph.node_to_idx)}")
    print(f"  Adjacency matrix: {adj_memory / 1024:.2f} KB")
    print(f"  Laplacian matrix: {lap_memory / 1024:.2f} KB")
    print(f"  Rho vector: {rho_memory / 1024:.2f} KB")
    print(f"  Total graph memory: {(adj_memory + lap_memory + rho_memory) / 1024:.2f} KB")
    print(f"  Bytes per token (graph): {(adj_memory + lap_memory + rho_memory) / n_tokens:.4f}")
    
    # 3. LSH
    print("\n  --- SeededLSH ---")
    lsh = SeededLSH(d=d_model, w=10.0, m=8, seed=42)
    
    lsh_memory = lsh.a.nbytes + lsh.b.nbytes
    print(f"  Hash vectors (a): {lsh.a.nbytes / 1024:.2f} KB")
    print(f"  Bias terms (b): {lsh.b.nbytes / 1024:.2f} KB")
    print(f"  Total LSH memory: {lsh_memory / 1024:.2f} KB")
    print(f"  Bytes per token (LSH): {lsh_memory / n_tokens:.4f}")
    
    # 4. Embeddings - using float16
    print("\n  --- Embeddings ---")
    n_stored_embeddings = n_pages  # One per page
    embedding_memory_f64 = n_stored_embeddings * d_model * 8  # float64
    embedding_memory_f16 = n_stored_embeddings * d_model * 2  # float16
    print(f"  Stored embeddings: {n_stored_embeddings}")
    print(f"  Memory (float64): {embedding_memory_f64 / 1024 / 1024:.2f} MB")
    print(f"  Memory (float16): {embedding_memory_f16 / 1024 / 1024:.2f} MB")
    print(f"  Bytes per token (float16): {embedding_memory_f16 / n_tokens:.4f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    # Use float16 for embeddings
    total_memory = pages_memory + adj_memory + lap_memory + rho_memory + lsh_memory + embedding_memory_f16
    print(f"\n  Total memory: {total_memory / 1024 / 1024:.2f} MB")
    print(f"  Bytes per token: {total_memory / n_tokens:.4f}")
    print(f"  Target: < 2 bytes/token")
    print(f"  Status: {'PASS' if total_memory / n_tokens < 2 else 'FAIL'}")
    
    # Breakdown
    print(f"\n  --- Breakdown ---")
    print(f"  Pages text:    {pages_memory / total_memory * 100:.1f}%")
    print(f"  Graph:         {(adj_memory + lap_memory + rho_memory) / total_memory * 100:.1f}%")
    print(f"  LSH:           {lsh_memory / total_memory * 100:.1f}%")
    print(f"  Embeddings:    {embedding_memory_f16 / total_memory * 100:.1f}%")
    
    # 25M token projection
    print("\n" + "=" * 70)
    print("  25M TOKEN PROJECTION")
    print("=" * 70)
    n_tokens_25m = 25_000_000
    n_pages_25m = n_tokens_25m // page_size
    cache_size_25m = 10  # Keep only 10 pages in RAM
    
    # Pages in RAM (fixed by cache_size)
    pages_memory_25m = cache_size_25m * page_size * 5  # ~5 bytes/token average
    
    # Graph scales with unique concepts, not tokens
    graph_memory_25m = (adj_memory + lap_memory + rho_memory)  # Fixed
    
    # LSH is fixed (d=512)
    lsh_memory_25m = lsh_memory  # Fixed
    
    # Embeddings: all stored on disk, only cache_size in RAM
    embeddings_in_ram = cache_size_25m * d_model * 2  # float16
    embeddings_on_disk = (n_pages_25m - cache_size_25m) * d_model * 2  # float16 on disk
    
    total_memory_25m = pages_memory_25m + graph_memory_25m + lsh_memory_25m + embeddings_in_ram
    
    print(f"\n  Corpus: {n_tokens_25m:,} tokens")
    print(f"  Pages: {n_pages_25m}")
    print(f"  Cache: {cache_size_25m} pages")
    print(f"\n  RAM usage:")
    print(f"    Pages text:     {pages_memory_25m / 1024 / 1024:.2f} MB")
    print(f"    Graph:          {graph_memory_25m / 1024 / 1024:.2f} MB")
    print(f"    LSH:            {lsh_memory_25m / 1024 / 1024:.2f} MB")
    print(f"    Embeddings:     {embeddings_in_ram / 1024 / 1024:.2f} MB")
    print(f"    Total RAM:      {total_memory_25m / 1024 / 1024:.2f} MB")
    print(f"\n  Disk usage:")
    print(f"    Pages:          {(n_pages_25m - cache_size_25m) * page_size * 5 / 1024 / 1024:.2f} MB")
    print(f"    Embeddings:     {embeddings_on_disk / 1024 / 1024:.2f} MB")
    print(f"\n  Bytes per token: {total_memory_25m / n_tokens_25m:.4f}")
    print(f"  Target: < 2 bytes/token")
    print(f"  Status: {'PASS' if total_memory_25m / n_tokens_25m < 2 else 'FAIL'}")

if __name__ == "__main__":
    analyze_memory()
