#!/usr/bin/env python3
"""Minimal benchmark: 1M to 10M tokens."""
import sys, os, tempfile, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(40))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from the_context.core import VirtualMemoryTree
from the_context.query import QueryEngine
from the_context.core import DeterministicKnowledgeGraph
from the_context.extraction import HeuristicExtractor
from the_context.core import SeededLSH

print("=" * 70)
print("  MINI BENCHMARK: 1M to 10M Tokens")
print("=" * 70)

for n in [1_000_000, 5_000_000, 10_000_000]:
    print(f"\n  Running {n:,} tokens...", end="", flush=True)
    
    corpus = []
    for i in range(n):
        if i % 500 == 0:
            corpus.extend("quantum memory spectral manifold representation beacon hierarchy".split())
        else:
            corpus.append(f"token_{i}")
    
    d = 128
    persist = tempfile.mkdtemp()
    lsh = SeededLSH(d=d, w=10.0, m=4, seed=42)
    tree = VirtualMemoryTree(page_size=1000, cache_size=10, persist_dir=persist)
    graph = DeterministicKnowledgeGraph(d_model=d)
    extractor = HeuristicExtractor()
    
    t0 = time.perf_counter()
    pids = list(tree.ingest_stream(iter(corpus)))
    ingest_ms = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    triples = 0
    for pid in pids:
        pt = tree.get_page(pid)
        if pt is None: continue
        for s, p, o in extractor.extract(pt):
            b1 = tree.get_beacon_for_page(pid)
            graph.add_triplet(s, p, o, beacon_id=b1 or "")
            triples += 1
    graph_ms = (time.perf_counter() - t0) * 1000
    
    if len(graph.node_to_idx) >= 2:
        graph.build_laplacian()
    
    mem = sum(len(t.encode()) for t in tree.pages.values())
    mem += len(tree.beacon_b1) * d * 2
    mem += len(tree.beacon_b2) * (d*2 + d*8)
    mem += len(tree.beacon_b3) * (5*8 + 10*5*8)
    if graph.A is not None:
        mem += graph.A.data.nbytes + graph.A.indices.nbytes + graph.A.indptr.nbytes
    if graph.L_sym is not None:
        mem += graph.L_sym.data.nbytes + graph.L_sym.indices.nbytes + graph.L_sym.indptr.nbytes
    mem += lsh.a.nbytes + lsh.b.nbytes
    
    gate = QueryEngine(tree=tree, graph=graph, lsh=lsh, d_model=d)
    for i in range(3):
        gate.collapse(query=f"test {i}", max_tokens=512)
    
    lats = []
    for i in range(20):
        r = gate.collapse(query=f"quantum memory {i % 5}", max_tokens=1024)
        lats.append(r.latency_ms)
    lats.sort()
    
    text_bytes = sum(len(t.encode()) for t in corpus)
    ratio = text_bytes / mem if mem > 0 else 0
    bpt = mem / n
    
    print(f" Done!")
    print(f"    Memory: {mem/1024/1024:.2f} MB | B/tok: {bpt:.4f} | Ratio: {ratio:.1f}x")
    print(f"    Ingest: {ingest_ms:.0f}ms ({n/(ingest_ms/1000):,.0f} tok/s) | Graph: {graph_ms:.0f}ms")
    print(f"    Queries: p50={lats[len(lats)//2]:.1f}ms p95={lats[int(len(lats)*0.95)]:.1f}ms")
    
    import shutil
    shutil.rmtree(persist, ignore_errors=True)

print("\n" + "=" * 70)
print("  DONE")
print("=" * 70)
