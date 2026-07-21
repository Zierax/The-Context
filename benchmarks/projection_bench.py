#!/usr/bin/env python3
"""Projection benchmark: 1M baseline + project to 25M, 100M, 500M."""
import sys, os, tempfile, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(40))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory_manager import VirtualMemoryTree
from query_engine import QueryEngine
from knowledge_graph import DeterministicKnowledgeGraph
from entity_extractor import HeuristicExtractor
from math_engine import SeededLSH

print("=" * 70)
print("  PROJECTION BENCHMARK: 1M Baseline → 25M, 100M, 500M")
print("=" * 70)

# Run 1M and 5M to establish scaling model
results = []
for n in [1_000_000, 5_000_000]:
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
    
    # Measure each component
    pages_text = sum(len(t.encode()) for t in tree.pages.values())
    b1_emb = len(tree.beacon_b1) * d * 2
    b2_pats = len(tree.beacon_b2) * (d*2 + d*8)
    b3_sigs = len(tree.beacon_b3) * (5*8 + 10*5*8)
    graph_mem = 0
    if graph.A is not None:
        graph_mem += graph.A.data.nbytes + graph.A.indices.nbytes + graph.A.indptr.nbytes
    if graph.L_sym is not None:
        graph_mem += graph.L_sym.data.nbytes + graph.L_sym.indices.nbytes + graph.L_sym.indptr.nbytes
    lsh_mem = lsh.a.nbytes + lsh.b.nbytes
    total_mem = pages_text + b1_emb + b2_pats + b3_sigs + graph_mem + lsh_mem
    
    gate = QueryEngine(tree=tree, graph=graph, lsh=lsh, d_model=d)
    for i in range(3):
        gate.collapse(query=f"test {i}", max_tokens=512)
    lats = []
    for i in range(20):
        r = gate.collapse(query=f"quantum memory {i % 5}", max_tokens=1024)
        lats.append(r.latency_ms)
    lats.sort()
    
    text_bytes = sum(len(t.encode()) for t in corpus)
    ratio = text_bytes / total_mem if total_mem > 0 else 0
    
    results.append({
        'n': n, 'mem': total_mem, 'pages': pages_text, 'b1': b1_emb, 'b2': b2_pats,
        'b3': b3_sigs, 'graph': graph_mem, 'lsh': lsh_mem, 'ingest_ms': ingest_ms,
        'graph_ms': graph_ms, 'p50': lats[len(lats)//2], 'p95': lats[int(len(lats)*0.95)],
        'ratio': ratio, 'bpt': total_mem/n, 'n_pages': len(pids), 'triples': triples,
        'concepts': len(graph.node_to_idx),
    })
    
    print(f" Done! {total_mem/1024/1024:.2f} MB, {total_mem/n:.4f} B/tok, {ratio:.1f}x")
    
    import shutil
    shutil.rmtree(persist, ignore_errors=True)

# Print detailed results
print("\n" + "=" * 70)
print("  DETAILED RESULTS")
print("=" * 70)

for r in results:
    print(f"\n  --- {r['n']:,} tokens ({r['n_pages']:,} pages, {r['concepts']:,} concepts, {r['triples']:,} triples) ---")
    print(f"  Memory breakdown:")
    print(f"    Pages text:  {r['pages']/1024/1024:>8.2f} MB ({r['pages']/r['mem']*100:.1f}%)")
    print(f"    B1 embeds:   {r['b1']/1024/1024:>8.2f} MB ({r['b1']/r['mem']*100:.1f}%)")
    print(f"    B2 patches:  {r['b2']/1024/1024:>8.2f} MB ({r['b2']/r['mem']*100:.1f}%)")
    print(f"    B3 sigs:     {r['b3']/1024/1024:>8.2f} MB ({r['b3']/r['mem']*100:.1f}%)")
    print(f"    Graph:       {r['graph']/1024/1024:>8.2f} MB ({r['graph']/r['mem']*100:.1f}%)")
    print(f"    LSH:         {r['lsh']/1024/1024:>8.2f} MB ({r['lsh']/r['mem']*100:.1f}%)")
    print(f"    Total:       {r['mem']/1024/1024:>8.2f} MB")
    print(f"  Performance:")
    print(f"    Ingest:  {r['ingest_ms']:.0f}ms ({r['n']/(r['ingest_ms']/1000):,.0f} tok/s)")
    print(f"    Graph:   {r['graph_ms']:.0f}ms")
    print(f"    Query:   p50={r['p50']:.1f}ms p95={r['p95']:.1f}ms")
    print(f"  Compression:")
    print(f"    Ratio:   {r['ratio']:.1f}x")
    print(f"    B/tok:   {r['bpt']:.4f}")

# Linear projection
print("\n" + "=" * 70)
print("  LINEAR PROJECTION")
print("=" * 70)

tokens = np.array([r['n'] for r in results])
memory = np.array([r['mem'] for r in results])

# Fit memory = a * tokens + b
a, b = np.polyfit(tokens, memory, 1)
print(f"\n  Memory model: {a:.6f} * tokens + {b:.0f} bytes")
print(f"  (a = {a*1e6:.2f} bytes per million tokens)")

# Project to targets
targets = [25_000_000, 100_000_000, 250_000_000, 500_000_000]
print(f"\n  {'Target':>15} {'Projected Memory':>18} {'Bytes/Tok':>12} {'Ratio':>10}")
print(f"  {'-'*15} {'-'*18} {'-'*12} {'-'*10}")

for target in targets:
    proj_mem = a * target + b
    proj_bpt = proj_mem / target
    text_bytes = target * 4.5  # ~4.5 bytes per token average
    proj_ratio = text_bytes / proj_mem
    print(f"  {target:>15,} {proj_mem/1024/1024:>16.2f} MB {proj_bpt:>11.4f} {proj_ratio:>9.1f}x")

# Component scaling
print("\n  Component scaling (bytes per token at scale):")
print(f"  At 25M tokens:")
for comp in ['pages', 'b1', 'b2', 'b3', 'graph', 'lsh']:
    # For pages and b1: scale with tokens
    # For b2, b3, graph: scale with pages (tokens/1000)
    # For lsh: fixed
    if comp == 'lsh':
        print(f"    {comp:>10}: {results[0][comp]/1024/1024:>8.2f} MB (fixed)")
    elif comp in ['pages', 'b1']:
        # Scale linearly
        scale = 25_000_000 / results[0]['n']
        val = results[0][comp] * scale
        print(f"    {comp:>10}: {val/1024/1024:>8.2f} MB ({val/25_000_000:.4f} B/tok)")
    else:
        # Scale with pages
        scale = 25_000_000 / results[0]['n']
        val = results[0][comp] * scale
        print(f"    {comp:>10}: {val/1024/1024:>8.2f} MB ({val/25_000_000:.4f} B/tok)")

print("\n" + "=" * 70)
print("  BENCHMARK COMPLETE")
print("=" * 70)
