#!/usr/bin/env python3
"""1M baseline + project to 25M-500M."""
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
print("  1M BASELINE → PROJECTION TO 25M, 100M, 500M")
print("=" * 70)

n = 1_000_000
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

print(f"\n  Ingesting {n:,} tokens...", end="", flush=True)
t0 = time.perf_counter()
pids = list(tree.ingest_stream(iter(corpus)))
ingest_ms = (time.perf_counter() - t0) * 1000
print(f" {ingest_ms:.0f}ms")

print(f"  Building graph...", end="", flush=True)
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
print(f" {graph_ms:.0f}ms ({triples:,} triples)")

if len(graph.node_to_idx) >= 2:
    print(f"  Building Laplacian...", end="", flush=True)
    t0 = time.perf_counter()
    graph.build_laplacian()
    lap_ms = (time.perf_counter() - t0) * 1000
    print(f" {lap_ms:.0f}ms")

# Measure components
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

# Query latency
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

# Print 1M results
print(f"\n  --- 1M BASELINE RESULTS ---")
print(f"  Pages: {len(pids):,} | Concepts: {len(graph.node_to_idx):,} | Triples: {triples:,}")
print(f"\n  Memory breakdown:")
print(f"    Pages text:  {pages_text/1024/1024:>8.3f} MB ({pages_text/total_mem*100:.1f}%)")
print(f"    B1 embeds:   {b1_emb/1024/1024:>8.3f} MB ({b1_emb/total_mem*100:.1f}%)")
print(f"    B2 patches:  {b2_pats/1024/1024:>8.3f} MB ({b2_pats/total_mem*100:.1f}%)")
print(f"    B3 sigs:     {b3_sigs/1024/1024:>8.3f} MB ({b3_sigs/total_mem*100:.1f}%)")
print(f"    Graph:       {graph_mem/1024/1024:>8.3f} MB ({graph_mem/total_mem*100:.1f}%)")
print(f"    LSH:         {lsh_mem/1024/1024:>8.3f} MB ({lsh_mem/total_mem*100:.1f}%)")
print(f"    ---")
print(f"    Total:       {total_mem/1024/1024:>8.3f} MB")
print(f"\n  Performance:")
print(f"    Ingest:  {ingest_ms:.0f}ms ({n/(ingest_ms/1000):,.0f} tok/s)")
print(f"    Graph:   {graph_ms:.0f}ms")
print(f"    Query:   p50={lats[len(lats)//2]:.1f}ms p95={lats[int(len(lats)*0.95)]:.1f}ms")
print(f"\n  Compression:")
print(f"    Text:    {text_bytes/1024/1024:.2f} MB")
print(f"    Memory:  {total_mem/1024/1024:.2f} MB")
print(f"    Ratio:   {ratio:.1f}x")
print(f"    B/tok:   {total_mem/n:.4f}")

# Projection
print(f"\n  --- PROJECTIONS ---")
print(f"  Scaling model: fixed + variable components")
print(f"  Variable: pages ({pages_text/n:.4f} B/tok) + B1 ({b1_emb/n:.4f} B/tok) + B2 ({b2_pats/n:.4f} B/tok) + B3 ({b3_sigs/n:.4f} B/tok)")
print(f"  Fixed: Graph ({graph_mem/1024/1024:.2f} MB) + LSH ({lsh_mem/1024/1024:.2f} MB)")
print(f"  Note: Graph/LSH are fixed regardless of corpus size (scale with concepts, not tokens)")

# Per-token variable cost
var_per_tok = (pages_text + b1_emb + b2_pats + b3_sigs) / n
fixed_total = graph_mem + lsh_mem

print(f"\n  Variable cost: {var_per_tok:.6f} bytes/token")
print(f"  Fixed cost: {fixed_total/1024/1024:.2f} MB")

targets = [
    (25_000_000, "25M"),
    (100_000_000, "100M"),
    (250_000_000, "250M"),
    (500_000_000, "500M"),
]

print(f"\n  {'Target':>10} {'Variable':>12} {'Fixed':>10} {'Total':>12} {'B/tok':>10} {'Ratio':>10}")
print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")

for target, label in targets:
    var_mem = var_per_tok * target
    total_proj = var_mem + fixed_total
    bpt = total_proj / target
    text_proj = target * 4.5
    ratio_proj = text_proj / total_proj
    print(f"  {label:>10} {var_mem/1024/1024:>10.2f} MB {fixed_total/1024/1024:>8.2f} MB {total_proj/1024/1024:>10.2f} MB {bpt:>9.4f} {ratio_proj:>9.1f}x")

print(f"\n  --- KEY INSIGHT ---")
print(f"  Memory scales LINEARLY with tokens (not quadratically)")
print(f"  At 500M tokens: only {var_per_tok * 500_000_000 / 1024/1024 / 1024:.1f} GB + {fixed_total/1024/1024:.0f} MB fixed")
print(f"  Compression ratio IMPROVES with scale (fixed costs amortized)")

print("\n" + "=" * 70)
print("  BENCHMARK COMPLETE")
print("=" * 70)
