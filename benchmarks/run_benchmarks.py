#!/usr/bin/env python3
"""Hard benchmarks for the Spectral Memory Manifold Co-Processor.

Tests:
1. Standard unit-level benchmarks (math operations)
2. Hard spectral graph benchmarks (large sparse matrices)
3. Real-world pipeline benchmarks (ingestion + query latency)
4. Stress test: large concept graphs
5. Determinism verification under load
"""

import gc
import math
import os
import sys
import tempfile
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from the_context.core import (
    SeededLSH,
    sinusoidal_encode,
    normalized_laplacian,
    spectral_signature,
    reconstruct_from_spectral,
    fokker_planck_step,
    submodular_pack,
    compute_gaussian_patch,
    estimate_token_count,
)
from the_context.core import DeterministicKnowledgeGraph
from the_context.core import VirtualMemoryTree
from the_context.extraction import HeuristicExtractor
from the_context.query import QueryEngine


def bench(label):
    class Timer:
        def __enter__(self):
            gc.disable()
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, *a):
            self.elapsed = (time.perf_counter() - self.t0) * 1000
            gc.enable()
            print(f"  {label:.<55} {self.elapsed:>10.2f} ms")
    return Timer()


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ─── BENCHMARK 1: Sinusoidal Encoding ───
separator("BENCHMARK 1: Sinusoidal Encoding Throughput")
for n in [100, 1000, 10000]:
    concepts = [f"concept_{i}_alpha" for i in range(n)]
    with bench(f"sinusoidal_encode(n={n}, d=512)"):
        emb = sinusoidal_encode(concepts, d_model=512)
    assert emb.shape == (n, 512)

# ─── BENCHMARK 2: LSH Hashing ───
separator("BENCHMARK 2: Seeded LSH Hashing")
for d, n in [(64, 10000), (256, 5000), (512, 2000)]:
    lsh = SeededLSH(d=d, w=10.0, m=8, seed=42)
    X = np.random.RandomState(0).randn(n, d).astype(np.float64)
    with bench(f"hash_batch(n={n}, d={d})"):
        buckets = lsh.hash_batch(X)
    assert len(buckets) == n

# ─── BENCHMARK 3: Large Sparse Laplacian ───
separator("BENCHMARK 3: Graph Laplacian (large sparse)")
for n in [500, 2000, 5000]:
    rng = np.random.RandomState(42)
    A_dense = rng.rand(n, n) * 0.1
    A_dense = (A_dense + A_dense.T) / 2
    A_dense[A_dense < 0.05] = 0
    A = sp.csr_matrix(A_dense.astype(np.float64))
    with bench(f"normalized_laplacian(n={n})"):
        L = normalized_laplacian(A)
    with bench(f"spectral_signature(n={n}, k=20)"):
        evals, evecs = spectral_signature(L, k=20)
    assert evals.shape == (20,)
    assert evecs.shape == (n, 20)

# ─── BENCHMARK 4: Fokker-Planck Diffusion ───
separator("BENCHMARK 4: Fokker-Planck Diffusion (multi-step)")
for n in [1000, 5000]:
    rng = np.random.RandomState(42)
    A_dense = rng.rand(n, n) * 0.05
    A_dense = (A_dense + A_dense.T) / 2
    A_dense[A_dense < 0.02] = 0
    A = sp.csr_matrix(A_dense.astype(np.float64))
    L = normalized_laplacian(A)
    rho = np.ones(n, dtype=np.float64) / n
    q = np.zeros(n, dtype=np.float64)
    q[:5] = 1.0
    with bench(f"fokker_planck x10 steps (n={n})"):
        for _ in range(10):
            rho = fokker_planck_step(rho, L, q)
    assert rho.shape == (n,)
    assert np.all(rho >= 0.0)

# ─── BENCHMARK 5: Submodular Packing (hard) ───
separator("BENCHMARK 5: Submodular Packing (stress)")
for n_cand, budget in [(100, 5000), (500, 20000), (2000, 50000)]:
    candidates = []
    rng = np.random.RandomState(42)
    for i in range(n_cand):
        n_concepts = rng.randint(2, 8)
        coverage = {f"c{j}": rng.random() for j in range(n_concepts)}
        candidates.append({
            "id": f"p{i}",
            "text": f"page_{i}",
            "token_count": rng.randint(50, 500),
            "concept_coverage": coverage,
            "strength": rng.random(),
        })
    with bench(f"submodular_pack(n={n_cand}, budget={budget})"):
        selected = submodular_pack(candidates, budget)
    total = sum(c["token_count"] for c in candidates if c["id"] in selected)
    assert total <= budget

# ─── BENCHMARK 6: Gaussian Patch ───
separator("BENCHMARK 6: Gaussian Patch (B2 compression)")
for n, d in [(10, 512), (50, 256), (100, 128)]:
    vectors = np.random.RandomState(0).randn(n, d).astype(np.float64)
    with bench(f"compute_gaussian_patch(n={n}, d={d})"):
        mu, Sigma_inv = compute_gaussian_patch(vectors)
    assert mu.shape == (d,)
    assert Sigma_inv.shape == (d, d)

# ─── BENCHMARK 7: Full Pipeline (small real-world) ───
separator("BENCHMARK 7: Full Pipeline — Small Corpus (50K tokens)")
d_model = 64
lsh = SeededLSH(d=d_model, w=10.0, m=4, seed=42)
tree = VirtualMemoryTree(page_size=1000, cache_size=200, persist_dir="/tmp/bench_small")
graph = DeterministicKnowledgeGraph(d_model=d_model)
extractor = HeuristicExtractor()

corpus_tokens = []
for i in range(50000):
    if i % 5000 == 0:
        corpus_tokens.extend(f"quantum memory is spectral manifold representation knowledge".split())
    elif i % 3000 == 0:
        corpus_tokens.extend(f"beacon hierarchy compresses B1 into B2 into B3 spectral".split())
    else:
        corpus_tokens.append(f"token_{i}")

with bench("ingest_stream(50K tokens)"):
    page_ids = list(tree.ingest_stream(iter(corpus_tokens)))

triple_count = 0
with bench("extract_triplets + build_graph"):
    for pid in page_ids:
        page_text = tree.get_page(pid)
        if page_text is None:
            continue
        triples = extractor.extract(page_text)
        for subject, predicate, obj in triples:
            b1_id = tree.get_beacon_for_page(pid)
            graph.add_triplet(subject, predicate, obj, weight=1.0, beacon_id=b1_id or "", page_id=pid)
            triple_count += 1

with bench("build_laplacian"):
    if len(graph.node_to_idx) >= 2:
        graph.build_laplacian()

gate = QueryEngine(tree=tree, graph=graph, lsh=lsh, d_model=d_model)

latencies = []
with bench("100 queries (sequential)"):
    for i in range(100):
        q = f"quantum memory spectral {i % 10}"
        result = gate.collapse(query=q, max_tokens=2048)
        latencies.append(result.latency_ms)

latencies_sorted = sorted(latencies)
p50 = latencies_sorted[len(latencies_sorted) // 2]
p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
print(f"\n  Query latency: p50={p50:.2f}ms  p95={p95:.2f}ms  min={min(latencies):.2f}ms  max={max(latencies):.2f}ms")
print(f"  Pages ingested: {len(page_ids)}  Triples: {triple_count}  Concepts: {len(graph.node_to_idx)}")

# ─── BENCHMARK 8: Full Pipeline (hard — 200K tokens) ───
separator("BENCHMARK 8: Full Pipeline — Hard Corpus (200K tokens)")
d_model = 64
lsh2 = SeededLSH(d=d_model, w=10.0, m=4, seed=42)
tree2 = VirtualMemoryTree(page_size=1000, cache_size=500, persist_dir="/tmp/bench_hard")
graph2 = DeterministicKnowledgeGraph(d_model=d_model)

corpus2 = []
concept_defs = {
    "quantum_memory": "Quantum memory is a spectral manifold representation of knowledge that encodes information in the eigenvalues of a graph Laplacian. It enables deterministic retrieval through hierarchical beacon compression.",
    "spectral_manifold": "A spectral manifold is a low-dimensional Riemannian manifold embedded in semantic space where the graph Laplacian serves as the metric tensor.",
    "beacon_hierarchy": "The beacon hierarchy consists of three levels: B1 as tangent vectors, B2 as Gaussian patches, and B3 as spectral signatures achieving information-theoretic compression.",
    "fokker_planck": "Fokker-Planck dynamics govern temporal memory evolution combining diffusion, drift, source, and decay terms.",
    "submodular_packing": "Submodular packing selects the optimal subset of memory pages within a token budget using greedy approximation.",
    "seeded_lsh": "Seeded Locality-Sensitive Hashing partitions the manifold into deterministic Voronoi cells using fixed random projections.",
    "deterministic_retrieval": "Deterministic retrieval ensures identical queries with identical memory state produce identical results.",
    "temporal_memory": "Temporal memory strength evolves without manual timestamps through Fokker-Planck equation intrinsically encoding temporal dynamics.",
}

# Definitions at start
for concept, defn in concept_defs.items():
    corpus2.extend(defn.split())
    corpus2.extend([f"filler_{concept}_{i}" for i in range(50)])

# Padding
corpus2.extend([f"pad_{i}" for i in range(50000)])

# Cross-references in the middle
for concept in list(concept_defs.keys())[:4]:
    corpus2.extend(f"Recall that {concept.replace('_', ' ')} uses spectral decomposition for compression".split())
    corpus2.extend([f"crossref_{concept}_{i}" for i in range(30)])

# More padding
while len(corpus2) < 200000:
    corpus2.append(f"corpus_token_{len(corpus2)}")
corpus2 = corpus2[:200000]

with bench("ingest_stream(200K tokens)"):
    page_ids2 = list(tree2.ingest_stream(iter(corpus2)))

triple_count2 = 0
with bench("extract_triplets + build_graph"):
    for pid in page_ids2:
        page_text = tree2.get_page(pid)
        if page_text is None:
            continue
        triples = extractor.extract(page_text)
        for subject, predicate, obj in triples:
            b1_id = tree2.get_beacon_for_page(pid)
            graph2.add_triplet(subject, predicate, obj, weight=1.0, beacon_id=b1_id or "", page_id=pid)
            triple_count2 += 1

with bench("build_laplacian"):
    if len(graph2.node_to_idx) >= 2:
        graph2.build_laplacian()

gate2 = QueryEngine(tree=tree2, graph=graph2, lsh=lsh2, d_model=d_model)

latencies2 = []
with bench("100 queries (sequential)"):
    for i in range(100):
        q = f"quantum memory spectral manifold {i % 10}"
        result = gate2.collapse(query=q, max_tokens=4096)
        latencies2.append(result.latency_ms)

latencies2_sorted = sorted(latencies2)
p50_2 = latencies2_sorted[len(latencies2_sorted) // 2]
p95_2 = latencies2_sorted[int(len(latencies2_sorted) * 0.95)]
print(f"\n  Query latency: p50={p50_2:.2f}ms  p95={p95_2:.2f}ms  min={min(latencies2):.2f}ms  max={max(latencies2):.2f}ms")
print(f"  Pages ingested: {len(page_ids2)}  Triples: {triple_count2}  Concepts: {len(graph2.node_to_idx)}")

# ─── BENCHMARK 9: Determinism Stress Test ───
separator("BENCHMARK 9: Determinism Under Load")
gate_ref = QueryEngine(tree=tree2, graph=graph2, lsh=lsh2, d_model=d_model)
query = "quantum memory spectral manifold beacon"
ref_result = gate_ref.collapse(query=query, max_tokens=4096)
mismatches = 0
with bench("determinism x50 (same query)"):
    for i in range(50):
        r = gate_ref.collapse(query=query, max_tokens=4096)
        if r.pages != ref_result.pages:
            mismatches += 1
print(f"  Mismatches: {mismatches}/50  {'PASS' if mismatches == 0 else 'FAIL'}")

# ─── BENCHMARK 10: Concurrent Throughput ───
separator("BENCHMARK 10: Concurrent Query Throughput")
import threading

gate_conc = QueryEngine(tree=tree2, graph=graph2, lsh=lsh2, d_model=d_model)
results_lock = threading.Lock()
conc_results = []

def conc_worker(idx):
    r = gate_conc.collapse(query=f"test concept {idx % 5}", max_tokens=1024)
    with results_lock:
        conc_results.append(r.latency_ms)

n_threads = 20
t0 = time.perf_counter()
threads = [threading.Thread(target=conc_worker, args=(i,)) for i in range(n_threads)]
for t in threads:
    t.start()
for t in threads:
    t.join()
wall_ms = (time.perf_counter() - t0) * 1000
print(f"  {n_threads} concurrent queries in {wall_ms:.2f}ms total")
print(f"  Per-query avg: {np.mean(conc_results):.2f}ms  p95: {sorted(conc_results)[int(len(conc_results)*0.95)]:.2f}ms")

separator("ALL BENCHMARKS COMPLETE")
