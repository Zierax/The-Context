#!/usr/bin/env python3
"""Test: verify different queries return different relevant pages."""
import sys, os, tempfile, warnings
import numpy as np
warnings.filterwarnings('ignore')
import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(40))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory_manager import VirtualMemoryTree
from query_engine import QueryEngine
from knowledge_graph import DeterministicKnowledgeGraph
from entity_extractor import HeuristicExtractor
from math_engine import SeededLSH, sinusoidal_encode

print("=" * 70)
print("  CONTEXT ENGINEERING: Topic Differentiation Test")
print("=" * 70)

# Each topic gets a FULL 1000-token page (padded to page_size)
topics = {
    "quantum_memory": "Quantum memory is a spectral manifold representation of knowledge that encodes information in the eigenvalues of a graph Laplacian. It enables deterministic retrieval through hierarchical beacon compression.",
    "spectral_manifold": "A spectral manifold is a low-dimensional Riemannian manifold embedded in semantic space. The graph Laplacian serves as the metric tensor for the spectral manifold. Vector similarity is geodesic distance.",
    "beacon_hierarchy": "The beacon hierarchy consists of three levels: B1 as tangent vectors, B2 as Gaussian patches, and B3 as spectral signatures. B1 beacons are 1000-token chunks. B2 compresses 10 B1. B3 compresses 10 B2.",
    "fokker_planck": "Fokker-Planck dynamics govern temporal memory evolution on the graph. The equation combines diffusion, drift, source, and decay terms. This enables intrinsic temporal reasoning without timestamp metadata.",
}

# Build corpus: each topic gets its own 1000-token page
corpus = []
for topic, text in topics.items():
    words = text.split()
    corpus.extend(words)
    corpus.extend([f"pad_{topic}_{i}" for i in range(1000 - len(words))])

# Pad to 5000
while len(corpus) < 5000:
    corpus.append(f"pad_{len(corpus)}")
corpus = corpus[:5000]

print(f"\n  Corpus: {len(corpus):,} tokens")
print(f"  Topics: {list(topics.keys())}")

# Ingest
d = 128
persist = tempfile.mkdtemp()
lsh = SeededLSH(d=d, w=10.0, m=4, seed=42)
tree = VirtualMemoryTree(page_size=1000, cache_size=10, persist_dir=persist)
graph = DeterministicKnowledgeGraph(d_model=d)
extractor = HeuristicExtractor()

pids = list(tree.ingest_stream(iter(corpus)))
print(f"  Pages: {len(pids)}")

# Extract knowledge
for pid in pids:
    pt = tree.get_page(pid)
    if pt is None:
        print(f"    {pid}: NULL (evicted)")
        continue
    b1 = tree.get_beacon_for_page(pid)
    triples = extractor.extract(pt)
    for s, p, o in triples:
        graph.add_triplet(s, p, o, beacon_id=b1 or "")
    preview = pt[:60].replace('\n', ' ')
    print(f"    {pid} (b1={b1}): {len(triples)} triples | {preview}...")

print(f"  Graph: {len(graph.node_to_idx)} concepts")
print(f"  beacon_to_concepts: {list(graph.beacon_to_concepts.keys())}")

if len(graph.node_to_idx) >= 2:
    graph.build_laplacian()

# Query
gate = QueryEngine(tree=tree, graph=graph, lsh=lsh, d_model=d)
queries = [
    ("What is quantum memory?", "quantum"),
    ("How does the beacon hierarchy work?", "beacon"),
    ("Explain Fokker-Planck dynamics", "fokker"),
    ("What is a spectral manifold?", "spectral"),
]

print(f"\n  --- QUERY RESULTS ---")
for query, keyword in queries:
    result = gate.collapse(query=query, max_tokens=4096)
    if result.pages:
        preview = result.pages[0][:120].replace('\n', ' ')
        has_kw = keyword in preview.lower()
        print(f"    Q: {query}")
        print(f"    Pages: {len(result.pages)} | Relevant: {'YES' if has_kw else 'NO'} | Preview: {preview}...")
    else:
        print(f"    Q: {query}")
        print(f"    Pages: 0 | Error: {result.error}")

import shutil
shutil.rmtree(persist, ignore_errors=True)

print("\n" + "=" * 70)
print("  TOPIC DIFFERENTIATION TEST COMPLETE")
print("=" * 70)
