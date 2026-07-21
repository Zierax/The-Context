#!/usr/bin/env python3
"""Debug submodular_pack fallback."""
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
from math_engine import SeededLSH, sinusoidal_encode, estimate_token_count

# Build same corpus as topic_test
topics = {
    "quantum_memory": "Quantum memory is a spectral manifold representation of knowledge that encodes information in the eigenvalues of a graph Laplacian. It enables deterministic retrieval through hierarchical beacon compression.",
    "spectral_manifold": "A spectral manifold is a low-dimensional Riemannian manifold embedded in semantic space. The graph Laplacian serves as the metric tensor for the spectral manifold. Vector similarity is geodesic distance.",
    "beacon_hierarchy": "The beacon hierarchy consists of three levels: B1 as tangent vectors, B2 as Gaussian patches, and B3 as spectral signatures. B1 beacons are 1000-token chunks. B2 compresses 10 B1. B3 compresses 10 B2.",
    "fokker_planck": "Fokker-Planck dynamics govern temporal memory evolution on the graph. The equation combines diffusion, drift, source, and decay terms. This enables intrinsic temporal reasoning without timestamp metadata.",
}

corpus = []
for topic, text in topics.items():
    words = text.split()
    corpus.extend(words)
    corpus.extend([f"pad_{topic}_{i}" for i in range(1000 - len(words))])
while len(corpus) < 5000:
    corpus.append(f"pad_{len(corpus)}")
corpus = corpus[:5000]

d = 128
persist = tempfile.mkdtemp()
lsh = SeededLSH(d=d, w=10.0, m=4, seed=42)
tree = VirtualMemoryTree(page_size=1000, cache_size=10, persist_dir=persist)
graph = DeterministicKnowledgeGraph(d_model=d)
extractor = HeuristicExtractor()

pids = list(tree.ingest_stream(iter(corpus)))
for pid in pids:
    pt = tree.get_page(pid)
    if pt is None:
        print(f"{pid}: EVICTED")
        continue
    b1 = tree.get_beacon_for_page(pid)
    for s, p, o in extractor.extract(pt):
        graph.add_triplet(s, p, o, beacon_id=b1 or "")

if len(graph.node_to_idx) >= 2:
    graph.build_laplacian()

gate = QueryEngine(tree=tree, graph=graph, lsh=lsh, d_model=d)
gate._ensure_caches()

# Simulate what collapse does
from math_engine import sinusoidal_encode
query = "What is quantum memory?"
query_tokens = query.lower().split()
query_embedding = sinusoidal_encode(query_tokens, d_model=d)
q = np.mean(query_embedding, axis=0).astype(np.float64)
diffused_rho = graph.concept_diffusion(list(graph.node_to_idx.keys())[:5], steps=3)

beacon_to_concepts = graph.beacon_to_concepts
print(f"beacon_to_concepts: {beacon_to_concepts}")
print(f"diffused_rho sum: {np.sum(diffused_rho):.6f}")

# Build candidates the same way collapse does
all_pages_set = set()
for b1_id in list(tree.beacon_b1.keys()):
    for page_id in tree.b1_to_pages.get(b1_id, []):
        if page_id in all_pages_set:
            continue
        all_pages_set.add(page_id)
        text = tree.get_page(page_id)
        if text is None:
            print(f"  {page_id}: EVICTED (can't load)")
            continue
        concept_coverage = {}
        for concept in beacon_to_concepts.get(b1_id, []):
            if concept in graph.node_to_idx:
                idx = graph.node_to_idx[concept]
                concept_coverage[concept] = float(diffused_rho[idx])
        token_count = estimate_token_count(text)
        density = sum(concept_coverage.values()) / max(token_count, 1)
        print(f"  {page_id}: concepts={list(concept_coverage.keys())} tokens={token_count} density={density:.6f} preview={text[:50]}")

import shutil
shutil.rmtree(persist, ignore_errors=True)
