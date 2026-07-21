#!/usr/bin/env python3
import sys, os, tempfile, warnings, numpy as np
warnings.filterwarnings('ignore')
import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(40))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory_manager import VirtualMemoryTree
from query_engine import QueryEngine
from knowledge_graph import DeterministicKnowledgeGraph
from entity_extractor import HeuristicExtractor
from math_engine import SeededLSH, sinusoidal_encode

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
    if pt is None: continue
    b1 = tree.get_beacon_for_page(pid)
    for s, p, o in extractor.extract(pt):
        graph.add_triplet(s, p, o, beacon_id=b1 or "")
graph.build_laplacian()

concepts = list(graph.node_to_idx.keys())
beacon_to_concepts = graph.beacon_to_concepts

# For each query, show what gets activated and the per-page coverage
for query in ["What is quantum memory?", "How does the beacon hierarchy work?", "Explain Fokker-Planck dynamics"]:
    query_tokens = query.lower().split()
    activated = set()
    for concept_name in concepts:
        concept_tokens = set(concept_name.lower().split())
        overlap = set(query_tokens) & concept_tokens
        if len(overlap) >= 2:
            activated.add(concept_name)
        elif any(w in concept_name.lower() for w in query_tokens if len(w) >= 4):
            activated.add(concept_name)

    rho = graph.concept_diffusion(list(activated), steps=3)

    print(f"\nQuery: {query}")
    print(f"  Activated concepts: {activated}")
    print(f"  Top rho: {[(concepts[i], f'{rho[i]:.4f}') for i in np.argsort(rho)[-8:]]}")

    # Per-page coverage
    for pid in pids:
        pt = tree.get_page(pid)
        if pt is None: continue
        b1 = tree.get_beacon_for_page(pid)
        concepts_for_page = beacon_to_concepts.get(b1, [])
        total_rho = sum(float(rho[graph.node_to_idx[c]]) for c in concepts_for_page if c in graph.node_to_idx)
        n_concepts = len(concepts_for_page)
        preview = pt[:40]
        print(f"  {pid}: {n_concepts} concepts, total_rho={total_rho:.4f} | {preview}...")

import shutil
shutil.rmtree(persist, ignore_errors=True)
