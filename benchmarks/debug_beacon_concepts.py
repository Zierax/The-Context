#!/usr/bin/env python3
"""Debug: trace beacon_to_concepts and query activation."""
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

persist = tempfile.mkdtemp()
d = 128
tree = VirtualMemoryTree(page_size=1000, cache_size=10, persist_dir=persist)
graph = DeterministicKnowledgeGraph(d_model=d)
extractor = HeuristicExtractor()
lsh = SeededLSH(d=d, w=10.0, m=4, seed=42)

# Ingest real knowledge pages
corpus = [
    "Quantum memory is a spectral manifold representation of knowledge that encodes information in the eigenvalues of a graph Laplacian.",
    "It enables deterministic retrieval through hierarchical beacon compression.",
    "The beacon hierarchy consists of three levels: B1 as tangent vectors, B2 as Gaussian patches, and B3 as spectral signatures.",
    "A spectral manifold is a low-dimensional Riemannian manifold embedded in semantic space.",
    "The graph Laplacian serves as the metric tensor for the spectral manifold.",
]
# Pad each concept with padding
full_corpus = []
for fact in corpus:
    full_corpus.extend(fact.split())
    full_corpus.extend([f"pad_{i}" for i in range(20)])

# Pad to 5000 tokens
while len(full_corpus) < 5000:
    full_corpus.append(f"pad_{len(full_corpus)}")
full_corpus = full_corpus[:5000]

print(f"Corpus: {len(full_corpus)} tokens")
pids = list(tree.ingest_stream(iter(full_corpus)))
print(f"Pages: {len(pids)}")

# Extract knowledge
for pid in pids:
    pt = tree.get_page(pid)
    if pt is None:
        print(f"  {pid}: NULL (on disk)")
        continue
    b1 = tree.get_beacon_for_page(pid)
    triples = extractor.extract(pt)
    preview = pt[:80].replace('\n', ' ')
    print(f"  {pid} (b1={b1}): {len(triples)} triples | {preview}...")
    for s, p, o in triples[:3]:
        graph.add_triplet(s, p, o, beacon_id=b1 or "")

print(f"\nGraph concepts: {len(graph.node_to_idx)}")
print(f"beacon_to_concepts: {graph.beacon_to_concepts}")

# Build Laplacian
if len(graph.node_to_idx) >= 2:
    graph.build_laplacian()

# Query
gate = QueryEngine(tree=tree, graph=graph, lsh=lsh, d_model=d)
result = gate.collapse(query="What is quantum memory?", max_tokens=2048)
print(f"\nQuery result: {len(result.pages)} pages, {result.tokens_used} tokens")
if result.pages:
    print(f"  First page: {result.pages[0][:100]}...")
if result.error:
    print(f"  Error: {result.error}")

import shutil
shutil.rmtree(persist, ignore_errors=True)
