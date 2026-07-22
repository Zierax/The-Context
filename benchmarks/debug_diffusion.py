#!/usr/bin/env python3
import sys, os, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from the_context.core import DeterministicKnowledgeGraph
from the_context.extraction import HeuristicExtractor

graph = DeterministicKnowledgeGraph(d_model=128)
ext = HeuristicExtractor()

texts = [
    ("quantum_memory", "Quantum memory is a spectral manifold representation of knowledge that encodes information in the eigenvalues of a graph Laplacian. It enables deterministic retrieval through hierarchical beacon compression."),
    ("spectral_manifold", "A spectral manifold is a low-dimensional Riemannian manifold embedded in semantic space. The graph Laplacian serves as the metric tensor for the spectral manifold. Vector similarity is geodesic distance."),
    ("beacon_hierarchy", "The beacon hierarchy consists of three levels: B1 as tangent vectors, B2 as Gaussian patches, and B3 as spectral signatures. B1 beacons are 1000-token chunks. B2 compresses 10 B1. B3 compresses 10 B2."),
    ("fokker_planck", "Fokker-Planck dynamics govern temporal memory evolution on the graph. The equation combines diffusion, drift, source, and decay terms. This enables intrinsic temporal reasoning without timestamp metadata."),
]

for topic, text in texts:
    for s, p, o in ext.extract(text):
        graph.add_triplet(s, p, o, beacon_id="b1_test")

graph.build_laplacian()
concepts = list(graph.node_to_idx.keys())
print(f"All {len(concepts)} concepts:")
for c in concepts:
    print(f"  {c}")

for query_concepts in [["quantum memory"], ["spectral manifold"], ["beacon hierarchy"], ["fokker"]]:
    rho = graph.concept_diffusion(query_concepts, steps=3)
    top_idx = np.argsort(rho)[-5:]
    print(f"\nQuery {query_concepts}: top = {[concepts[i] for i in top_idx]}")
    print(f"  rho: {[f'{rho[i]:.6f}' for i in top_idx]}")
