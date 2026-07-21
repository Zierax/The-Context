#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from entity_extractor import HeuristicExtractor
ext = HeuristicExtractor()

texts = [
    "Quantum memory is a spectral manifold representation of knowledge that encodes information in the eigenvalues of a graph Laplacian. It enables deterministic retrieval.",
    "A spectral manifold is a low-dimensional Riemannian manifold embedded in semantic space. The graph Laplacian serves as the metric tensor.",
    "The beacon hierarchy consists of three levels: B1 as tangent vectors, B2 as Gaussian patches, and B3 as spectral signatures.",
    "Fokker-Planck dynamics govern temporal memory evolution on the graph. The equation combines diffusion, drift, source, and decay terms.",
]
for i, t in enumerate(texts):
    triples = ext.extract(t)
    print(f"Text {i}: {len(triples)} triples")
    for s, p, o in triples[:5]:
        print(f"  ({s}, {p}, {o})")
