#!/usr/bin/env python3
"""Debug: Why are all triples on same page?"""
import sys, os, tempfile, warnings
warnings.filterwarnings('ignore')
import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(40))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory_manager import VirtualMemoryTree
from entity_extractor import HeuristicExtractor

print("=" * 70)
print("  DEBUG: Triple Extraction Per Page")
print("=" * 70)

# Simple corpus: 3 pages with different content
corpus = []
# Page 1: 1000 tokens about quantum memory
corpus.extend("Quantum memory is a spectral manifold representation of knowledge that encodes information in the eigenvalues of a graph Laplacian. It enables deterministic retrieval through hierarchical beacon compression.".split())
corpus.extend([f"filler_1_{i}" for i in range(200)])
# Page 2: 1000 tokens about spectral manifold
corpus.extend("A spectral manifold is a low-dimensional Riemannian manifold embedded in semantic space where the graph Laplacian serves as the metric tensor. Vector similarity is geodesic distance.".split())
corpus.extend([f"filler_2_{i}" for i in range(200)])
# Page 3: 1000 tokens about beacon hierarchy
corpus.extend("The beacon hierarchy consists of three levels: B1 as tangent vectors, B2 as Gaussian patches, and B3 as spectral signatures achieving information-theoretic compression.".split())
corpus.extend([f"filler_3_{i}" for i in range(200)])

# Pad to 5000 tokens
while len(corpus) < 5000:
    corpus.append(f"pad_{len(corpus)}")
corpus = corpus[:5000]

print(f"\n  Corpus: {len(corpus)} tokens")

# Ingest
persist = tempfile.mkdtemp()
tree = VirtualMemoryTree(page_size=1000, cache_size=10, persist_dir=persist)
pids = list(tree.ingest_stream(iter(corpus)))
print(f"  Pages: {len(pids)}")

# Extract triples from each page
extractor = HeuristicExtractor()
for i, pid in enumerate(pids):
    pt = tree.get_page(pid)
    if pt is None:
        print(f"\n  Page {i+1} ({pid}): NOT FOUND (evicted)")
        continue
    
    triples = extractor.extract(pt)
    print(f"\n  Page {i+1} ({pid}):")
    print(f"    Text preview: {pt[:150]}...")
    print(f"    Triples: {len(triples)}")
    for s, p, o in triples[:5]:
        print(f"      ({s}, {p}, {o})")
    if len(triples) > 5:
        print(f"      ... and {len(triples)-5} more")

import shutil
shutil.rmtree(persist, ignore_errors=True)

print("\n" + "=" * 70)
print("  DEBUG COMPLETE")
print("=" * 70)
