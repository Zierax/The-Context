#!/usr/bin/env python3
"""Debug query pipeline: trace concept → beacon → page mapping."""
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
from the_context.core import SeededLSH, sinusoidal_encode

print("=" * 70)
print("  DEBUG: Query Pipeline Trace")
print("=" * 70)

# Generate small corpus with known structure
corpus = []
# Page 1: Definition of quantum memory
corpus.extend("Quantum memory is a spectral manifold representation of knowledge that encodes information in the eigenvalues of a graph Laplacian.".split())
corpus.extend([f"filler_1_{i}" for i in range(50)])
# Page 2: Definition of spectral manifold  
corpus.extend("A spectral manifold is a low-dimensional Riemannian manifold embedded in semantic space.".split())
corpus.extend([f"filler_2_{i}" for i in range(50)])
# Page 3: Cross-reference
corpus.extend("Recall that quantum memory uses spectral decomposition of the graph Laplacian for compression and retrieval.".split())
corpus.extend([f"filler_3_{i}" for i in range(50)])
# Pad
while len(corpus) < 10000:
    corpus.append(f"pad_{len(corpus)}")
corpus = corpus[:10000]

print(f"\n  Corpus: {len(corpus):,} tokens")
print(f"  Pages with definitions: quantum_memory, spectral_manifold")
print(f"  Page with cross-reference: quantum_memory reference")

# Setup
d = 64
persist = tempfile.mkdtemp()
lsh = SeededLSH(d=d, w=10.0, m=4, seed=42)
tree = VirtualMemoryTree(page_size=1000, cache_size=10, persist_dir=persist)
graph = DeterministicKnowledgeGraph(d_model=d)
extractor = HeuristicExtractor()

# Ingest
pids = list(tree.ingest_stream(iter(corpus)))
print(f"\n  Ingested: {len(pids)} pages")

# Extract triples
triples = 0
for pid in pids:
    pt = tree.get_page(pid)
    if pt is None: continue
    for s, p, o in extractor.extract(pt):
        b1 = tree.get_beacon_for_page(pid)
        graph.add_triplet(s, p, o, beacon_id=b1 or "")
        triples += 1
print(f"  Triples: {triples}")
print(f"  Concepts: {list(graph.node_to_idx.keys())}")

# Build Laplacian
if len(graph.node_to_idx) >= 2:
    graph.build_laplacian()

# Debug: Check beacon → page mapping
print(f"\n  --- Beacon → Page Mapping ---")
for b1_id, page_ids in tree.b1_to_pages.items():
    print(f"  {b1_id}: {len(page_ids)} pages")

# Debug: Check concept → beacon mapping
print(f"\n  --- Concept → Beacon Mapping ---")
for concept, beacons in graph.beacon_map.items():
    print(f"  {concept}: {beacons}")

# Debug: Check beacon → concepts mapping
print(f"\n  --- Beacon → Concepts Mapping ---")
for beacon_id, concepts in graph.beacon_to_concepts.items():
    print(f"  {beacon_id}: {concepts}")

# Query
gate = QueryEngine(tree=tree, graph=graph, lsh=lsh, d_model=d)

query = "What is quantum memory?"
print(f"\n  Query: {query}")

# Step 1: Embed query
query_tokens = query.lower().split()
query_embedding = sinusoidal_encode(query_tokens, d_model=d)
q = np.mean(query_embedding, axis=0).astype(np.float64)
print(f"  Query embedding shape: {query_embedding.shape}")

# Step 2: Hash to LSH bucket
bucket = lsh.hash_vector(q)
print(f"  LSH bucket: {bucket}")

# Step 3: Find concepts in same bucket
gate._ensure_caches()
print(f"  Cached concepts: {len(gate._cached_concepts)}")
print(f"  Cached buckets: {len(gate._cached_buckets)}")

# Check which concepts are in the same bucket
same_bucket = []
for i, cb in enumerate(gate._cached_buckets):
    if cb == bucket:
        same_bucket.append(gate._cached_concepts[i])
print(f"  Concepts in same bucket: {same_bucket}")

# Step 4: Compute proximity
candidate_indices = []
for i, cb in enumerate(gate._cached_buckets):
    if cb == bucket:
        candidate_indices.append(i)

if not candidate_indices:
    print("  No concepts in bucket! Using fallback...")
    # Fallback: find nearest concepts
    concept_embeddings = gate._cached_embeddings
    diffs_all = concept_embeddings - q[np.newaxis, :]
    sq_dists_all = np.sum(diffs_all ** 2, axis=1)
    nearest_dist = np.min(sq_dists_all)
    threshold = nearest_dist + 2.0 * max(np.std(sq_dists_all), 1.0)
    candidate_indices = list(np.where(sq_dists_all <= threshold)[0])
    print(f"  Fallback candidates: {[gate._cached_concepts[i] for i in candidate_indices]}")

# Step 5: Run full collapse
result = gate.collapse(query=query, max_tokens=2048)
print(f"\n  --- Collapse Result ---")
print(f"  Pages: {len(result.pages)}")
print(f"  Tokens used: {result.tokens_used}")
print(f"  Compression: {result.compression_ratio}")
print(f"  Confidence: {result.confidence_score}")
print(f"  Concepts activated: {result.concepts_activated}")
if result.pages:
    print(f"  First page preview: {result.pages[0][:200]}...")

import shutil
shutil.rmtree(persist, ignore_errors=True)

print("\n" + "=" * 70)
print("  DEBUG COMPLETE")
print("=" * 70)
