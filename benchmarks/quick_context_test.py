#!/usr/bin/env python3
"""Quick test of context engineering pipeline."""
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
print("  QUICK CONTEXT ENGINEERING TEST")
print("=" * 70)

# Step 1: Generate small corpus with REAL knowledge
print("\n  [1/6] Generating corpus...")

corpus_pages = {
    "quantum_memory": [
        "Quantum memory is a spectral manifold representation of knowledge that encodes information in the eigenvalues of a graph Laplacian.",
        "It enables deterministic retrieval through hierarchical beacon compression.",
        "The beacon hierarchy consists of three levels: B1 as tangent vectors, B2 as Gaussian patches, and B3 as spectral signatures.",
    ],
    "spectral_manifold": [
        "A spectral manifold is a low-dimensional Riemannian manifold embedded in semantic space.",
        "The graph Laplacian serves as the metric tensor for the spectral manifold.",
        "Vector similarity is geodesic distance on the spectral manifold.",
    ],
    "beacon_hierarchy": [
        "The beacon hierarchy consists of three levels: B1, B2, and B3.",
        "B1 beacons are 1000-token chunks as tangent vectors.",
        "B2 beacons compress 10 B1 beacons into Gaussian patches.",
        "B3 beacons compress 10 B2 beacons into spectral signatures.",
    ],
    "fokker_planck": [
        "Fokker-Planck dynamics govern temporal memory evolution on the graph.",
        "The equation combines diffusion, drift, source, and decay terms.",
        "This enables intrinsic temporal reasoning without timestamp metadata.",
    ],
}

# Build corpus with interleaved content
corpus = []
for concept, facts in corpus_pages.items():
    for fact in facts:
        corpus.extend(fact.split())
        corpus.extend([f"filler_{concept}_{i}" for i in range(10)])

# Pad to 5K tokens
while len(corpus) < 5_000:
    corpus.append(f"pad_{len(corpus)}")
corpus = corpus[:5_000]

print(f"    Corpus: {len(corpus):,} tokens")
print(f"    Concepts: {list(corpus_pages.keys())}")

# Step 2: Ingest
print("\n  [2/6] Ingesting into memory tree...")
d = 128
persist = tempfile.mkdtemp()
lsh = SeededLSH(d=d, w=10.0, m=4, seed=42)
tree = VirtualMemoryTree(page_size=1000, cache_size=10, persist_dir=persist)
graph = DeterministicKnowledgeGraph(d_model=d)
extractor = HeuristicExtractor()

t0 = time.perf_counter()
pids = list(tree.ingest_stream(iter(corpus)))
ingest_ms = (time.perf_counter() - t0) * 1000
print(f"    Created {len(pids):,} pages in {ingest_ms:.0f}ms")

# Step 3: Extract knowledge
print("\n  [3/6] Extracting knowledge from pages...")
t0 = time.perf_counter()
triples = 0
for pid in pids:
    pt = tree.get_page(pid)
    if pt is None:
        print(f"    {pid}: NULL (evicted)")
        continue
    b1 = tree.get_beacon_for_page(pid)
    extracted = extractor.extract(pt)
    for s, p, o in extracted:
        graph.add_triplet(s, p, o, beacon_id=b1 or "")
        triples += 1
    if extracted:
        print(f"    {pid} (b1={b1}): {len(extracted)} triples | {pt[:60]}...")
graph_ms = (time.perf_counter() - t0) * 1000
print(f"    Extracted {triples:,} triples in {graph_ms:.0f}ms")
print(f"    Graph has {len(graph.node_to_idx):,} unique concepts")
print(f"    beacon_to_concepts keys: {list(graph.beacon_to_concepts.keys())}")

# Step 4: Build knowledge graph
print("\n  [4/6] Building knowledge graph...")
if len(graph.node_to_idx) >= 2:
    graph.build_laplacian()
    print(f"    Laplacian built ({graph.A.nnz:,} edges)")
else:
    print("    WARNING: Too few nodes for Laplacian")

# Step 5: Query with knowledge
print("\n  [5/6] Querying with knowledge graph...")
gate = QueryEngine(tree=tree, graph=graph, lsh=lsh, d_model=d)

queries = [
    "What is quantum memory?",
    "How does the beacon hierarchy work?",
    "Explain Fokker-Planck dynamics",
    "What is a spectral manifold?",
]

results = []
for query in queries:
    t0 = time.perf_counter()
    result = gate.collapse(query=query, max_tokens=2048)
    ms = (time.perf_counter() - t0) * 1000
    
    # Debug: show all returned pages
    for i, page in enumerate(result.pages):
        preview = page[:80].replace('\n', ' ')
        print(f"    [{i}] {preview}...")
    
    # Check if result contains relevant content
    has_relevant = False
    if result.pages:
        for page in result.pages:
            if any(kw in page.lower() for kw in ["quantum", "spectral", "beacon", "fokker"]):
                has_relevant = True
                break
    
    results.append({
        'query': query,
        'pages': len(result.pages),
        'tokens_used': result.tokens_used,
        'compression': result.compression_ratio,
        'confidence': result.confidence_score,
        'latency': ms,
        'has_relevant': has_relevant,
        'preview': result.pages[0][:100] if result.pages else "N/A",
    })
    
    print(f"\n    Query: {query}")
    print(f"    Pages: {len(result.pages)} | Tokens: {result.tokens_used:,} | Compression: {result.compression_ratio:.1f}x")
    print(f"    Confidence: {result.confidence_score:.4f} | Latency: {ms:.1f}ms")
    print(f"    Relevant: {'YES' if has_relevant else 'NO'}")
    if result.pages:
        print(f"    Preview: {result.pages[0][:200]}...")

# Step 6: Measure compression
print("\n  [6/6] Measuring compression...")
mem = sum(len(t.encode()) for t in tree.pages.values())
mem += len(tree.beacon_b1) * d * 2
mem += len(tree.beacon_b2) * (d*2 + d*8)
mem += len(tree.beacon_b3) * (5*8 + 10*5*8)
if graph.A is not None:
    mem += graph.A.data.nbytes + graph.A.indices.nbytes + graph.A.indptr.nbytes
if graph.L_sym is not None:
    mem += graph.L_sym.data.nbytes + graph.L_sym.indices.nbytes + graph.L_sym.indptr.nbytes
mem += lsh.a.nbytes + lsh.b.nbytes

text_bytes = sum(len(t.encode()) for t in corpus)
ratio = text_bytes / mem if mem > 0 else 0

print(f"\n  --- SUMMARY ---")
print(f"  Corpus: {len(corpus):,} tokens")
print(f"  Memory: {mem/1024:.2f} KB ({mem/len(corpus):.4f} bytes/token)")
print(f"  Compression: {ratio:.1f}x (text -> memory)")
print(f"  Knowledge: {len(graph.node_to_idx):,} concepts, {triples:,} triples")
print(f"\n  --- QUERY RESULTS ---")
relevant_count = sum(1 for r in results if r['has_relevant'])
print(f"  Queries with relevant content: {relevant_count}/{len(results)}")
print(f"  Average confidence: {np.mean([r['confidence'] for r in results]):.4f}")
print(f"  Average latency: {np.mean([r['latency'] for r in results]):.1f}ms")

import shutil
shutil.rmtree(persist, ignore_errors=True)

print("\n" + "=" * 70)
print("  CONTEXT ENGINEERING COMPLETE")
print("=" * 70)
