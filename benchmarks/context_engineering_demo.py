#!/usr/bin/env python3
"""Context Engineering Demo: 100K tokens → query retrieval."""
import sys, os, tempfile, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(40))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory_manager import VirtualMemoryTree
from query_engine import QueryEngine
from knowledge_graph import DeterministicKnowledgeGraph
from entity_extractor import HeuristicExtractor
from math_engine import SeededLSH

print("=" * 70)
print("  CONTEXT ENGINEERING DEMO: 100K tokens → query retrieval")
print("=" * 70)

# Step 1: Generate corpus with real text
print("\n  [1/5] Generating corpus with real text patterns...")

def generate_realistic_corpus(n_tokens):
    """Generate corpus with real English sentences (not token_12345)."""
    corpus = []
    concepts = {
        "quantum memory": "Quantum memory is a spectral manifold representation of knowledge that encodes information in the eigenvalues of a graph Laplacian. It enables deterministic retrieval through hierarchical beacon compression.",
        "spectral manifold": "A spectral manifold is a low-dimensional Riemannian manifold embedded in semantic space where the graph Laplacian serves as the metric tensor.",
        "beacon hierarchy": "The beacon hierarchy consists of three levels: B1 as tangent vectors, B2 as Gaussian patches, and B3 as spectral signatures achieving information-theoretic compression.",
        "Fokker-Planck dynamics": "Fokker-Planck dynamics govern temporal memory evolution combining diffusion, drift, source, and decay terms.",
        "submodular packing": "Submodular packing selects the optimal subset of memory pages within a token budget using greedy approximation.",
        "seeded LSH": "Seeded Locality-Sensitive Hashing partitions the manifold into deterministic Voronoi cells using fixed random projections.",
    }
    
    cross_refs = [
        "Recall that quantum memory uses spectral decomposition of the graph Laplacian for compression and retrieval.",
        "As previously defined, the spectral manifold unifies vectors, graphs, temporal dynamics, and compression.",
        "The beacon hierarchy mentioned earlier enables the 100x+ token efficiency improvement over naive retrieval.",
        "The Fokker-Planck equation described above gives intrinsic temporal reasoning without timestamp metadata.",
    ]
    
    idx = 0
    for concept, definition in concepts.items():
        # Add definition
        corpus.extend(definition.split())
        # Add filler
        corpus.extend([f"filler_{idx}_{i}" for i in range(100)])
        idx += 1
    
    # Add cross-references
    for ref in cross_refs:
        corpus.extend(ref.split())
        corpus.extend([f"padding_{idx}_{i}" for i in range(50)])
        idx += 1
    
    # Pad to target
    while len(corpus) < n_tokens:
        corpus.append(f"padding_token_{len(corpus)}")
    
    return corpus[:n_tokens]

corpus = generate_realistic_corpus(100_000)
print(f"    Generated {len(corpus):,} tokens")

# Step 2: Ingest
print("\n  [2/5] Ingesting into memory tree...")
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

# Step 3: Extract triples
print("\n  [3/5] Extracting entities and building knowledge graph...")
t0 = time.perf_counter()
triples = 0
for pid in pids:
    pt = tree.get_page(pid)
    if pt is None: continue
    for s, p, o in extractor.extract(pt):
        b1 = tree.get_beacon_for_page(pid)
        graph.add_triplet(s, p, o, beacon_id=b1 or "")
        triples += 1
graph_ms = (time.perf_counter() - t0) * 1000
print(f"    Extracted {triples:,} triples in {graph_ms:.0f}ms")
print(f"    Graph has {len(graph.node_to_idx):,} unique concepts")

# Step 4: Build Laplacian
print("\n  [4/5] Building graph Laplacian...")
if len(graph.node_to_idx) >= 2:
    graph.build_laplacian()
    print(f"    Laplacian built ({graph.A.nnz:,} edges)")
else:
    print("    WARNING: Too few nodes for Laplacian")

# Step 5: Query
print("\n  [5/5] Executing queries...")
gate = QueryEngine(tree=tree, graph=graph, lsh=lsh, d_model=d)

queries = [
    "What is quantum memory?",
    "How does spectral manifold work?",
    "Explain the beacon hierarchy",
    "What is Fokker-Planck dynamics?",
    "How does seeded LSH work?",
]

for query in queries:
    t0 = time.perf_counter()
    result = gate.collapse(query=query, max_tokens=2048)
    ms = (time.perf_counter() - t0) * 1000
    
    print(f"\n    Query: {query}")
    print(f"    Pages returned: {len(result.pages)}")
    print(f"    Tokens used: {result.tokens_used:,} / {result.tokens_total:,}")
    print(f"    Compression: {result.compression_ratio:.1f}x")
    print(f"    Confidence: {result.confidence_score:.4f}")
    print(f"    Latency: {ms:.1f}ms")
    if result.pages:
        preview = result.pages[0][:100] + "..." if len(result.pages[0]) > 100 else result.pages[0]
        print(f"    Preview: {preview}")

# Memory
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
print(f"  Corpus: 100,000 tokens")
print(f"  Memory: {mem/1024/1024:.2f} MB ({mem/100_000:.4f} bytes/token)")
print(f"  Compression: {ratio:.1f}x (text → memory)")
print(f"  Concepts: {len(graph.node_to_idx):,}")
print(f"  Triples: {triples:,}")

import shutil
shutil.rmtree(persist, ignore_errors=True)

print("\n" + "=" * 70)
print("  CONTEXT ENGINEERING DEMO COMPLETE")
print("=" * 70)
