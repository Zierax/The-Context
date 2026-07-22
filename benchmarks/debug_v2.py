import sys
sys.path.insert(0, '.')

from memory_manager import VirtualMemoryTree
from knowledge_graph import DeterministicKnowledgeGraph
from entity_extractor import HeuristicExtractor
from entity_extractor_v2 import EmbeddingEntityExtractor
from embedding_engine import HashEmbeddingProvider
from query_engine import QueryEngine
from query_engine_v2 import QueryEngineV2
from math_engine import SeededLSH

import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(0))

corpus = "The capital of France is Paris. Paris is a beautiful city. The Eiffel Tower is in Paris. The Louvre museum is in Paris. " * 50
corpus_tokens = corpus.split()
print(f"Corpus tokens: {len(corpus_tokens)}")

provider = HashEmbeddingProvider(dimension=384)

tree = VirtualMemoryTree(cache_size=200, page_size=100, persist_dir="/tmp/debug_v2")
pids = list(tree.ingest_stream(iter(corpus_tokens)))
print(f"Tree pages created: {len(pids)}")
print(f"Tree pages: {list(tree.pages)}")

graph = DeterministicKnowledgeGraph(d_model=128)
extractor = HeuristicExtractor()

for page_id in list(tree.pages):
    text = tree.get_page(page_id)
    if text:
        triples = extractor.extract(text)
        print(f"  Triples from {page_id}: {len(triples)}")
        for t in triples[:2]:
            print(f"    {t}")
        for subj, pred, obj in triples:
            graph.add_triplet(subj, pred, obj, page_id=page_id, beacon_id=f"b1_{page_id}")

print(f"Graph nodes: {len(graph.node_to_idx)}")
if len(graph.node_to_idx) >= 2:
    graph.build_laplacian()
    print("Laplacian built")

lsh = SeededLSH(d=128, m=4, seed=42)
engine_v1 = QueryEngine(tree=tree, graph=graph, lsh=lsh, d_model=128)

result_v1 = engine_v1.collapse(query="What is the capital of France?", max_tokens=4096)
print(f"\nV1 Result: {len(result_v1.pages)} pages")
print(f"  Error: {result_v1.error}")
print(f"  Concepts: {result_v1.concepts_activated}")

engine_v2 = QueryEngineV2(tree=tree, graph=graph, provider=provider, d_model=128)

result_v2 = engine_v2.collapse(query="What is the capital of France?", max_tokens=4096)
print(f"\nV2 Result: {len(result_v2.pages)} pages")
print(f"  Error: {result_v2.error}")
print(f"  Concepts: {result_v2.concepts_activated}")
