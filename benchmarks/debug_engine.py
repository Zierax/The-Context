import sys
sys.path.insert(0, '.')

from memory_manager import VirtualMemoryTree
from knowledge_graph import DeterministicKnowledgeGraph
from entity_extractor import HeuristicExtractor
from entity_extractor_v2 import EmbeddingEntityExtractor
from embedding_engine import HashEmbeddingProvider

import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(0))

corpus = "The capital of France is Paris. Paris is a beautiful city. The Eiffel Tower is in Paris."
provider = HashEmbeddingProvider(dimension=384)

tree = VirtualMemoryTree(cache_size=200, page_size=1000, persist_dir="/tmp/debug_test")
tree.ingest_stream(iter([corpus]))

print(f"Tree pages: {list(tree.pages)}")
print(f"Tree pages count: {len(list(tree.pages))}")

for page_id in tree.pages:
    text = tree.get_page(page_id)
    print(f"  Page {page_id}: {len(text) if text else 0} chars")
    if text:
        print(f"    Preview: {text[:100]}...")

graph = DeterministicKnowledgeGraph(d_model=128)
extractor = HeuristicExtractor()

for page_id in tree.pages:
    text = tree.get_page(page_id)
    if text:
        triples = extractor.extract(text)
        print(f"  Triples from {page_id}: {len(triples)}")
        for t in triples[:3]:
            print(f"    {t}")
        for subj, pred, obj in triples:
            graph.add_triplet(subj, pred, obj, page_id=page_id, beacon_id=f"b1_{page_id}")

print(f"Graph nodes: {len(graph.node_to_idx)}")
if len(graph.node_to_idx) >= 2:
    graph.build_laplacian()
    print("Laplacian built")

from query_engine import QueryEngine
from math_engine import SeededLSH

lsh = SeededLSH(d=128, m=4, seed=42)
engine = QueryEngine(tree=tree, graph=graph, lsh=lsh, d_model=128)

result = engine.collapse(query="What is the capital of France?", max_tokens=4096)
print(f"V1 Result: {len(result.pages)} pages, error={result.error}")
if result.pages:
    print(f"  First page: {result.pages[0][:100]}...")

from query_engine_v2 import QueryEngineV2

engine_v2 = QueryEngineV2(tree=tree, graph=graph, provider=provider, d_model=128)
result_v2 = engine_v2.collapse(query="What is the capital of France?", max_tokens=4096)
print(f"V2 Result: {len(result_v2.pages)} pages, error={result_v2.error}")
if result_v2.pages:
    print(f"  First page: {result_v2.pages[0][:100]}...")
