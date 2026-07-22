import sys, os
sys.path.insert(0, '.')
os.environ['PYTHONPATH'] = '.'

from query_engine import QueryEngine, CollapseRequest
from memory_manager import VirtualMemoryTree
from knowledge_graph import DeterministicKnowledgeGraph
from entity_extractor import HeuristicExtractor

# Build adversarial corpus
import random
random.seed(42)

countries = [
    ("France", "Paris", "Germany", "Berlin"),
    ("Germany", "Berlin", "Spain", "Madrid"),
    ("Spain", "Madrid", "Italy", "Rome"),
    ("Italy", "Rome", "Japan", "Tokyo"),
    ("Japan", "Tokyo", "Brazil", "Brasilia"),
]

corpus_parts = []
for i, (country1, capital1, country2, capital2) in enumerate(countries):
    part = f"""
    {country1} is a country in Europe. The capital of {country1} is {capital1}.
    {capital1} is known for its architecture and culture. Many tourists visit {capital1} every year.
    The population of {country1} is over 60 million people. {country1} has a rich history.
    {country1} produces wine and cheese. The language spoken in {country1} is French.
    {country2} is another country. The capital of {country2} is {capital2}.
    {capital2} is a large city with many attractions. People live in {capital2}.
    """
    corpus_parts.append(part)

full_corpus = "\n".join(corpus_parts)

# Build memory tree
tree = VirtualMemoryTree(cache_size=200, page_size=1000, persist_dir="/tmp/tmp_adversarial_debug")
tree.ingest_stream(iter([full_corpus]))

# Build knowledge graph
graph = DeterministicKnowledgeGraph(d_model=128)
extractor = HeuristicExtractor()

for page_id in tree.pages:
    text = tree.get_page(page_id)
    if text:
        triples = extractor.extract(text)
        for subj, pred, obj in triples:
            graph.add_triplet(subj, pred, obj, page_id=page_id, beacon_id=f"b1_debug")

graph.build_laplacian()

# Build QueryEngine
from math_engine import SeededLSH
lsh = SeededLSH(d=128, m=4, seed=42)
engine = QueryEngine(tree=tree, graph=graph, lsh=lsh, d_model=128)

# Test adversarial query
query = "What is the capital of France?"
result = engine.collapse(query=query, max_tokens=4096)

print(f"\n{'='*60}")
print(f"QUERY: {query}")
print(f"PAGES RETURNED: {len(result.pages)}")
print(f"CONCEPTS ACTIVATED: {result.concepts_activated}")
print(f"\nPage contents (first 200 chars each):")
for i, page in enumerate(result.pages):
    preview = page[:200].replace('\n', ' ')
    has_paris = "paris" in page.lower()
    has_berlin = "berlin" in page.lower()
    print(f"  Page {i}: {'[HAS PARIS]' if has_paris else ''} {'[HAS BERLIN]' if has_berlin else ''} {preview}...")

print(f"\nCorrect answer 'Paris' found: {any('paris' in p.lower() for p in result.pages)}")
print(f"Wrong answer 'Berlin' found: {any('berlin' in p.lower() for p in result.pages)}")
