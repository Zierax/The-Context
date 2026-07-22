import sys
sys.path.insert(0, '.')
from memory_manager import VirtualMemoryTree

tree = VirtualMemoryTree(cache_size=200, page_size=100, persist_dir='/tmp/test_short')
corpus = 'The capital of France is Paris. Paris is a beautiful city. The Eiffel Tower is in Paris. '.split() * 20
print(f'Corpus tokens: {len(corpus)}')
pids = list(tree.ingest_stream(iter(corpus)))
print(f'Pages created: {len(pids)}')
for pid in tree.pages:
    text = tree.get_page(pid)
    print(f'  {pid}: {len(text)} chars') if text else print(f'  {pid}: None')
