import sys
sys.path.insert(0, "src")
from benchmarks.ruler_bench import build_adversarial_corpus
corpus, facts = build_adversarial_corpus(num_facts=2, distractor_tokens=5000)
print("Facts:")
for f in facts:
    print(f"  {f}")
print(f"Corpus tokens: {len(corpus)}")
# Find where facts appear
for i, token in enumerate(corpus):
    if token.lower() in ["paris", "berlin", "france", "germany", "spain", "madrid", "italy", "rome", "japan", "tokyo"]:
        print(f"  Token {i}: {token}")
