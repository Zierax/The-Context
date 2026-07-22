import sys
sys.path.insert(0, '.')

from entity_extractor import HeuristicExtractor

extractor = HeuristicExtractor()

text = "The capital of France is Paris. Paris is known for its architecture and culture. Many tourists visit Paris every year. The population of France is over 60 million people."

triples = extractor.extract(text)
print(f"Triples from sample text: {len(triples)}")
for t in triples:
    print(f"  {t}")

# Check if the SVO pattern matches
import re
TRIPLE_PATTERN = re.compile(
    r"(\b(?:\w+\s+){0,8}?\w+)\s+"
    r"(is|are|was|were|has|have|had|contains?|includes?|refers?|"
    r"calls?|depends?|requires?|uses?|defines?|implements?|extends?|"
    r"produces?|creates?|transforms?|computes?|links?|maps?|connects?|"
    r"triggers?|generates?|stores?|retrieves?|processes?)\s+"
    r"(\b(?:\w+\s+){0,8}?\w+)",
    re.IGNORECASE,
)
matches = TRIPLE_PATTERN.findall(text)
print(f"\nTRIPLE_PATTERN matches: {len(matches)}")
for m in matches[:10]:
    print(f"  {m}")

# Check sentence splitting
SENTENCE_PATTERN = re.compile(r"[^.!?]+[.!?]", re.UNICODE)
sentences = SENTENCE_PATTERN.findall(text)
print(f"\nSentences: {len(sentences)}")
for s in sentences:
    print(f"  '{s.strip()}' ({len(s.split())} words)")
    if len(s.split()) > 30:
        print(f"    SKIPPED (>30 words)")
