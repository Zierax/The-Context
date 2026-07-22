"""
Benchmark comparing QueryEngine V1 (LSH-based) vs V2 (Embedding-based).

Tests on RULER-style tasks to measure improvement.
"""

import sys
import os
import time
import random
import json
import numpy as np

sys.path.insert(0, '.')

from memory_manager import VirtualMemoryTree
from knowledge_graph import DeterministicKnowledgeGraph
from entity_extractor import HeuristicExtractor
from entity_extractor_v2 import EmbeddingEntityExtractor
from embedding_engine import HashEmbeddingProvider

import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(0))


# ============================================================================
# Corpus Generation
# ============================================================================

def generate_distractor(n_tokens: int, seed: int = 42) -> str:
    """Generate random distractor text."""
    random.seed(seed)
    words = [
        "the", "a", "an", "is", "are", "was", "were", "have", "has", "had",
        "system", "process", "method", "structure", "concept", "element",
        "data", "information", "knowledge", "analysis", "evaluation",
        "computational", "mathematical", "statistical", "logical", "physical",
        "algorithm", "function", "parameter", "variable", "constant",
        "network", "graph", "tree", "matrix", "vector", "tensor",
        "encoding", "decoding", "compression", "retrieval", "storage",
        "transformation", "projection", "simulation", "optimization",
    ]
    return " ".join(random.choices(words, k=n_tokens))


def build_single_needle_corpus(
    needle_position: float = 0.5,
    haystack_tokens: int = 8000,
    seed: int = 42,
) -> tuple[list[str], dict]:
    """Build corpus with single needle at specified position."""
    random.seed(seed)
    np.random.seed(seed)

    needle = "The secret code is ALPHA-BRAVO-CHARLIE."
    needle_tokens = needle.split()

    before_tokens = int(haystack_tokens * needle_position)
    after_tokens = haystack_tokens - before_tokens

    corpus = []
    corpus.extend(generate_distractor(before_tokens, seed).split())
    corpus.extend(needle_tokens)
    corpus.extend(generate_distractor(after_tokens, seed + 1).split())

    fact = {
        "query": "What is the secret code?",
        "answer": "alpha-bravo-charlie",
        "position": needle_position,
    }

    return corpus, fact


def build_adversarial_corpus(
    num_facts: int = 5,
    seed: int = 42,
) -> tuple[list[str], list[dict]]:
    """Build corpus with correct facts and similar distractors."""
    random.seed(seed)
    np.random.seed(seed)

    countries = ["France", "Germany", "Spain", "Italy", "Japan"]
    capitals = ["Paris", "Berlin", "Madrid", "Rome", "Tokyo"]

    facts = []
    corpus = []

    for fact_id in range(num_facts):
        distractor = generate_distractor(2000, seed + fact_id * 3)
        corpus.extend(distractor.split())

        correct = f"The capital of {countries[fact_id]} is {capitals[fact_id]}."
        corpus.extend(correct.split())

        distractor2 = generate_distractor(2000, seed + fact_id * 3 + 1)
        corpus.extend(distractor2.split())

        wrong_idx = (fact_id + 1) % len(countries)
        distractor_fact = f"The capital of {countries[wrong_idx]} is {capitals[fact_id]}."
        corpus.extend(distractor_fact.split())

        distractor3 = generate_distractor(2000, seed + fact_id * 3 + 2)
        corpus.extend(distractor3.split())

        facts.append({
            "fact_id": fact_id,
            "query": f"What is the capital of {countries[fact_id]}?",
            "answer": capitals[fact_id].lower(),
            "wrong_answer": capitals[wrong_idx].lower(),
        })

    return corpus, facts


# ============================================================================
# Evaluation
# ============================================================================

def answer_in_pages(answer: str, pages: list[str]) -> bool:
    """Check if answer appears in any page."""
    answer_lower = answer.lower()
    for page in pages:
        if answer_lower in page.lower():
            return True
    return False


def run_single_needle_test(graph_v1, graph_v2, provider) -> dict:
    """Test single needle retrieval."""
    from query_engine import QueryEngine
    from query_engine_v2 import QueryEngineV2
    from math_engine import SeededLSH

    corpus, fact = build_single_needle_corpus()

    tree_v1 = VirtualMemoryTree(cache_size=200, page_size=100, persist_dir="/tmp/bench_v1_sn")
    tree_v2 = VirtualMemoryTree(cache_size=200, page_size=100, persist_dir="/tmp/bench_v2_sn")

    list(tree_v1.ingest_stream(iter(corpus)))
    list(tree_v2.ingest_stream(iter(corpus)))

    extractor_v1 = HeuristicExtractor()
    extractor_v2 = EmbeddingEntityExtractor(provider=provider)

    for page_id in list(tree_v1.pages):
        text = tree_v1.get_page(page_id)
        if text:
            triples = extractor_v1.extract(text)
            for subj, pred, obj in triples:
                graph_v1.add_triplet(subj, pred, obj, page_id=page_id, beacon_id=f"b1_{page_id}")

    for page_id in list(tree_v2.pages):
        text = tree_v2.get_page(page_id)
        if text:
            triples = extractor_v2.extract_triples(text)
            for subj, pred, obj in triples:
                graph_v2.add_triplet(subj, pred, obj, page_id=page_id, beacon_id=f"b1_{page_id}")

    if len(graph_v1.node_to_idx) >= 2:
        graph_v1.build_laplacian()
    if len(graph_v2.node_to_idx) >= 2:
        graph_v2.build_laplacian()

    lsh = SeededLSH(d=128, m=4, seed=42)
    engine_v1 = QueryEngine(tree=tree_v1, graph=graph_v1, lsh=lsh, d_model=128)
    engine_v2 = QueryEngineV2(tree=tree_v2, graph=graph_v2, provider=provider, d_model=128)

    start = time.perf_counter()
    result_v1 = engine_v1.collapse(query=fact["query"], max_tokens=4096)
    time_v1 = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    result_v2 = engine_v2.collapse(query=fact["query"], max_tokens=4096)
    time_v2 = (time.perf_counter() - start) * 1000

    found_v1 = answer_in_pages(fact["answer"], result_v1.pages)
    found_v2 = answer_in_pages(fact["answer"], result_v2.pages)

    return {
        "task": "single_needle",
        "v1": {"found": found_v1, "time_ms": time_v1, "pages": len(result_v1.pages)},
        "v2": {"found": found_v2, "time_ms": time_v2, "pages": len(result_v2.pages)},
    }


def run_adversarial_test(graph_v1, graph_v2, provider) -> dict:
    """Test adversarial retrieval."""
    from query_engine import QueryEngine
    from query_engine_v2 import QueryEngineV2
    from math_engine import SeededLSH

    corpus, facts = build_adversarial_corpus()

    tree_v1 = VirtualMemoryTree(cache_size=200, page_size=100, persist_dir="/tmp/bench_v1_adv")
    tree_v2 = VirtualMemoryTree(cache_size=200, page_size=100, persist_dir="/tmp/bench_v2_adv")

    list(tree_v1.ingest_stream(iter(corpus)))
    list(tree_v2.ingest_stream(iter(corpus)))

    extractor_v1 = HeuristicExtractor()
    extractor_v2 = EmbeddingEntityExtractor(provider=provider)

    for page_id in list(tree_v1.pages):
        text = tree_v1.get_page(page_id)
        if text:
            triples = extractor_v1.extract(text)
            for subj, pred, obj in triples:
                graph_v1.add_triplet(subj, pred, obj, page_id=page_id, beacon_id=f"b1_{page_id}")

    for page_id in list(tree_v2.pages):
        text = tree_v2.get_page(page_id)
        if text:
            triples = extractor_v2.extract_triples(text)
            for subj, pred, obj in triples:
                graph_v2.add_triplet(subj, pred, obj, page_id=page_id, beacon_id=f"b1_{page_id}")

    if len(graph_v1.node_to_idx) >= 2:
        graph_v1.build_laplacian()
    if len(graph_v2.node_to_idx) >= 2:
        graph_v2.build_laplacian()

    lsh = SeededLSH(d=128, m=4, seed=42)
    engine_v1 = QueryEngine(tree=tree_v1, graph=graph_v1, lsh=lsh, d_model=128)
    engine_v2 = QueryEngineV2(tree=tree_v2, graph=graph_v2, provider=provider, d_model=128)

    results_v1 = []
    results_v2 = []

    for fact in facts:
        start = time.perf_counter()
        result_v1 = engine_v1.collapse(query=fact["query"], max_tokens=4096)
        time_v1 = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        result_v2 = engine_v2.collapse(query=fact["query"], max_tokens=4096)
        time_v2 = (time.perf_counter() - start) * 1000

        correct_v1 = answer_in_pages(fact["answer"], result_v1.pages)
        distractor_v1 = answer_in_pages(fact["wrong_answer"], result_v1.pages)
        clean_v1 = correct_v1 and not distractor_v1

        correct_v2 = answer_in_pages(fact["answer"], result_v2.pages)
        distractor_v2 = answer_in_pages(fact["wrong_answer"], result_v2.pages)
        clean_v2 = correct_v2 and not distractor_v2

        results_v1.append({
            "correct": correct_v1,
            "distractor": distractor_v1,
            "clean": clean_v1,
            "time_ms": time_v1,
        })
        results_v2.append({
            "correct": correct_v2,
            "distractor": distractor_v2,
            "clean": clean_v2,
            "time_ms": time_v2,
        })

    n = len(facts)
    return {
        "task": "adversarial",
        "v1": {
            "clean_accuracy": sum(1 for r in results_v1 if r["clean"]) / n,
            "any_accuracy": sum(1 for r in results_v1 if r["correct"]) / n,
            "distractor_rate": sum(1 for r in results_v1 if r["distractor"]) / n,
            "avg_time_ms": np.mean([r["time_ms"] for r in results_v1]),
        },
        "v2": {
            "clean_accuracy": sum(1 for r in results_v2 if r["clean"]) / n,
            "any_accuracy": sum(1 for r in results_v2 if r["correct"]) / n,
            "distractor_rate": sum(1 for r in results_v2 if r["distractor"]) / n,
            "avg_time_ms": np.mean([r["time_ms"] for r in results_v2]),
        },
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("  Query Engine V1 vs V2 Benchmark")
    print("=" * 70)

    provider = HashEmbeddingProvider(dimension=384)

    # Single Needle Test
    print("\n[SINGLE_NEEDLE] Running...")
    graph_v1 = DeterministicKnowledgeGraph(d_model=128)
    graph_v2 = DeterministicKnowledgeGraph(d_model=128)

    result_sn = run_single_needle_test(graph_v1, graph_v2, provider)
    print(f"  V1: {'FOUND' if result_sn['v1']['found'] else 'MISSED'} ({result_sn['v1']['time_ms']:.1f}ms, {result_sn['v1']['pages']} pages)")
    print(f"  V2: {'FOUND' if result_sn['v2']['found'] else 'MISSED'} ({result_sn['v2']['time_ms']:.1f}ms, {result_sn['v2']['pages']} pages)")

    # Adversarial Test
    print("\n[ADVERSARIAL] Running...")
    graph_v1 = DeterministicKnowledgeGraph(d_model=128)
    graph_v2 = DeterministicKnowledgeGraph(d_model=128)

    result_adv = run_adversarial_test(graph_v1, graph_v2, provider)
    print(f"  V1: clean={result_adv['v1']['clean_accuracy']:.0%} any={result_adv['v1']['any_accuracy']:.0%} distractor={result_adv['v1']['distractor_rate']:.0%} ({result_adv['v1']['avg_time_ms']:.1f}ms)")
    print(f"  V2: clean={result_adv['v2']['clean_accuracy']:.0%} any={result_adv['v2']['any_accuracy']:.0%} distractor={result_adv['v2']['distractor_rate']:.0%} ({result_adv['v2']['avg_time_ms']:.1f}ms)")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Single Needle:  V1={'100%' if result_sn['v1']['found'] else '0%'}  V2={'100%' if result_sn['v2']['found'] else '0%'}")
    print(f"  Adversarial:    V1={result_adv['v1']['clean_accuracy']:.0%} clean  V2={result_adv['v2']['clean_accuracy']:.0%} clean")
    print("=" * 70)

    results = {
        "single_needle": result_sn,
        "adversarial": result_adv,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/v1_vs_v2_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to results/v1_vs_v2_benchmark.json")


if __name__ == "__main__":
    main()
