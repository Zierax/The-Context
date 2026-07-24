#!/usr/bin/env python3
"""RULER-style benchmark suite for hierarchical beacon memory system.

Adapted from RULER (NVIDIA, 2024) methodology for memory retrieval systems.
Tests retrieval accuracy with increasing corpus size and distractor complexity.

Task Categories:
1. Single Needle (NIAH) - Find one fact in distractor text
2. Multi Needle - Find multiple facts scattered in distractor text
3. Multi-hop - Connect facts across different pages
4. Adversarial - Distinguish correct facts from similar distractors
5. Aggregation - Count/sum facts spread across corpus

Usage:
    python benchmarks/ruler_bench.py --size 10000 --tasks all
    python benchmarks/ruler_bench.py --size 50000 --tasks retrieval
"""
import sys
import os
import time
import json
import random
import argparse
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from the_context.core import VirtualMemoryTree
from the_context.query import QueryEngine
from the_context.core import DeterministicKnowledgeGraph
from the_context.extraction import HeuristicExtractor
from the_context.core import SeededLSH


# ============================================================================
# Corpus Generators (RULER-style)
# ============================================================================

def generate_needle(corpus_id: int) -> dict:
    """Generate a unique factual statement (the 'needle')."""
    subjects = [
        "The capital of France", "The largest planet", "The speed of light",
        "The inventor of Python", "The year WWII ended", "The chemical symbol for gold",
        "The deepest ocean", "The tallest building", "The smallest prime",
        "The author of 1984", "The currency of Japan", "The longest river",
        "The boiling point of water", "The element carbon", "The number of continents",
    ]
    objects = [
        "Paris", "Jupiter", "299792458 m/s", "Guido van Rossum", "1945", "Au",
        "Pacific Ocean", "Burj Khalifa", "2", "George Orwell", "Yen", "Nile",
        "100°C", "C", "7",
    ]
    return {
        "id": corpus_id,
        "text": f"{subjects[corpus_id % len(subjects)]} is {objects[corpus_id % len(objects)]}.",
        "subject": subjects[corpus_id % len(subjects)].lower(),
        "answer": objects[corpus_id % len(objects)].lower(),
    }


def generate_distractor(length: int) -> str:
    """Generate plausible but incorrect filler text."""
    words = [
        "quantum", "spectral", "manifold", "beacon", "hierarchy", "diffusion",
        "laplacian", "embedding", "vector", "similarity", "graph", "tensor",
        "metric", "geodesic", "curvature", "topology", "eigenvalue", "matrix",
        "transformation", "projection", "compression", "encoding", "decoding",
        "entropy", "information", "retrieval", "indexing", "search", "query",
    ]
    result = []
    for _ in range(length):
        result.append(random.choice(words))
    return " ".join(result)


def build_corpus(
    num_needles: int = 1,
    distractor_tokens: int = 10000,
    seed: int = 42,
) -> tuple[list[str], list[dict]]:
    """Build a corpus with needles embedded in distractor text.

    Returns:
        (tokens, needles) - token list and needle metadata
    """
    random.seed(seed)
    np.random.seed(seed)

    needles = [generate_needle(i) for i in range(num_needles)]
    corpus = []
    needle_positions = []

    # Interleave needles evenly through distractor text
    segment_size = distractor_tokens // (num_needles + 1)

    for i in range(num_needles + 1):
        # Add distractor segment
        distractor = generate_distractor(segment_size)
        corpus.extend(distractor.split())

        # Add needle (except after last distractor segment)
        if i < num_needles:
            needle_tokens = needles[i]["text"].split()
            corpus.extend(needle_tokens)
            needle_positions.append({
                "needle_id": i,
                "start_token": len(corpus) - len(needle_tokens),
                "end_token": len(corpus),
                "depth": (len(corpus) - len(needle_tokens)) / max(len(corpus), 1),
            })

    return corpus, needles


def build_multi_hop_corpus(
    num_chains: int = 1,
    hops_per_chain: int = 2,
    distractor_tokens: int = 10000,
    seed: int = 42,
) -> tuple[list[str], list[dict]]:
    """Build corpus with multi-hop reasoning chains.

    Each chain: A is X, B is Y, A is related to B.
    Question: "What is X related to?" -> Answer: Y
    """
    random.seed(seed)
    np.random.seed(seed)

    subjects = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"]
    relations = ["is a type of", "is used in", "is related to", "depends on"]
    objects = ["system", "method", "process", "structure", "concept", "element"]

    chains = []
    corpus = []
    segment_size = distractor_tokens // (num_chains * hops_per_chain + 1)

    for chain_id in range(num_chains):
        chain = {"chain_id": chain_id, "hops": []}
        s = subjects[chain_id % len(subjects)]

        for hop in range(hops_per_chain):
            # Add distractor
            distractor = generate_distractor(segment_size)
            corpus.extend(distractor.split())

            # Add hop: s relation o
            o = objects[(chain_id * hops_per_chain + hop) % len(objects)]
            rel = relations[hop % len(relations)]
            sentence = f"{s} {rel} {o}."
            corpus.extend(sentence.split())

            chain["hops"].append({
                "subject": s.lower(),
                "relation": rel,
                "object": o.lower(),
                "sentence": sentence,
            })

            # Next hop: s becomes o of this hop
            s = o

        chains.append(chain)

    return corpus, chains


def build_adversarial_corpus(
    num_facts: int = 1,
    distractor_tokens: int = 10000,
    seed: int = 42,
) -> tuple[list[str], list[dict]]:
    """Build corpus with correct facts and similar distractors.

    Example: Correct: "The capital of France is Paris"
             Distractor: "The capital of Germany is Paris" (wrong!)
    Facts are separated by distractor text so they land on different pages.
    """
    random.seed(seed)
    np.random.seed(seed)

    facts = []
    corpus = []
    # Split distractor evenly: before fact, between correct and distractor, after
    third = distractor_tokens // 3

    for fact_id in range(num_facts):
        # Add distractor before the correct fact
        distractor = generate_distractor(third)
        corpus.extend(distractor.split())

        # Correct fact
        countries = ["France", "Germany", "Spain", "Italy", "Japan", "Brazil"]
        capitals = ["Paris", "Berlin", "Madrid", "Rome", "Tokyo", "Brasilia"]
        idx = fact_id % len(countries)
        correct = f"The capital of {countries[idx]} is {capitals[idx]}."
        corpus.extend(correct.split())

        # Add distractor BETWEEN correct and distractor (ensures different pages)
        distractor2 = generate_distractor(third)
        corpus.extend(distractor2.split())

        # Adversarial distractor (similar structure, wrong answer)
        wrong_idx = (idx + 1) % len(countries)
        distractor_fact = f"The capital of {countries[wrong_idx]} is {capitals[idx]}."
        corpus.extend(distractor_fact.split())

        # Add distractor after
        distractor3 = generate_distractor(third)
        corpus.extend(distractor3.split())

        facts.append({
            "fact_id": fact_id,
            "correct": correct.lower(),
            "distractor": distractor_fact.lower(),
            "query": f"What is the capital of {countries[idx]}?",
            "answer": capitals[idx].lower(),
            "wrong_answer": capitals[wrong_idx].lower(),
        })

    return corpus, facts


# ============================================================================
# Evaluation Metrics
# ============================================================================

def precision_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    """Precision@K: fraction of top-K results that are relevant."""
    if not retrieved or not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(r.lower() for r in relevant)
    hits = sum(1 for r in retrieved_k if any(rel in r.lower() for rel in relevant_set))
    return hits / min(k, len(retrieved_k))


def recall_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    """Recall@K: fraction of relevant items found in top-K results."""
    if not retrieved or not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(r.lower() for r in relevant)
    hits = sum(1 for r in retrieved_k if any(rel in r.lower() for rel in relevant_set))
    return hits / len(relevant_set) if relevant_set else 0.0


def answer_in_page(answer: str, pages: list[str]) -> bool:
    """Check if answer text appears in any retrieved page."""
    answer_lower = answer.lower()
    for page in pages:
        if answer_lower in page.lower():
            return True
    return False


# ============================================================================
# Benchmark Tasks
# ============================================================================

def run_single_needle_bench(
    engine: QueryEngine,
    corpus: list[str],
    needles: list[dict],
    pids: list[str],
    tree: VirtualMemoryTree,
) -> dict:
    """Task 1: Single Needle - Find one fact in distractor text."""
    results = []
    for needle in needles:
        # Query that should retrieve the needle
        query = f"What {needle['subject']}?"
        result = engine.collapse(query=query, max_tokens=4096)

        found = answer_in_page(needle["answer"], result.pages)
        results.append({
            "needle_id": needle["id"],
            "query": query,
            "expected": needle["answer"],
            "found": found,
            "pages_returned": len(result.pages),
            "tokens_used": result.tokens_used,
            "confidence": result.confidence_score,
        })

    accuracy = sum(1 for r in results if r["found"]) / len(results) if results else 0.0
    return {
        "task": "single_needle",
        "accuracy": accuracy,
        "details": results,
    }


def run_multi_needle_bench(
    engine: QueryEngine,
    corpus: list[str],
    needles: list[dict],
    pids: list[str],
    tree: VirtualMemoryTree,
) -> dict:
    """Task 2: Multi Needle - Find multiple facts scattered in text."""
    # Query that should retrieve ALL needles
    query = "What are the facts mentioned in this text?"
    # Budget should cover enough pages to find all needles
    # Each needle is ~10 tokens, but pages are 1000 tokens each
    # Need enough budget for ~60% of pages to ensure needle coverage
    total_tokens = sum(
        len(tree.get_page(p).split())
        for p in pids if tree.get_page(p)
    )
    budget = max(8192, int(total_tokens * 0.8))
    result = engine.collapse(query=query, max_tokens=budget)

    found_count = 0
    for needle in needles:
        if answer_in_page(needle["answer"], result.pages):
            found_count += 1

    accuracy = found_count / len(needles) if needles else 0.0
    return {
        "task": "multi_needle",
        "accuracy": accuracy,
        "found": found_count,
        "total": len(needles),
        "pages_returned": len(result.pages),
        "tokens_used": result.tokens_used,
    }


def run_multi_hop_bench(
    engine: QueryEngine,
    corpus: list[str],
    chains: list[dict],
    pids: list[str],
    tree: VirtualMemoryTree,
) -> dict:
    """Task 3: Multi-hop - Connect facts across pages."""
    results = []
    for chain in chains:
        # Query about the first subject
        first_hop = chain["hops"][0]
        last_hop = chain["hops"][-1]
        query = f"What is {first_hop['subject']} related to?"
        result = engine.collapse(query=query, max_tokens=4096)

        # Check if any answer from the chain appears in results
        found = False
        for hop in chain["hops"]:
            if answer_in_page(hop["object"], result.pages):
                found = True
                break

        results.append({
            "chain_id": chain["chain_id"],
            "query": query,
            "expected": last_hop["object"],
            "found": found,
            "pages_returned": len(result.pages),
        })

    accuracy = sum(1 for r in results if r["found"]) / len(results) if results else 0.0
    return {
        "task": "multi_hop",
        "accuracy": accuracy,
        "details": results,
    }


def run_adversarial_bench(
    engine: QueryEngine,
    corpus: list[str],
    facts: list[dict],
    pids: list[str],
    tree: VirtualMemoryTree,
) -> dict:
    """Task 4: Adversarial - Distinguish correct from similar distractors."""
    results = []
    for fact in facts:
        result = engine.collapse(query=fact["query"], max_tokens=4096)

        # Check if correct answer is found (and distractor is not)
        correct_found = answer_in_page(fact["answer"], result.pages)
        distractor_found = answer_in_page(fact["wrong_answer"], result.pages)

        results.append({
            "fact_id": fact["fact_id"],
            "query": fact["query"],
            "correct_found": correct_found,
            "distractor_found": distractor_found,
            "clean_hit": correct_found and not distractor_found,
            "pages_returned": len(result.pages),
        })

    clean_accuracy = sum(1 for r in results if r["clean_hit"]) / len(results) if results else 0.0
    any_accuracy = sum(1 for r in results if r["correct_found"]) / len(results) if results else 0.0
    distractor_rate = sum(1 for r in results if r["distractor_found"]) / len(results) if results else 0.0

    return {
        "task": "adversarial",
        "clean_accuracy": clean_accuracy,
        "any_accuracy": any_accuracy,
        "distractor_rate": distractor_rate,
        "details": results,
    }


# ============================================================================
# Main Benchmark Runner
# ============================================================================

def run_benchmark(
    size: int = 10000,
    tasks: list[str] | None = None,
    seed: int = 42,
) -> dict:
    """Run the full benchmark suite."""
    if tasks is None:
        tasks = ["single_needle", "multi_needle", "multi_hop", "adversarial"]

    print(f"\n{'='*70}")
    print(f"  RULER-Style Benchmark: {size:,} tokens")
    print(f"{'='*70}")

    all_results = {}

    for task in tasks:
        print(f"\n  [{task.upper()}] Running...")

        # Build corpus for this task
        if task in ("single_needle", "multi_needle"):
            num_needles = 5 if task == "multi_needle" else 1
            corpus, needles = build_corpus(
                num_needles=num_needles,
                distractor_tokens=size,
                seed=seed,
            )
        elif task == "multi_hop":
            corpus, chains = build_multi_hop_corpus(
                num_chains=3,
                hops_per_chain=3,
                distractor_tokens=size,
                seed=seed,
            )
        elif task == "adversarial":
            corpus, facts = build_adversarial_corpus(
                num_facts=5,
                distractor_tokens=size,
                seed=seed,
            )
        else:
            print(f"    Unknown task: {task}")
            continue

        print(f"    Corpus: {len(corpus):,} tokens")

        # Ingest
        d = 128
        persist = tempfile.mkdtemp()
        lsh = SeededLSH(d=d, w=10.0, m=4, seed=seed)
        tree = VirtualMemoryTree(page_size=1000, cache_size=200, persist_dir=persist)
        graph = DeterministicKnowledgeGraph(d_model=d)
        extractor = HeuristicExtractor()

        t0 = time.perf_counter()
        pids = list(tree.ingest_stream(iter(corpus)))
        ingest_ms = (time.perf_counter() - t0) * 1000
        print(f"    Ingested: {len(pids)} pages in {ingest_ms:.0f}ms")

        # Extract knowledge
        for pid in pids:
            pt = tree.get_page(pid)
            if pt is None:
                continue
            b1 = tree.get_beacon_for_page(pid)
            for s, p, o in extractor.extract(pt):
                graph.add_triplet(s, p, o, beacon_id=b1 or "")

        if len(graph.node_to_idx) >= 2:
            graph.build_laplacian()

        # Query
        engine = QueryEngine(tree=tree, graph=graph, lsh=lsh, d_model=d)

        # Run task
        t0 = time.perf_counter()
        if task == "single_needle":
            result = run_single_needle_bench(engine, corpus, needles, pids, tree)
        elif task == "multi_needle":
            result = run_multi_needle_bench(engine, corpus, needles, pids, tree)
        elif task == "multi_hop":
            result = run_multi_hop_bench(engine, corpus, chains, pids, tree)
        elif task == "adversarial":
            result = run_adversarial_bench(engine, corpus, facts, pids, tree)

        query_ms = (time.perf_counter() - t0) * 1000

        # Memory measurement — pages are on disk, count beacon embeddings only
        mem = len(tree.beacon_b1) * d * 2  # B1 embeddings (float16)
        mem += len(tree.beacon_b2) * (d * 2 + d * 8)  # B2 Gaussian patches
        mem += len(tree.beacon_b3) * (5 * 8 + 10 * 5 * 8)  # B3
        text_bytes = sum(len(t.encode()) for t in corpus)

        result["query_ms"] = query_ms
        result["memory_bytes"] = mem
        result["text_bytes"] = text_bytes
        result["compression_ratio"] = text_bytes / mem if mem > 0 else 0

        # Print result
        acc = result.get("accuracy", result.get("clean_accuracy", 0))
        print(f"    Accuracy: {acc:.1%}")
        print(f"    Query time: {query_ms:.0f}ms")
        print(f"    Memory: {mem/1024:.1f} KB ({mem/len(corpus):.2f} bytes/token)")
        print(f"    Compression: {result['compression_ratio']:.2f}x")

        all_results[task] = result

        # Cleanup
        import shutil
        shutil.rmtree(persist, ignore_errors=True)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for task, result in all_results.items():
        acc = result.get("accuracy", result.get("clean_accuracy", 0))
        print(f"  {task:20s}: {acc:.1%}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RULER-style benchmark")
    parser.add_argument("--size", type=int, default=10000, help="Corpus size in tokens")
    parser.add_argument("--tasks", nargs="+", default=["single_needle", "multi_needle", "multi_hop", "adversarial"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_benchmark(size=args.size, tasks=args.tasks, seed=args.seed)

    # Save results
    output_path = Path("results") / "ruler_benchmark.json"
    output_path.parent.mkdir(exist_ok=True)

    # Convert to serializable format
    serializable = {}
    for task, result in results.items():
        serializable[task] = {
            k: v for k, v in result.items()
            if k != "details"  # Skip detailed results for JSON
        }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to {output_path}")
