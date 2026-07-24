#!/usr/bin/env python3
"""Wikipedia real-world benchmark for The Context memory system.

Tests compression and retrieval on real noisy data from Wikipedia.
Downloads a Wikipedia subset and evaluates:
- Compression ratio (total tokens / retrieved tokens)
- Retrieval accuracy (can we find relevant facts?)
- Latency (query time at scale)
- Memory efficiency (bytes per token)

Usage:
    PYTHONPATH=src python3 benchmarks/wikipedia_bench.py --size 1000000
    PYTHONPATH=src python3 benchmarks/wikipedia_bench.py --size 500000 --max-articles 1000
"""
import sys
import os
import time
import json
import re
import random
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from the_context.core import VirtualMemoryTree, DeterministicKnowledgeGraph
from the_context.query import QueryEngine
from the_context.extraction import HeuristicExtractor
from the_context.core import SeededLSH


# ============================================================================
# Wikipedia Data Loading
# ============================================================================

def load_wikipedia_subset(
    size_tokens: int = 1_000_000,
    max_articles: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Load a Wikipedia subset from HuggingFace.

    Args:
        size_tokens: Target number of tokens to load
        max_articles: Maximum number of articles (None = no limit)
        seed: Random seed for reproducibility

    Returns:
        List of article dicts with keys: id, title, text, tokens
    """
    from datasets import load_dataset

    print(f"  Loading Wikipedia from HuggingFace...")
    ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

    articles = []
    total_tokens = 0
    rng = random.Random(seed)

    for i, item in enumerate(ds):
        if max_articles and i >= max_articles:
            break
        if total_tokens >= size_tokens:
            break

        text = item["text"]
        tokens = len(text.split())

        # Skip very short articles
        if tokens < 100:
            continue

        articles.append({
            "id": item["id"],
            "title": item["title"],
            "text": text,
            "tokens": tokens,
        })
        total_tokens += tokens

        if (i + 1) % 1000 == 0:
            print(f"    Loaded {i+1:,} articles, {total_tokens:,} tokens...")

    print(f"  Loaded {len(articles):,} articles, {total_tokens:,} tokens")
    return articles


# ============================================================================
# Retrieval Evaluation
# ============================================================================

def generate_retrieval_queries(articles: list[dict], n_queries: int = 100, seed: int = 42) -> list[dict]:
    """Generate retrieval queries from Wikipedia articles.

    Each query asks about a specific fact from an article.
    The ground truth is the article containing that fact.

    Args:
        articles: List of article dicts
        n_queries: Number of queries to generate
        seed: Random seed

    Returns:
        List of query dicts with keys: query, answer, article_id, article_title
    """
    rng = random.Random(seed)
    queries = []

    for _ in range(n_queries):
        # Pick a random article
        article = rng.choice(articles)
        text = article["text"]
        sentences = re.split(r'[.!?]+', text)

        # Pick a meaningful sentence (not too short, not too long)
        meaningful = [s.strip() for s in sentences if 10 < len(s.split()) < 30]
        if not meaningful:
            continue

        sentence = rng.choice(meaningful)
        words = sentence.split()

        # Create a query by masking a key phrase
        if len(words) >= 5:
            # Mask a noun phrase (simple heuristic: words after "is/was/are")
            for i, w in enumerate(words):
                if w.lower() in ("is", "was", "are", "were") and i + 1 < len(words):
                    # Extract the predicate (what comes after the verb)
                    predicate = " ".join(words[i + 1:])
                    question_words = words[:i]
                    query = "What is " + " ".join(question_words).lower().rstrip(",") + "?"
                    answer = predicate.strip().rstrip(".")
                    break
            else:
                # Fallback: use first few words as query
                query = "What is " + " ".join(words[:3]).lower() + "?"
                answer = " ".join(words[3:]).strip().rstrip(".")
        else:
            query = f"Tell me about {article['title']}"
            answer = article["title"]

        if query and answer:
            queries.append({
                "query": query,
                "answer": answer,
                "article_id": article["id"],
                "article_title": article["title"],
            })

    return queries[:n_queries]


def compute_retrieval_accuracy(
    engine: QueryEngine,
    queries: list[dict],
    articles: list[dict],
    top_k: int = 5,
) -> dict:
    """Compute retrieval accuracy metrics.

    Args:
        engine: QueryEngine instance
        queries: List of query dicts
        articles: List of article dicts
        top_k: Number of top results to consider

    Returns:
        Dictionary with accuracy metrics
    """
    article_id_to_title = {a["id"]: a["title"] for a in articles}

    hits_at_1 = 0
    hits_at_5 = 0
    mrr = 0.0
    latencies = []

    for q in queries:
        t0 = time.perf_counter()
        result = engine.collapse(query=q["query"], max_tokens=4096)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)

        # Check if the target article is in retrieved pages
        # (approximate: check if article title appears in any page)
        target_title = q["article_title"].lower()
        found_rank = None

        for rank, page_text in enumerate(result.pages[:top_k]):
            if target_title in page_text.lower():
                found_rank = rank + 1
                break

        if found_rank == 1:
            hits_at_1 += 1
        if found_rank is not None:
            hits_at_5 += 1
            mrr += 1.0 / found_rank

    n = len(queries)
    return {
        "hits_at_1": hits_at_1 / n if n > 0 else 0.0,
        "hits_at_5": hits_at_5 / n if n > 0 else 0.0,
        "mrr": mrr / n if n > 0 else 0.0,
        "avg_latency_ms": np.mean(latencies),
        "p50_latency_ms": np.percentile(latencies, 50),
        "p95_latency_ms": np.percentile(latencies, 95),
    }


# ============================================================================
# Main Benchmark
# ============================================================================

def run_wikipedia_benchmark(
    size_tokens: int = 1_000_000,
    max_articles: int | None = None,
    n_queries: int = 100,
    page_size: int = 1000,
    seed: int = 42,
) -> dict:
    """Run Wikipedia real-world benchmark.

    Args:
        size_tokens: Target corpus size in tokens
        max_articles: Maximum articles to load
        n_queries: Number of retrieval queries
        page_size: Page size for VirtualMemoryTree
        seed: Random seed

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*70}")
    print(f"  Wikipedia Real-World Benchmark — The Context Memory System")
    print(f"{'='*70}")

    # Load data
    articles = load_wikipedia_subset(
        size_tokens=size_tokens,
        max_articles=max_articles,
        seed=seed,
    )

    if not articles:
        print("  ERROR: No articles loaded")
        return {}

    # Generate queries
    queries = generate_retrieval_queries(articles, n_queries=n_queries, seed=seed)
    print(f"  Generated {len(queries)} retrieval queries")

    # Build corpus: flatten articles into page-sized chunks
    corpus_tokens = []
    page_to_article = {}  # page_id -> article

    for article in articles:
        words = article["text"].split()
        for i in range(0, len(words), page_size):
            page_text = " ".join(words[i:i + page_size])
            corpus_tokens.extend(page_text.split())
            page_id = len(corpus_tokens)
            page_to_article[page_id] = article

    total_tokens = len(corpus_tokens)
    print(f"  Corpus: {total_tokens:,} tokens in {len(page_to_article):,} pages")

    # Ingest into The Context
    d = 128
    persist = tempfile.mkdtemp()
    lsh = SeededLSH(d=d, w=10.0, m=4, seed=seed)
    tree = VirtualMemoryTree(page_size=page_size, cache_size=500, persist_dir=persist)
    graph = DeterministicKnowledgeGraph(d_model=d)
    extractor = HeuristicExtractor()

    print(f"\n  Ingesting corpus...")
    t0 = time.perf_counter()
    pids = list(tree.ingest_stream(iter(corpus_tokens)))
    ingest_ms = (time.perf_counter() - t0) * 1000
    print(f"  Ingested: {len(pids)} pages in {ingest_ms:.0f}ms")

    # Knowledge graph extraction skipped for speed
    # The retrieval relies on LSH bucketing, not graph structure
    extract_ms = 0.0
    print(f"  Knowledge graph extraction skipped for benchmarking speed")

    # Initialize query engine
    engine = QueryEngine(tree=tree, graph=graph, lsh=lsh, d_model=d)

    # Run retrieval evaluation
    print(f"\n  Running {len(queries)} retrieval queries...")
    retrieval_metrics = compute_retrieval_accuracy(engine, queries, articles, top_k=5)

    # Compute compression metrics
    avg_tokens_used = np.mean([
        engine.collapse(query=q["query"], max_tokens=4096).tokens_used
        for q in queries[:10]  # Sample for speed
    ])

    # Memory measurement
    mem = len(tree.beacon_b1) * d * 2 + len(tree.beacon_b2) * (d * 2 + d * 8)

    summary = {
        "benchmark": "wikipedia_real_world",
        "corpus_tokens": total_tokens,
        "corpus_pages": len(pids),
        "n_articles": len(articles),
        "n_queries": len(queries),
        "ingest_ms": ingest_ms,
        "extract_ms": extract_ms,
        "memory_bytes": mem,
        "bytes_per_token": mem / total_tokens if total_tokens > 0 else 0,
        "compression_ratio": total_tokens / avg_tokens_used if avg_tokens_used > 0 else 0,
        "token_reduction_pct": (1 - avg_tokens_used / total_tokens) * 100 if total_tokens > 0 else 0,
        "retrieval": retrieval_metrics,
    }

    # Print results
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Corpus: {summary['corpus_tokens']:,} tokens, {summary['corpus_pages']:,} pages")
    print(f"  Articles: {summary['n_articles']:,}")
    print(f"  ")
    print(f"  Compression:")
    print(f"    Avg Tokens Retrieved: {avg_tokens_used:.0f}")
    print(f"    Compression Ratio:    {summary['compression_ratio']:.2f}x")
    print(f"    Token Reduction:      {summary['token_reduction_pct']:.1f}%")
    print(f"    Memory:               {summary['memory_bytes']/1024/1024:.1f} MB")
    print(f"    Bytes/Token:          {summary['bytes_per_token']:.2f}")
    print(f"  ")
    print(f"  Retrieval Accuracy:")
    print(f"    Hits@1:               {retrieval_metrics['hits_at_1']:.4f}")
    print(f"    Hits@5:               {retrieval_metrics['hits_at_5']:.4f}")
    print(f"    MRR:                  {retrieval_metrics['mrr']:.4f}")
    print(f"  ")
    print(f"  Latency:")
    print(f"    Avg:                  {retrieval_metrics['avg_latency_ms']:.1f}ms")
    print(f"    P50:                  {retrieval_metrics['p50_latency_ms']:.1f}ms")
    print(f"    P95:                  {retrieval_metrics['p95_latency_ms']:.1f}ms")

    # Cleanup
    import shutil
    shutil.rmtree(persist, ignore_errors=True)

    # Save results
    output_path = Path("results") / "wikipedia_benchmark.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wikipedia real-world benchmark")
    parser.add_argument("--size", type=int, default=1_000_000, help="Target corpus size in tokens")
    parser.add_argument("--max-articles", type=int, default=None, help="Max articles to load")
    parser.add_argument("--queries", type=int, default=100, help="Number of retrieval queries")
    parser.add_argument("--page-size", type=int, default=1000, help="Page size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_wikipedia_benchmark(
        size_tokens=args.size,
        max_articles=args.max_articles,
        n_queries=args.queries,
        page_size=args.page_size,
        seed=args.seed,
    )
