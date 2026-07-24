#!/usr/bin/env python3
"""HotPotQA benchmark for The Context memory system.

Evaluates multi-hop retrieval accuracy on real-world noisy data.
HotPotQA: ~90K multi-hop QA pairs from Wikipedia, requiring reasoning
across 2+ supporting documents.

Metrics:
- F1: Token-level precision × recall / (precision + recall)
- EM: Exact Match (case-insensitive, whitespace-normalized)
- Recall@k: Fraction of relevant pages found in top-k results
- Compression Ratio: Total tokens / Retrieved tokens

Usage:
    PYTHONPATH=src python3 benchmarks/hotpotqa_bench.py --split validation --max-questions 500
    PYTHONPATH=src python3 benchmarks/hotpotqa_bench.py --split validation --max-questions 100 --size 50000
"""
import sys
import os
import time
import json
import re
import tempfile
import warnings
from pathlib import Path
from collections import Counter
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from the_context.core import VirtualMemoryTree, DeterministicKnowledgeGraph
from the_context.query import QueryEngine
from the_context.extraction import HeuristicExtractor
from the_context.core import SeededLSH


# ============================================================================
# HotPotQA Data Loading
# ============================================================================

def load_hotpotqa(split: str = "validation", max_questions: int | None = None) -> list[dict]:
    """Load HotPotQA dataset from HuggingFace.

    Args:
        split: Dataset split ('train' or 'validation')
        max_questions: Maximum number of questions to load (None = all)

    Returns:
        List of question dicts with keys: id, question, answer, supporting_facts, context
    """
    from datasets import load_dataset

    print(f"  Loading HotPotQA {split} split from HuggingFace (streaming)...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, streaming=True)

    questions = []
    for i, item in enumerate(ds):
        if max_questions and i >= max_questions:
            break

        # Build context: list of (title, paragraph) tuples
        context_titles = item["context"]["title"]
        context_sentences = item["context"]["sentences"]

        # Flatten paragraphs
        paragraphs = []
        for title, sentences in zip(context_titles, context_sentences):
            paragraphs.append((title, " ".join(sentences)))

        # Get supporting facts
        supporting_facts = []
        for sf_title, sf_sent_idx in zip(
            item["supporting_facts"]["title"],
            item["supporting_facts"]["sent_id"]
        ):
            # Find the paragraph
            for title, sentences in zip(context_titles, context_sentences):
                if title == sf_title:
                    # sf_sent_idx is the sentence index in that paragraph
                    if sf_sent_idx < len(sentences):
                        supporting_facts.append({
                            "title": title,
                            "sentence": sentences[sf_sent_idx],
                        })
                    break

        questions.append({
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "type": item["type"],
            "level": item["level"],
            "supporting_facts": supporting_facts,
            "context": paragraphs,
            "n_paragraphs": len(paragraphs),
        })

    print(f"  Loaded {len(questions)} questions ({sum(q['n_paragraphs'] for q in questions):,} paragraphs)")
    return questions


# ============================================================================
# F1 / EM Scoring
# ============================================================================

def normalize_answer(text: str) -> str:
    """Normalize answer for comparison (lowercase, strip articles/whitespace)."""
    text = text.lower()
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    # Count common tokens
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_em(prediction: str, ground_truth: str) -> float:
    """Compute Exact Match (1.0 if normalized strings match, 0.0 otherwise)."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def extract_answer_from_pages(pages: list[str], question: str) -> str:
    """Extract the most likely answer from retrieved pages.

    Uses question-type-aware extraction with multiple strategies:
    1. Detect question type (who/where/when/what/how/which)
    2. Extract question entities as anchors
    3. Find answer spans near anchors using NER and patterns
    4. Score and rank candidates
    """
    if not pages:
        return ""

    full_text = " ".join(pages)
    question_lower = question.lower()

    # Detect question type
    question_type = detect_question_type(question_lower)

    # Extract question entities (nouns, proper nouns)
    question_entities = extract_question_entities(question_lower, question)

    # Find sentences containing question entities
    sentences = re.split(r'[.!?]+', full_text)
    candidate_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence.split()) < 2:
            continue

        sentence_lower = sentence.lower()

        # Score based on question entity overlap
        entity_overlap = sum(1 for e in question_entities if e in sentence_lower)
        question_word_overlap = sum(1 for w in question_lower.split() if w in sentence_lower)

        score = entity_overlap * 2.0 + question_word_overlap * 0.5

        if score > 0:
            candidate_sentences.append((score, sentence))

    # Sort by score (highest first)
    candidate_sentences.sort(key=lambda x: x[0], reverse=True)

    # Try extraction strategies in order of specificity
    for _, sentence in candidate_sentences[:5]:  # Top 5 candidates
        # Strategy 1: Question-type specific extraction
        answer = extract_by_question_type(sentence, question_type, question_entities)
        if answer and is_valid_answer(answer, question_type):
            return answer

        # Strategy 2: Copula extraction (X is/was/are Y)
        answer = extract_copula_answer(sentence, question_entities)
        if answer and is_valid_answer(answer, question_type):
            return answer

        # Strategy 3: Named entity extraction
        answer = extract_named_entity(sentence, question_type)
        if answer and is_valid_answer(answer, question_type):
            return answer

        # Strategy 4: Number/date extraction
        answer = extract_number_or_date(sentence, question_type)
        if answer and is_valid_answer(answer, question_type):
            return answer

    # Fallback: Return first meaningful chunk from best sentence
    if candidate_sentences:
        best_sentence = candidate_sentences[0][1]
        return extract_meaningful_chunk(best_sentence)

    return ""


def detect_question_type(question_lower: str) -> str:
    """Detect the type of question being asked."""
    if question_lower.startswith("who ") or "who " in question_lower:
        return "who"
    elif question_lower.startswith("where ") or "where " in question_lower:
        return "where"
    elif question_lower.startswith("when ") or "when " in question_lower:
        return "when"
    elif question_lower.startswith("how many") or "how much" in question_lower:
        return "quantity"
    elif question_lower.startswith("how ") or "how " in question_lower:
        return "how"
    elif question_lower.startswith("which ") or "which " in question_lower:
        return "which"
    elif " or " in question_lower and ("?" in question_lower[-5:]):
        return "choice"  # "Is X or Y?" type questions
    elif question_lower.startswith("is ") or question_lower.startswith("are ") or question_lower.startswith("was ") or question_lower.startswith("were "):
        return "yesno"  # Yes/no questions
    else:
        return "what"


def extract_question_entities(question_lower: str, question: str = "") -> list[str]:
    """Extract meaningful entities from the question (nouns, proper nouns)."""
    # Remove common question words
    stop_words = {"who", "what", "where", "when", "how", "which", "is", "are", "was", "were",
                  "the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by",
                  "does", "do", "did", "has", "have", "had", "can", "could", "would", "should"}

    words = question_lower.split()
    original_words = question.split() if question else words
    entities = []

    for i, word in enumerate(words):
        # Skip question words and stop words
        if word in stop_words or len(word) <= 2:
            continue

        # Include word and possibly next word if it's a multi-word entity
        entity = word
        if i + 1 < len(words) and words[i + 1] not in stop_words:
            # Check if next word starts with capital letter in original question
            if i + 1 < len(original_words) and original_words[i + 1][0:1].isupper():
                entity = word + " " + words[i + 1]

        entities.append(entity)

    return entities


def extract_by_question_type(sentence: str, question_type: str, question_entities: list[str]) -> str:
    """Extract answer based on question type."""
    sentence_lower = sentence.lower()

    if question_type == "who":
        # Look for person names (capitalized words, possibly with titles)
        person_patterns = [
            r'(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b',  # Full names
            r'(?:born|named|called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        for pattern in person_patterns:
            matches = re.findall(pattern, sentence)
            for match in matches:
                if len(match.split()) <= 4 and not any(w in match.lower() for w in ["the", "and", "or"]):
                    return match

    elif question_type == "where":
        # Look for locations (cities, countries, places)
        location_patterns = [
            r'(?:in|at|near|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,\s*[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*)\b',  # "City, Country"
            r'(?:located|situated|based)\s+(?:in|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        for pattern in location_patterns:
            matches = re.findall(pattern, sentence)
            for match in matches:
                if len(match.split()) <= 5:
                    return match

    elif question_type == "when":
        # Look for dates, years, time periods
        date_patterns = [
            r'\b(\d{4})\b',  # Year
            r'(?:in|on|from|during)\s+(\w+\s+\d{4})',  # "Month Year"
            r'(?:in|on|from|during)\s+(\d{1,2}\s+\w+\s+\d{4})',  # "DD Month YYYY"
            r'(\d{4}\s*[-–]\s*\d{4})',  # Range "2000-2010"
            r'(?:since|from|until|till)\s+(\w+\s+\d{4})',
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            if matches:
                return matches[0]

    elif question_type == "quantity":
        # Look for numbers with units
        quantity_patterns = [
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|thousand|hundred)',
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:people|inhabitants|residents|citizens|miles|km|km²|m²)',
            r'(\d+(?:,\d{3})*(?:\.\d+)?)',
        ]
        for pattern in quantity_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple):
                    return " ".join(matches[0])
                return matches[0]

    elif question_type == "choice":
        # For "X or Y" questions, look for the mentioned entity
        for entity in question_entities:
            if entity in sentence_lower:
                # Find the full named entity in the sentence
                pattern = rf'\b{re.escape(entity)}\b'
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    # Extract surrounding context
                    start = max(0, match.start() - 20)
                    end = min(len(sentence), match.end() + 20)
                    chunk = sentence[start:end]
                    # Clean up to get just the entity
                    words = chunk.split()
                    entity_words = []
                    for w in words:
                        if w.lower() == entity.split()[0]:
                            entity_words = []
                            for ew in entity.split():
                                entity_words.append(ew)
                            break
                    if entity_words:
                        return " ".join(entity_words)

    elif question_type == "yesno":
        # Look for yes/no indicators
        yes_patterns = [r'\b(yes)\b', r'\b(indeed)\b', r'\b(correct)\b', r'\b(true)\b']
        no_patterns = [r'\b(no)\b', r'\b(not)\b', r'\b(false)\b']

        for pattern in yes_patterns:
            if re.search(pattern, sentence_lower):
                return "yes"

        for pattern in no_patterns:
            if re.search(pattern, sentence_lower):
                return "no"

    return ""


def extract_copula_answer(sentence: str, question_entities: list[str]) -> str:
    """Extract answer from copula constructions (X is/was/are Y)."""
    # Patterns for "X is/was/are Y"
    copula_patterns = [
        r'(?:is|was|are|were)\s+(?:a|an|the)?\s*(.+?)(?:\.|,|\(|$)',
        r'(?:known as|called|named)\s+(.+?)(?:\.|,|\(|$)',
        r'(?:consists of|includes|contains)\s+(.+?)(?:\.|,|\(|$)',
    ]

    for pattern in copula_patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Clean up
            answer = re.sub(r'\s*\(.*?\)\s*', ' ', answer)  # Remove parenthetical
            answer = re.sub(r'\s*,\s*.*$', '', answer)  # Take first clause
            answer = re.sub(r'\s+and\s+.*$', '', answer)  # Take first item
            answer = answer.strip()

            if len(answer.split()) <= 10 and len(answer) > 1:
                return answer

    return ""


def extract_named_entity(sentence: str, question_type: str) -> str:
    """Extract named entities from sentence."""
    # Find all named entities (capitalized word sequences)
    entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', sentence)

    # Filter based on question type
    if question_type == "who":
        # Prefer person-like entities (2-3 words)
        person_entities = [e for e in entities if 1 <= len(e.split()) <= 3]
        if person_entities:
            return person_entities[0]
    elif question_type == "where":
        # Prefer location-like entities
        loc_entities = [e for e in entities if len(e.split()) <= 3]
        if loc_entities:
            return loc_entities[0]
    else:
        # Return first meaningful entity
        for entity in entities:
            if len(entity.split()) <= 3 and entity.lower() not in ["the", "and", "or", "but"]:
                return entity

    return ""


def extract_number_or_date(sentence: str, question_type: str) -> str:
    """Extract numbers or dates from sentence."""
    # Look for years
    years = re.findall(r'\b(1[89]\d{2}|20[0-2]\d)\b', sentence)
    if years and question_type in ["when", "quantity"]:
        return years[0]

    # Look for other numbers
    numbers = re.findall(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b', sentence)
    if numbers:
        # Prefer larger numbers for quantity questions
        if question_type == "quantity":
            return max(numbers, key=lambda x: float(x.replace(',', '')))
        return numbers[0]

    return ""


def is_valid_answer(answer: str, question_type: str) -> bool:
    """Check if extracted answer is valid for the question type."""
    if not answer or len(answer) < 2:
        return False

    answer_lower = answer.lower()

    # Invalid answers (too generic)
    invalid_answers = {"the", "a", "an", "it", "this", "that", "they", "he", "she",
                       "his", "her", "its", "their", "was", "is", "are", "were",
                       "has", "have", "had", "do", "does", "did"}
    if answer_lower in invalid_answers:
        return False

    # Type-specific validation
    if question_type == "who":
        # Should contain at least one capitalized word
        if not any(w[0].isupper() for w in answer.split() if w):
            return False
    elif question_type == "when":
        # Should contain a number or date-related word
        if not re.search(r'\d|year|month|day|century|era|period', answer_lower):
            return False
    elif question_type == "where":
        # Should look like a location
        if len(answer.split()) > 5:
            return False
    elif question_type == "quantity":
        # Should contain a number
        if not re.search(r'\d', answer):
            return False

    return True


def extract_meaningful_chunk(sentence: str) -> str:
    """Extract a meaningful chunk from a sentence as fallback."""
    # Take first N words
    words = sentence.split()
    if len(words) <= 10:
        return sentence

    # Try to find a natural break point
    for i in range(min(10, len(words) - 1)):
        if words[i].endswith(','):
            return " ".join(words[:i + 1]).rstrip(',')

    return " ".join(words[:8]) + "..."


# ============================================================================
# Recall@k Metric
# ============================================================================

def compute_recall_at_k(
    retrieved_pages: list[str],
    supporting_facts: list[dict],
    k: int = 5,
) -> float:
    """Compute Recall@k: fraction of supporting facts found in top-k pages.

    A supporting fact is considered "found" if its sentence appears
    (partially or fully) in any of the top-k retrieved pages.
    """
    if not supporting_facts:
        return 1.0  # No facts to find

    retrieved_top_k = retrieved_pages[:k]
    retrieved_text = " ".join(retrieved_top_k).lower()

    found = 0
    for sf in supporting_facts:
        sf_sentence = sf["sentence"].lower()
        # Check if the supporting fact sentence (or a significant portion)
        # appears in the retrieved text
        sf_words = set(sf_sentence.split())
        retrieved_words = set(retrieved_text.split())
        overlap = len(sf_words & retrieved_words)

        # Consider found if >50% of words overlap
        if overlap > len(sf_words) * 0.5:
            found += 1

    return found / len(supporting_facts)


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_hotpotqa_benchmark(
    split: str = "validation",
    max_questions: int = 500,
    target_size_tokens: int | None = None,
    page_size: int = 2000,
    seed: int = 42,
) -> dict:
    """Run HotPotQA benchmark on The Context memory system.

    Args:
        split: HotPotQA split ('validation' or 'train')
        max_questions: Maximum questions to evaluate
        target_size_tokens: Target corpus size in tokens (None = use all context)
        page_size: Page size for VirtualMemoryTree
        seed: Random seed for reproducibility

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*70}")
    print(f"  HotPotQA Benchmark — The Context Memory System")
    print(f"{'='*70}")

    # Load data
    questions = load_hotpotqa(split=split, max_questions=max_questions)

    # Extract all context paragraphs
    all_paragraphs = []
    for q in questions:
        for title, text in q["context"]:
            all_paragraphs.append((title, text))

    print(f"\n  Total paragraphs: {len(all_paragraphs):,}")

    # If target size specified, limit paragraphs
    if target_size_tokens:
        total_tokens = sum(len(text.split()) for _, text in all_paragraphs)
        if total_tokens > target_size_tokens:
            # Sample paragraphs to hit target size
            ratio = target_size_tokens / total_tokens
            n_keep = max(1, int(len(all_paragraphs) * ratio))
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(all_paragraphs), size=n_keep, replace=False)
            all_paragraphs = [all_paragraphs[i] for i in sorted(indices)]
            print(f"  Sampled to {len(all_paragraphs):,} paragraphs (~{target_size_tokens:,} tokens)")

    # Build corpus: each paragraph is one page (simpler, faster)
    corpus_tokens = []
    page_to_paragraph = {}  # page_id -> paragraph (title, text)
    pid_counter = 0

    for title, text in all_paragraphs:
        words = text.split()
        corpus_tokens.extend(words)
        pid_counter += 1
        page_to_paragraph[pid_counter] = (title, text)

    total_tokens = len(corpus_tokens)
    print(f"  Corpus: {total_tokens:,} tokens in {pid_counter:,} pages")

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

    # Run queries
    print(f"\n  Running {len(questions)} queries...")
    results = {
        "f1_scores": [],
        "em_scores": [],
        "recall_at_5": [],
        "recall_at_1": [],
        "latency_ms": [],
        "tokens_used": [],
        "compression_ratios": [],
        "answers": [],
    }

    t0 = time.perf_counter()
    for i, q in enumerate(questions):
        if (i + 1) % 50 == 0:
            elapsed = (time.perf_counter() - t0) * 1000
            avg_ms = elapsed / (i + 1)
            print(f"    [{i+1}/{len(questions)}] avg {avg_ms:.0f}ms/query")

        # Query the system
        q_start = time.perf_counter()
        result = engine.collapse(query=q["question"], max_tokens=4096)
        q_ms = (time.perf_counter() - q_start) * 1000

        # Extract answer from retrieved pages
        predicted_answer = extract_answer_from_pages(result.pages, q["question"])

        # Compute metrics
        f1 = compute_f1(predicted_answer, q["answer"])
        em = compute_em(predicted_answer, q["answer"])
        r5 = compute_recall_at_k(result.pages, q["supporting_facts"], k=5)
        r1 = compute_recall_at_k(result.pages, q["supporting_facts"], k=1)

        results["f1_scores"].append(f1)
        results["em_scores"].append(em)
        results["recall_at_5"].append(r5)
        results["recall_at_1"].append(r1)
        results["latency_ms"].append(q_ms)
        results["tokens_used"].append(result.tokens_used)
        results["compression_ratios"].append(result.compression_ratio)
        results["answers"].append({
            "id": q["id"],
            "question": q["question"][:100],
            "gold": q["answer"],
            "predicted": predicted_answer[:100],
            "f1": f1,
            "em": em,
        })

    total_ms = (time.perf_counter() - t0) * 1000

    # Compute summary statistics
    summary = {
        "benchmark": "hotpotqa",
        "split": split,
        "n_questions": len(questions),
        "corpus_tokens": total_tokens,
        "corpus_pages": len(pids),
        "ingest_ms": ingest_ms,
        "extract_ms": extract_ms,
        "total_query_ms": total_ms,
        "avg_query_ms": np.mean(results["latency_ms"]),
        "p50_query_ms": np.percentile(results["latency_ms"], 50),
        "p95_query_ms": np.percentile(results["latency_ms"], 95),
        "p99_query_ms": np.percentile(results["latency_ms"], 99),
        "f1_mean": np.mean(results["f1_scores"]),
        "f1_std": np.std(results["f1_scores"]),
        "f1_median": np.median(results["f1_scores"]),
        "em_mean": np.mean(results["em_scores"]),
        "recall_at_1": np.mean(results["recall_at_1"]),
        "recall_at_5": np.mean(results["recall_at_5"]),
        "avg_tokens_used": np.mean(results["tokens_used"]),
        "avg_compression": np.mean(results["compression_ratios"]),
        "memory_bytes": len(tree.beacon_b1) * d * 2 + len(tree.beacon_b2) * (d * 2 + d * 8),
        "compression_ratio_tokens": total_tokens / max(np.mean(results["tokens_used"]), 1),
        "by_type": {},
        "by_level": {},
    }

    # Breakdown by question type
    for q, f1, em in zip(questions, results["f1_scores"], results["em_scores"]):
        qtype = q["type"]
        if qtype not in summary["by_type"]:
            summary["by_type"][qtype] = {"f1": [], "em": [], "count": 0}
        summary["by_type"][qtype]["f1"].append(f1)
        summary["by_type"][qtype]["em"].append(em)
        summary["by_type"][qtype]["count"] += 1

    for qtype in summary["by_type"]:
        summary["by_type"][qtype]["f1_mean"] = np.mean(summary["by_type"][qtype]["f1"])
        summary["by_type"][qtype]["em_mean"] = np.mean(summary["by_type"][qtype]["em"])
        del summary["by_type"][qtype]["f1"]
        del summary["by_type"][qtype]["em"]

    # Breakdown by difficulty level
    for q, f1, em in zip(questions, results["f1_scores"], results["em_scores"]):
        level = q["level"]
        if level not in summary["by_level"]:
            summary["by_level"][level] = {"f1": [], "em": [], "count": 0}
        summary["by_level"][level]["f1"].append(f1)
        summary["by_level"][level]["em"].append(em)
        summary["by_level"][level]["count"] += 1

    for level in summary["by_level"]:
        summary["by_level"][level]["f1_mean"] = np.mean(summary["by_level"][level]["f1"])
        summary["by_level"][level]["em_mean"] = np.mean(summary["by_level"][level]["em"])
        del summary["by_level"][level]["f1"]
        del summary["by_level"][level]["em"]

    # Print results
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    print(f"  Questions: {summary['n_questions']}")
    print(f"  Corpus: {summary['corpus_tokens']:,} tokens, {summary['corpus_pages']:,} pages")
    print(f"  ")
    print(f"  F1:              {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"  Exact Match:     {summary['em_mean']:.4f}")
    print(f"  Recall@1:        {summary['recall_at_1']:.4f}")
    print(f"  Recall@5:        {summary['recall_at_5']:.4f}")
    print(f"  ")
    print(f"  Avg Query Time:  {summary['avg_query_ms']:.1f}ms")
    print(f"  P50 Query Time:  {summary['p50_query_ms']:.1f}ms")
    print(f"  P95 Query Time:  {summary['p95_query_ms']:.1f}ms")
    print(f"  ")
    print(f"  Avg Tokens Used: {summary['avg_tokens_used']:.0f}")
    print(f"  Compression:     {summary['avg_compression']:.2f}x")
    print(f"  Token Reduction: {(1 - summary['avg_tokens_used']/summary['corpus_tokens'])*100:.1f}%")
    print(f"  ")
    print(f"  By Question Type:")
    for qtype, stats in summary["by_type"].items():
        print(f"    {qtype:20s}: F1={stats['f1_mean']:.4f} EM={stats['em_mean']:.4f} (n={stats['count']})")
    print(f"  ")
    print(f"  By Difficulty Level:")
    for level, stats in summary["by_level"].items():
        print(f"    {level:20s}: F1={stats['f1_mean']:.4f} EM={stats['em_mean']:.4f} (n={stats['count']})")

    # Compare with competitors
    print(f"\n{'='*70}")
    print(f"  COMPETITOR COMPARISON")
    print(f"{'='*70}")
    print(f"  {'System':<25s} {'F1':>8s} {'EM':>8s} {'Recall@5':>10s}")
    print(f"  {'-'*53}")
    print(f"  {'The Context':<25s} {summary['f1_mean']:>8.4f} {summary['em_mean']:>8.4f} {summary['recall_at_5']:>10.4f}")
    print(f"  {'Cognee (24Q)':<25s} {'0.84':>8s} {'0.69':>8s} {'N/A':>10s}")
    print(f"  {'Graphiti':<25s} {'0.74':>8s} {'N/A':>8s} {'N/A':>10s}")
    print(f"  {'LightRAG':<25s} {'0.67':>8s} {'N/A':>8s} {'N/A':>10s}")
    print(f"  {'Mem0':<25s} {'0.54':>8s} {'N/A':>8s} {'N/A':>10s}")

    # Cleanup
    import shutil
    shutil.rmtree(persist, ignore_errors=True)

    # Save results
    summary["top_answers"] = results["answers"][:20]

    output_path = Path("results") / "hotpotqa_benchmark.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HotPotQA benchmark for The Context")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"])
    parser.add_argument("--max-questions", type=int, default=500, help="Max questions to evaluate")
    parser.add_argument("--size", type=int, default=None, help="Target corpus size in tokens")
    parser.add_argument("--page-size", type=int, default=1000, help="Page size for VirtualMemoryTree")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_hotpotqa_benchmark(
        split=args.split,
        max_questions=args.max_questions,
        target_size_tokens=args.size,
        page_size=args.page_size,
        seed=args.seed,
    )
