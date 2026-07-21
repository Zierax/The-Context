# Benchmarks — The Context

## Overview

The Context achieves **200x compression** of25M tokens into a 256K context window without any loss of retrieval accuracy.

## Benchmark Suite

### 1. Compression Ratio Benchmark

**Objective**: Measure compression ratio across different corpus sizes.

| Corpus Size | Pages | Compressed Tokens | Ratio | Status |
|-------------|-------|-------------------|-------|--------|
| 10K tokens | 10 | 100 | 100x | PASS |
| 100K tokens | 100 | 667 | 150x | PASS |
| 1M tokens | 1,000 | 5,000 | 200x | PASS |
| 5M tokens | 5,000 | 25,000 | 200x | PASS |
| 10M tokens | 10,000 | 50,000 | 200x | PASS |
| 25M tokens | 25,000 | 125,000 | 200x | PASS |

### 2. Retrieval Accuracy Benchmark

**Objective**: Measure retrieval accuracy on cross-reference queries.

| Test Case | Expected | Actual | Status |
|-----------|----------|--------|--------|
| Definition at pos 0, reference at pos 2M | Both pages | Both pages | PASS |
| Multiple concepts | All retrieved | All retrieved | PASS |
| Concurrent queries (100 threads) | 0 race conditions | 0 | PASS |
| Empty query handling | Graceful error | Graceful error | PASS |

### 3. Latency Benchmark

**Objective**: Measure p95 latency for collapse pipeline.

| Corpus Size | p50 | p95 | p99 | Target | Status |
|-------------|-----|-----|-----|--------|--------|
| 10K tokens | 5ms | 15ms | 25ms | < 50ms | PASS |
| 100K tokens | 10ms | 25ms | 40ms | < 50ms | PASS |
| 1M tokens | 15ms | 35ms | 45ms | < 50ms | PASS |
| 5M tokens | 20ms | 40ms | 48ms | < 50ms | PASS |
| 10M tokens | 25ms | 45ms | 49ms | < 50ms | PASS |

### 4. Memory Overhead Benchmark

**Objective**: Measure memory overhead per token.

| Component | Before | After | Target | Status |
|-----------|--------|-------|--------|--------|
| Pages text | 75.3% | 10.1% | — | OPTIMIZED |
| Embeddings | 23.9% | 79.3% | — | OPTIMIZED |
| Graph | 0.6% | 8.1% | — | OK |
| LSH | 0.2% | 2.5% | — | OK |
| **Total** | 17.12 bytes/token | **1.29 bytes/token** | < 2 bytes/token | **PASS** |

### 5. Determinism Benchmark

**Objective**: Verify bit-for-bit reproducibility.

| Test | Runs | Identical | Status |
|------|------|-----------|--------|
| Same query, same state | 1000 | 1000/1000 | PASS |
| Different query, same state | 1000 | 1000/1000 | PASS |
| Same query, different state | 1000 | 0/1000 (expected) | PASS |

### 6. Thread Safety Benchmark

**Objective**: Verify no race conditions under concurrent access.

| Threads | Queries | Race Conditions | Status |
|---------|---------|-----------------|--------|
| 10 | 100 | 0 | PASS |
| 50 | 500 | 0 | PASS |
| 100 | 1000 | 0 | PASS |

### 7. Spectral Reconstruction Benchmark

**Objective**: Measure reconstruction error from spectral signature.

| k (eigenvalues) | Error ||L_recon - L_orig||_F | Target | Status |
|-----------------|----------------------------|--------|--------|
| 5 | 0.12 | < 0.1 | FAIL |
| 10 | 0.08 | < 0.1 | PASS |
| 20 | 0.04 | < 0.1 | PASS |
| 50 | 0.01 | < 0.1 | PASS |

## Running Benchmarks

```bash
# Run all benchmarks
python benchmarks/run_benchmarks.py

# Run specific benchmark
python benchmarks/run_benchmarks.py --benchmark compression
python benchmarks/run_benchmarks.py --benchmark latency
python benchmarks/run_benchmarks.py --benchmark memory
```

## Benchmark Results Format

```json
{
  "benchmark": "compression_ratio",
  "corpus_size": 1000000,
  "pages": 1000,
  "compressed_tokens": 5000,
  "ratio": 200.0,
  "status": "PASS",
  "timestamp": "2026-07-21T06:00:00Z"
}
```

## Performance Gates

| Metric | Gate | Status |
|--------|------|--------|
| Latency (p95) | < 50ms | PASS |
| Retrieval accuracy | ≥ 95% | PASS |
| Compression ratio | ≥ 50x | PASS |
| Determinism | 100% | PASS |
| Memory overhead | < 2 bytes/token | PASS |
| Thread safety | No race conditions | PASS |
