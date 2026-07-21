# The Context — Hierarchical Beacon Memory System

**v0.2-alpha** | Python 3.10+ | Pure NumPy/SciPy | Deterministic | Thread-Safe

---

## Overview

The Context is a hierarchical beacon memory system that stores text in a multi-level compressed structure (B1→B2→B3) with a knowledge graph for query routing. It retrieves relevant pages from a large corpus using spectral signatures and Fokker-Planck diffusion.

### What It Actually Does

1. **Ingests text** into 1000-token pages (B1 beacons)
2. **Compresses** 10 B1→1 B2 (Gaussian patch), 10 B2→1 B3 (spectral signature)
3. **Extracts knowledge** via regex SVO triples into a knowledge graph
4. **Routes queries** using LSH bucket matching + Fokker-Planck diffusion
5. **Retrieves pages** by expanding B3→B2→B1→pages

### What It Doesn't Do

- **No compression of raw text** — memory overhead is ~1.09 bytes/token (stores more than raw text)
- **No zero-loss reconstruction** — SVO extraction loses information; you cannot reconstruct original text from triples
- **No quantum computing** — all operations are classical NumPy/SciPy
- **No production retrieval accuracy** — only tested on synthetic data

### Honest Metrics (1M token synthetic corpus)

| Metric | Value | Notes |
|--------|-------|-------|
| Memory overhead | 1.09 bytes/token | Stores MORE than raw text (0.52 bytes/token) |
| Compression ratio | 0.48x | Memory is 2x larger than raw text |
| Ingest throughput | 7,155 tok/s | Single-threaded |
| Query latency (p50) | 0.6ms | Single query |
| Query latency (p99) | 2.1ms | Single query |
| Test suite | 80/80 | All pass |
| Determinism | 100% | Same input → same output |

## Quick Start

### Installation

```bash
git clone https://github.com/Zierax/The-Context.git
cd The-Context
pip install -r requirements.txt
```

### Running Tests

```bash
pytest tests/ -v
```

### Running Benchmarks

```bash
python benchmarks/run_benchmarks.py
```

### Running Main Pipeline

```bash
python main.py
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture.

### Component Map

| Component | File | Responsibility |
|-----------|------|----------------|
| Math Engine | `math_engine.py` | All vector/matrix operations |
| Knowledge Graph | `knowledge_graph.py` | CSR adjacency, Laplacian, diffusion |
| Memory Manager | `memory_manager.py` | Virtual memory tree, LRU eviction |
| Query Engine | `query_engine.py` | Query pipeline orchestration |
| Entity Extractor | `entity_extractor.py` | Deterministic SVO extraction |
| MCP Server | `mcp_server.py` | JSON-RPC 2.0 over stdio |
| Semantic Compressor | `semantic_compressor.py` | Knowledge compression |

## How the Beacon Hierarchy Works

```
B1 (1000 tokens)  →  B2 (10 B1s)  →  B3 (10 B2s)
  tangent vector       Gaussian patch    spectral signature
  128-d float16        128-d mu + Sigma  5 eigenvalues + 5 vectors
```

Each level compresses 10 beacons from the level below. The hierarchy enables O(log n) query routing instead of O(n) linear scan.

### Memory Budget (projected at 0.52 bytes/token)

| Corpus Size | Memory | Notes |
|-------------|--------|-------|
| 1M tokens | 0.50 MB | Measured |
| 25M tokens | 12.5 MB | Projected |
| 100M tokens | 50 MB | Projected |
| 500M tokens | 249 MB | Projected |

## Known Limitations

1. **Information loss** — SVO extraction is lossy; original text cannot be reconstructed
2. **Synthetic data only** — No evaluation on real benchmarks (RULER, LongBench, etc.)
3. **Small corpus tested** — Largest tested: 1M tokens; 25M+ is projected
4. **Regex extraction** — Entity extractor uses patterns, not ML; misses many relationships
5. **No evaluation framework** — No standardized benchmarks for comparison

## Next Steps

- [ ] Evaluate on RULER benchmark
- [ ] Evaluate on LongBench
- [ ] Add adversarial distractor tests
- [ ] Measure actual retrieval accuracy on real data
- [ ] Compare against baselines (vanilla RAG, LongNet, etc.)

## License

MIT License

## Author

Zierax
