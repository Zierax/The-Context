# Architecture — The Context

## Overview

The Context implements a **Spectral Memory Manifold** architecture that unifies vector similarity, graph topology, temporal dynamics, and hierarchical compression into a single mathematical object.

## Mathematical Foundation

### The Foundational Axiom

All knowledge lives on a low-dimensional Riemannian manifold ℳ embedded in semantic space ℝ^d. The graph Laplacian L IS the metric tensor. Vector similarity IS geodesic distance. Temporal evolution IS heat diffusion.

### The Beacon Hierarchy

```
┌─────────┬────────────────────┬──────────────────────────────────────────────┐
│ LEVEL   │ REPRESENTATION     │ MATHEMATICAL OPERATION                       │
├─────────┼────────────────────┼──────────────────────────────────────────────┤
│ B1      │ Token chunk        │ Tangent vector v_i ∈ ℝ^d (embedding)         │
│         │ (1,000 tokens)     │                                              │
├─────────┼────────────────────┼──────────────────────────────────────────────┤
│ B2      │ 10 B1 beacons      │ Gaussian patch on tangent space:             │
│         │ (10,000 tokens)    │ G_j = (μ_j, Σ_j) where μ_j = mean(v_i),      │
│         │                    │ Σ_j = precision matrix from local covariance │
├─────────┼────────────────────┼──────────────────────────────────────────────┤
│ B3      │ 10 B2 beacons      │ Spectral signature: eigenvalues λ of         │
│         │ (100,000 tokens)   │ normalized Laplacian L_sym = D^(-1/2)LD^(-1/2)│
│         │                    │ λ_2 (Fiedler value) bounds Cheeger constant  │
│         │                    │ → global connectivity preserved              │
└─────────┴────────────────────┴──────────────────────────────────────────────┘
```

## Data Flow

```
Raw Tokens → Tokenizer → Page Slicer (1K chunks)
    ↓
B1 Encoder ──→ Tangent vectors v_i ∈ ℝ^d
    ↓
Entity Extractor ──→ Triples (S,P,O) with Beacon_ID, Page_ID
    ↓
Sparse Adjacency A (CSR) ←── Concept co-occurrence weights
    ↓
B2 Compressor ──→ Gaussian patches G_j = (μ_j, Σ_j)
    ↓
B3 Compressor ──→ Spectral signatures λ(L_sym)
    ↓
┌─────────────────────────────────────────────────────────────┐
│                    SPECTRAL MEMORY MANIFOLD ℳ                │
│  (Unified: vectors + graph + temporal + compression)        │
└─────────────────────────────────────────────────────────────┘
    ↑
Query Q ──→ Embed q ──→ LSH bucket ──→ Concept diffusion ──→
B3 rank ──→ B2 expand ──→ B1 expand ──→ Submodular pack ──→
MCP JSON-RPC response
```

## Component Details

### 1. Math Engine (`math_engine.py`)

The mathematical heart of the system. All functions are vectorized NumPy.

#### Sinusoidal Concept Encoder
```python
def sinusoidal_encode(concepts: list[str], d_model: int = 512, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Deterministic static embedding for raw concept strings.
    Uses a fixed random Fourier feature map:
        z(x) = [cos(w_1·x + b_1), ..., cos(w_d·x + b_d)]
    where w_i ~ N(0, σ²I), b_i ~ Uniform(0, 2π)
    Seeds: w_seed=42, b_seed=43. NEVER change.
    """
```

#### Seeded LSH Voronoi Partition
```python
class SeededLSH:
    def __init__(self, d: int, w: float = 10.0, m: int = 8, seed: int = 42):
        """d: dimension, w: bucket width, m: hash functions, seed: fixed seed"""
    
    def hash_vector(self, x: np.ndarray) -> tuple[int, ...]:
        """Return m-dimensional bucket index."""
    
    def hash_batch(self, X: np.ndarray) -> list[tuple[int, ...]]:
        """Vectorized batch hashing."""
```

#### Graph Laplacian Operations
```python
def normalized_laplacian(A: sp.csr_matrix) -> sp.csr_matrix:
    """Compute L_sym = I - D^(-1/2) A D^(-1/2)"""

def spectral_signature(L_sym: sp.csr_matrix, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Compute top-k eigenvalues of L_sym using scipy.sparse.linalg.eigsh."""

def reconstruct_from_spectral(eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                                n_nodes: int) -> sp.csr_matrix:
    """Reconstruct adjacency from spectral signature."""
```

#### Fokker-Planck Diffusion
```python
def fokker_planck_step(rho: np.ndarray, L: sp.csr_matrix,
                        q: np.ndarray, alpha: float = 0.1,
                        beta: float = 0.5, gamma: float = 0.01) -> np.ndarray:
    """
    One discrete step of Fokker-Planck on the graph.
    ρ_{t+1} = ρ_t + α·(L·ρ_t) + β·(q⊙ρ_t) - γ·ρ_t
    """
```

#### Submodular Context Packing
```python
def submodular_pack(candidates: list[dict], budget: int) -> list[str]:
    """
    Select optimal subset of memory pages within token budget.
    Uses greedy algorithm with (1-1/e) approximation guarantee.
    """
```

### 2. Knowledge Graph (`knowledge_graph.py`)

Deterministic, thread-safe knowledge graph with CSR sparse adjacency.

```python
class DeterministicKnowledgeGraph:
    def __init__(self, d_model: int = 512):
        self.node_to_idx: dict[str, int] = {}
        self.idx_to_node: dict[int, str] = {}
        self.beacon_map: dict[str, list[str]] = {}
        self.A: sp.csr_matrix | None = None
        self.L_sym: sp.csr_matrix | None = None
        self.rho: np.ndarray | None = None
    
    def add_triplet(self, subject: str, predicate: str, object: str,
                     weight: float = 1.0, beacon_id: str = "", page_id: str = "") -> None:
        """Add (S,P,O) to graph. Create nodes if absent."""
    
    def build_laplacian(self) -> None:
        """Recompute L_sym from current A."""
    
    def concept_diffusion(self, query_concepts: list[str], steps: int = 3) -> np.ndarray:
        """Execute heat diffusion from query concepts."""
    
    def get_active_beacons(self, top_k: int = 10) -> list[str]:
        """Return top-k beacon IDs by aggregated ρ·weight."""
```

### 3. Memory Manager (`memory_manager.py`)

Virtual memory tree with hierarchical beacon compression and LRU eviction.

```python
class VirtualMemoryTree:
    def __init__(self, page_size: int = 1000, cache_size: int = 10):
        self.pages: OrderedDict[str, str] = OrderedDict()
        self.beacon_b1: dict[str, np.ndarray] = {}
        self.beacon_b2: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.beacon_b3: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    
    def ingest_stream(self, token_stream: Iterator[str]) -> Iterator[str]:
        """Consume token stream, chunk into pages, yield page IDs."""
    
    def allocate_b1(self, page_id: str, text: str, embedding: np.ndarray) -> str:
        """Store B1 beacon. Evict lowest-score page if cache full."""
    
    def compress_b2(self, b1_beacons: list[str]) -> str:
        """Group 10 B1 beacons, compute Gaussian patch."""
    
    def compress_b3(self, b2_beacons: list[str]) -> str:
        """Group 10 B2 beacons, compute spectral signature."""
```

### 4. Quantum Gate (`quantum_gate.py`)

Orchestration layer for the query collapse pipeline.

```python
class QuantumGate:
    def __init__(self, tree: VirtualMemoryTree, graph: DeterministicKnowledgeGraph,
                  lsh: SeededLSH, d_model: int = 512):
        self.tree = tree
        self.graph = graph
        self.lsh = lsh
    
    def collapse(self, query: str, max_tokens: int = 4096,
                  required_concepts: list[str] | None = None) -> CollapseResult:
        """Execute the full collapse pipeline."""
```

### 5. Entity Extractor (`entity_extractor.py`)

Deterministic heuristic extractor using regex patterns.

```python
class HeuristicExtractor(EntityExtractor):
    def extract(self, text: str) -> list[tuple[str, str, str]]:
        """Extract (Subject, Predicate, Object) triples from text."""
```

### 6. Semantic Compressor (`semantic_compressor.py`)

**NEW**: Compresses natural language into semantic knowledge.

```python
class SemanticKnowledgeCompressor:
    def __init__(self, d_model: int = 512, max_entities: int = 100_000,
                  max_relations: int = 500_000, max_facts: int = 1_000_000):
        self.entities: dict[str, SemanticEntity] = {}
        self.relations: dict[str, SemanticRelation] = {}
        self.facts: dict[str, SemanticFact] = {}
    
    def extract_from_text(self, text: str) -> dict[str, int]:
        """Extract semantic knowledge from text."""
    
    def query_entities(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Query entities by semantic similarity."""
    
    def compute_compression_ratio(self) -> float:
        """Compute the compression ratio achieved."""
```

## Thread Safety

All components use `threading.RLock()` for thread safety:
- `DeterministicKnowledgeGraph._lock`
- `VirtualMemoryTree._lock`
- `SemanticKnowledgeCompressor._lock`

## Determinism Guarantees

1. **Seeded Randomness**: All random operations use fixed seeds (42, 43)
2. **Exact Sparse Operations**: CSR matrix operations are deterministic
3. **No Approximate NN**: LSH with exact bucket matching
4. **No Stochastic Algorithms**: All algorithms are deterministic

## Compression Hierarchy

```
Raw Text (100 MB)
    ↓ Semantic Extraction
Knowledge Graph (5 MB)
    ↓ Spectral Compression
Compressed Knowledge (500 KB)
    ↓ Context Window
Query Results (256 KB)
```

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Entity Extraction | O(n) | Regex-based, single pass |
| Graph Construction | O(E) | E = number of edges |
| Laplacian Build | O(n²) | Sparse, only non-zero entries |
| Spectral Signature | O(n·k·iter) | k eigenvalues, iter iterations |
| Fokker-Planck Step | O(nnz) | Sparse matrix-vector product |
| LSH Hash | O(d·m) | d dimensions, m hash functions |
| Submodular Pack | O(n·B) | n candidates, B budget |

## Memory Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY LAYOUT                             │
├─────────────────────────────────────────────────────────────┤
│ B1 Embeddings (float16):    512 × 2 bytes = 1 KB per page  │
│ B2 Patches (float64):       512 × 8 × 2 = 8 KB per patch   │
│ B3 Signatures (float64):    10 × 8 = 80 bytes per signature │
│ Graph Adjacency (CSR):      ~8 bytes per edge               │
│ LSH Hash Vectors (float64): 512 × 8 = 4 KB                 │
└─────────────────────────────────────────────────────────────┘
```
