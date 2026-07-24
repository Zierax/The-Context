from .math_engine import (
    SeededLSH, sinusoidal_encode, fokker_planck_step,
    submodular_pack, estimate_token_count, tokenize,
    normalized_laplacian, spectral_signature, reconstruct_from_spectral,
    compute_gaussian_patch, svd_compress, svd_decompress, lsh_deduplicate,
    random_projection_compress, random_projection_decompress,
)
from .knowledge_graph import DeterministicKnowledgeGraph
from .memory_manager import VirtualMemoryTree
