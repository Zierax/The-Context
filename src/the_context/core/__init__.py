from .math_engine import (
    SeededLSH, sinusoidal_encode, fokker_planck_step,
    submodular_pack, estimate_token_count, tokenize,
    normalized_laplacian, spectral_signature, reconstruct_from_spectral,
    compute_gaussian_patch,
)
from .knowledge_graph import DeterministicKnowledgeGraph
from .memory_manager import VirtualMemoryTree
