"""
Pacote de representações múltiplas (Multi-vector) — Passo 20.
"""
from src.representation.multi_vector import (
    ChunkRepresentation,
    RepresentationType,
    generate_representations,
    index_multi_vector,
    multi_vector_hybrid_search,
    search_multi_vector,
)

__all__ = [
    "ChunkRepresentation",
    "RepresentationType",
    "generate_representations",
    "index_multi_vector",
    "search_multi_vector",
    "multi_vector_hybrid_search",
]
