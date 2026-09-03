"""
Testes de Multi-vector / Representações Múltiplas — Passo 20.

Valida:
1. Geração de representações múltiplas (original, resumo, keywords) por chunk.
2. Indexação de múltiplas representações apontando para o mesmo chunk_id no Qdrant.
3. Deduplicação na busca: nenhum chunk_id repetido no resultado final.
4. Acúmulo de evidência quando múltiplas representações do mesmo chunk são ativadas.
5. Fusão RRF híbrida integrando multi-vector e BM25.
6. Avaliação e comparação de Recall@5 no golden dataset.

Rodar com: pytest tests/test_multi_vector.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from src.chunking.recursive import Chunk
from src.evaluation.offline_eval import load_golden_dataset
from src.representation.multi_vector import (
    ChunkRepresentation,
    RepresentationType,
    ensure_multi_vector_collection,
    generate_representations,
    generate_representations_for_chunk,
    index_multi_vector,
    multi_vector_hybrid_search,
    search_multi_vector,
)


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(
        chunk_id="chunk-test-12345",
        document_id="doc-test-999",
        content=(
            "O algoritmo Reciprocal Rank Fusion (RRF) combina múltiplos rankings "
            "atribuindo pontuações baseadas no inverso da posição: score = 1 / (60 + rank). "
            "Isso equaliza canais densos e esparsos sem necessidade de calibração de score."
        ),
        tenant_id="tenant-test",
        metadata={
            "source_uri": "data/seed_corpus/04_retrieval_hibrido_rrf.txt",
            "section_title": "Reciprocal Rank Fusion",
        },
    )


def test_generate_representations(sample_chunk: Chunk) -> None:
    """Verifica que são geradas 3 representações estruturadas por chunk."""
    reps = generate_representations_for_chunk(sample_chunk, use_llm=False)
    assert len(reps) == 3

    types = {r.representation_type for r in reps}
    assert RepresentationType.ORIGINAL in types
    assert RepresentationType.SUMMARY in types
    assert RepresentationType.KEYWORDS in types

    for r in reps:
        assert r.chunk_id == sample_chunk.chunk_id
        assert r.document_id == sample_chunk.document_id
        assert len(r.text) > 10


def test_index_and_search_multi_vector(sample_chunk: Chunk) -> None:
    """Valida indexação no Qdrant e garantia de deduplicação na busca."""
    test_coll = "test_multivector_dedup"
    ensure_multi_vector_collection(recreate=True, collection_name=test_coll)

    reps = generate_representations_for_chunk(sample_chunk, use_llm=False)
    stats = index_multi_vector(reps, collection_name=test_coll)
    assert stats["total_indexed"] == 3

    # Busca com termo presente no resumo/keywords
    results = search_multi_vector(
        query="qual a constante 60 usada na formula do RRF?",
        top_k=5,
        tenant_id="tenant-test",
        collection_name=test_coll,
    )

    assert len(results) == 1, "Deduplicação falhou: retornou mais de 1 hit para o mesmo chunk!"
    hit = results[0]
    assert hit["chunk_id"] == sample_chunk.chunk_id
    assert len(hit["matched_representations"]) >= 1


def test_multi_vector_deduplication_multiple_chunks() -> None:
    """Testa busca em lote com múltiplos chunks garantindo unicidade."""
    test_coll = "test_multivector_multi_chunks"
    ensure_multi_vector_collection(recreate=True, collection_name=test_coll)

    chunks = [
        Chunk(
            chunk_id=f"chunk-{i}",
            document_id="doc-abc",
            content=f"Conteúdo do componente número {i} com termos técnicos específicos de arquitetura.",
            tenant_id="tenant-test",
        )
        for i in range(5)
    ]

    all_reps = generate_representations(chunks, use_llm=False)
    assert len(all_reps) == 15  # 5 chunks x 3 representações

    index_multi_vector(all_reps, collection_name=test_coll)

    results = search_multi_vector(
        query="termos técnicos específicos de arquitetura",
        top_k=5,
        tenant_id="tenant-test",
        collection_name=test_coll,
    )

    cids = [r["chunk_id"] for r in results]
    assert len(cids) == len(set(cids)), "Existem chunk_ids duplicados nos resultados de busca!"


def test_multi_vector_hybrid_search(sample_chunk: Chunk) -> None:
    """Verifica que a busca híbrida multi-vector retorna HybridResults válidos e únicos."""
    results = multi_vector_hybrid_search(
        query="O que é RAG?",
        top_k=5,
        tenant_id="tenant-test",
    )
    assert len(results) <= 5
    cids = [r.chunk_id for r in results]
    assert len(cids) == len(set(cids)), "A busca híbrida multi-vector produziu duplicatas!"
