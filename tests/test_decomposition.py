"""
Testes de Query Decomposition — Passo 22.

Valida:
1. Decomposição sintática e por regras de queries compostas e comparativas.
2. Fusão multi-query com RRF ponderado e bônus de cobertura.
3. Execução ponta a ponta da busca decomposta (decomposed_search).
4. Cobertura de múltiplos tópicos em perguntas compostas.

Rodar com: pytest tests/test_decomposition.py -v
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest

from src.retrieval.acl_filter import ANY_PRINCIPAL
from src.retrieval.decomposition import (
    decompose_query,
    decomposed_search,
    multi_query_fusion,
)
from src.retrieval.hybrid import HybridResult


def test_decompose_comparison_query() -> None:
    """Verifica decomposição de perguntas comparativas."""
    q = "Compare a busca vetorial densa e a busca BM25 léxica"
    subs = decompose_query(q, use_llm=False)
    assert len(subs) >= 2
    combined = " ".join(subs).lower()
    assert "busca vetorial densa" in combined
    assert "bm25" in combined


def test_decompose_conjunction_query() -> None:
    """Verifica divisão de perguntas com conjunções explícitas."""
    q = "Como funciona o Tesseract OCR e qual é a função do modelo Claude Vision?"
    subs = decompose_query(q, use_llm=False)
    assert len(subs) >= 2


def test_decompose_atomic_query() -> None:
    """Queries curtas e atômicas não devem ser subdivididas desnecessariamente."""
    q = "O que é RRF?"
    subs = decompose_query(q, use_llm=False)
    assert len(subs) == 1
    assert subs[0] == q


def test_multi_query_fusion_deduplication_and_bonus() -> None:
    """Testa que a fusão multi-query deduplica e pontua melhor chunks multi-tópico."""
    res_q1 = [
        HybridResult(chunk_id="chunk-A", content="Sobre Tesseract OCR", rrf_score=0.016),
        HybridResult(chunk_id="chunk-B", content="Sobre PDF Parser", rrf_score=0.015),
    ]
    res_q2 = [
        HybridResult(chunk_id="chunk-A", content="Sobre Tesseract OCR", rrf_score=0.016),
        HybridResult(chunk_id="chunk-C", content="Sobre Claude Vision", rrf_score=0.014),
    ]

    fused = multi_query_fusion([res_q1, res_q2], top_k=5)
    cids = [r.chunk_id for r in fused]
    assert len(cids) == len(set(cids)), "Existem chunks duplicados na fusão multi-query!"
    # chunk-A apareceu em ambas as subqueries -> deve ser o primeiro
    assert fused[0].chunk_id == "chunk-A"


@pytest.mark.integration
def test_decomposed_search_end_to_end(require_qdrant, require_postgres) -> None:
    """Testa busca decomposta ponta a ponta com retrieval real."""
    q = "Como funciona o Tesseract OCR e qual é a função do modelo Claude Vision no pipeline?"
    results, subqueries = decomposed_search(q, top_k=5, tenant_id="tenant-test", principals=ANY_PRINCIPAL, use_llm=False)
    assert len(subqueries) >= 2
    assert len(results) > 0
    assert results[0].chunk_id is not None
