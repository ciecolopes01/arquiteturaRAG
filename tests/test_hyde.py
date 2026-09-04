"""
Testes de HyDE (Hypothetical Document Embeddings) — Passo 23.

Valida:
1. Geração de resposta/documento hipotético estruturado para queries.
2. Execução de busca puramente baseada no vetor hipotético (hyde_search).
3. Fusão RRF tri-canal: Dense + BM25 + HyDE (hyde_hybrid_search).
4. Avaliação e recuperação no golden dataset.

Rodar com: pytest tests/test_hyde.py -v
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest

from src.retrieval.acl_filter import ANY_PRINCIPAL
from src.retrieval.hyde import (
    generate_hypothetical_document,
    hyde_hybrid_search,
    hyde_search,
)


def test_generate_hypothetical_document() -> None:
    """Verifica que a resposta hipotética é gerada com densidade semântica adequada."""
    q = "Como funciona o reranking com Cross-Encoder?"
    hypo = generate_hypothetical_document(q, use_llm=False)
    assert isinstance(hypo, str)
    assert len(hypo) > 50
    assert "Cross-Encoder" in hypo or "RAG" in hypo or "recuperação" in hypo


@pytest.mark.integration
def test_hyde_search_dense_vector(require_qdrant) -> None:
    """Valida busca dense via embedding do documento hipotético."""
    q = "resolução do OCR e taxa de acurácia"
    hits, hypo = hyde_search(q, top_k=3, tenant_id="tenant-test", principals=ANY_PRINCIPAL, use_llm=False)
    assert len(hypo) > 20
    assert len(hits) > 0
    assert "chunk_id" in hits[0]


@pytest.mark.integration
def test_hyde_hybrid_search_tri_channel(require_qdrant, require_postgres) -> None:
    """Testa fusão tripla Dense + BM25 + HyDE sem duplicatas."""
    q = "Em que página aparece o diagrama do pipeline multimodal RAG?"
    results, hypo = hyde_hybrid_search(
        query=q,
        top_k=5,
        tenant_id="tenant-test",
        principals=ANY_PRINCIPAL,
        use_llm=False,
    )
    assert len(results) > 0
    cids = [r.chunk_id for r in results]
    assert len(cids) == len(set(cids)), "Fusão tri-canal produziu chunks duplicados!"
    assert any("16_arquitetura_multimodal_rag.pdf" in (r.source_uri or "") for r in results)
