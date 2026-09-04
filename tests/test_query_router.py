"""
Testes do Query Router Completo — Passo 21.

Valida:
1. Classificação de intenções de busca (fact_lookup, table_lookup, semantic_explanation, multimodal).
2. Cálculo de pesos de canal e garantia do piso mínimo (min_weight_floor).
3. Fusão RRF ponderada.
4. Execução de busca híbrida com routing suave.
5. Avaliação offline segmentada no golden dataset.

Rodar com: pytest tests/test_query_router.py -v
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")

from src.evaluation.offline_eval import load_golden_dataset
from src.retrieval.acl_filter import ANY_PRINCIPAL
from src.retrieval.query_router import (
    MIN_WEIGHT_FLOOR,
    QueryIntent,
    classify_intent,
    get_channel_weights,
    routed_hybrid_search,
    weighted_rrf,
)


@pytest.mark.parametrize(
    "query,expected_intent",
    [
        ("Quanto custa a licença do modelo de embedding por mês?", QueryIntent.TABLE_LOOKUP),
        ("Qual foi o MRR e ARR no mês de Janeiro na planilha financeira?", QueryIntent.TABLE_LOOKUP),
        ("Qual o valor do orçamento do projeto Alpha na planilha?", QueryIntent.TABLE_LOOKUP),
        ("Em que página aparece o diagrama do pipeline multimodal RAG?", QueryIntent.MULTIMODAL),
        ("Qual a resolução e gráfico de acurácia do OCR no relatório?", QueryIntent.MULTIMODAL),
        ("Explique como funciona a fusão de rankings no algoritmo RRF", QueryIntent.SEMANTIC_EXPLANATION),
        ("Qual a diferença entre chunking recursivo e semântico?", QueryIntent.SEMANTIC_EXPLANATION),
        ("Quais são as vantagens e trade-offs da busca híbrida?", QueryIntent.SEMANTIC_EXPLANATION),
        ("Qual a porta padrão do serviço Qdrant no docker-compose?", QueryIntent.FACT_LOOKUP),
        ("Qual a versão do Tesseract utilizada?", QueryIntent.FACT_LOOKUP),
    ],
)
def test_classify_intent(query: str, expected_intent: QueryIntent) -> None:
    """Valida a precisão da classificação de intenção."""
    assert classify_intent(query) == expected_intent


def test_get_channel_weights_respects_floor() -> None:
    """Verifica que o piso mínimo é respeitado e nenhum canal cai a zero."""
    for intent in QueryIntent:
        weights = get_channel_weights(intent)
        assert weights["dense"] >= MIN_WEIGHT_FLOOR
        assert weights["bm25"] >= MIN_WEIGHT_FLOOR

    # Testa com pesos customizados extremos (zero ou negativo)
    custom = {
        QueryIntent.FACT_LOOKUP: {"dense": 0.0, "bm25": -0.5}
    }
    w = get_channel_weights(QueryIntent.FACT_LOOKUP, boosts=custom, min_weight_floor=0.15)
    assert w["dense"] == 0.15
    assert w["bm25"] == 0.15


def test_weighted_rrf_unification() -> None:
    """Valida cálculo de fusão do weighted_rrf integrado ao reciprocal_rank_fusion."""
    dense = [{"chunk_id": "c1", "content": "dense doc", "score": 0.9}]
    bm25 = [{"chunk_id": "c2", "content": "bm25 doc", "score": 12.5}]
    graph = [{"chunk_id": "c1", "content": "graph doc", "score": 1.0}]

    weights = {"dense": 1.5, "bm25": 1.0, "graph": 0.5}
    fused = weighted_rrf(
        dense_hits=dense,
        bm25_hits=bm25,
        graph_hits=graph,
        weights=weights,
        top_k=2,
    )
    assert len(fused) == 2
    # c1 tem votos no canal dense e no canal graph -> maior score RRF
    assert fused[0].chunk_id == "c1"
    assert fused[0].rrf_score > fused[1].rrf_score
    assert fused[0].dense_rank == 1
    assert fused[0].graph_rank == 1


@pytest.mark.integration
def test_routed_hybrid_search_end_to_end(require_qdrant, require_postgres) -> None:
    """Executa busca roteada ponta a ponta e verifica retorno de intenção e resultados."""
    results, intent, weights = routed_hybrid_search(
        query="Em que página aparece o fluxo de indexação de imagens?",
        top_k=3,
        tenant_id="tenant-test",
        principals=ANY_PRINCIPAL,
    )
    assert intent == QueryIntent.MULTIMODAL
    assert weights["dense"] > weights["bm25"]
    assert len(results) > 0
    assert results[0].chunk_id is not None


@pytest.mark.integration
def test_evaluate_golden_dataset_with_router(require_qdrant, require_postgres) -> None:
    """Avalia o golden dataset usando routed_hybrid_search."""
    dataset = load_golden_dataset()
    sample = [d for d in dataset if d["id"] in ["gd-44", "gd-45", "gd-46", "gd-47", "gd-48"]]
    assert len(sample) == 5

    hits = 0
    for item in sample:
        q = item["question"]
        expected = set(item["expected_chunk_ids"])
        results, _, _ = routed_hybrid_search(q, top_k=5, tenant_id="tenant-test", principals=ANY_PRINCIPAL)
        retrieved_ids = {r.chunk_id for r in results}
        if expected & retrieved_ids:
            hits += 1

    recall = hits / len(sample)
    assert recall >= 0.8, f"Recall com router abaixo da meta: {recall}"
    assert hits == 5
