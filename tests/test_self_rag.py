"""
Testes unitários e de integração para Adaptive Retrieval e Self-RAG — Passo 29.

Valida:
1. Avaliação de suficiência e cálculo de confiança (SelfRAGDecision).
2. Reescrita de query para 2ª rodada de retrieval em casos limítrofes.
3. Critério de "pronto": Reconhecimento de perguntas fora do corpus com recusa
   explícita ("não sei / não encontrei"), prevenindo alucinações em 100% dos testes.
4. Preservação da resposta e citações em perguntas dentro do corpus (in-domain).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from src.generation.self_rag import (
    CONFIDENCE_THRESHOLD,
    REFUSAL_MESSAGE,
    RETRY_THRESHOLD,
    SelfRAGDecision,
    adaptive_answer,
    evaluate_context_sufficiency,
    rewrite_query_for_retry,
)
from src.retrieval.reranker import RerankResult


def test_evaluate_context_sufficiency_thresholds():
    """Valida a transição de estados de decisão conforme o score do reranker."""
    # 1. Caso suficiente (score positivo alto)
    high_chunk = RerankResult(
        chunk_id="c1",
        content="Retrieval-Augmented Generation combina busca vetorial e LLMs.",
        rerank_score=3.8,
        original_rank=1,
        new_rank=1,
    )
    dec, conf = evaluate_context_sufficiency("o que é rag", [high_chunk])
    assert dec == SelfRAGDecision.SUFFICIENT
    assert conf > 0.90

    # 2. Caso limítrofe (score intermediário -> retry)
    mid_chunk = RerankResult(
        chunk_id="c2",
        content="Documento sobre infraestrutura geral de banco de dados.",
        rerank_score=-1.2,
        original_rank=1,
        new_rank=1,
    )
    dec, conf = evaluate_context_sufficiency("algoritmo rrf", [mid_chunk])
    assert dec == SelfRAGDecision.RETRY_RETRIEVAL
    assert 0.10 <= conf <= 0.50

    # 3. Caso insuficiente (score muito baixo -> recusa)
    low_chunk = RerankResult(
        chunk_id="c3",
        content="Instruções de segurança predial e extintores.",
        rerank_score=-6.5,
        original_rank=1,
        new_rank=1,
    )
    dec, conf = evaluate_context_sufficiency("receita culinária", [low_chunk])
    assert dec == SelfRAGDecision.INSUFFICIENT
    assert conf < 0.05

    # 4. Caso vazio
    dec_empty, conf_empty = evaluate_context_sufficiency("query", [])
    assert dec_empty == SelfRAGDecision.INSUFFICIENT
    assert conf_empty == 0.0


def test_query_rewriting_for_retry():
    """Valida a simplificação e remoção de pronomes/stopwords para retry."""
    query = "Por favor, como que funciona exatamente o algoritmo RRF na prática?"
    rewritten = rewrite_query_for_retry(query)
    assert "por" not in rewritten
    assert "funciona" in rewritten
    assert "rrf" in rewritten


def test_out_of_domain_queries_refusal():
    """
    Critério de "pronto" do Passo 29:
    Nas perguntas sem resposta no corpus, o sistema responde "não sei / não encontrei"
    em vez de alucinar uma resposta inventada.
    """
    out_of_domain_questions = [
        "Qual é a receita caseira para preparar um bolo de cenoura com cobertura de chocolate?",
        "Quem foi o artilheiro da Copa do Mundo de futebol masculino de 1970 no México?",
        "Qual a velocidade máxima de cruzeiro e altitude do caça militar F-22 Raptor?",
        "Quais são os principais sintomas clínicos da gripe aviária em mamíferos marinhos?",
        "Como plantar, adubar e podar orquídeas raras em clima tropical úmido?",
    ]

    refusals = 0
    for q in out_of_domain_questions:
        res = adaptive_answer(
            query=q,
            tenant_id="tenant-test",
            top_k=3,
            confidence_threshold=0.0,
            retry_threshold=-2.0,
        )
        # O sistema deve recusar responder por falta de contexto
        is_refused = res.is_refusal or res.decision == SelfRAGDecision.INSUFFICIENT or REFUSAL_MESSAGE in res.answer
        if is_refused:
            refusals += 1

    taxa_recusa = refusals / len(out_of_domain_questions)
    print(f"\n[Self-RAG] Taxa de identificação correta fora de domínio: {taxa_recusa * 100:.1f}% ({refusals}/{len(out_of_domain_questions)})")
    assert taxa_recusa >= 0.80, f"Taxa de recusa abaixo do esperado: {taxa_recusa}"
    assert refusals == len(out_of_domain_questions), "Todas as perguntas fora do corpus devem ser recusadas."


def test_in_domain_query_is_answered():
    """Valida que perguntas relevantes cobertas pelo corpus são identificadas como SUFFICIENT."""
    in_domain_q = "Qual a constante k padrão utilizada no algoritmo RRF na fusão de rankings?"
    res = adaptive_answer(
        query=in_domain_q,
        tenant_id="tenant-test",
        top_k=3,
        confidence_threshold=-1.0,
    )
    assert res.decision == SelfRAGDecision.SUFFICIENT
    assert res.is_refusal is False
    assert len(res.answer) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
