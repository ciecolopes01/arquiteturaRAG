"""
Módulo de Adaptive Retrieval e Self-RAG — Passo 29.

Avalia dinamicamente a suficiência e a relevância do contexto recuperado
antes de acionar a geração pelo LLM, prevenindo alucinações.

Fluxo:
1. Executa retrieval híbrido + reranking do estágio 1.
2. Avalia a suficiência via score de confiança do reranker top-1:
   - Se score >= CONFIDENCE_THRESHOLD: SUFFICIENT -> gera resposta com citações.
   - Se RETRY_THRESHOLD <= score < CONFIDENCE_THRESHOLD: RETRY -> reescreve a query
     (expansão de termos) e tenta uma segunda rodada de busca.
   - Se score < RETRY_THRESHOLD (ou retry não melhorou): INSUFFICIENT -> responde
     imediatamente que não encontrou informação suficiente nos documentos, sem alucinar.
3. Retorna SelfRAGResult com decisão, score de confiança e resposta final.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("rag.self_rag")


from src.config import settings
from src.generation.answer import AnswerResult, answer
from src.retrieval.hybrid import HybridResult, hybrid_search
from src.retrieval.query_router import routed_hybrid_search
from src.retrieval.reranker import RerankResult, rerank

# Observabilidade (Passo 14)
try:
    from src.observability.instrumentation import get_current_trace
except ImportError:
    def get_current_trace():
        return None


class SelfRAGDecision(str, Enum):
    SUFFICIENT = "sufficient"        # Contexto relevante e suficiente para responder
    RETRY_RETRIEVAL = "retry"        # Contexto limítrofe; aciona 2ª rodada com query expandida
    INSUFFICIENT = "insufficient"    # Pergunta sem resposta no corpus; recusa explícita sem alucinar


# ---------------------------------------------------------------------------
# Thresholds de Confiança
# ---------------------------------------------------------------------------
# BGE-reranker produz scores logit não-normalizados (tipicamente > 0 para relevantes,
# e < -2.0 para irrelevantes).
CONFIDENCE_THRESHOLD = 0.0    # Score mínimo no top-1 para considerar suficiente
RETRY_THRESHOLD = -2.5        # Abaixo disso é irrelevante; entre -2.5 e 0.0 aciona retry

REFUSAL_MESSAGE = "Não encontrei informações suficientes nos documentos fornecidos para responder a essa pergunta."


@dataclass
class SelfRAGResult:
    """Resultado do pipeline adaptativo Self-RAG."""
    query: str
    decision: SelfRAGDecision
    confidence_score: float
    answer: str
    chunks_used: List[Any] = field(default_factory=list)
    retried: bool = False
    is_refusal: bool = False
    latency_ms: float = 0.0
    source_uris: List[str] = field(default_factory=list)


def _sigmoid(x: float) -> float:
    """Normaliza o score logit do reranker para a escala [0, 1]."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def evaluate_context_sufficiency(
    query: str,
    ranked_chunks: List[Any],
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    retry_threshold: float = RETRY_THRESHOLD,
) -> Tuple[SelfRAGDecision, float]:
    """
    Avalia a suficiência do contexto recuperado para responder à query.
    Utiliza o score do top-1 resultado como sinal primário de confiança.
    """
    if not ranked_chunks:
        return SelfRAGDecision.INSUFFICIENT, 0.0

    top_chunk = ranked_chunks[0]

    # Extrai score dependendo se é RerankResult ou HybridResult
    if isinstance(top_chunk, RerankResult):
        raw_score = top_chunk.rerank_score
    elif hasattr(top_chunk, "rerank_score") and getattr(top_chunk, "rerank_score") is not None:
        raw_score = getattr(top_chunk, "rerank_score")
    elif isinstance(top_chunk, HybridResult):
        # Para HybridResult puro (RRF), mapeia score RRF relativo
        raw_score = (top_chunk.rrf_score * 100.0) - 2.0
    elif isinstance(top_chunk, dict):
        raw_score = top_chunk.get("score", -5.0)
    else:
        raw_score = -5.0

    confidence = round(_sigmoid(raw_score), 4)

    if raw_score >= confidence_threshold:
        return SelfRAGDecision.SUFFICIENT, confidence
    elif raw_score >= retry_threshold:
        return SelfRAGDecision.RETRY_RETRIEVAL, confidence
    else:
        return SelfRAGDecision.INSUFFICIENT, confidence


def rewrite_query_for_retry(query: str) -> str:
    """
    Reescreve e simplifica a query para uma segunda tentativa de retrieval,
    removendo pronomes e ruído pontual para focar nos termos-chave.
    """
    import re
    stopwords = {"qual", "quais", "onde", "como", "quando", "por", "que", "para", "pela", "pelo", "seria", "favor"}
    tokens = [w for w in re.findall(r"\w+", query.lower()) if w not in stopwords and len(w) > 2]
    if tokens:
        return " ".join(tokens)
    return query


def adaptive_answer(
    query: str,
    top_k: int = 5,
    tenant_id: Optional[str] = None,
    principals: Optional[List[str]] = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    retry_threshold: float = RETRY_THRESHOLD,
    allow_retry: bool = True,
) -> SelfRAGResult:
    """
    Ponto de entrada do pipeline Self-RAG:
    1. Retrieval híbrido roteado com filtro de tenant e ACL (top_k ampliado para reranking).
    2. Reranking com cross-encoder BGE.
    3. Avaliação de suficiência:
       - SUFFICIENT: gera resposta.
       - RETRY: reescreve query e reexecuta busca.
       - INSUFFICIENT: recusa educadamente sem alucinar.
    """
    t0 = time.perf_counter()

    # 1. Primeira rodada de retrieval (pool de candidatos para reranker com filtro de ACL)
    candidates, intent, _ = routed_hybrid_search(
        query=query,
        top_k=max(top_k * 3, 15),
        tenant_id=tenant_id,
        principals=principals,
    )

    reranked = []
    if candidates:
        try:
            reranked = rerank(query, candidates, top_k=top_k)
        except Exception as e:
            logger.warning("Falha no reranker do Self-RAG (%s); usando fallback de candidatos brutos", e)
            reranked = candidates[:top_k]

    decision, confidence = evaluate_context_sufficiency(
        query,
        reranked,
        confidence_threshold=confidence_threshold,
        retry_threshold=retry_threshold,
    )

    retried = False

    # 2. Segunda rodada se necessário e permitido
    if decision == SelfRAGDecision.RETRY_RETRIEVAL and allow_retry:
        retried = True
        rewritten_q = rewrite_query_for_retry(query)
        retry_candidates, _, _ = routed_hybrid_search(
            query=rewritten_q,
            top_k=max(top_k * 4, 20),
            tenant_id=tenant_id,
            principals=principals,
        )
        if retry_candidates:
            try:
                reranked_retry = rerank(query, retry_candidates, top_k=top_k)
            except Exception as e:
                logger.warning("Falha no reranker do Self-RAG na 2ª rodada (%s); usando fallback", e)
                reranked_retry = retry_candidates[:top_k]

            decision_retry, conf_retry = evaluate_context_sufficiency(
                query,
                reranked_retry,
                confidence_threshold=confidence_threshold,
                retry_threshold=retry_threshold,
            )
            if decision_retry == SelfRAGDecision.SUFFICIENT or conf_retry > confidence:
                reranked = reranked_retry
                decision = decision_retry
                confidence = conf_retry

    # 3. Decisão final de geração vs recusa
    if decision == SelfRAGDecision.INSUFFICIENT:
        elapsed = (time.perf_counter() - t0) * 1000
        trace = get_current_trace()
        if trace:
            trace.metadata["self_rag_decision"] = decision.value
            trace.metadata["self_rag_confidence"] = confidence
            trace.metadata["self_rag_refusal"] = True

        return SelfRAGResult(
            query=query,
            decision=SelfRAGDecision.INSUFFICIENT,
            confidence_score=confidence,
            answer=REFUSAL_MESSAGE,
            chunks_used=[],
            retried=retried,
            is_refusal=True,
            latency_ms=round(elapsed, 2),
            source_uris=[],
        )

    # 4. Suficiente: Gera resposta através do Claude (answer) usando os chunks validados
    ans_res = answer(
        query=query,
        top_k=top_k,
        tenant_id=tenant_id,
        principals=principals,
        prefetched_chunks=reranked,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    trace = get_current_trace()
    if trace:
        trace.metadata["self_rag_decision"] = SelfRAGDecision.SUFFICIENT.value
        trace.metadata["self_rag_confidence"] = confidence
        trace.metadata["self_rag_refusal"] = False

    return SelfRAGResult(
        query=query,
        decision=SelfRAGDecision.SUFFICIENT,
        confidence_score=confidence,
        answer=ans_res.answer,
        chunks_used=ans_res.chunks_used,
        retried=retried,
        is_refusal=False,
        latency_ms=round(elapsed, 2),
        source_uris=ans_res.source_uris,
    )
