"""
Query Decomposition — Passo 22.

Decompõe perguntas complexas e multi-facetadas em sub-perguntas atômicas,
executa retrieval híbrido para cada uma e funde os resultados via RRF multi-query,
garantindo alta cobertura antes do reranking.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.config import settings
from src.retrieval.hybrid import HybridResult, RRF_K, hybrid_search
from src.retrieval.query_router import routed_hybrid_search

# Observabilidade (Passo 14)
try:
    from src.observability.instrumentation import get_current_trace
except ImportError:
    def get_current_trace():
        return None


# ---------------------------------------------------------------------------
# Decomposição de Queries (LLM + Fallback Heurístico)
# ---------------------------------------------------------------------------

_SPLIT_CONJUNCTIONS = re.compile(
    r"\s+(?:e\s+tamb[ée]m|al[ée]m\s+de|bem\s+como|e\s+qual\s+[ée]|e\s+como|e\s+por\s+que|e\s+onde|;\s*|\.\s+)\s*",
    re.IGNORECASE,
)

_COMPARE_PATTERNS = [
    re.compile(r"^(?:compare|qual\s+a\s+diferen[çc]a\s+entre)\s+(.+?)\s+e\s+(.+?)(?:\?|$)", re.IGNORECASE),
    re.compile(r"^(.+?)\s+versus\s+(.+?)(?:\?|$)", re.IGNORECASE),
]


def _heuristic_decompose(query: str) -> List[str]:
    """Decompõe queries usando padrões sintáticos de comparação e conjunção."""
    q = query.strip()

    # 1. Padrões de comparação: "compare X e Y" ou "diferença entre X e Y"
    for pat in _COMPARE_PATTERNS:
        m = pat.search(q)
        if m:
            item1 = m.group(1).strip()
            item2 = m.group(2).strip()
            return [
                f"O que é {item1} e quais são suas características?",
                f"O que é {item2} e quais são suas características?",
            ]

    # 2. Divisão por conjunções de perguntas duplas
    parts = _SPLIT_CONJUNCTIONS.split(q)
    subqueries = [p.strip().rstrip("?").strip() + "?" for p in parts if len(p.strip()) > 10]
    if len(subqueries) > 1:
        return subqueries

    # Se não for composta, retorna a própria query
    return [q]


def decompose_query(query: str, use_llm: bool = True) -> List[str]:
    """
    Decompõe uma query em sub-perguntas atômicas e complementares.
    Usa Claude com prompt rigoroso e recorre à heurística sintática caso offline.
    """
    query_clean = query.strip()
    if len(query_clean.split()) <= 4:
        return [query_clean]

    if use_llm and settings.anthropic_api_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            prompt = (
                "Você é um especialista em recuperação de informação. Decomponha a pergunta abaixo "
                "em 2 a 3 sub-perguntas atômicas, diretas e independentes para busca vetorial. "
                "Se a pergunta já for simples e atômica, retorne apenas ela mesma.\n\n"
                f"Pergunta: {query_clean}\n\n"
                "Responda ESTRITAMENTE em formato JSON com a lista de strings, sem qualquer texto adicional:\n"
                '["sub-pergunta 1", "sub-pergunta 2"]'
            )

            resp = client.messages.create(
                model=getattr(settings, "anthropic_vision_model", "claude-3-5-sonnet-20241022"),
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = "\n".join(b.text for b in resp.content if hasattr(b, "text")).strip()

            # Tenta extrair array JSON
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                items = json.loads(m.group(0))
                if isinstance(items, list) and items:
                    return [str(it).strip() for it in items if str(it).strip()]
        except Exception:
            pass

    return _heuristic_decompose(query_clean)


# ---------------------------------------------------------------------------
# Multi-Query Retrieval & RRF Fusion
# ---------------------------------------------------------------------------

def multi_query_fusion(
    subquery_results: List[List[HybridResult]],
    k: int = RRF_K,
    top_k: int = 5,
) -> List[HybridResult]:
    """
    Funde os resultados recuperados para cada sub-pergunta via RRF multi-query:
        score(chunk) = Σ_{q}  1 / (k + rank_q(chunk))
    """
    scores: Dict[str, float] = {}
    best_ranks: Dict[str, int] = {}
    occurrences: Dict[str, int] = {}
    chunk_meta: Dict[str, HybridResult] = {}

    for q_idx, results in enumerate(subquery_results):
        for rank, res in enumerate(results, start=1):
            cid = res.chunk_id
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))
            occurrences[cid] = occurrences.get(cid, 0) + 1
            if cid not in best_ranks or rank < best_ranks[cid]:
                best_ranks[cid] = rank
            if cid not in chunk_meta:
                chunk_meta[cid] = res

    # Bônus de relevância para chunks encontrados em múltiplas sub-perguntas
    for cid in scores:
        if occurrences[cid] > 1:
            scores[cid] *= 1.0 + (0.1 * (occurrences[cid] - 1))

    sorted_cids = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)

    final_results: List[HybridResult] = []
    for cid in sorted_cids[:top_k]:
        base = chunk_meta[cid]
        final_results.append(
            HybridResult(
                chunk_id=cid,
                content=base.content,
                rrf_score=scores[cid],
                dense_rank=base.dense_rank,
                bm25_rank=base.bm25_rank,
                source_uri=base.source_uri,
                section_title=base.section_title,
                document_title=base.document_title,
            )
        )

    return final_results


def decomposed_search(
    query: str,
    top_k: int = 5,
    tenant_id: Optional[str] = None,
    use_llm: bool = True,
    use_router: bool = True,
) -> Tuple[List[HybridResult], List[str]]:
    """
    Executa o pipeline completo de Query Decomposition:
    1. Decompõe a query em sub-perguntas.
    2. Executa busca híbrida (com ou sem router) para cada sub-pergunta.
    3. Funde todos os resultados com RRF multi-query e bônus de cobertura.
    """
    t0 = time.perf_counter()
    subqueries = decompose_query(query, use_llm=use_llm)

    subquery_results: List[List[HybridResult]] = []
    for sq in subqueries:
        if use_router:
            res, _, _ = routed_hybrid_search(sq, top_k=top_k * 2, tenant_id=tenant_id)
        else:
            res = hybrid_search(sq, top_k=top_k * 2, tenant_id=tenant_id)
        subquery_results.append(res)

    fused = multi_query_fusion(subquery_results, top_k=top_k)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    trace = get_current_trace()
    if trace:
        trace.metadata["decomposed_subqueries"] = subqueries
        trace.metadata["decomposition_latency_ms"] = elapsed_ms

    return fused, subqueries
