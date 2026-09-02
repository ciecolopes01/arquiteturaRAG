"""
Fast path / Full path + routing suave — Passo 13.

Nem toda query precisa do pipeline completo (rerank cross-encoder incluso).
`estimate_complexity` classifica a query com heurística leve — sem chamada
de LLM, que anularia o ganho de latência que o fast path deveria trazer
(mesma razão documentada em data/seed_corpus/09_fast_full_path_routing.pdf,
escrito no Passo 10).

Heurística (score em [0, 1], threshold = 0.6 — mesmo valor do PDF seed):
  - termos comparativos ("diferença entre", "comparado a", "versus"...): +0.4
  - nº de "entidades" (heurística: siglas/acrônimos, termos com maiúscula
    fora do início da frase, tokens com dígitos como "BM25", "k=60"):
    até +0.3
  - comprimento da pergunta (em palavras): até +0.3

Dois caminhos, não dois sistemas (nenhum canal de busca é eliminado):
  - fast: hybrid_search (dense + BM25 + RRF uniforme), sem reranker.
  - full: fast + reranker cross-encoder sobre os candidatos.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import List, Optional

# Observabilidade (Passo 14)
try:
    from src.observability.instrumentation import record_routing_decision
except ImportError:
    def record_routing_decision(path, complexity):
        pass

from src.retrieval.hybrid import HybridResult, hybrid_search
from src.retrieval.reranker import RerankResult, rerank

# ---------------------------------------------------------------------------
# Estimativa de complexidade
# ---------------------------------------------------------------------------

COMPLEXITY_THRESHOLD = 0.6

_COMPARATIVE_RE = re.compile(
    r"diferen[çc]a entre|comparad[oa]s?\s+(a|com)|\bversus\b|\bvs\.?\b|"
    r"\bcompare\b|melhor\s+(que|do que)",
    re.IGNORECASE,
)
# "Entidade": sigla/acrônimo (BM25, RRF, ACL), termo técnico com dígito
# (k=60), ou termo composto com hífen (bi-encoder, cross-encoder) — todos
# sinais de vocabulário específico/técnico na query.
_ENTITY_RE = re.compile(r"\b[A-Z]{2,}[A-Za-z0-9]*\b|\b\w*\d\w*\b|\b\w+-\w+(-\w+)*\b")


def _count_entities(query: str) -> int:
    # Ignora a primeira palavra (maiúscula por ser início de frase, não
    # necessariamente uma entidade).
    rest = query.split(" ", 1)[1] if " " in query else ""
    return len(_ENTITY_RE.findall(rest))


def estimate_complexity(query: str) -> float:
    """
    Heurística leve de complexidade, sem chamada de LLM.

    Retorna score em [0, 1]. >= COMPLEXITY_THRESHOLD -> full path.
    """
    score = 0.0

    if _COMPARATIVE_RE.search(query):
        score += 0.45

    n_entities = _count_entities(query)
    score += min(n_entities / 3, 1.0) * 0.3

    n_words = len(query.split())
    score += min(n_words / 25, 1.0) * 0.3

    return round(min(score, 1.0), 4)


def choose_path(query: str, threshold: float = COMPLEXITY_THRESHOLD) -> str:
    """Retorna "fast" ou "full" para a query."""
    return "full" if estimate_complexity(query) >= threshold else "fast"


# ---------------------------------------------------------------------------
# Resultado da rota
# ---------------------------------------------------------------------------

@dataclass
class RoutedResult:
    path: str                    # "fast" | "full"
    complexity: float
    results: List                # list[HybridResult] (fast) ou list[RerankResult] (full)
    latency_ms: float
    retrieval_candidates: int    # nº de candidatos antes do reranker (0 no fast path)


# ---------------------------------------------------------------------------
# Execução roteada
# ---------------------------------------------------------------------------

def route_search(
    query: str,
    top_k: int = 5,
    tenant_id: Optional[str] = None,
    principals: Optional[List[str]] = None,
    threshold: float = COMPLEXITY_THRESHOLD,
    full_path_candidates: int = 10,
    dense_candidates: int = 20,
    bm25_candidates: int = 20,
) -> RoutedResult:
    """
    Roteia a query para o fast path ou o full path com base na
    complexidade estimada, e executa o caminho escolhido.

    full_path_candidates é deliberadamente menor que dense/bm25_candidates:
    o reranker (BGE-reranker-v2-m3, Passo 9) mede ~1.5-1.8s/chunk em CPU —
    rerankear os 20 candidatos completos custaria ~30s/query. Rerankear só
    os top-10 do RRF equilibra ganho de precisão com latência aceitável.
    """
    complexity = estimate_complexity(query)
    path = "full" if complexity >= threshold else "fast"

    # Registra decisionão de roteamento no trace ativo (Passo 14)
    record_routing_decision(path, complexity)

    t0 = time.time()

    if path == "fast":
        results = hybrid_search(
            query, top_k=top_k, tenant_id=tenant_id, principals=principals,
            dense_candidates=dense_candidates, bm25_candidates=bm25_candidates,
        )
        candidates = 0
    else:
        candidates_list = hybrid_search(
            query, top_k=full_path_candidates, tenant_id=tenant_id, principals=principals,
            dense_candidates=dense_candidates, bm25_candidates=bm25_candidates,
        )
        results = rerank(query, candidates_list, top_k=top_k)
        candidates = len(candidates_list)

    latency_ms = (time.time() - t0) * 1000

    return RoutedResult(
        path=path,
        complexity=complexity,
        results=results,
        latency_ms=latency_ms,
        retrieval_candidates=candidates,
    )
