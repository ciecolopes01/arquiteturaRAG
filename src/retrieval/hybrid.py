"""
Retrieval Híbrido + RRF uniforme — Passo 6.

Combina os dois canais de busca (dense Qdrant + BM25 Postgres) usando
Reciprocal Rank Fusion (RRF) com k=60 e pesos iguais (uniforme).

Fórmula (conforme arquitetura-rag-revisada.md §5):
    score(chunk) = Σ_i  weight_i / (k + rank_i(chunk))

Nesta fase (V1 uniforme):
    weight_i = 1  para todos os canais
    k        = 60 (constante canônica da literatura)

Canais:
    - dense  : busca vetorial coseno no Qdrant (BGE-M3)
    - bm25   : busca full-text tsvector no Postgres

Filtro de tenant aplicado em ambos os canais ANTES do ranking.

Módulo principal:
    hybrid_search(query, top_k, tenant_id) -> list[HybridResult]

HybridResult é um dataclass simples com:
    chunk_id, content, rrf_score, sources (quais canais contribuíram),
    dense_rank, bm25_rank, source_uri, section_title, document_title
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Observabilidade (Passo 14)
try:
    from src.observability.instrumentation import get_current_trace
except ImportError:
    def get_current_trace():
        return None

from src.embedding.embed_index import search_dense
from src.retrieval.bm25_index import search_bm25

# ---------------------------------------------------------------------------
# Constantes RRF
# ---------------------------------------------------------------------------

RRF_K = 60          # constante padrão — raramente precisa ser ajustada
RRF_WEIGHT = 1.0    # peso uniforme para ambos os canais nesta fase


# ---------------------------------------------------------------------------
# Schema de resultado
# ---------------------------------------------------------------------------

@dataclass
class HybridResult:
    """Resultado fundido pelo RRF a partir de múltiplos canais."""
    chunk_id: str
    content: str
    rrf_score: float
    source_uri: str = ""
    document_title: str = ""
    section_title: str = ""
    dense_rank: Optional[int] = None    # posição no ranking dense (1-indexed)
    bm25_rank: Optional[int] = None     # posição no ranking BM25 (1-indexed)
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    parent_chunk_id: Optional[str] = None
    parent_content: Optional[str] = None

    @property
    def sources(self) -> list[str]:
        """Canais que contribuíram para este resultado."""
        s = []
        if self.dense_rank is not None:
            s.append("dense")
        if self.bm25_rank is not None:
            s.append("bm25")
        return s


# ---------------------------------------------------------------------------
# RRF core
# ---------------------------------------------------------------------------

def _rrf_score(ranks: Dict[str, int], k: int = RRF_K, weight: float = RRF_WEIGHT) -> float:
    """
    Calcula o score RRF de um chunk dado seu rank em cada canal.

    ranks: {"dense": 1, "bm25": 3}  — rank 1-indexed
    """
    return sum(weight / (k + r) for r in ranks.values())


def reciprocal_rank_fusion(
    ranked_lists: Dict[str, List[dict]],
    top_k: int,
    k: int = RRF_K,
    weight: float = RRF_WEIGHT,
) -> List[HybridResult]:
    """
    Funde múltiplas listas de ranking via RRF.

    Parâmetros:
    - ranked_lists : {"dense": [...], "bm25": [...]}
                     cada lista é uma lista de dicts com "chunk_id", "content", "score"
    - top_k        : número de resultados finais
    - k            : constante de suavização (padrão 60)
    - weight       : peso de cada canal (padrão 1.0 uniforme)

    Retorna lista de HybridResult ordenada por rrf_score desc.
    """
    # Acumula scores e metadados por chunk_id
    scores: Dict[str, float] = {}
    meta: Dict[str, dict] = {}      # chunk_id -> payload do primeiro canal que o viu
    ranks: Dict[str, Dict[str, int]] = {}   # chunk_id -> {canal: rank}
    raw_scores: Dict[str, Dict[str, float]] = {}

    for channel_name, results in ranked_lists.items():
        for rank_0, item in enumerate(results):
            cid = item["chunk_id"]
            rank_1 = rank_0 + 1  # 1-indexed

            contribution = weight / (k + rank_1)
            scores[cid] = scores.get(cid, 0.0) + contribution

            if cid not in meta:
                meta[cid] = item
            if cid not in ranks:
                ranks[cid] = {}
            ranks[cid][channel_name] = rank_1

            if cid not in raw_scores:
                raw_scores[cid] = {}
            raw_scores[cid][channel_name] = item.get("score", 0.0)

    # Ordena por score RRF desc e pega top_k
    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]

    results_out = []
    for cid in sorted_ids:
        m = meta[cid]
        r = ranks[cid]
        rs = raw_scores[cid]
        results_out.append(
            HybridResult(
                chunk_id=cid,
                content=m.get("content", ""),
                rrf_score=round(scores[cid], 6),
                source_uri=m.get("source_uri", ""),
                document_title=m.get("document_title", ""),
                section_title=m.get("section_title", ""),
                dense_rank=r.get("dense"),
                bm25_rank=r.get("bm25"),
                dense_score=rs.get("dense"),
                bm25_score=rs.get("bm25"),
                parent_chunk_id=m.get("parent_chunk_id"),
                parent_content=m.get("parent_content"),
            )
        )

    return results_out


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    top_k: int = 5,
    tenant_id: Optional[str] = None,
    principals: Optional[List[str]] = None,
    dense_candidates: int = 20,
    bm25_candidates: int = 20,
) -> List[HybridResult]:
    """
    Retrieval híbrido: dense (Qdrant) + BM25 (Postgres) fundidos via RRF.

    Parâmetros:
    - query            : pergunta do usuário.
    - top_k            : número de resultados finais após fusão.
    - tenant_id        : filtro de tenant aplicado em AMBOS os canais.
    - principals       : filtro de ACL (Passo 11) aplicado em AMBOS os canais —
                          pré-filtro, antes do ranking. None = sem filtro de ACL
                          (compatibilidade com chamadas anteriores ao Passo 11).
    - dense_candidates : quantos candidatos buscar no canal dense antes da fusão.
    - bm25_candidates  : quantos candidatos buscar no canal BM25 antes da fusão.

    Buscamos mais candidatos do que top_k para que a fusão RRF tenha material
    suficiente — chunks que aparecem em ambos os canais sobem no ranking.

    Retorna lista de HybridResult ordenada por rrf_score desc.
    """
    # Busca em paralelo (sequencial nesta fase — paralelismo real entra em V1.1)
    dense_results = search_dense(
        query, top_k=dense_candidates, tenant_id=tenant_id, principals=principals,
    )
    bm25_results = search_bm25(
        query, top_k=bm25_candidates, tenant_id=tenant_id, principals=principals,
    )

    ranked_lists = {}
    if dense_results:
        ranked_lists["dense"] = dense_results
    if bm25_results:
        ranked_lists["bm25"] = bm25_results

    if not ranked_lists:
        return []

    # Fusão RRF — mede latência separadamente (embed+qdrant já foram medidos)
    _t_fusion = time.time()
    result = reciprocal_rank_fusion(ranked_lists, top_k=top_k)
    _fusion_ms = (time.time() - _t_fusion) * 1000
    trace = get_current_trace()
    if trace is not None:
        trace.record_stage(
            "retrieval.fusion", _fusion_ms,
            n_dense=len(dense_results),
            n_bm25=len(bm25_results),
            top_k=top_k,
        )
    return result
