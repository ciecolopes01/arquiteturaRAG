"""
Reranker local BGE-reranker-v2-m3 — Passo 9.

Cross-encoder que reordena os candidatos do retrieval híbrido antes de
passar ao LLM. Usa FlagEmbedding (mesma lib do BGE-M3 do Passo 4).

Por que cross-encoder melhora o ranking?
- O bi-encoder (BGE-M3) codifica query e chunk SEPARADAMENTE → aproximação.
- O cross-encoder recebe (query, chunk) JUNTOS → vê interações entre termos,
  muito mais preciso, mas 10-100x mais lento — ok pois opera só em top-N.

Singleton de módulo: modelo carregado uma única vez (mesmo padrão do Passo 4).

Funções:
  rerank(query, chunks, top_k) -> list[RerankResult]
  reranker_dimension()          -> str  (nome do modelo)

RerankResult:
  chunk_id, content, rerank_score, original_rank, new_rank,
  source_uri, section_title, document_title

Dependências: FlagEmbedding>=1.2.10 (já instalado no Passo 4), torch
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Union

# Observabilidade (Passo 14)
try:
    from src.observability.instrumentation import get_current_trace
except ImportError:
    def get_current_trace():
        return None

from src.config import settings
from src.retrieval.hybrid import HybridResult

# ---------------------------------------------------------------------------
# Singleton do modelo
# ---------------------------------------------------------------------------

_reranker = None
_reranker_model_name: str = ""


def _get_reranker():
    """Carrega FlagReranker uma única vez (singleton de módulo)."""
    global _reranker, _reranker_model_name

    if _reranker is not None and _reranker_model_name == settings.reranker_model:
        return _reranker

    try:
        from FlagEmbedding import FlagReranker
    except ImportError:
        raise ImportError(
            "FlagEmbedding nao instalado.\n"
            "Execute: pip install FlagEmbedding torch"
        )

    print(
        f"[Reranker] Carregando {settings.reranker_model} "
        f"em {settings.embedding_device}... (primeira vez pode demorar)"
    )
    t0 = time.time()
    _reranker = FlagReranker(
        settings.reranker_model,
        use_fp16=(settings.embedding_device != "cpu"),
        device=settings.embedding_device,
    )
    _reranker_model_name = settings.reranker_model
    print(f"[Reranker] Modelo carregado em {time.time() - t0:.1f}s")
    return _reranker


# ---------------------------------------------------------------------------
# Schema de resultado
# ---------------------------------------------------------------------------

@dataclass
class RerankResult:
    """Chunk após reranking com scores originais e novos."""
    chunk_id: str
    content: str
    rerank_score: float       # score do cross-encoder (logit, pode ser negativo)
    original_rank: int        # posição antes do reranking (1-indexed)
    new_rank: int             # posição após reranking (1-indexed)
    source_uri: str = ""
    section_title: str = ""
    document_title: str = ""
    rrf_score: float = 0.0    # score RRF original, para comparação
    parent_chunk_id: Optional[str] = None
    parent_content: Optional[str] = None

    @property
    def rank_change(self) -> int:
        """Positivo = subiu, negativo = desceu."""
        return self.original_rank - self.new_rank


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    chunks: List[HybridResult],
    top_k: int = 5,
) -> List[RerankResult]:
    """
    Reordena os chunks com cross-encoder BGE-reranker-v2-m3.

    Parâmetros:
    - query  : pergunta original do usuário.
    - chunks : lista de HybridResult do hybrid_search (candidatos).
    - top_k  : número de resultados retornados após reranking.

    Retorna lista de RerankResult ordenada por rerank_score desc.

    Nota: se chunks estiver vazio, retorna lista vazia sem carregar o modelo.
    """
    if not chunks:
        return []

    reranker = _get_reranker()

    # Pares (query, chunk_content) para o cross-encoder
    pairs = [[query, c.content] for c in chunks]

    t0 = time.time()
    scores = reranker.compute_score(pairs, normalize=False)
    elapsed = time.time() - t0

    # scores pode ser float (1 par) ou list[float]
    if isinstance(scores, float):
        scores = [scores]

    print(
        f"[Reranker] {len(chunks)} chunks rerankeados em {elapsed:.2f}s "
        f"({elapsed/max(len(chunks),1)*1000:.0f}ms/chunk)"
    )

    # Registra no trace ativo (Passo 14)
    trace = get_current_trace()
    if trace is not None:
        trace.record_stage(
            "retrieval.reranking",
            elapsed * 1000,
            n_chunks=len(chunks),
            ms_per_chunk=round(elapsed / max(len(chunks), 1) * 1000, 1),
        )

    # Ordena por score desc e seleciona top_k
    scored = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    original_ranks = {c.chunk_id: idx for idx, c in enumerate(chunks, 1)}

    results = []
    for new_rank, (chunk, score) in enumerate(scored, 1):
        original_rank = original_ranks.get(chunk.chunk_id, new_rank)
        results.append(
            RerankResult(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                rerank_score=round(score, 4),
                original_rank=original_rank,
                new_rank=new_rank,
                source_uri=chunk.source_uri,
                section_title=chunk.section_title,
                document_title=chunk.document_title,
                rrf_score=chunk.rrf_score,
                parent_chunk_id=getattr(chunk, "parent_chunk_id", None),
                parent_content=getattr(chunk, "parent_content", None),
            )
        )

    return results


def reranker_model_name() -> str:
    """Retorna o nome do modelo de reranking configurado."""
    return settings.reranker_model
