"""
HyDE (Hypothetical Document Embeddings) — Passo 23.

Gera um documento hipotético plausível via LLM respondendo à query e busca pelo
embedding desse documento hipotético (em vez da query original curta/ambígua),
aproximando a representação vetorial da distribuição dos documentos reais no corpus.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.config import settings
from src.embedding.embed_index import search_dense
from src.embedding.model import embed_query
from src.retrieval.bm25_index import search_bm25
from src.retrieval.hybrid import HybridResult, RRF_K

# Observabilidade (Passo 14)
try:
    from src.observability.instrumentation import get_current_trace
except ImportError:
    def get_current_trace():
        return None


def generate_hypothetical_document(query: str, use_llm: bool = True) -> str:
    """
    Gera um parágrafo hipotético em resposta à query.
    Utiliza Claude 3.5 Sonnet ou fallback heurístico expansivo se offline.
    """
    if use_llm and settings.anthropic_api_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            prompt = (
                "Você é um engenheiro sênior de inteligência artificial. "
                "Escreva uma resposta técnica hipotética, informativa e concisa (3 a 5 frases) "
                "para a seguinte pergunta. Use vocabulário técnico rigoroso apropriado para "
                "arquiteturas RAG, bancos vetoriais, embeddings e processamento de dados corporativos:\n\n"
                f"Pergunta: {query.strip()}\n\n"
                "Resposta Hipotética:"
            )

            resp = client.messages.create(
                model=getattr(settings, "anthropic_vision_model", "claude-3-5-sonnet-20241022"),
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "\n".join(b.text for b in resp.content if hasattr(b, "text")).strip()
            if len(text) > 40:
                return text
        except Exception:
            pass

    # Fallback heurístico determinístico (expansão semântica de domínio RAG)
    domain_terms = [
        "pipeline de recuperação",
        "banco vetorial Qdrant",
        "índice de busca BM25",
        "embeddings densos BGE-M3",
        "arquitetura RAG corporativa",
        "chunking e metadados",
    ]
    return (
        f"Em relação a '{query.strip()}', o sistema opera através de {domain_terms[0]} "
        f"integrado ao {domain_terms[1]} e {domain_terms[2]}. Os componentes utilizam "
        f"{domain_terms[3]} para calcular a similaridade de cosseno e ranquear os trechos mais relevantes "
        f"dentro da {domain_terms[4]}, aplicando {domain_terms[5]}."
    )


def hyde_search(
    query: str,
    top_k: int = 5,
    tenant_id: Optional[str] = None,
    use_llm: bool = True,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Gera documento hipotético e busca chunks usando o embedding dessa hipótese.
    Retorna a lista de candidatos e o texto hipotético gerado.
    """
    hypo_doc = generate_hypothetical_document(query, use_llm=use_llm)
    hits = search_dense(query=hypo_doc, top_k=top_k, tenant_id=tenant_id)
    return hits, hypo_doc


def hyde_hybrid_search(
    query: str,
    top_k: int = 5,
    tenant_id: Optional[str] = None,
    use_llm: bool = True,
    w_dense: float = 1.0,
    w_bm25: float = 1.0,
    w_hyde: float = 1.2,
) -> Tuple[List[HybridResult], str]:
    """
    Fusão RRF de 3 canais:
    1. Dense padrão (vetor da query direta)
    2. BM25 (termos léxicos exatos)
    3. HyDE (vetor da resposta hipotética)

    Equação:
        score(c) = w_dense/(k + rank_dense) + w_bm25/(k + rank_bm25) + w_hyde/(k + rank_hyde)
    """
    t0 = time.perf_counter()

    dense_hits = search_dense(query=query, top_k=top_k * 3, tenant_id=tenant_id)
    bm25_hits = search_bm25(query=query, top_k=top_k * 3, tenant_id=tenant_id)
    hyde_hits, hypo_doc = hyde_search(query=query, top_k=top_k * 3, tenant_id=tenant_id, use_llm=use_llm)

    rrf_scores: Dict[str, float] = {}
    dense_ranks: Dict[str, int] = {}
    bm25_ranks: Dict[str, int] = {}
    hyde_ranks: Dict[str, int] = {}
    chunk_data: Dict[str, Any] = {}

    for rank, h in enumerate(dense_hits, start=1):
        cid = h.get("chunk_id") if isinstance(h, dict) else getattr(h, "chunk_id", None)
        if cid:
            dense_ranks[cid] = rank
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (w_dense / (RRF_K + rank))
            chunk_data[cid] = h

    for rank, h in enumerate(bm25_hits, start=1):
        cid = h.get("chunk_id") if isinstance(h, dict) else getattr(h, "chunk_id", None)
        if cid:
            bm25_ranks[cid] = rank
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (w_bm25 / (RRF_K + rank))
            if cid not in chunk_data:
                chunk_data[cid] = h

    for rank, h in enumerate(hyde_hits, start=1):
        cid = h.get("chunk_id") if isinstance(h, dict) else getattr(h, "chunk_id", None)
        if cid:
            hyde_ranks[cid] = rank
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (w_hyde / (RRF_K + rank))
            if cid not in chunk_data:
                chunk_data[cid] = h

    sorted_cids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)

    results: List[HybridResult] = []
    for cid in sorted_cids[:top_k]:
        ch = chunk_data[cid]
        content = ch.get("content", "") if isinstance(ch, dict) else getattr(ch, "content", "")
        source_uri = ch.get("source_uri") if isinstance(ch, dict) else getattr(ch, "source_uri", None)
        section_title = ch.get("section_title") if isinstance(ch, dict) else getattr(ch, "section_title", None)
        document_title = ch.get("document_title") if isinstance(ch, dict) else getattr(ch, "document_title", None)

        results.append(
            HybridResult(
                chunk_id=cid,
                content=content,
                rrf_score=rrf_scores[cid],
                dense_rank=dense_ranks.get(cid),
                bm25_rank=bm25_ranks.get(cid),
                source_uri=source_uri,
                section_title=section_title,
                document_title=document_title,
            )
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    trace = get_current_trace()
    if trace:
        trace.metadata["hyde_document"] = hypo_doc
        trace.metadata["hyde_latency_ms"] = elapsed_ms

    return results, hypo_doc
