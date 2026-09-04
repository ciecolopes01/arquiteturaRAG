"""
Multi-vector / Representações Múltiplas — Passo 20.

Permite indexar várias representações vetoriais do mesmo chunk:
1. Texto original bruto (fiel aos termos do documento).
2. Resumo sintético (linguagem natural densa, cobre queries conceituais).
3. Keywords e entidades (facilita casamento semântico e vocabulário alternativo).
4. Pergunta hipotética (HyDE antecipado no chunk / reverse HyDE).

Na busca:
- Qdrant retorna os candidatos mais próximos em qualquer representação.
- Deduplicação inteligente: hits em representações diferentes do mesmo `chunk_id`
  são combinados como evidência adicional (acumulação de score e preservação
  do melhor rank), nunca retornando o mesmo chunk duplicado.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.chunking.recursive import Chunk
from src.config import settings
from src.embedding.model import embed_query, embed_texts, embedding_dimension
from src.retrieval.bm25_index import search_bm25
from src.retrieval.hybrid import RRF_K, HybridResult


class RepresentationType(str, Enum):
    ORIGINAL = "original"
    SUMMARY = "summary"
    KEYWORDS = "keywords"
    HYPOTHETICAL = "hypothetical"


@dataclass
class ChunkRepresentation:
    """Uma representação textual de um chunk a ser indexada separadamente."""

    id: str
    chunk_id: str
    document_id: str
    representation_type: RepresentationType
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tenant_id: Optional[str] = None
    acl: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Geradores de Representações
# ---------------------------------------------------------------------------

def _extract_fallback_summary(text: str) -> str:
    """Extrai um resumo heurístico do texto quando LLM não está disponível."""
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    if not lines:
        return text[:150]
    # Pega as primeiras 2 sentenças
    clean = " ".join(lines)
    parts = re.split(r"(?<=[.!?])\s+", clean)
    summary = " ".join(parts[:2]).strip()
    return summary if len(summary) > 20 else clean[:180]


def _extract_fallback_keywords(text: str) -> str:
    """Extrai palavras-chave heurísticas por frequência e termos capitalizados."""
    stopwords = {
        "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "é", "com",
        "não", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como",
        "mas", "ao", "ele", "das", "à", "seu", "sua", "ou", "quando", "muito",
        "nos", "já", "eu", "também", "só", "pelo", "pela", "até", "isso", "ela",
        "entre", "depois", "sem", "mesmo", "aos", "seus", "quem", "nas", "me",
        "esse", "eles", "você", "essa", "num", "nem", "suas", "meu", "às", "minha",
        "têm", "numa", "pelos", "elas", "qual", "nós", "lhe", "deles", "essas",
        "esses", "pelas", "este", "dele", "tu", "te", "vocês", "vos", "lhes",
        "meus", "minhas", "teu", "tua", "teus", "tuas", "nosso", "nossa", "nossos",
        "nossas", "dela", "delas", "esta", "estes", "estas", "aquele", "aquela",
        "aqueles", "aquelas", "isto", "aquilo", "estou", "está", "estamos", "estão",
        "seção", "figura", "página", "tabela", "imagem", "texto", "descrição",
    }
    words = re.findall(r"\b[A-Za-z0-9Á-ú\-_]{3,}\b", text)
    filtered = [w for w in words if w.lower() not in stopwords]
    # Termos únicos preservando ordem
    seen = set()
    keywords = []
    for w in filtered:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl)
            keywords.append(w)
        if len(keywords) >= 10:
            break
    return ", ".join(keywords) if keywords else text[:100]


def generate_representations_for_chunk(
    chunk: Chunk,
    use_llm: bool = True,
) -> List[ChunkRepresentation]:
    """
    Gera representações múltiplas para um determinado chunk:
    1. ORIGINAL: o conteúdo bruto do chunk.
    2. SUMMARY: resumo conciso de 1-2 sentenças.
    3. KEYWORDS: termos-chave e conceitos centrais.
    """
    reps: List[ChunkRepresentation] = []

    # 1. Representação Original
    reps.append(
        ChunkRepresentation(
            id=str(uuid.uuid4()),
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            representation_type=RepresentationType.ORIGINAL,
            text=chunk.content,
            metadata=chunk.metadata,
            tenant_id=chunk.tenant_id,
            acl=chunk.acl,
        )
    )

    summary_text = None
    keywords_text = None

    # Tenta usar Claude se habilitado e configurado
    if use_llm and settings.anthropic_api_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

            # Prompt conjunto para economizar chamadas de API
            prompt = (
                "Você é um engenheiro de recuperação de informação. Para o texto abaixo:\n"
                "1. Escreva um RESUMO em 1 ou 2 frases densas em conceitos.\n"
                "2. Liste 6 a 10 PALAVRAS-CHAVE separadas por vírgula.\n\n"
                f"Texto:\n{chunk.content[:1500]}\n\n"
                "Formato da resposta:\n"
                "RESUMO: <resumo>\n"
                "KEYWORDS: <keywords>"
            )

            resp = client.messages.create(
                model=getattr(settings, "anthropic_vision_model", "claude-3-5-sonnet-20241022"),
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            out = "\n".join(b.text for b in resp.content if hasattr(b, "text"))

            m_sum = re.search(r"RESUMO:\s*(.+?)(?=\nKEYWORDS:|$)", out, re.DOTALL | re.IGNORECASE)
            m_kw = re.search(r"KEYWORDS:\s*(.+?)$", out, re.DOTALL | re.IGNORECASE)
            if m_sum:
                summary_text = m_sum.group(1).strip()
            if m_kw:
                keywords_text = m_kw.group(1).strip()
        except Exception:
            pass

    # Fallbacks estruturados se LLM não gerou
    if not summary_text:
        summary_text = _extract_fallback_summary(chunk.content)
    if not keywords_text:
        keywords_text = _extract_fallback_keywords(chunk.content)

    # 2. Representação SUMMARY
    reps.append(
        ChunkRepresentation(
            id=str(uuid.uuid4()),
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            representation_type=RepresentationType.SUMMARY,
            text=f"[Resumo do Chunk]: {summary_text}",
            metadata={**chunk.metadata, "is_representation": True},
            tenant_id=chunk.tenant_id,
            acl=chunk.acl,
        )
    )

    # 3. Representação KEYWORDS
    reps.append(
        ChunkRepresentation(
            id=str(uuid.uuid4()),
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            representation_type=RepresentationType.KEYWORDS,
            text=f"[Palavras-Chave e Conceitos]: {keywords_text}",
            metadata={**chunk.metadata, "is_representation": True},
            tenant_id=chunk.tenant_id,
            acl=chunk.acl,
        )
    )

    return reps


def generate_representations(
    chunks: List[Chunk],
    use_llm: bool = True,
) -> List[ChunkRepresentation]:
    """Gera representações múltiplas para uma lista inteira de chunks."""
    all_reps: List[ChunkRepresentation] = []
    for c in chunks:
        all_reps.extend(generate_representations_for_chunk(c, use_llm=use_llm))
    return all_reps


# ---------------------------------------------------------------------------
# Indexação Multi-Vector no Qdrant
# ---------------------------------------------------------------------------

MULTI_VECTOR_COLLECTION = "rag_chunks_multivector"


def ensure_multi_vector_collection(
    recreate: bool = False,
    collection_name: str = MULTI_VECTOR_COLLECTION,
) -> None:
    """Garante que a coleção de representações múltiplas existe no Qdrant."""
    client = QdrantClient(url=settings.qdrant_url)
    dim = embedding_dimension()
    exists = client.collection_exists(collection_name)

    if exists and recreate:
        client.delete_collection(collection_name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


def index_multi_vector(
    representations: List[ChunkRepresentation],
    collection_name: str = MULTI_VECTOR_COLLECTION,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Gera embeddings de cada representação e indexa no Qdrant.
    Todos os pontos apontam para o mesmo chunk_id original via payload.
    """
    ensure_multi_vector_collection(collection_name=collection_name)
    client = QdrantClient(url=settings.qdrant_url)

    texts = [r.text for r in representations]
    embeddings = embed_texts(texts)

    points = []
    for r, vec in zip(representations, embeddings):
        acl_principals = []
        if isinstance(r.acl, list):
            for e in r.acl:
                if isinstance(e, dict) and e.get("principal"):
                    acl_principals.append(e["principal"])
                elif isinstance(e, str):
                    acl_principals.append(e)

        payload = {
            "representation_id": r.id,
            "chunk_id": r.chunk_id,
            "document_id": r.document_id,
            "representation_type": r.representation_type.value,
            "text": r.text,
            "tenant_id": r.tenant_id or "",
            "acl": r.acl,
            "acl_principals": acl_principals,
            **r.metadata,
        }
        points.append(
            PointStruct(
                id=r.id,
                vector=vec,
                payload=payload,
            )
        )

    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=collection_name,
            points=points[i : i + batch_size],
        )

    return {
        "total_indexed": len(points),
        "collection": collection_name,
        "representations_per_chunk": len(points) / max(1, len({r.chunk_id for r in representations})),
    }


# ---------------------------------------------------------------------------
# Busca Multi-Vector com Deduplicação
# ---------------------------------------------------------------------------

def search_multi_vector(
    query: str,
    top_k: int = 5,
    tenant_id: Optional[str] = None,
    principals: Optional[List[str]] = None,
    collection_name: str = MULTI_VECTOR_COLLECTION,
    fetch_multiplier: int = 4,
) -> List[Dict[str, Any]]:
    """
    Executa busca densa sobre todas as representações no Qdrant e realiza
    DEDUPLICAÇÃO por `chunk_id`.

    Múltiplos hits no mesmo chunk somam evidência (preserva o melhor score e
    lista os tipos de representação correspondentes). Retorna chunks únicos.
    """
    client = QdrantClient(url=settings.qdrant_url)
    if not client.collection_exists(collection_name):
        return []

    q_vec = embed_query(query)

    conditions = []
    if tenant_id:
        conditions.append(
            FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))
        )
    if principals:
        conditions.append(
            FieldCondition(key="acl_principals", match=MatchAny(any=principals))
        )
    query_filter = Filter(must=conditions) if conditions else None

    # Busca pool ampliado para acomodar representações múltiplas
    limit = max(top_k * fetch_multiplier, 20)

    results = client.query_points(
        collection_name=collection_name,
        query=q_vec,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )

    # Deduplicação agrupada por chunk_id
    grouped: Dict[str, Dict[str, Any]] = {}

    for rank, point in enumerate(results.points, start=1):
        payload = point.payload or {}
        cid = payload.get("chunk_id", str(point.id))
        rtype = payload.get("representation_type", "original")
        score = point.score

        if cid not in grouped:
            grouped[cid] = {
                "chunk_id": cid,
                "score": score,
                "best_rank": rank,
                "best_representation": rtype,
                "matched_representations": [rtype],
                "content": payload.get("content") or payload.get("text", ""),
                "document_id": payload.get("document_id"),
                "source_uri": payload.get("source_uri"),
                "section_title": payload.get("section_title"),
                "tenant_id": payload.get("tenant_id"),
                "metadata": payload,
            }
        else:
            # Chunk já encontrado em outra representação -> combina evidência
            entry = grouped[cid]
            entry["matched_representations"].append(rtype)
            # Bônus suave de consistência multi-representação (+5% no score do melhor)
            if score > entry["score"]:
                entry["score"] = score
                entry["best_representation"] = rtype
            entry["score"] = entry["score"] * 1.05

    # Ordena chunks únicos pelo score combinado
    unique_sorted = sorted(grouped.values(), key=lambda x: x["score"], reverse=True)
    return unique_sorted[:top_k]


# ---------------------------------------------------------------------------
# Busca Híbrida Multi-Vector (Dense Multi-Vector + BM25 via RRF)
# ---------------------------------------------------------------------------

def multi_vector_hybrid_search(
    query: str,
    top_k: int = 5,
    tenant_id: Optional[str] = None,
    principals: Optional[List[str]] = None,
    collection_name: str = MULTI_VECTOR_COLLECTION,
) -> List[HybridResult]:
    """
    Fusão RRF entre:
    1. Canal Dense com representações múltiplas (deduplicado).
    2. Canal Léxico BM25 (Postgres).
    Garante que nenhum chunk apareça duplicado no ranking final.
    """
    dense_hits = search_multi_vector(
        query=query,
        top_k=top_k * 2,
        tenant_id=tenant_id,
        principals=principals,
        collection_name=collection_name,
    )

    bm25_hits = search_bm25(query=query, top_k=top_k * 2, tenant_id=tenant_id, principals=principals)

    rrf_scores: Dict[str, float] = {}
    dense_ranks: Dict[str, int] = {}
    bm25_ranks: Dict[str, int] = {}
    chunk_data: Dict[str, Dict[str, Any]] = {}

    for rank, h in enumerate(dense_hits, start=1):
        cid = h["chunk_id"]
        dense_ranks[cid] = rank
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (RRF_K + rank))
        chunk_data[cid] = h

    for rank, h in enumerate(bm25_hits, start=1):
        cid = h["chunk_id"]
        bm25_ranks[cid] = rank
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (RRF_K + rank))
        if cid not in chunk_data:
            chunk_data[cid] = h

    sorted_cids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)

    results: List[HybridResult] = []
    for cid in sorted_cids[:top_k]:
        info = chunk_data[cid]
        sources = []
        if cid in dense_ranks:
            sources.append("dense_multivector")
        if cid in bm25_ranks:
            sources.append("bm25")

        results.append(
            HybridResult(
                chunk_id=cid,
                content=info.get("content", ""),
                rrf_score=rrf_scores[cid],
                sources=sources,
                dense_rank=dense_ranks.get(cid),
                bm25_rank=bm25_ranks.get(cid),
                source_uri=info.get("source_uri"),
                section_title=info.get("section_title"),
                document_title=info.get("document_title"),
            )
        )

    return results
