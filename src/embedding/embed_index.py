"""
Indexação de embeddings no Qdrant — Passo 4.

Pega os Chunks gerados pelo chunker (Passo 3), gera vetores com o BGE-M3
(model.py) e grava na coleção do Qdrant — um ponto por chunk.

Payload de cada ponto (campos consultáveis via filtro Qdrant):
  - chunk_id      : str  — ID único do chunk
  - document_id   : str  — ID do Document de origem
  - tenant_id     : str  — para filtro de ACL (Passo 11 completa isso)
  - acl           : list — entradas de controle de acesso
  - source_uri    : str
  - document_title: str
  - section_title : str
  - chunk_index   : int
  - embedding_model_version: str — para migração futura (Passo arquitetura §8)

Função principal: embed_and_index(chunks, recreate_collection=False)
Função de busca:  search_dense(query, top_k, tenant_id) -> list[dict]
"""

from __future__ import annotations

import time
from typing import List, Optional

# Observabilidade (Passo 14) — import condicional para não quebrar
# módulos que rodam sem o pipeline de trace ativo.
try:
    from src.observability.instrumentation import get_current_trace
    _OBS_AVAILABLE = True
except ImportError:
    _OBS_AVAILABLE = False
    def get_current_trace():
        return None

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

# ---------------------------------------------------------------------------
# Cliente Qdrant (singleton de módulo)
# ---------------------------------------------------------------------------

_client: Optional[QdrantClient] = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url)
    return _client


# ---------------------------------------------------------------------------
# Gestão da coleção
# ---------------------------------------------------------------------------

def ensure_collection(recreate: bool = False) -> None:
    """
    Cria (ou recria) a coleção no Qdrant com os parâmetros corretos.

    - recreate=False: cria somente se não existir (seguro para re-execuções).
    - recreate=True:  apaga e recria (útil ao mudar dimensão ou config do índice).
    """
    client = _get_client()
    dim = embedding_dimension()
    collection = settings.qdrant_collection

    existing = {c.name for c in client.get_collections().collections}

    if collection in existing:
        if recreate:
            print(f"[Qdrant] Recriando coleção '{collection}'...")
            client.delete_collection(collection)
        else:
            print(f"[Qdrant] Coleção '{collection}' já existe — mantendo.")
            return

    print(f"[Qdrant] Criando coleção '{collection}' (dim={dim}, cosine)...")
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"[Qdrant] Coleção criada.")


# ---------------------------------------------------------------------------
# Indexação
# ---------------------------------------------------------------------------

def embed_and_index(
    chunks: List[Chunk],
    batch_size: int = 12,
    recreate_collection: bool = False,
) -> dict:
    """
    Gera embeddings para todos os chunks e indexa no Qdrant.

    Retorna um dict com estatísticas:
      - total_chunks: int
      - total_time_s: float
      - ms_per_chunk: float
      - embedding_model: str
      - collection: str

    Exemplo:
        from src.parsing.simple_parsers import parse_file
        from src.chunking.recursive import chunk_recursive
        from src.embedding.embed_index import embed_and_index

        doc = parse_file("data/seed_corpus/01_introducao_rag.txt")
        chunks = chunk_recursive(doc)
        stats = embed_and_index(chunks)
    """
    ensure_collection(recreate=recreate_collection)
    client = _get_client()
    collection = settings.qdrant_collection
    model_version = settings.embedding_model

    print(f"\n[embed_and_index] Indexando {len(chunks)} chunks em '{collection}'...")
    t_total = time.time()

    # Processa em batches para não sobrecarregar memória
    points: List[PointStruct] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.content for c in batch]

        vectors = embed_texts(texts, batch_size=batch_size)

        for chunk, vector in zip(batch, vectors):
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "tenant_id": chunk.tenant_id or "",
                "acl": chunk.acl,
                # Lista achatada de principals (Passo 11) — Qdrant faz match
                # eficiente em campos de array com MatchAny; a estrutura rica
                # em "acl" (principal+permission) fica para auditoria/exibição.
                "acl_principals": [e.get("principal") for e in chunk.acl if e.get("principal")],
                "source_uri": chunk.metadata.get("source_uri", ""),
                "document_title": chunk.metadata.get("document_title", ""),
                "section_title": chunk.metadata.get("section_title", ""),
                "chunk_index": chunk.metadata.get("chunk_index", i),
                "content": chunk.content,          # guardamos o texto para inspeção
                "embedding_model_version": model_version,
            }
            points.append(
                PointStruct(
                    id=chunk.chunk_id,   # Qdrant aceita UUID como ID
                    vector=vector,
                    payload=payload,
                )
            )

        print(f"  batch {i // batch_size + 1}: {len(batch)} chunks embedados")

    # Upsert em lote único (mais eficiente que um por um)
    client.upsert(collection_name=collection, points=points)

    total_time = time.time() - t_total
    stats = {
        "total_chunks": len(chunks),
        "total_time_s": round(total_time, 2),
        "ms_per_chunk": round(total_time / max(len(chunks), 1) * 1000, 1),
        "embedding_model": model_version,
        "collection": collection,
        "device": settings.embedding_device,
    }
    print(
        f"[embed_and_index] Concluído: {len(chunks)} chunks em {total_time:.1f}s "
        f"({stats['ms_per_chunk']}ms/chunk) — device={settings.embedding_device}"
    )
    return stats


# ---------------------------------------------------------------------------
# Busca densa
# ---------------------------------------------------------------------------

def search_dense(
    query: str,
    top_k: int = 10,
    tenant_id: Optional[str] = None,
    principals: Optional[List[str]] = None,
) -> List[dict]:
    """
    Busca vetorial por similaridade coseno no Qdrant.

    Parâmetros:
    - query      : texto da pergunta.
    - top_k      : número de resultados retornados.
    - tenant_id  : se fornecido, filtra por tenant (pré-filtro, não pós-filtro).
    - principals : se fornecido (Passo 11), exige que pelo menos um principal
                   do usuário (ex: "user:joao", "grupo:financeiro") esteja
                   presente no acl_principals do chunk. Combinado com
                   tenant_id via AND — os dois pré-filtros rodam ANTES do
                   ranking de similaridade, nunca depois.

    Retorna lista de dicts com: chunk_id, score, content, metadata.
    """
    client = _get_client()
    trace = get_current_trace()

    # --- Embedding da query (sub-estágio instrumentado) ---
    _t_embed = time.time()
    query_vector = embed_query(query)
    _embed_ms = (time.time() - _t_embed) * 1000
    if trace is not None:
        trace.record_stage("retrieval.embedding", _embed_ms)

    # Filtros pré-ranking no Qdrant (tenant + ACL) — nunca pós-filtro
    conditions = []
    if tenant_id:
        conditions.append(
            FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))
        )
    if principals:
        conditions.append(
            FieldCondition(key="acl_principals", match=MatchAny(any=principals))
        )
    search_filter = Filter(must=conditions) if conditions else None

    # --- Busca vetorial Qdrant (sub-estágio instrumentado) ---
    _t_qdrant = time.time()
    # qdrant-client >= 1.7 usa query_points() em vez de search()
    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        query_filter=search_filter,
        with_payload=True,
    )
    _qdrant_ms = (time.time() - _t_qdrant) * 1000
    if trace is not None:
        trace.record_stage("retrieval.dense", _qdrant_ms, top_k=top_k)

    # query_points retorna QueryResponse com .points (lista de ScoredPoint)
    points = response.points if hasattr(response, "points") else response

    return [
        {
            "chunk_id": r.payload.get("chunk_id"),
            "score": round(r.score, 4),
            "content": r.payload.get("content", ""),
            "source_uri": r.payload.get("source_uri", ""),
            "section_title": r.payload.get("section_title", ""),
            "document_title": r.payload.get("document_title", ""),
        }
        for r in points
    ]


# ---------------------------------------------------------------------------
# Invalidação seletiva (Passo 12)
# ---------------------------------------------------------------------------

def delete_document(document_id: str) -> int:
    """
    Remove do Qdrant todos os pontos (chunks) de um document_id.

    Usado pela reindexação incremental: ao detectar que um documento mudou,
    seus chunks antigos são removidos antes de indexar os novos — nenhum
    outro documento é tocado.

    Retorna o número de pontos removidos.
    """
    client = _get_client()
    collection = settings.qdrant_collection

    doc_filter = Filter(
        must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
    )
    count_before = client.count(
        collection_name=collection, count_filter=doc_filter, exact=True
    ).count
    if count_before == 0:
        return 0

    client.delete(collection_name=collection, points_selector=doc_filter)
    return count_before
