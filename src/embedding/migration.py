"""
Migração de Modelo de Embedding Sem Downtime — Passo 24.

Permite trocar o modelo de embedding com processo repetível e zero downtime:
1. Provisionamento de nova coleção isolada (namespace novo no Qdrant).
2. Dual-write / reprocessamento em paralelo mantendo a coleção ativa intacta.
3. Avaliação lado a lado (side-by-side evaluation) no golden dataset.
4. Cutover suave com período de graça (rollback capability) antes do expurgo.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from src.chunking.recursive import Chunk
from src.config import settings
from src.embedding.model import embed_query, embed_texts, embedding_dimension


@dataclass
class MigrationPlan:
    source_collection: str
    target_collection: str
    source_model: str
    target_model: str
    vector_dimension: int
    created_at: float = field(default_factory=time.time)
    status: str = "initialized"  # initialized -> dual_writing -> evaluated -> cutover -> completed


def prepare_target_collection(
    target_collection: str,
    dimension: Optional[int] = None,
    recreate: bool = False,
) -> bool:
    """Cria a nova coleção para o novo modelo no Qdrant sem afetar a existente."""
    client = QdrantClient(url=settings.qdrant_url)
    dim = dimension or embedding_dimension()

    exists = client.collection_exists(target_collection)
    if exists and recreate:
        client.delete_collection(target_collection)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=target_collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        return True
    return False


def _ensure_valid_point_id(raw_id: str) -> str:
    try:
        return str(uuid.UUID(str(raw_id)))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(raw_id)))


def dual_write_chunks(
    chunks: List[Chunk],
    target_collection: str,
    batch_size: int = 32,
) -> int:
    """
    Grava os chunks na nova coleção enquanto a primária continua servindo tráfego.
    """
    prepare_target_collection(target_collection=target_collection)
    client = QdrantClient(url=settings.qdrant_url)

    texts = [c.content for c in chunks]
    embeddings = embed_texts(texts)

    points = []
    for c, vec in zip(chunks, embeddings):
        payload = {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "content": c.content,
            "tenant_id": c.tenant_id or "",
            "acl": c.acl,
            "migrated_at": time.time(),
            **c.metadata,
        }
        point_id = _ensure_valid_point_id(c.chunk_id)
        points.append(PointStruct(id=point_id, vector=vec, payload=payload))

    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=target_collection,
            points=points[i : i + batch_size],
        )

    return len(points)


def side_by_side_eval(
    dataset: List[dict],
    primary_collection: str,
    target_collection: str,
    k: int = 5,
    tenant_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compara o desempenho de recuperação (Recall@K) lado a lado entre
    a coleção atual e a coleção candidata da migração.
    """
    client = QdrantClient(url=settings.qdrant_url)

    primary_hits = 0
    target_hits = 0
    total = len(dataset)

    for item in dataset:
        q = item["question"]
        expected = set(item["expected_chunk_ids"])
        q_vec = embed_query(q)

        # Busca na primária
        res_prim = client.query_points(
            collection_name=primary_collection,
            query=q_vec,
            limit=k,
            with_payload=True,
        )
        p_ids = {p.payload.get("chunk_id", str(p.id)) for p in res_prim.points}
        if expected & p_ids:
            primary_hits += 1

        # Busca na target
        res_targ = client.query_points(
            collection_name=target_collection,
            query=q_vec,
            limit=k,
            with_payload=True,
        )
        t_ids = {p.payload.get("chunk_id", str(p.id)) for p in res_targ.points}
        if expected & t_ids:
            target_hits += 1

    prim_recall = primary_hits / max(1, total)
    targ_recall = target_hits / max(1, total)

    return {
        "total_queries": total,
        "k": k,
        "primary_collection": primary_collection,
        "primary_recall": prim_recall,
        "target_collection": target_collection,
        "target_recall": targ_recall,
        "delta_recall": targ_recall - prim_recall,
        "recommendation": "PROCEED_CUTOVER" if targ_recall >= prim_recall else "REVIEW_OR_ABORT",
    }


def perform_cutover(
    plan: MigrationPlan,
    side_by_side_result: Dict[str, Any],
    grace_period_days: int = 30,
) -> Dict[str, Any]:
    """
    Executa o cutover lógico:
    - Marca o novo namespace como ativo.
    - Preserva o namespace legado como fallback de rollback durante o grace period.
    """
    if side_by_side_result.get("recommendation") != "PROCEED_CUTOVER":
        raise ValueError(
            f"Cutover abortado: desempenho insuficiente na coleção target: "
            f"{side_by_side_result.get('delta_recall', 0.0):+.4f}"
        )

    plan.status = "cutover"
    return {
        "status": "CUTOVER_SUCCESSFUL",
        "active_collection": plan.target_collection,
        "legacy_collection": plan.source_collection,
        "grace_period_days": grace_period_days,
        "rollback_available": True,
        "evaluated_at": time.time(),
    }
