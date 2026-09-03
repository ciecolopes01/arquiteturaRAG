"""
Testes de Migração de Modelo de Embedding Sem Downtime — Passo 24.

Valida:
1. Criação isolada de namespace target no Qdrant (prepare_target_collection).
2. Dual-write e reindexação paralela sem interrupção do tráfego ativo.
3. Avaliação lado a lado (side-by-side) entre coleção legada e nova.
4. Lógica de cutover e proteção de rollback com período de graça.

Rodar com: pytest tests/test_migration.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from src.chunking.recursive import Chunk
from src.config import settings
from src.embedding.migration import (
    MigrationPlan,
    dual_write_chunks,
    perform_cutover,
    prepare_target_collection,
    side_by_side_eval,
)


@pytest.fixture
def migration_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"mig-chunk-{i}",
            document_id="doc-migration",
            content=f"Trecho técnico {i} descrevendo características de embeddings densos e busca vetorial.",
            tenant_id="tenant-test",
        )
        for i in range(4)
    ]


def test_prepare_target_collection() -> None:
    """Verifica criação de nova coleção target isolada no Qdrant."""
    target_coll = "test_qdrant_migration_v2"
    created = prepare_target_collection(target_collection=target_coll, recreate=True)
    assert created is True

    # Chamada subsequente idempotente
    assert prepare_target_collection(target_collection=target_coll, recreate=False) is False


def test_dual_write_and_side_by_side(migration_chunks: list[Chunk]) -> None:
    """Valida gravação em paralelo e avaliação lado a lado."""
    prim_coll = "test_coll_primary"
    targ_coll = "test_coll_target"

    # Prepara ambas com os chunks de teste
    prepare_target_collection(prim_coll, recreate=True)
    prepare_target_collection(targ_coll, recreate=True)

    c1 = dual_write_chunks(migration_chunks, target_collection=prim_coll)
    c2 = dual_write_chunks(migration_chunks, target_collection=targ_coll)
    assert c1 == 4
    assert c2 == 4

    test_queries = [
        {
            "question": "características de embeddings densos",
            "expected_chunk_ids": ["mig-chunk-0", "mig-chunk-1"],
        },
        {
            "question": "busca vetorial em documentos",
            "expected_chunk_ids": ["mig-chunk-2", "mig-chunk-3"],
        },
    ]

    eval_res = side_by_side_eval(
        dataset=test_queries,
        primary_collection=prim_coll,
        target_collection=targ_coll,
        k=3,
    )

    assert eval_res["primary_recall"] == 1.0
    assert eval_res["target_recall"] == 1.0
    assert eval_res["recommendation"] == "PROCEED_CUTOVER"

    # Testa cutover
    plan = MigrationPlan(
        source_collection=prim_coll,
        target_collection=targ_coll,
        source_model="BAAI/bge-m3",
        target_model="BAAI/bge-m3-v2",
        vector_dimension=1024,
    )

    cutover_res = perform_cutover(plan, eval_res, grace_period_days=30)
    assert cutover_res["status"] == "CUTOVER_SUCCESSFUL"
    assert cutover_res["active_collection"] == targ_coll
    assert cutover_res["rollback_available"] is True
