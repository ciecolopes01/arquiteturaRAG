"""
Testes de ACL/Tenant — Passo 11.

Segue o critério de segurança definido em data/seed_corpus/07_acl_tenancy.html
(parte do corpus seed, escrito no Passo 2): criar um chunk com conteúdo
único, dar permissão apenas a um tenant/grupo, e verificar que uma query de
outro tenant ou grupo sobre exatamente esse conteúdo retorna zero
resultados — zero falsos negativos de segurança, mesmo com query
adversarial (busca pelo texto exato do chunk restrito).

Também valida que o backfill de acl_principals nos chunks pré-existentes
(indexados antes do Passo 11) não quebrou o retrieval normal, e que chamadas
sem `principals` (compatibilidade com Passos 4-10) continuam funcionando.

Rodar com: python tests/test_acl.py
(requer Qdrant e Postgres rodando, com o corpus já indexado)
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, ".")

from src.chunking.recursive import Chunk
from src.embedding.embed_index import embed_and_index
from src.retrieval.bm25_index import index_bm25
from src.retrieval.acl_filter import UserContext, secure_search
from src.evaluation.offline_eval import evaluate_recall_at_k, load_golden_dataset

# Conteúdo único (nonce) — evita colisão com qualquer coisa já no corpus,
# tornando o teste de "zero falsos negativos" inequívoco.
NONCE_A = "zx7Qplatypus-financeiro-9c2e"
NONCE_B = "zx7Qmarmoset-rh-4f81"

TENANT_A = "tenant-acl-A"
TENANT_B = "tenant-acl-B"


def _make_test_chunks() -> list[Chunk]:
    doc_id_a = str(uuid.uuid4())
    doc_id_b = str(uuid.uuid4())
    chunk_a = Chunk(
        document_id=doc_id_a,
        content=f"Relatório confidencial do departamento financeiro. Código de referência: {NONCE_A}.",
        tenant_id=TENANT_A,
        acl=[{"principal": "grupo:financeiro", "permission": "read"}],
        metadata={"source_uri": "test/acl_a.txt", "document_title": "ACL Test A", "section_title": None, "chunk_index": 0},
    )
    chunk_b = Chunk(
        document_id=doc_id_b,
        content=f"Relatório confidencial do departamento de RH. Código de referência: {NONCE_B}.",
        tenant_id=TENANT_B,
        acl=[{"principal": "grupo:rh", "permission": "read"}],
        metadata={"source_uri": "test/acl_b.txt", "document_title": "ACL Test B", "section_title": None, "chunk_index": 0},
    )
    return [chunk_a, chunk_b]


def _index_test_chunks(chunks: list[Chunk]) -> None:
    embed_and_index(chunks, recreate_collection=False)
    index_bm25(chunks, clear_first=False)


def _contains_nonce(results, nonce: str) -> bool:
    return any(nonce in r.content for r in results)


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

def test_correct_tenant_and_group_sees_content() -> bool:
    print("  [*] test_correct_tenant_and_group_sees_content...")
    ctx = UserContext(tenant_id=TENANT_A, user_id="alice", groups=["financeiro"])
    results = secure_search(NONCE_A, ctx, top_k=5)
    ok = _contains_nonce(results, NONCE_A)
    print(f"  [{'OK' if ok else 'FALHOU'}] tenant A + grupo financeiro -> {len(results)} resultado(s)")
    return ok


def test_wrong_group_same_tenant_sees_nothing() -> bool:
    """Mesmo tenant, grupo sem permissão -> 0 resultados (enforcement de ACL, não só tenant)."""
    print("  [*] test_wrong_group_same_tenant_sees_nothing...")
    ctx = UserContext(tenant_id=TENANT_A, user_id="bob", groups=["rh"])  # RH não tem ACL no chunk A
    results = secure_search(NONCE_A, ctx, top_k=5)
    ok = not _contains_nonce(results, NONCE_A)
    print(f"  [{'OK' if ok else 'FALHOU'}] tenant A + grupo rh (sem permissão) -> {len(results)} resultado(s)")
    return ok


def test_wrong_tenant_sees_nothing() -> bool:
    """Tenant diferente, mesmo com query exata do conteúdo -> 0 resultados."""
    print("  [*] test_wrong_tenant_sees_nothing...")
    ctx = UserContext(tenant_id=TENANT_B, user_id="carol", groups=["financeiro"])
    results = secure_search(NONCE_A, ctx, top_k=5)
    ok = not _contains_nonce(results, NONCE_A)
    print(f"  [{'OK' if ok else 'FALHOU'}] tenant B (mesmo com grupo financeiro) -> {len(results)} resultado(s)")
    return ok


def test_adversarial_exact_query_still_blocked() -> bool:
    """Query adversarial: busca pelo texto EXATO do chunk restrito, de outro tenant."""
    print("  [*] test_adversarial_exact_query_still_blocked...")
    ctx = UserContext(tenant_id=TENANT_B, user_id="mallory", groups=["financeiro", "rh", "admin"])
    exact_query = f"Relatório confidencial do departamento financeiro. Código de referência: {NONCE_A}."
    results = secure_search(exact_query, ctx, top_k=10)
    ok = not _contains_nonce(results, NONCE_A)
    print(f"  [{'OK' if ok else 'FALHOU'}] query adversarial exata de outro tenant -> {len(results)} resultado(s)")
    return ok


def test_cross_contamination_between_a_and_b() -> bool:
    """Tenant B só vê o conteúdo B; tenant A só vê o conteúdo A."""
    print("  [*] test_cross_contamination_between_a_and_b...")
    ctx_b = UserContext(tenant_id=TENANT_B, user_id="dave", groups=["rh"])
    results_b = secure_search(NONCE_B, ctx_b, top_k=5)
    ok_b_sees_b = _contains_nonce(results_b, NONCE_B)
    ok_b_not_a = not _contains_nonce(results_b, NONCE_A)
    ok = ok_b_sees_b and ok_b_not_a
    print(f"  [{'OK' if ok else 'FALHOU'}] tenant B vê B={ok_b_sees_b}, não vê A={ok_b_not_a}")
    return ok


def test_backfilled_legacy_content_still_searchable() -> bool:
    """
    Chunks indexados antes do Passo 11 (sem ACL explícito) foram backfilled
    com acl_principals=['grupo:geral']. Um usuário desse grupo deve
    continuar recuperando esse conteúdo normalmente.
    """
    print("  [*] test_backfilled_legacy_content_still_searchable...")
    ctx = UserContext(tenant_id="tenant-test", user_id="erin", groups=["geral"])
    results = secure_search("O que é RAG e quais são seus componentes?", ctx, top_k=5)
    ok = len(results) > 0
    print(f"  [{'OK' if ok else 'FALHOU'}] conteúdo legado (backfill) -> {len(results)} resultado(s)")
    return ok


def test_legacy_content_blocked_without_group_geral() -> bool:
    """Usuário do tenant certo mas sem o grupo 'geral' não deve ver o conteúdo legado."""
    print("  [*] test_legacy_content_blocked_without_group_geral...")
    ctx = UserContext(tenant_id="tenant-test", user_id="frank", groups=["outro-grupo-sem-acesso"])
    results = secure_search("O que é RAG e quais são seus componentes?", ctx, top_k=5)
    ok = len(results) == 0
    print(f"  [{'OK' if ok else 'FALHOU'}] tenant certo, grupo sem ACL -> {len(results)} resultado(s)")
    return ok


def test_backward_compatible_without_principals() -> bool:
    """hybrid_search/offline_eval sem passar `principals` continua funcionando (Passos 4-10)."""
    print("  [*] test_backward_compatible_without_principals...")
    dataset = load_golden_dataset()
    result = evaluate_recall_at_k(dataset, k=5, mode="hybrid", tenant_id="tenant-test", verbose=False)
    ok = result["recall_at_k"] >= 0.90
    print(f"  [{'OK' if ok else 'FALHOU'}] Recall@5 sem filtro de ACL = {result['recall_at_k']}")
    return ok


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("Testando ACL/Tenant (Passo 11)...\n")

    print("Indexando chunks de teste (tenant A / tenant B)...")
    _index_test_chunks(_make_test_chunks())
    print()

    tests = [
        test_correct_tenant_and_group_sees_content,
        test_wrong_group_same_tenant_sees_nothing,
        test_wrong_tenant_sees_nothing,
        test_adversarial_exact_query_still_blocked,
        test_cross_contamination_between_a_and_b,
        test_backfilled_legacy_content_still_searchable,
        test_legacy_content_blocked_without_group_geral,
        test_backward_compatible_without_principals,
    ]

    results = {}
    for fn in tests:
        results[fn.__name__] = fn()
        print()

    print("Resumo:")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FALHOU'}")

    if all(results.values()):
        print("\nPasso 11 concluído — ACL aplicado antes do ranking, zero falsos negativos de segurança.")
        sys.exit(0)
    else:
        failed = [n for n, ok in results.items() if not ok]
        print(f"\nPasso 11 incompleto — {len(failed)} teste(s) falharam: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
