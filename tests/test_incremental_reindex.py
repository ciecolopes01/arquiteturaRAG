"""
Teste de reindexação incremental — Passo 12.

Usa um diretório isolado (data/incremental_test/), fora do corpus seed real,
para poder editar arquivos livremente sem arriscar os chunk_ids referenciados
pelo golden dataset de 33 perguntas (ver risco documentado no Passo 10).

Valida:
1. Primeira indexação de 2 documentos (A, B) -> status "indexed".
2. Rodar de novo sem mudar nada -> status "unchanged", 0 chunks tocados.
3. Editar só o documento A -> reindexar A remove/recria APENAS os chunks
   de A; B permanece com os mesmos chunk_ids, mesma contagem.
4. document_registry reflete o novo content_hash de A.
5. Busca pelo conteúdo novo de A encontra o chunk atualizado; o conteúdo
   antigo de A não é mais encontrado (foi de fato invalidado, não duplicado).

Rodar com: python tests/test_incremental_reindex.py
(requer Qdrant e Postgres rodando)
"""

import sys
from pathlib import Path

sys.path.insert(0, ".")

import pytest

from src.ingestion.incremental_reindex import (
    _get_registry_entry,
    incremental_reindex_file,
    stable_document_id,
)
from src.retrieval.acl_filter import ANY_PRINCIPAL
from src.retrieval.bm25_index import _get_conn
from src.retrieval.hybrid import hybrid_search

TEST_DIR = Path("data/incremental_test")
TENANT = "tenant-incremental-test"

DOC_A_ORIGINAL = (
    "Documento A\n\n"
    "Este é o conteúdo original do documento A. Código único: alpha-9f31."
)
DOC_A_EDITED = (
    "Documento A\n\n"
    "Este é o conteúdo EDITADO do documento A. Código único: alpha-EDITADO-7c02."
)
DOC_B_CONTENT = (
    "Documento B\n\n"
    "Este é o conteúdo do documento B, que nunca deve ser tocado. Código único: beta-3d18."
)


def _chunk_ids_for_document(document_id: str) -> set:
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute("SELECT chunk_id FROM bm25_chunks WHERE document_id = %s;", (document_id,))
        return {row[0] for row in cur.fetchall()}


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

def setup() -> tuple[Path, Path]:
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    path_a = TEST_DIR / "doc_a.txt"
    path_b = TEST_DIR / "doc_b.txt"
    _write(path_a, DOC_A_ORIGINAL)
    _write(path_b, DOC_B_CONTENT)
    return path_a, path_b


def _run_first_index_reports_indexed(path_a: Path, path_b: Path) -> bool:
    print("  [*] _run_first_index_reports_indexed...")
    r_a = incremental_reindex_file(path_a, tenant_id=TENANT)
    r_b = incremental_reindex_file(path_b, tenant_id=TENANT)
    ok = r_a["status"] == "indexed" and r_b["status"] == "indexed" and r_a["chunks_added"] > 0 and r_b["chunks_added"] > 0
    print(f"  [{'OK' if ok else 'FALHOU'}] A={r_a['status']}({r_a['chunks_added']} chunks) | B={r_b['status']}({r_b['chunks_added']} chunks)")
    return ok


def _run_rerun_unchanged_touches_nothing(path_a: Path, path_b: Path) -> bool:
    print("  [*] _run_rerun_unchanged_touches_nothing...")
    r_a = incremental_reindex_file(path_a, tenant_id=TENANT)
    r_b = incremental_reindex_file(path_b, tenant_id=TENANT)
    ok = (
        r_a["status"] == "unchanged" and r_a["chunks_removed"] == 0 and r_a["chunks_added"] == 0
        and r_b["status"] == "unchanged" and r_b["chunks_removed"] == 0 and r_b["chunks_added"] == 0
    )
    print(f"  [{'OK' if ok else 'FALHOU'}] A={r_a['status']} | B={r_b['status']}")
    return ok


def _run_edit_a_only_touches_a(path_a: Path, path_b: Path) -> bool:
    print("  [*] _run_edit_a_only_touches_a...")

    doc_id_b = stable_document_id(str(path_b))
    chunks_b_before = _chunk_ids_for_document(doc_id_b)

    _write(path_a, DOC_A_EDITED)
    r_a = incremental_reindex_file(path_a, tenant_id=TENANT)

    chunks_b_after = _chunk_ids_for_document(doc_id_b)

    ok_a_reindexed = r_a["status"] == "reindexed" and r_a["chunks_removed"] > 0 and r_a["chunks_added"] > 0
    ok_b_untouched = chunks_b_before == chunks_b_after and len(chunks_b_before) > 0

    print(f"  [{'OK' if ok_a_reindexed else 'FALHOU'}] A reindexado: {r_a['status']}, removed={r_a['chunks_removed']}, added={r_a['chunks_added']}")
    print(f"  [{'OK' if ok_b_untouched else 'FALHOU'}] B intocado: {len(chunks_b_before)} chunk_ids idênticos antes/depois = {ok_b_untouched}")
    return ok_a_reindexed and ok_b_untouched


def _run_registry_reflects_new_hash(path_a: Path) -> bool:
    print("  [*] _run_registry_reflects_new_hash...")
    from src.parsing.simple_parsers import parse_file
    doc = parse_file(path_a, tenant_id=TENANT, source_uri=str(path_a))
    entry = _get_registry_entry(str(path_a))
    ok = entry is not None and entry["content_hash"] == doc.content_hash
    print(f"  [{'OK' if ok else 'FALHOU'}] hash no registry bate com o hash atual do arquivo")
    return ok


def _run_search_finds_new_content_not_old(path_a: Path) -> bool:
    print("  [*] _run_search_finds_new_content_not_old...")
    results_new = hybrid_search("alpha-EDITADO-7c02", top_k=5, tenant_id=TENANT, principals=ANY_PRINCIPAL)
    results_old = hybrid_search("alpha-9f31", top_k=5, tenant_id=TENANT, principals=ANY_PRINCIPAL)

    found_new = any("alpha-EDITADO-7c02" in r.content for r in results_new)
    # conteúdo antigo não pode mais aparecer (foi de fato invalidado)
    found_old = any("alpha-9f31" in r.content and "EDITADO" not in r.content for r in results_old)

    ok = found_new and not found_old
    print(f"  [{'OK' if ok else 'FALHOU'}] conteúdo novo encontrado={found_new} | conteúdo antigo (não editado) ainda presente={found_old}")
    return ok


@pytest.mark.integration
def test_incremental_reindex_pipeline(require_qdrant, require_postgres) -> None:
    path_a, path_b = setup()
    try:
        assert _run_first_index_reports_indexed(path_a, path_b)
        assert _run_rerun_unchanged_touches_nothing(path_a, path_b)
        assert _run_edit_a_only_touches_a(path_a, path_b)
        assert _run_registry_reflects_new_hash(path_a)
        assert _run_search_finds_new_content_not_old(path_a)
    finally:
        teardown(path_a, path_b)


def teardown(path_a: Path, path_b: Path) -> None:
    """Remove os chunks de teste para não poluir o índice entre execuções."""
    from src.embedding.embed_index import delete_document as qdrant_delete
    from src.retrieval.bm25_index import delete_document as bm25_delete

    for p in (path_a, path_b):
        doc_id = stable_document_id(str(p))
        qdrant_delete(doc_id)
        bm25_delete(doc_id)

    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM document_registry WHERE source_uri IN (%s, %s);",
            (str(path_a), str(path_b)),
        )
    conn.commit()

    for p in (path_a, path_b):
        p.unlink(missing_ok=True)
    try:
        TEST_DIR.rmdir()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("Testando reindexação incremental (Passo 12)...\n")

    path_a, path_b = setup()

    tests_results = {}
    try:
        tests_results["test_first_index_reports_indexed"] = _run_first_index_reports_indexed(path_a, path_b)
        print()
        tests_results["test_rerun_unchanged_touches_nothing"] = _run_rerun_unchanged_touches_nothing(path_a, path_b)
        print()
        tests_results["test_edit_a_only_touches_a"] = _run_edit_a_only_touches_a(path_a, path_b)
        print()
        tests_results["test_registry_reflects_new_hash"] = _run_registry_reflects_new_hash(path_a)
        print()
        tests_results["test_search_finds_new_content_not_old"] = _run_search_finds_new_content_not_old(path_a)
        print()
    finally:
        teardown(path_a, path_b)

    print("Resumo:")
    for name, ok in tests_results.items():
        print(f"  {name}: {'OK' if ok else 'FALHOU'}")

    if all(tests_results.values()):
        print("\nPasso 12 concluído — reindexação incremental funciona, editar 1 doc não toca nos demais.")
        sys.exit(0)
    else:
        failed = [n for n, ok in tests_results.items() if not ok]
        print(f"\nPasso 12 incompleto — {len(failed)} teste(s) falharam: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
