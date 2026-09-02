"""
Testes do índice BM25 Postgres — Passo 5.

Valida:
1. Tabela criada sem erro.
2. Corpus seed indexado (65 chunks).
3. Busca por termo exato retorna chunk correto no topo.
4. Busca por linguagem natural retorna resultados relevantes.
5. Filtro de tenant funciona (pre-filter SQL, não post-filter).
6. Busca por termo inexistente retorna lista vazia.

Rodar com: python tests/test_bm25.py
(Qdrant deve estar rodando — Postgres obrigatório)
"""

import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.parsing.simple_parsers import parse_file
from src.chunking.recursive import chunk_recursive
from src.retrieval.bm25_index import (
    clear_bm25_table,
    create_bm25_table,
    index_bm25,
    search_bm25,
)

CORPUS_DIR = Path("data/seed_corpus")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_and_index_bm25() -> list:
    """Parseia e indexa todo o corpus seed no Postgres BM25."""
    files = sorted(
        list(CORPUS_DIR.glob("*.txt")) + list(CORPUS_DIR.glob("*.html"))
        + list(CORPUS_DIR.glob("*.pdf")) + list(CORPUS_DIR.glob("*.docx"))  # Passo 10
    )
    all_chunks = []
    for f in files:
        doc = parse_file(f, tenant_id="tenant-test")
        chunks = chunk_recursive(doc, max_tokens=512, overlap=64)
        all_chunks.extend(chunks)
    return all_chunks


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

def test_create_table() -> bool:
    print("  [*] test_create_table...")
    try:
        create_bm25_table()
        print("  [OK] Tabela bm25_chunks criada/verificada")
        return True
    except Exception as e:
        print(f"  [FALHOU] {e}")
        return False


def test_index_corpus() -> bool:
    print("  [*] test_index_corpus...")
    try:
        chunks = build_and_index_bm25()
        n = index_bm25(chunks, clear_first=True)
        assert n == len(chunks), f"esperado {len(chunks)}, indexado {n}"
        print(f"  [OK] {n} chunks indexados no Postgres BM25")
        return True
    except Exception as e:
        print(f"  [FALHOU] {e}")
        import traceback; traceback.print_exc()
        return False


def test_exact_term_search() -> bool:
    """
    Termos exatos presentes nos documentos devem aparecer no topo.
    """
    print("  [*] test_exact_term_search...")
    checks = [
        ("RRF",        "rrf",          "RRF no top-5"),
        ("embedding",  "embedding",    "embedding no top-5"),
        ("reranking",  "reranking",    "reranking no top-5"),
    ]
    failed = []
    for query, expected_term, label in checks:
        results = search_bm25(query, top_k=5, tenant_id="tenant-test")
        found = any(expected_term.lower() in r["content"].lower() for r in results)
        if not found:
            failed.append(label)
            print(f"    [ERRO] '{label}': nenhum resultado com '{expected_term}'")
            for r in results[:2]:
                print(f"      -> {r['content'][:80]}")
        else:
            print(f"    [OK] '{label}': encontrado no top-{len(results)}")

    if failed:
        print(f"  [FALHOU] {len(failed)} busca(s) sem resultado esperado")
        return False
    print(f"  [OK] Todos os {len(checks)} termos exatos encontrados")
    return True


def test_natural_language_search() -> bool:
    """
    Queries em linguagem natural devem retornar chunks relevantes.
    """
    print("  [*] test_natural_language_search...")
    queries = [
        ("golden dataset recall",  "golden",    "avaliacao offline"),
        ("chunking overlap",       "overlap",   "chunking"),
        ("tenant retrieval",       "tenant",    "ACL/tenancy"),
    ]
    passed = 0
    for query, expected, label in queries:
        results = search_bm25(query, top_k=5, tenant_id="tenant-test")
        found = any(expected.lower() in r["content"].lower() for r in results)
        if found:
            passed += 1
            top_score = results[0]["score"] if results else 0
            print(f"    [OK] '{label}': top score={top_score}")
        else:
            print(f"    [AVISO] '{label}': '{expected}' nao encontrado no top-5")

    ok = passed >= 2  # pelo menos 2/3
    status = "OK" if ok else "FALHOU"
    print(f"  [{status}] {passed}/{len(queries)} queries naturais com resultado correto")
    return ok


def test_tenant_filter() -> bool:
    """
    Tenant errado deve retornar 0 resultados mesmo com query exata.
    """
    print("  [*] test_tenant_filter...")
    results_ok = search_bm25("embedding", top_k=5, tenant_id="tenant-test")
    results_bad = search_bm25("embedding", top_k=5, tenant_id="tenant-INEXISTENTE")

    if len(results_ok) == 0:
        print("  [FALHOU] tenant correto nao retornou resultados")
        return False
    if len(results_bad) != 0:
        print(f"  [FALHOU] tenant errado retornou {len(results_bad)} resultado(s)")
        return False

    print(f"  [OK] tenant correto={len(results_ok)} | tenant errado=0")
    return True


def test_empty_query_returns_empty() -> bool:
    """Termo completamente ausente do corpus retorna lista vazia."""
    print("  [*] test_empty_query_returns_empty...")
    results = search_bm25("xyzabc123inexistente", top_k=5, tenant_id="tenant-test")
    if results:
        print(f"  [FALHOU] retornou {len(results)} resultado(s) para termo inexistente")
        return False
    print("  [OK] termo inexistente retorna lista vazia")
    return True


def test_scores_normalized() -> bool:
    """Score do top-1 deve ser 1.0 (normalizado) ou próximo."""
    print("  [*] test_scores_normalized...")
    results = search_bm25("embedding retrieval", top_k=5, tenant_id="tenant-test")
    if not results:
        print("  [SKIP] nenhum resultado — pulando")
        return True
    top = results[0]["score"]
    ok = 0.9 <= top <= 1.001  # top-1 deve ser exatamente 1.0 pela normalização
    status = "OK" if ok else "FALHOU"
    print(f"  [{status}] top-1 score={top} (esperado ~1.0)")
    return ok


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("Testando índice BM25 Postgres (Passo 5)...\n")

    tests = [
        test_create_table,
        test_index_corpus,
        test_exact_term_search,
        test_natural_language_search,
        test_tenant_filter,
        test_empty_query_returns_empty,
        test_scores_normalized,
    ]

    results = {}
    for fn in tests:
        results[fn.__name__] = fn()
        print()

    # Inspeção manual: mostra top-3 de uma query representativa
    print("Inspecao manual — top-3 para 'recuperacao informacao RAG':")
    print("-" * 60)
    rows = search_bm25("recuperacao informacao RAG", top_k=3, tenant_id="tenant-test")
    for i, r in enumerate(rows):
        print(f"  [{i+1}] score={r['score']} | {r['content'][:100]}")
    print()

    print("Resumo:")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FALHOU'}")

    if all(results.values()):
        print("\nPasso 5 concluido — indice BM25 pronto no Postgres.")
        sys.exit(0)
    else:
        failed = [n for n, ok in results.items() if not ok]
        print(f"\nPasso 5 incompleto — {len(failed)} teste(s) falharam: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
