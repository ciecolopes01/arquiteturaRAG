"""
Teste de embedding + indexação Qdrant — Passo 4.

Fluxo:
1. Parseia todo o corpus seed (8 arquivos).
2. Chunkeia (recursive, max_tokens=512).
3. Gera embeddings com BGE-M3 local e indexa no Qdrant.
4. Faz 5 queries de teste e verifica se o chunk correto aparece no top-3.
5. Registra tempo de embedding por chunk (referência para CPU vs GPU).

Rodar com: python tests/test_embedding.py
(primeira execução baixa o modelo BGE-M3 do HuggingFace ~2GB)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from src.parsing.simple_parsers import parse_file
from src.chunking.recursive import chunk_recursive
from src.embedding.embed_index import embed_and_index, search_dense

CORPUS_DIR = Path("data/seed_corpus")

# ---------------------------------------------------------------------------
# Queries de teste com resposta esperada (string que deve estar no chunk)
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    {
        "query": "O que é Recall@5 e qual valor é considerado bom?",
        "expected_snippet": "Recall@5",   # deve aparecer no top-3
        "description": "Métrica de avaliação de retrieval",
    },
    {
        "query": "Qual é a constante k usada no algoritmo RRF?",
        "expected_snippet": "k=60",
        "description": "Parâmetro do RRF",
    },
    {
        "query": "O que é um cross-encoder e para que serve no reranking?",
        "expected_snippet": "cross-encoder",
        "description": "Conceito de reranker",
    },
    {
        "query": "Por que o controle de acesso deve ser aplicado antes do ranking?",
        "expected_snippet": "antes",
        "description": "ACL pre-filter",
    },
    {
        "query": "Qual o tamanho típico de chunk em tokens para documentos corporativos?",
        "expected_snippet": "512",
        "description": "Parâmetro de chunking",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_and_index_corpus() -> dict:
    """Parseia, chunkeia e indexa todo o corpus seed."""
    files = sorted(
        list(CORPUS_DIR.glob("*.txt")) + list(CORPUS_DIR.glob("*.html"))
        + list(CORPUS_DIR.glob("*.pdf")) + list(CORPUS_DIR.glob("*.docx"))  # Passo 10
    )
    all_chunks = []
    for f in files:
        doc = parse_file(f, tenant_id="tenant-test")
        chunks = chunk_recursive(doc, max_tokens=512, overlap=64)
        all_chunks.extend(chunks)

    print(f"\n[corpus] {len(files)} documentos -> {len(all_chunks)} chunks")
    stats = embed_and_index(all_chunks, recreate_collection=True)
    return stats


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

def test_indexing() -> bool:
    """Corpus seed indexado sem erro; stats retornadas."""
    print("  [*] test_indexing...")
    try:
        stats = build_and_index_corpus()
        assert stats["total_chunks"] > 0
        assert stats["total_time_s"] > 0
        print(f"  [OK] {stats['total_chunks']} chunks | "
              f"{stats['total_time_s']}s | "
              f"{stats['ms_per_chunk']}ms/chunk | "
              f"device={stats['device']}")
        return True, stats
    except Exception as e:
        print(f"  [FALHOU] {e}")
        import traceback; traceback.print_exc()
        return False, {}


def test_search_returns_results(stats: dict) -> bool:
    """Busca retorna resultados não-vazios para queries simples."""
    print("  [*] test_search_returns_results...")
    results = search_dense("O que é RAG?", top_k=5)
    if not results:
        print("  [FALHOU] nenhum resultado retornado")
        return False
    print(f"  [OK] {len(results)} resultado(s) | top score={results[0]['score']}")
    return True


def test_relevant_chunks_in_top3() -> bool:
    """Para cada query de teste, o chunk esperado aparece no top-3."""
    print("  [*] test_relevant_chunks_in_top3...")
    passed = 0
    for q in TEST_QUERIES:
        results = search_dense(q["query"], top_k=5)
        found = any(
            q["expected_snippet"].lower() in r["content"].lower()
            for r in results[:3]
        )
        status = "✓" if found else "✗"
        print(f"    {status} [{q['description']}]: '{q['query'][:60]}…'")
        if found:
            passed += 1
        else:
            # mostra o que veio
            for i, r in enumerate(results[:3]):
                print(f"      top{i+1}: {r['content'][:80]}")

    rate = passed / len(TEST_QUERIES)
    # Critério: pelo menos 3/5 no top-3 (corpus seed pequeno, mais leniente)
    ok = passed >= 3
    status = "OK" if ok else "FALHOU"
    print(f"  [{status}] {passed}/{len(TEST_QUERIES)} queries com chunk correto no top-3")
    return ok


def test_tenant_filter() -> bool:
    """Pre-filtro de tenant_id funciona: outro tenant → 0 resultados."""
    print("  [*] test_tenant_filter...")
    results_with = search_dense("RAG", top_k=5, tenant_id="tenant-test")
    results_without = search_dense("RAG", top_k=5, tenant_id="tenant-INEXISTENTE")

    if len(results_with) == 0:
        print("  [FALHOU] tenant correto não retornou resultados")
        return False
    if len(results_without) != 0:
        print(f"  [FALHOU] tenant errado retornou {len(results_without)} resultado(s)")
        return False

    print(f"  [OK] tenant correto={len(results_with)} resultados | "
          f"tenant errado=0 resultados")
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("Testando Embedding + Índice Qdrant (Passo 4)...\n")
    print("AVISO: primeira execução baixa BGE-M3 do HuggingFace (~2GB).\n")

    # Indexa o corpus antes dos testes de busca
    ok_index, stats = test_indexing()
    print()

    if not ok_index:
        print("Indexação falhou — abortando testes de busca.")
        sys.exit(1)

    results = {
        "test_indexing": ok_index,
        "test_search_returns_results": test_search_returns_results(stats),
        "test_relevant_chunks_in_top3": test_relevant_chunks_in_top3(),
        "test_tenant_filter": test_tenant_filter(),
    }

    print("\n" + "=" * 60)
    print("Resumo:")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FALHOU'}")

    print(f"\nReferência de performance:")
    print(f"  Modelo     : {stats.get('embedding_model', '?')}")
    print(f"  Device     : {stats.get('device', '?')}")
    print(f"  Total chunks: {stats.get('total_chunks', '?')}")
    print(f"  Tempo total : {stats.get('total_time_s', '?')}s")
    print(f"  ms/chunk    : {stats.get('ms_per_chunk', '?')}ms")

    if all(results.values()):
        print("\nPasso 4 concluído — embeddings indexados no Qdrant.")
        sys.exit(0)
    else:
        failed = [n for n, ok in results.items() if not ok]
        print(f"\nPasso 4 incompleto — {len(failed)} teste(s) falharam: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
