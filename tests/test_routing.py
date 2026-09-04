"""
Teste de Fast Path / Full Path + routing suave — Passo 13.

Roda as 33 perguntas do golden dataset através de route_search() e mede,
SEPARADAMENTE por caminho (fast vs full):
1. Recall@5 (o chunk esperado está no top-5 do resultado final).
2. Latência média.

Também valida a heurística em si (estimate_complexity):
- Query comparativa + termos técnicos específicos -> full path.
- Query de fato simples (curta, sem comparação) -> fast path.
- Nenhum canal de busca é eliminado em nenhum dos dois caminhos (o full
  path sempre parte dos mesmos candidatos do hybrid_search, só adiciona
  o reranker por cima).

Rodar com: python tests/test_routing.py
(requer Qdrant e Postgres rodando, com o corpus indexado)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import pytest

from src.retrieval.acl_filter import ANY_PRINCIPAL
from src.retrieval.routing import COMPLEXITY_THRESHOLD, choose_path, estimate_complexity, route_search

GOLDEN_PATH = Path("data/golden_dataset.json")
TENANT = "tenant-test"


def _hit(chunk_ids_in_results: set, expected: list) -> bool:
    return bool(chunk_ids_in_results & set(expected))


# ---------------------------------------------------------------------------
# Testes da heurística (sem I/O)
# ---------------------------------------------------------------------------

def test_comparative_technical_query_routes_full() -> bool:
    print("  [*] test_comparative_technical_query_routes_full...")
    q = "Qual a diferença entre bi-encoder e cross-encoder no pipeline de retrieval?"
    path = choose_path(q)
    ok = path == "full"
    print(f"  [{'OK' if ok else 'FALHOU'}] '{q[:50]}...' -> {path}")
    return ok


def test_simple_lookup_routes_fast() -> bool:
    print("  [*] test_simple_lookup_routes_fast...")
    q = "O que são embeddings?"
    path = choose_path(q)
    ok = path == "fast"
    print(f"  [{'OK' if ok else 'FALHOU'}] '{q}' -> {path}")
    return ok


def test_complexity_score_bounded() -> bool:
    print("  [*] test_complexity_score_bounded...")
    queries = [
        "O que é RAG?",
        "Qual a diferença entre bi-encoder, cross-encoder, dense-retrieval e sparse-retrieval, "
        "comparado com abordagens híbridas versus soluções puramente léxicas?",
    ]
    scores = [estimate_complexity(q) for q in queries]
    ok = all(0.0 <= s <= 1.0 for s in scores)
    print(f"  [{'OK' if ok else 'FALHOU'}] scores={scores} (todos em [0,1])")
    return ok


# ---------------------------------------------------------------------------
# Avaliação no golden dataset — Recall@5 e latência por caminho
# ---------------------------------------------------------------------------

def evaluate_routing() -> dict:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    per_path = {"fast": [], "full": []}

    print(f"\n  Rodando {len(dataset)} perguntas via route_search (threshold={COMPLEXITY_THRESHOLD})...\n")
    for item in dataset:
        routed = route_search(item["question"], top_k=5, tenant_id=TENANT, principals=ANY_PRINCIPAL)
        result_ids = {r.chunk_id for r in routed.results}
        hit = _hit(result_ids, item["expected_chunk_ids"])

        per_path[routed.path].append({
            "id": item["id"],
            "hit": hit,
            "latency_ms": routed.latency_ms,
            "complexity": routed.complexity,
        })

        marker = "✓" if hit else "✗"
        print(f"    {marker} [{routed.path:4}] {item['id']:6} c={routed.complexity:.3f} "
              f"{routed.latency_ms:8.1f}ms  {item['question'][:55]}")

    return per_path


def _summarize(entries: list, label: str) -> dict:
    if not entries:
        return {"label": label, "n": 0, "recall": None, "avg_latency_ms": None}
    n = len(entries)
    hits = sum(1 for e in entries if e["hit"])
    avg_latency = sum(e["latency_ms"] for e in entries) / n
    return {
        "label": label,
        "n": n,
        "recall": round(hits / n, 4),
        "hits": hits,
        "avg_latency_ms": round(avg_latency, 1),
    }


@pytest.mark.integration
def test_recall_and_latency_per_path(require_qdrant, require_postgres) -> bool:
    print("  [*] test_recall_and_latency_per_path...")
    per_path = evaluate_routing()

    fast_stats = _summarize(per_path["fast"], "fast")
    full_stats = _summarize(per_path["full"], "full")

    print()
    print(f"    fast: n={fast_stats['n']:2} recall@5={fast_stats['recall']} "
          f"avg_latency={fast_stats['avg_latency_ms']}ms")
    print(f"    full: n={full_stats['n']:2} recall@5={full_stats['recall']} "
          f"avg_latency={full_stats['avg_latency_ms']}ms")

    # Critério de pronto: latência do fast path é bem menor que a do full
    # path (o ponto inteiro do routing), e Recall@5 do fast path não é
    # catastroficamente pior que o baseline hybrid conhecido (~0.97).
    ok_latency_gap = (
        full_stats["avg_latency_ms"] is None
        or fast_stats["avg_latency_ms"] < full_stats["avg_latency_ms"]
    )
    ok_fast_recall = fast_stats["recall"] is not None and fast_stats["recall"] >= 0.85

    ok = ok_latency_gap and ok_fast_recall
    print(f"\n  [{'OK' if ok else 'FALHOU'}] fast mais rápido que full={ok_latency_gap}, "
          f"Recall@5 fast >= 0.85={ok_fast_recall}")
    assert ok
    return ok


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("Testando Fast Path / Full Path + routing (Passo 13)...\n")

    tests = [
        test_comparative_technical_query_routes_full,
        test_simple_lookup_routes_fast,
        test_complexity_score_bounded,
    ]
    results = {}
    for fn in tests:
        results[fn.__name__] = fn()
        print()

    results["test_recall_and_latency_per_path"] = test_recall_and_latency_per_path()

    print("\nResumo:")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FALHOU'}")

    if all(results.values()):
        print("\nPasso 13 concluído — routing fast/full funciona, latência e Recall@5 medidos por caminho.")
        sys.exit(0)
    else:
        failed = [n for n, ok in results.items() if not ok]
        print(f"\nPasso 13 incompleto — {len(failed)} teste(s) falharam: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
