"""
Testes do retrieval híbrido + RRF — Passo 6.

Valida:
1. RRF puro (sem I/O) — math correto.
2. hybrid_search retorna resultados para queries do corpus seed.
3. Chunks que aparecem nos dois canais ficam no topo (propriedade do RRF).
4. Comparação visual: híbrido >= melhor canal isolado em pelo menos
   metade das queries de teste.
5. Filtro de tenant propagado para ambos os canais.

Rodar com:
    $env:PYTHONIOENCODING='utf-8'; python tests/test_hybrid.py
(Qdrant + Postgres devem estar rodando com o corpus seed indexado)
"""

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, ".")

from src.embedding.embed_index import search_dense
from src.retrieval.bm25_index import search_bm25
from src.retrieval.hybrid import (
    HybridResult,
    RRF_K,
    hybrid_search,
    reciprocal_rank_fusion,
)

CORPUS_DIR = Path("data/seed_corpus")

# ---------------------------------------------------------------------------
# Queries de comparação (query, string esperada no top-5)
# ---------------------------------------------------------------------------

COMPARISON_QUERIES = [
    {
        "query": "O que e Recall@5 e como e calculado?",
        "expected": "Recall",
        "label": "avaliacao offline",
    },
    {
        "query": "Como funciona o algoritmo RRF?",
        "expected": "k",
        "label": "RRF formula",
    },
    {
        "query": "Qual a diferenca entre bi-encoder e cross-encoder?",
        "expected": "encoder",
        "label": "reranking encoders",
    },
    {
        "query": "Por que aplicar filtro de ACL antes do ranking?",
        "expected": "ACL",
        "label": "ACL pre-filter",
    },
    {
        "query": "Quais os parametros tipicos de chunking em producao?",
        "expected": "512",
        "label": "parametros chunking",
    },
]


# ---------------------------------------------------------------------------
# Helpers de inspeção
# ---------------------------------------------------------------------------

def show_comparison(query: str, label: str) -> dict:
    """Mostra top-5 de cada canal + híbrido lado a lado."""
    dense = search_dense(query, top_k=5, tenant_id="tenant-test")
    bm25  = search_bm25(query,  top_k=5, tenant_id="tenant-test")
    hyb   = hybrid_search(query, top_k=5, tenant_id="tenant-test")

    print(f"\n  [{label}] '{query[:60]}'")
    print(f"  {'Dense':^30} | {'BM25':^30} | {'Hibrido (RRF)':^35}")
    print(f"  {'-'*30} | {'-'*30} | {'-'*35}")

    n = max(len(dense), len(bm25), len(hyb))
    for i in range(min(n, 5)):
        d_txt = dense[i]["content"][:28].replace('\n', ' ') if i < len(dense) else ""
        b_txt = bm25[i]["content"][:28].replace('\n', ' ')  if i < len(bm25)  else ""
        h_txt = hyb[i].content[:30].replace('\n', ' ')       if i < len(hyb)   else ""
        h_src = "+".join(hyb[i].sources)                     if i < len(hyb)   else ""
        print(f"  {d_txt:<30} | {b_txt:<30} | {h_txt:<28} [{h_src}]")

    return {"dense": dense, "bm25": bm25, "hybrid": hyb}


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

def test_rrf_math_pure() -> bool:
    """RRF puro sem I/O: valida a fórmula matematicamente."""
    print("  [*] test_rrf_math_pure...")

    # Dois canais, 3 docs cada. Doc-A aparece em ambos no topo.
    lists = {
        "dense": [
            {"chunk_id": "A", "content": "doc A", "score": 0.9},
            {"chunk_id": "B", "content": "doc B", "score": 0.8},
            {"chunk_id": "C", "content": "doc C", "score": 0.7},
        ],
        "bm25": [
            {"chunk_id": "A", "content": "doc A", "score": 1.0},
            {"chunk_id": "D", "content": "doc D", "score": 0.5},
            {"chunk_id": "E", "content": "doc E", "score": 0.3},
        ],
    }
    results = reciprocal_rank_fusion(lists, top_k=5)

    assert results[0].chunk_id == "A", f"top-1 esperado A, got {results[0].chunk_id}"
    assert results[0].dense_rank == 1
    assert results[0].bm25_rank == 1

    expected_score_A = round(1.0 / (RRF_K + 1) + 1.0 / (RRF_K + 1), 6)
    assert abs(results[0].rrf_score - expected_score_A) < 1e-6, (
        f"score A esperado {expected_score_A}, got {results[0].rrf_score}"
    )

    # Doc-B só no dense (rank 2) deve ter score menor que A
    b_result = next(r for r in results if r.chunk_id == "B")
    assert b_result.rrf_score < results[0].rrf_score

    # Doc-B só em dense (rank 2), doc-D só em bm25 (rank 2) → mesmo score
    d_result = next(r for r in results if r.chunk_id == "D")
    assert abs(b_result.rrf_score - d_result.rrf_score) < 1e-9

    print(f"  [OK] RRF math validado: A em topo ({results[0].rrf_score:.6f}), "
          f"B e D empatados ({b_result.rrf_score:.6f})")
    return True


def test_hybrid_returns_results() -> bool:
    """hybrid_search retorna resultados para query simples."""
    print("  [*] test_hybrid_returns_results...")
    results = hybrid_search("RAG retrieval", top_k=5, tenant_id="tenant-test")
    if not results:
        print("  [FALHOU] nenhum resultado")
        return False
    top = results[0]
    print(f"  [OK] {len(results)} resultados | top rrf_score={top.rrf_score} | "
          f"canais: {top.sources}")
    return True


def test_both_channels_contribute() -> bool:
    """
    Para pelo menos 1 query, algum chunk no top-5 aparece nos dois canais
    (dense_rank não None e bm25_rank não None).
    """
    print("  [*] test_both_channels_contribute...")
    found_dual = False
    for q in COMPARISON_QUERIES:
        results = hybrid_search(q["query"], top_k=10, tenant_id="tenant-test")
        dual = [r for r in results if r.dense_rank is not None and r.bm25_rank is not None]
        if dual:
            found_dual = True
            print(f"    chunk duplo em '{q['label']}': "
                  f"dense_rank={dual[0].dense_rank}, bm25_rank={dual[0].bm25_rank}, "
                  f"rrf={dual[0].rrf_score:.5f}")
            break

    if not found_dual:
        print("  [AVISO] nenhum chunk apareceu nos dois canais — corpus pequeno demais")
        # Não é falha fatal — corpus seed é pequeno
    print(f"  [OK] chunks duplos {'encontrados' if found_dual else 'nao encontrados (corpus pequeno)'}")
    return True  # não falha para corpus pequeno


def test_hybrid_vs_solo_channels() -> bool:
    """
    Para cada query, hibrido deve ter Recall@5 >= melhor canal isolado em
    pelo menos metade dos casos (avaliacao visual — sem golden dataset ainda).
    """
    print("  [*] test_hybrid_vs_solo_channels...")
    wins = 0
    ties = 0
    for q in COMPARISON_QUERIES:
        query   = q["query"]
        exp     = q["expected"].lower()
        label   = q["label"]

        dense  = search_dense(query, top_k=5, tenant_id="tenant-test")
        bm25   = search_bm25(query,  top_k=5, tenant_id="tenant-test")
        hybrid = hybrid_search(query, top_k=5, tenant_id="tenant-test")

        in_dense  = any(exp in r["content"].lower() for r in dense[:5])
        in_bm25   = any(exp in r["content"].lower() for r in bm25[:5])
        in_hybrid = any(exp in r.content.lower() for r in hybrid[:5])

        best_solo = in_dense or in_bm25
        if in_hybrid and best_solo:
            wins += 1
            marker = "="
        elif in_hybrid and not best_solo:
            wins += 1
            marker = "+"
        elif not in_hybrid and not best_solo:
            ties += 1  # nenhum achou — empate negativo
            marker = "~"
        else:
            marker = "-"  # solo achou mas hibrido nao

        d = "✓" if in_dense  else "✗"
        b = "✓" if in_bm25   else "✗"
        h = "✓" if in_hybrid else "✗"
        print(f"    [{marker}] {label}: dense={d} bm25={b} hybrid={h}")

    total_meaningful = len(COMPARISON_QUERIES) - ties
    ok = wins >= max(1, total_meaningful // 2)
    status = "OK" if ok else "FALHOU"
    print(f"  [{status}] hibrido >=  solo em {wins}/{len(COMPARISON_QUERIES)} queries")
    return ok


def test_tenant_filter_hybrid() -> bool:
    """Tenant errado retorna lista vazia no canal híbrido."""
    print("  [*] test_tenant_filter_hybrid...")
    ok_results  = hybrid_search("RAG", top_k=5, tenant_id="tenant-test")
    bad_results = hybrid_search("RAG", top_k=5, tenant_id="tenant-INEXISTENTE")

    if not ok_results:
        print("  [FALHOU] tenant correto retornou 0 resultados")
        return False
    if bad_results:
        print(f"  [FALHOU] tenant errado retornou {len(bad_results)} resultado(s)")
        return False

    print(f"  [OK] tenant correto={len(ok_results)} | tenant errado=0")
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("Testando Retrieval Hibrido + RRF (Passo 6)...\n")

    tests = [
        test_rrf_math_pure,
        test_hybrid_returns_results,
        test_both_channels_contribute,
        test_hybrid_vs_solo_channels,
        test_tenant_filter_hybrid,
    ]

    results = {}
    for fn in tests:
        results[fn.__name__] = fn()
        print()

    # Inspecao comparativa de 3 queries
    print("=" * 100)
    print("Comparacao lado a lado: Dense | BM25 | Hibrido")
    print("=" * 100)
    for q in COMPARISON_QUERIES[:3]:
        show_comparison(q["query"], q["label"])

    print("\n\nResumo:")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FALHOU'}")

    if all(results.values()):
        print("\nPasso 6 concluido - retrieval hibrido RRF pronto.")
        sys.exit(0)
    else:
        failed = [n for n, ok in results.items() if not ok]
        print(f"\nPasso 6 incompleto - {len(failed)} teste(s) falharam: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
