"""
Teste de geração RAG — Passo 8.

Roda answer() para 5 perguntas do golden dataset e avalia:
1. Pipeline end-to-end funciona sem erro.
2. Resposta contém informação do chunk esperado (verificação heurística).
3. Resposta inclui marcação de fonte [Fonte N].
4. Latência total (retrieval + geração) registrada.

Em modo simulado (sem API key): valida que retrieval funciona e imprime
os chunks que seriam passados ao LLM — suficiente para confirmar o pipeline.

Rodar com:
    $env:PYTHONIOENCODING='utf-8'; python tests/test_generation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.generation.answer import answer, DEFAULT_MODEL

TENANT = "tenant-test"

# 5 perguntas representativas do golden dataset
TEST_CASES = [
    {
        "id": "gd-01",
        "question": "O que é RAG e quais são seus dois componentes principais?",
        "expected_keywords": ["recuperação", "geração", "retrieval", "LLM", "linguagem"],
        "expected_source": "01_introducao_rag.txt",
    },
    {
        "id": "gd-10",
        "question": "O que é Parent-Child chunking e qual vantagem oferece?",
        "expected_keywords": ["parent", "child", "filho", "pai", "chunk"],
        "expected_source": "03_chunking.txt",
    },
    {
        "id": "gd-15",
        "question": "Qual é o valor mínimo aceitável de Recall@5 para um sistema RAG corporativo?",
        "expected_keywords": ["0.75", "recall", "75"],
        "expected_source": "05_avaliacao_offline.txt",
    },
    {
        "id": "gd-18",
        "question": "Qual a diferença entre bi-encoder e cross-encoder no pipeline de retrieval?",
        "expected_keywords": ["bi-encoder", "cross-encoder", "query", "documento"],
        "expected_source": "06_reranking.html",
    },
    {
        "id": "gd-22",
        "question": "O que é tenant_id e como ele é usado no controle de acesso multi-tenant?",
        "expected_keywords": ["tenant", "acesso", "filtro", "cliente"],
        "expected_source": "07_acl_tenancy.html",
    },
]


def evaluate_answer(result, case: dict) -> dict:
    """Avalia heuristicamente se a resposta é correta."""
    answer_lower = result.answer.lower()

    # Verifica keywords esperadas
    keywords_found = [kw for kw in case["expected_keywords"] if kw.lower() in answer_lower]
    keyword_rate = len(keywords_found) / len(case["expected_keywords"])

    # Verifica citação de fonte
    has_citation = "[fonte" in answer_lower or "[source" in answer_lower

    # Verifica se a fonte certa foi recuperada
    source_retrieved = any(
        case["expected_source"].lower() in (c.source_uri or "").lower()
        for c in result.chunks_used
    )

    return {
        "keyword_rate": keyword_rate,
        "keywords_found": keywords_found,
        "has_citation": has_citation,
        "source_retrieved": source_retrieved,
        "simulated": result.simulated,
        "latency_ms": result.latency_ms,
        "tokens": result.total_tokens,
    }


def main() -> None:
    print("Testando Geracao RAG — Passo 8\n")
    print("=" * 70)

    all_results = []
    for case in TEST_CASES:
        print(f"\n[{case['id']}] {case['question'][:65]}")
        print("-" * 70)

        result = answer(
            case["question"],
            top_k=5,
            tenant_id=TENANT,
            stream=False,
        )

        eval_r = evaluate_answer(result, case)
        all_results.append(eval_r)

        # Mostra a resposta
        result.pretty_print()

        # Avaliação
        kw_ok = "OK" if eval_r["keyword_rate"] >= 0.5 else "FRACO"
        cit_ok = "OK" if eval_r["has_citation"] else "SEM CITACAO"
        src_ok = "OK" if eval_r["source_retrieved"] else "MISS"
        print(f"  Avaliacao: keywords={kw_ok}({eval_r['keyword_rate']:.0%}) | "
              f"citacao={cit_ok} | fonte={src_ok}")

    # Resumo
    print("\n" + "=" * 70)
    print("RESUMO FINAL — Passo 8")
    print("=" * 70)
    simulated = all_results[0]["simulated"]
    mode = "SIMULADO (sem API key)" if simulated else f"REAL ({DEFAULT_MODEL})"
    print(f"Modo: {mode}")

    avg_kw   = sum(r["keyword_rate"]      for r in all_results) / len(all_results)
    n_cit    = sum(1 for r in all_results if r["has_citation"])
    n_src    = sum(1 for r in all_results if r["source_retrieved"])
    avg_lat  = sum(r["latency_ms"]        for r in all_results) / len(all_results)

    print(f"\n  Keywords corretas (>=50%): {sum(1 for r in all_results if r['keyword_rate']>=0.5)}/{len(all_results)}")
    print(f"  Media keyword rate       : {avg_kw:.0%}")
    print(f"  Com citacao [Fonte N]    : {n_cit}/{len(all_results)}")
    print(f"  Fonte correta recuperada : {n_src}/{len(all_results)}")
    print(f"  Latencia media           : {avg_lat:.0f}ms")
    if not simulated:
        avg_tok = sum(r["tokens"] for r in all_results) / len(all_results)
        print(f"  Tokens medio por query   : {avg_tok:.0f}")

    # Critério de pronto: fonte correta recuperada em >= 4/5 casos
    ok = n_src >= 4
    if ok:
        print(f"\nPasso 8 concluido — pipeline RAG end-to-end funcional.")
        sys.exit(0)
    else:
        print(f"\nPasso 8 incompleto — fonte correta em apenas {n_src}/5 queries.")
        sys.exit(1)


if __name__ == "__main__":
    main()
