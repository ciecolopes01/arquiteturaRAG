"""
Teste de avaliação offline + baseline Recall@K — Passo 7.

Roda a avaliação nos 3 modos (hybrid, dense, bm25) e registra o baseline.

Rodar com:
    $env:PYTHONIOENCODING='utf-8'; python tests/test_evaluation.py
"""

import sys

sys.path.insert(0, ".")

from src.evaluation.offline_eval import evaluate_recall_at_k, load_golden_dataset


def test_golden_dataset_structure():
    """Valida integridade e schema do golden dataset."""
    dataset = load_golden_dataset()
    assert len(dataset) > 0
    for item in dataset:
        assert "question" in item
        assert "expected_chunk_ids" in item
        assert isinstance(item["expected_chunk_ids"], list)


def test_offline_eval_calculation(monkeypatch):
    """Valida métricas Recall@K e MRR da função evaluate_recall_at_k."""
    dataset = [
        {"question": "Q1", "expected_chunk_ids": ["c1", "c2"]},
        {"question": "Q2", "expected_chunk_ids": ["c3"]},
        {"question": "Q3", "expected_chunk_ids": ["c99"]},
    ]

    # Simula retrieved chunks:
    # Q1 -> acerto no rank 1 (MRR += 1.0)
    # Q2 -> acerto no rank 2 (MRR += 0.5)
    # Q3 -> erro (MRR += 0.0)
    def fake_retrieve(query, top_k, mode, *args, **kwargs):
        if query == "Q1":
            return ["c1", "other"]
        elif query == "Q2":
            return ["other", "c3"]
        return ["other1", "other2"]


    monkeypatch.setattr("src.evaluation.offline_eval._retrieve", fake_retrieve)

    metrics = evaluate_recall_at_k(dataset, k=2, mode="hybrid")
    assert metrics["hits"] == 2
    assert metrics["n_questions"] == 3
    assert abs(metrics["recall_at_k"] - (2 / 3)) < 1e-3
    assert abs(metrics["mrr"] - (1.5 / 3)) < 1e-3



def main() -> None:
    print("Avaliacao offline (Passo 7) — Baseline Recall@K\n")

    dataset = load_golden_dataset()
    print(f"Golden dataset carregado: {len(dataset)} perguntas\n")

    results = {}
    for mode in ["dense", "bm25", "hybrid"]:
        print(f"\n{'='*60}")
        print(f"  Modo: {mode.upper()}")
        print(f"{'='*60}")
        try:
            r = evaluate_recall_at_k(
                dataset, k=5, mode=mode, tenant_id="tenant-test", verbose=True
            )
            results[mode] = r
        except Exception as e:
            print(f"  [ERRO] {mode}: {e}")
            results[mode] = None

    print("\n\n" + "=" * 70)
    print("BASELINE FINAL")
    print("=" * 70)
    print(f"{'Modo':<10} {'Recall@5':>10} {'MRR':>8} {'Hits':>6} {'ms/query':>10}")
    print("-" * 70)
    for mode, r in results.items():
        if r:
            print(f"{mode:<10} {r['recall_at_k']:>10.4f} {r['mrr']:>8.4f} "
                  f"{r['hits']:>3}/{r['n_questions']:<3} {r['ms_per_query']:>10.1f}ms")
        else:
            print(f"{mode:<10} {'ERRO':>10}")

    print("\nNota: BM25 tem Recall baixo em queries conceituais (esperado —")
    print("o stemmer portugues nao indexa termos tecnicos em ingles).")
    print("O hibrido compensa via canal dense.")

    hybrid = results.get("hybrid")
    if hybrid and hybrid["recall_at_k"] >= 0.50:
        print(f"\nPasso 7 concluido — Baseline Recall@5 (hybrid)={hybrid['recall_at_k']:.4f}")
        sys.exit(0)
    else:
        r5 = hybrid["recall_at_k"] if hybrid else "N/A"
        print(f"\nRecall@5 baixo ({r5}) — revisar chunk IDs no golden dataset.")
        sys.exit(1)


if __name__ == "__main__":
    main()
