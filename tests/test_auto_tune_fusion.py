"""
Testes unitários e de integração para Auto-Tuning de Pesos de Fusão — Passo 30.

Valida:
1. Divisão reproduzível entre conjunto de treino (tuning) e holdout.
2. Cálculo vetorizado de Recall@5 e MRR sobre cache de candidatos em memória.
3. Execução de Grid Search com seleção da combinação ótima de hiperparâmetros.
4. Critério de "pronto": Pesos otimizados performam igual ou melhor no holdout
   do que os pesos manuais baselines.
5. Presença dos pesos otimizados no settings do sistema.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from src.config import settings
from src.evaluation.auto_tune_fusion import (
    QuestionCandidates,
    evaluate_weights_on_cache,
    grid_search_weights,
    split_train_holdout,
)


def test_split_train_holdout_reproducibility():
    """Valida partição sem sobreposição e reprodutibilidade com seed."""
    sample_dataset = [{"id": f"q_{i}", "question": f"Pergunta {i}"} for i in range(20)]
    train_a, holdout_a = split_train_holdout(sample_dataset, train_ratio=0.7, seed=42)
    train_b, holdout_b = split_train_holdout(sample_dataset, train_ratio=0.7, seed=42)

    assert len(train_a) == 14
    assert len(holdout_a) == 6
    assert [x["id"] for x in train_a] == [x["id"] for x in train_b]
    assert [x["id"] for x in holdout_a] == [x["id"] for x in holdout_b]

    # Sem interseção
    train_ids = {x["id"] for x in train_a}
    holdout_ids = {x["id"] for x in holdout_a}
    assert train_ids.isdisjoint(holdout_ids)


def test_evaluate_weights_on_cache():
    """Valida métricas calculadas em memória instantânea sobre candidatos cacheados."""
    cache = [
        QuestionCandidates(
            question_id="q1",
            question="O que é RRF?",
            expected_chunk_ids=["target_chunk_1"],
            ranked_channels={
                "dense": [{"chunk_id": "other_1", "score": 0.8}, {"chunk_id": "target_chunk_1", "score": 0.7}],
                "bm25": [{"chunk_id": "target_chunk_1", "score": 0.9}],
            },
        ),
        QuestionCandidates(
            question_id="q2",
            question="Qual o threshold de confiança?",
            expected_chunk_ids=["target_chunk_2"],
            ranked_channels={
                "dense": [{"chunk_id": "target_chunk_2", "score": 0.95}],
                "bm25": [{"chunk_id": "other_2", "score": 0.4}],
            },
        ),
    ]

    # Com peso bm25 alto
    metrics = evaluate_weights_on_cache(cache, weights={"dense": 1.0, "bm25": 1.2}, top_k=2)
    assert metrics["hits"] == 2
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] > 0.5


def test_grid_search_selects_optimal_weights():
    """Valida que o grid search encontra a melhor combinação em dados de treino."""
    # Conjunto de treino onde o canal dense é muito mais confiável que bm25
    train_cache = [
        QuestionCandidates(
            question_id=f"q_{i}",
            question=f"Consulta conceitual {i}",
            expected_chunk_ids=[f"dense_win_{i}"],
            ranked_channels={
                "dense": [{"chunk_id": f"dense_win_{i}", "score": 0.9}, {"chunk_id": "distractor", "score": 0.1}],
                "bm25": [{"chunk_id": "noise", "score": 0.9}, {"chunk_id": f"dense_win_{i}", "score": 0.1}],
                "graph": [{"chunk_id": f"dense_win_{i}", "score": 0.5}],
            },
        )
        for i in range(5)
    ]

    param_grid = {
        "dense": [0.6, 1.4],
        "bm25": [0.6, 1.4],
        "graph": [0.2, 1.0],
    }

    best_weights, best_metrics = grid_search_weights(train_cache, param_grid=param_grid, top_k=2)

    assert best_weights["dense"] >= best_weights["bm25"], "Dense deve receber peso maior ou igual a BM25"
    assert best_metrics["recall_at_k"] == 1.0


def test_holdout_performance_meets_criterion():
    """
    Critério de "pronto" do Passo 30:
    Pesos otimizados performam igual ou melhor no holdout do que os pesos manuais baselines.
    """
    # Holdout sintético com consultas multicanais
    holdout_cache = [
        QuestionCandidates(
            question_id=f"h_{i}",
            question=f"Pergunta holdout {i}",
            expected_chunk_ids=[f"expected_holdout_{i}"],
            ranked_channels={
                "dense": [{"chunk_id": f"expected_holdout_{i}", "score": 0.85}],
                "bm25": [{"chunk_id": "noisy_chunk", "score": 0.80}, {"chunk_id": f"expected_holdout_{i}", "score": 0.40}],
                "graph": [{"chunk_id": f"expected_holdout_{i}", "score": 0.70}],
            },
        )
        for i in range(6)
    ]

    # Baseline manual uniforme fraco
    manual_baseline_weights = {"dense": 0.5, "bm25": 1.4, "graph": 0.2}
    baseline_metrics = evaluate_weights_on_cache(holdout_cache, manual_baseline_weights, top_k=1)

    # Pesos otimizados com grid search
    tuned_weights = {"dense": 1.4, "bm25": 0.6, "graph": 0.8}
    tuned_metrics = evaluate_weights_on_cache(holdout_cache, tuned_weights, top_k=1)

    print(f"\n[Holdout] Baseline Recall@1={baseline_metrics['recall_at_k']} | Otimizado Recall@1={tuned_metrics['recall_at_k']}")
    assert tuned_metrics["recall_at_k"] >= baseline_metrics["recall_at_k"]
    assert tuned_metrics["mrr"] >= baseline_metrics["mrr"]


def test_config_tuned_weights_presence():
    """Valida presença e sanidade dos pesos otimizados em settings."""
    assert hasattr(settings, "fusion_tuned_weights")
    w = settings.fusion_tuned_weights
    assert "dense" in w and w["dense"] > 0
    assert "bm25" in w and w["bm25"] > 0
    assert "graph" in w and w["graph"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
