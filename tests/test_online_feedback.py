"""
Testes de Avaliação Online e Loop de Feedback — Passo 25.

Valida:
1. Registro de feedback de usuários (upvotes/downvotes) com metadados.
2. Identificação e filtragem de queries insatisfatórias (downvotes).
3. Promoção automática de feedback para o golden dataset oficial.
4. Cálculo da taxa de satisfação do usuário.
5. Emissão de alerta em caso de degradação de recall rolling (>= 5 p.p.).

Rodar com: pytest tests/test_online_feedback.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from src.evaluation.online_feedback import (
    FeedbackRating,
    calculate_satisfaction_rate,
    check_degradation_alert,
    get_downvoted_queries,
    load_all_feedback,
    promote_feedback_to_golden_dataset,
    record_feedback,
)


@pytest.fixture
def temp_feedback_log(tmp_path: Path) -> Path:
    return tmp_path / "online_feedback_test.jsonl"


@pytest.fixture
def temp_golden_dataset(tmp_path: Path) -> Path:
    p = tmp_path / "golden_dataset_test.json"
    data = [
        {
            "id": "gd-01",
            "question": "Pergunta inicial de teste",
            "expected_chunk_ids": ["c-1"],
            "source_doc": "doc-1",
            "notes": "teste",
        }
    ]
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def test_record_and_load_feedback(temp_feedback_log: Path) -> None:
    """Verifica registro e recuperação de feedback estruturado."""
    fb1 = record_feedback(
        query="Como funciona a fusão RRF?",
        retrieved_chunk_ids=["c-1", "c-2"],
        answer="O RRF combina listas ranqueadas usando k=60...",
        rating=FeedbackRating.UPVOTE,
        comment="Excelente resposta!",
        log_path=temp_feedback_log,
    )
    assert fb1.id.startswith("fb-")

    fb2 = record_feedback(
        query="Qual a cotação do dólar na planilha de custos?",
        retrieved_chunk_ids=["c-9"],
        answer="Não foi encontrada a informação de cotação.",
        rating=FeedbackRating.DOWNVOTE,
        comment="Faltou valor exato",
        log_path=temp_feedback_log,
    )

    all_fb = load_all_feedback(log_path=temp_feedback_log)
    assert len(all_fb) == 2

    downvoted = get_downvoted_queries(log_path=temp_feedback_log)
    assert len(downvoted) == 1
    assert downvoted[0].id == fb2.id


def test_promote_feedback_to_golden_dataset(
    temp_feedback_log: Path,
    temp_golden_dataset: Path,
) -> None:
    """Valida o loop de feedback enriquecendo o golden dataset com caso real."""
    fb = record_feedback(
        query="Qual o tempo de retenção da coleção legada após migração?",
        retrieved_chunk_ids=["c-20"],
        answer="Resposta incompleta...",
        rating=FeedbackRating.DOWNVOTE,
        log_path=temp_feedback_log,
    )

    res = promote_feedback_to_golden_dataset(
        feedback_id=fb.id,
        expected_chunk_ids=["c-expected-30-days"],
        notes="Usuário questionou período de graça após migração",
        golden_dataset_path=temp_golden_dataset,
        log_path=temp_feedback_log,
    )

    assert res["status"] == "PROMOTED"
    assert res["golden_id"] == "gd-2"

    # Confirma que foi adicionado ao arquivo
    with open(temp_golden_dataset, "r", encoding="utf-8") as f:
        gd = json.load(f)
    assert len(gd) == 2
    assert gd[1]["id"] == "gd-2"
    assert gd[1]["question"] == fb.query

    # Confirma que a flag promovida foi atualizada no log de feedback
    downvoted = get_downvoted_queries(log_path=temp_feedback_log)
    assert len(downvoted) == 0


def test_calculate_satisfaction_rate(temp_feedback_log: Path) -> None:
    """Verifica taxa de satisfação (% upvotes)."""
    record_feedback("Q1", ["c1"], "A1", FeedbackRating.UPVOTE, log_path=temp_feedback_log)
    record_feedback("Q2", ["c2"], "A2", FeedbackRating.UPVOTE, log_path=temp_feedback_log)
    record_feedback("Q3", ["c3"], "A3", FeedbackRating.DOWNVOTE, log_path=temp_feedback_log)

    entries = load_all_feedback(log_path=temp_feedback_log)
    stats = calculate_satisfaction_rate(entries)
    assert stats["total_feedback"] == 3
    assert stats["upvotes"] == 2
    assert stats["downvotes"] == 1
    assert stats["satisfaction_rate"] == pytest.approx(0.6667, rel=1e-3)


def test_check_degradation_alert() -> None:
    """Verifica gatilho de alerta para queda de qualidade."""
    # Queda de 6 p.p. (> 5% threshold)
    alert = check_degradation_alert(current_recall=0.88, baseline_recall=0.94, threshold_pct=0.05)
    assert alert["has_drift"] is True
    assert alert["status"] == "ALERT_DEGRADATION"

    # Estável (queda de 2 p.p. < 5%)
    stable = check_degradation_alert(current_recall=0.93, baseline_recall=0.95, threshold_pct=0.05)
    assert stable["has_drift"] is False
    assert stable["status"] == "STABLE"
