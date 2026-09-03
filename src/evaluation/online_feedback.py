"""
Avaliação Online + Loop de Feedback — Passo 25.

Fecha o ciclo de melhoria contínua a partir do uso real:
1. Captura de feedback explícito (upvote/downvote) e metadados da consulta.
2. Triage de queries insatisfatórias (downvotes) para enriquecimento do golden dataset.
3. Monitoramento de série temporal com detecção de degradação de recall.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.offline_eval import load_golden_dataset

FEEDBACK_LOG_PATH = Path("logs/online_feedback.jsonl")
GOLDEN_DATASET_PATH = Path("data/golden_dataset.json")


class FeedbackRating(str, Enum):
    UPVOTE = "upvote"
    DOWNVOTE = "downvote"


@dataclass
class FeedbackEntry:
    id: str
    query: str
    retrieved_chunk_ids: List[str]
    answer: str
    rating: FeedbackRating
    comment: Optional[str] = None
    tenant_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    promoted_to_golden: bool = False


def record_feedback(
    query: str,
    retrieved_chunk_ids: List[str],
    answer: str,
    rating: FeedbackRating | str,
    comment: Optional[str] = None,
    tenant_id: Optional[str] = None,
    log_path: Path = FEEDBACK_LOG_PATH,
) -> FeedbackEntry:
    """Registra uma interação com feedback do usuário final em logs/online_feedback.jsonl."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(rating, str):
        rating = FeedbackRating(rating.lower())

    entry = FeedbackEntry(
        id=f"fb-{uuid.uuid4().hex[:8]}",
        query=query.strip(),
        retrieved_chunk_ids=retrieved_chunk_ids,
        answer=answer.strip(),
        rating=rating,
        comment=comment,
        tenant_id=tenant_id,
    )

    data = asdict(entry)
    data["rating"] = entry.rating.value

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

    return entry


def load_all_feedback(log_path: Path = FEEDBACK_LOG_PATH) -> List[FeedbackEntry]:
    """Carrega todas as entradas de feedback registradas."""
    if not log_path.exists():
        return []

    entries: List[FeedbackEntry] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                entries.append(
                    FeedbackEntry(
                        id=d["id"],
                        query=d["query"],
                        retrieved_chunk_ids=d.get("retrieved_chunk_ids", []),
                        answer=d.get("answer", ""),
                        rating=FeedbackRating(d["rating"]),
                        comment=d.get("comment"),
                        tenant_id=d.get("tenant_id"),
                        created_at=d.get("created_at", time.time()),
                        promoted_to_golden=d.get("promoted_to_golden", False),
                    )
                )
            except Exception:
                continue
    return entries


def get_downvoted_queries(log_path: Path = FEEDBACK_LOG_PATH) -> List[FeedbackEntry]:
    """Filtra queries com feedback negativo que ainda não viraram caso de teste."""
    entries = load_all_feedback(log_path=log_path)
    return [e for e in entries if e.rating == FeedbackRating.DOWNVOTE and not e.promoted_to_golden]


def promote_feedback_to_golden_dataset(
    feedback_id: str,
    expected_chunk_ids: List[str],
    notes: str,
    golden_dataset_path: Path = GOLDEN_DATASET_PATH,
    log_path: Path = FEEDBACK_LOG_PATH,
) -> Dict[str, Any]:
    """
    Promove uma query insatisfatória registrada pelo usuário para o golden dataset oficial.
    """
    entries = load_all_feedback(log_path=log_path)
    target = None
    for e in entries:
        if e.id == feedback_id:
            target = e
            break

    if not target:
        raise ValueError(f"Feedback ID '{feedback_id}' não encontrado.")

    # Carrega golden dataset atual
    with open(golden_dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Novo ID para golden dataset
    existing_nums = []
    for d in dataset:
        qid = d.get("id", "")
        if qid.startswith("gd-"):
            try:
                existing_nums.append(int(qid.split("-")[1]))
            except ValueError:
                pass
    next_num = (max(existing_nums) + 1) if existing_nums else 1
    new_gd_id = f"gd-{next_num}"

    new_item = {
        "id": new_gd_id,
        "question": target.query,
        "expected_chunk_ids": expected_chunk_ids,
        "source_doc": "promoted_from_online_feedback",
        "notes": f"Passo 25 (Feedback loop): {notes}",
    }
    dataset.append(new_item)

    with open(golden_dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    # Marca feedback como promovido
    for e in entries:
        if e.id == feedback_id:
            e.promoted_to_golden = True

    with open(log_path, "w", encoding="utf-8") as f:
        for e in entries:
            d = asdict(e)
            d["rating"] = e.rating.value
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    return {
        "status": "PROMOTED",
        "golden_id": new_gd_id,
        "feedback_id": feedback_id,
        "question": target.query,
    }


def calculate_satisfaction_rate(entries: Optional[List[FeedbackEntry]] = None) -> Dict[str, Any]:
    """Calcula taxa de satisfação do usuário (% de upvotes)."""
    items = entries if entries is not None else load_all_feedback()
    if not items:
        return {"total_feedback": 0, "satisfaction_rate": 1.0, "upvotes": 0, "downvotes": 0}

    upvotes = sum(1 for e in items if e.rating == FeedbackRating.UPVOTE)
    downvotes = sum(1 for e in items if e.rating == FeedbackRating.DOWNVOTE)
    total = len(items)

    return {
        "total_feedback": total,
        "upvotes": upvotes,
        "downvotes": downvotes,
        "satisfaction_rate": round(upvotes / total, 4),
    }


def check_degradation_alert(
    current_recall: float,
    baseline_recall: float,
    threshold_pct: float = 0.05,
) -> Dict[str, Any]:
    """
    Detecta se houve queda de qualidade superior ao limite estipulado (ex: 5 p.p.).
    Retorna alerta estruturado.
    """
    delta = current_recall - baseline_recall
    has_drift = delta < -threshold_pct

    return {
        "has_drift": has_drift,
        "current_recall": round(current_recall, 4),
        "baseline_recall": round(baseline_recall, 4),
        "delta": round(delta, 4),
        "threshold": threshold_pct,
        "status": "ALERT_DEGRADATION" if has_drift else "STABLE",
        "message": (
            f"ALERTA: Degradação de qualidade detectada! Queda de {abs(delta)*100:.2f}% "
            f"(baseline={baseline_recall:.4f}, atual={current_recall:.4f})"
            if has_drift
            else f"Qualidade estável: delta={delta:+.4f} dentro do limiar de {threshold_pct*100:.1f}%."
        ),
    }
