"""
Avaliação offline (Recall@K) — Passo 7.

Calcula métricas de retrieval contra o golden dataset usando o pipeline
de retrieval configurado (híbrido, dense isolado, ou BM25 isolado).

Métricas implementadas:
  - Recall@K      : proporção de perguntas com chunk correto no top-K.
  - MRR           : Mean Reciprocal Rank (posição do 1º chunk correto).
  - Hit Rate@K    : alias de Recall@K (mesmo cálculo, nome comum em RAG).

Modos de retrieval:
  - "hybrid" (padrão) : dense + BM25 via RRF
  - "dense"            : só Qdrant
  - "bm25"             : só Postgres
  - "rerank"           : hybrid (candidatos maiores) + cross-encoder (Passo 9)

Uso:
    from src.evaluation.offline_eval import evaluate_recall_at_k, load_golden_dataset

    dataset = load_golden_dataset()
    result = evaluate_recall_at_k(dataset, k=5, mode="hybrid", tenant_id="tenant-test")
    print(result)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

# Observabilidade (Passo 14)
try:
    from src.observability.instrumentation import (
        check_recall_drift,
        log_recall_to_history,
    )
    _OBS_AVAILABLE = True
except ImportError:
    _OBS_AVAILABLE = False
    def log_recall_to_history(*a, **kw): pass
    def check_recall_drift(*a, **kw): return None

GOLDEN_DATASET_PATH = Path("data/golden_dataset.json")


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def load_golden_dataset(path: str | Path = GOLDEN_DATASET_PATH) -> List[dict]:
    """Carrega o golden dataset de disco."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Golden dataset não encontrado em '{path}'. "
            "Execute o Passo 7 para criá-lo."
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Helpers de retrieval por modo
# ---------------------------------------------------------------------------

def _retrieve(
    query: str,
    top_k: int,
    mode: str,
    tenant_id: Optional[str] = None,
    principals: Optional[List[str]] = None,
    rerank_candidates: int = 20,
) -> List[str]:
    """
    Executa o retrieval no modo especificado e retorna lista de chunk_ids
    ordenada por relevância (do mais relevante ao menos).
    """
    if mode == "hybrid":
        from src.retrieval.hybrid import hybrid_search
        results = hybrid_search(query, top_k=top_k, tenant_id=tenant_id, principals=principals)
        return [r.chunk_id for r in results]

    elif mode == "dense":
        from src.embedding.embed_index import search_dense
        results = search_dense(query, top_k=top_k, tenant_id=tenant_id, principals=principals)
        return [r["chunk_id"] for r in results]

    elif mode == "bm25":
        from src.retrieval.bm25_index import search_bm25
        results = search_bm25(query, top_k=top_k, tenant_id=tenant_id, principals=principals)
        return [r["chunk_id"] for r in results]

    elif mode == "rerank":
        from src.retrieval.hybrid import hybrid_search
        from src.retrieval.reranker import rerank
        candidates = hybrid_search(
            query,
            top_k=rerank_candidates,
            tenant_id=tenant_id,
            principals=principals,
            dense_candidates=rerank_candidates,
            bm25_candidates=rerank_candidates,
        )
        results = rerank(query, candidates, top_k=top_k)
        return [r.chunk_id for r in results]

    else:
        raise ValueError(f"Modo desconhecido: '{mode}'. Use 'hybrid', 'dense', 'bm25' ou 'rerank'.")


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def _reciprocal_rank(retrieved: List[str], expected: List[str]) -> float:
    """Retorna 1/rank do primeiro acerto, ou 0 se não houver acerto."""
    expected_set = set(expected)
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in expected_set:
            return 1.0 / rank
    return 0.0


def _hit(retrieved: List[str], expected: List[str]) -> bool:
    """True se pelo menos um chunk esperado aparece nos resultados."""
    return bool(set(retrieved) & set(expected))


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def evaluate_recall_at_k(
    dataset: List[dict],
    k: int = 5,
    mode: str = "hybrid",
    tenant_id: Optional[str] = None,
    principals: Optional[List[str]] = None,
    verbose: bool = True,
    rerank_candidates: int = 20,
) -> Dict[str, float]:
    """
    Calcula Recall@K e MRR sobre o golden dataset.

    Parâmetros:
    - dataset   : lista de entradas do golden dataset (de load_golden_dataset).
    - k         : tamanho do top-K para avaliação.
    - mode      : "hybrid" | "dense" | "bm25" | "rerank".
    - tenant_id : filtro de tenant aplicado em todos os retrievals.
    - principals: lista de identificadores de acesso do usuário.
    - verbose   : imprime resultado por pergunta se True.
    - rerank_candidates : nº de candidatos do hybrid_search antes do cross-encoder (modo "rerank").

    Retorna dict com:
    - recall_at_k : float [0,1]
    - mrr         : float [0,1]
    - hit_rate_at_k: float (alias de recall_at_k)
    - n_questions : int
    - k           : int
    - mode        : str
    - total_time_s: float
    - ms_per_query: float
    """
    hits = 0
    rr_sum = 0.0
    t0 = time.time()

    if verbose:
        print(f"\n[eval] Avaliando {len(dataset)} perguntas | modo={mode} | k={k}")
        print(f"{'#':<4} {'Hit':^5} {'RR':^6} {'Pergunta':<55} {'Esperado encontrado'}")
        print("-" * 100)

    for entry in dataset:
        qid      = entry.get("id", "?")
        question = entry["question"]
        expected = entry["expected_chunk_ids"]

        retrieved = _retrieve(
            question, top_k=k, mode=mode, tenant_id=tenant_id, principals=principals,
            rerank_candidates=rerank_candidates,
        )

        hit     = _hit(retrieved, expected)
        rr      = _reciprocal_rank(retrieved, expected)

        hits    += int(hit)
        rr_sum  += rr

        if verbose:
            hit_str = "OK" if hit else "--"
            q_short = question[:53]
            found   = [c for c in expected if c in retrieved]
            print(f"{qid:<4} {hit_str:^5} {rr:^6.3f} {q_short:<55} {found}")

    total_time = time.time() - t0
    n = len(dataset)

    result = {
        "recall_at_k":   round(hits / n, 4),
        "hit_rate_at_k": round(hits / n, 4),
        "mrr":           round(rr_sum / n, 4),
        "hits":          hits,
        "n_questions":   n,
        "k":             k,
        "mode":          mode,
        "total_time_s":  round(total_time, 2),
        "ms_per_query":  round(total_time / n * 1000, 1),
    }

    if verbose:
        print("-" * 100)
        print(f"\n[eval] Recall@{k}={result['recall_at_k']:.4f} | "
              f"MRR={result['mrr']:.4f} | "
              f"{hits}/{n} hits | "
              f"{result['ms_per_query']}ms/query | "
              f"modo={mode}")

    # Persiste no histórico de Recall@5 (Passo 14)
    log_recall_to_history(
        recall_at_5=result["recall_at_k"],
        mrr=result["mrr"],
        n_questions=n,
        mode=mode,
        ms_per_query=result["ms_per_query"],
    )
    # Verifica se houve degradação desde a melhor execução histórica
    drift_alert = check_recall_drift(threshold_drop=0.05)
    if drift_alert:
        print(f"\n{drift_alert}")

    return result


# ---------------------------------------------------------------------------
# CLI simples
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    mode      = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
    k         = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    tenant_id = sys.argv[3] if len(sys.argv) > 3 else "tenant-test"

    dataset = load_golden_dataset()
    result  = evaluate_recall_at_k(dataset, k=k, mode=mode, tenant_id=tenant_id)

    print(f"\nBaseline Recall@{k} ({mode}): {result['recall_at_k']:.4f}")
    print(f"Baseline MRR ({mode}):         {result['mrr']:.4f}")
