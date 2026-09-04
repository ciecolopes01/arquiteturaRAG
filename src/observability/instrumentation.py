"""
Observabilidade do pipeline RAG — Passo 14.

Instrumenta cada estágio do pipeline para medir:
  - Latência por estágio (ms)
  - Custo estimado por query (USD)
  - Tokens consumidos (LLM)
  - Path escolhido (fast/full)
  - Recall@5 histórico

Design: decorator @trace_stage + contexto de trace thread-local.
Não exige reescrever a lógica dos módulos existentes — apenas
adicionar @trace_stage nas funções a instrumentar.

Uso básico:
    from src.observability.instrumentation import pipeline_trace, get_current_trace

    with pipeline_trace(query="O que é RAG?") as trace:
        results = hybrid_search(query)
        answer_result = answer(query)
    # trace.to_json() — log estruturado do que aconteceu

Uso com decorator (para funções internas de cada estágio):
    from src.observability.instrumentation import trace_stage

    @trace_stage("retrieval.dense")
    def search_dense(query, top_k):
        ...

Log JSON estruturado gerado por log_trace():
    {
      "ts":            "2026-09-01T21:00:00Z",
      "query_hash":    "a3f9...",   # hash curto da query (não armazena a query raw)
      "path":          "fast",
      "stages": {
        "retrieval.embedding": 320,
        "retrieval.dense":     85,
        "retrieval.bm25":      42,
        "retrieval.fusion":    5,
        "retrieval.reranking": 0,
        "generation":          1850
      },
      "tokens":        {"input": 820, "output": 312, "total": 1132},
      "cost_usd":      0.00041,
      "recall_at_5_latest": null   # preenchido quando avaliação offline é rodada
    }
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuração do logger estruturado
# ---------------------------------------------------------------------------

LOG_DIR = Path("logs")

def _setup_logger() -> logging.Logger:
    """Configura logger que emite JSON em arquivo + console."""
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / "pipeline.jsonl"

    logger = logging.getLogger("rag.observability")
    logger.setLevel(logging.DEBUG)

    # Evita duplicar handlers se o módulo for recarregado
    if logger.handlers:
        return logger

    # Handler para arquivo JSONL (um JSON por linha)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(message)s"))  # mensagem já é JSON
    logger.addHandler(fh)

    # Handler para console (legível)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[obs] %(message)s"))
    logger.addHandler(ch)

    return logger


_logger = _setup_logger()


# ---------------------------------------------------------------------------
# Estimativa de custo (preços Anthropic, dez/2024)
# ---------------------------------------------------------------------------

# USD por 1M tokens (entrada / saída)
_COST_TABLE: Dict[str, Dict[str, float]] = {
    "claude-3-5-haiku-20241022":   {"input": 0.80,  "output": 4.00},
    "claude-3-5-sonnet-20241022":  {"input": 3.00,  "output": 15.00},
    "claude-3-opus-20240229":      {"input": 15.00, "output": 75.00},
    "claude-haiku-3-5":            {"input": 0.80,  "output": 4.00},
    # fallback genérico
    "__default__":                 {"input": 1.00,  "output": 5.00},
}


def estimate_llm_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estima o custo em USD de uma chamada ao LLM.

    Parâmetros:
    - model         : nome do modelo Claude usado.
    - input_tokens  : tokens de entrada (prompt).
    - output_tokens : tokens de saída (resposta).

    Retorna custo em USD (float, arredondado para 6 casas).
    """
    pricing = _COST_TABLE.get(model, _COST_TABLE["__default__"])
    cost = (
        input_tokens  / 1_000_000 * pricing["input"]
        + output_tokens / 1_000_000 * pricing["output"]
    )
    return round(cost, 6)


# ---------------------------------------------------------------------------
# PipelineTrace: acumula métricas de uma query
# ---------------------------------------------------------------------------

@dataclass
class StageMetric:
    """Métrica de um estágio individual."""
    name: str
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineTrace:
    """
    Contexto de observabilidade de uma query ponta a ponta.

    Criado pelo context manager `pipeline_trace(query)` e preenchido
    pelos decorators @trace_stage das funções instrumentadas.
    """
    query: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    path: str = "unknown"          # "fast" | "full" | "unknown"
    stages: List[StageMetric] = field(default_factory=list)
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    recall_at_5: Optional[float] = None   # preenchido por log_recall_to_trace()
    extra: Dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------------
    # Manipulação de estágios
    # --------------------------------------------------------

    def record_stage(self, name: str, latency_ms: float, **meta) -> None:
        """Registra o resultado de um estágio."""
        self.stages.append(StageMetric(name=name, latency_ms=latency_ms, metadata=meta))

    def total_latency_ms(self) -> float:
        """Soma a latência de todos os estágios registrados."""
        return sum(s.latency_ms for s in self.stages)

    def stage_dict(self) -> Dict[str, float]:
        """Retorna dict {nome_estágio: latência_ms} para serialização."""
        return {s.name: round(s.latency_ms, 1) for s in self.stages}

    # --------------------------------------------------------
    # Serialização
    # --------------------------------------------------------

    def _query_hash(self) -> str:
        """Hash curto da query (8 chars). Não armazena texto da query no log."""
        return hashlib.md5(self.query.encode()).hexdigest()[:8]

    def to_dict(self) -> dict:
        """Serializa para dict adequado ao log JSON."""
        return {
            "ts":            self.started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "query_hash":    self._query_hash(),
            "path":          self.path,
            "stages":        self.stage_dict(),
            "total_ms":      round(self.total_latency_ms(), 1),
            "tokens": {
                "input":  self.tokens_input,
                "output": self.tokens_output,
                "total":  self.tokens_input + self.tokens_output,
            },
            "cost_usd":      self.cost_usd,
            "recall_at_5_latest": self.recall_at_5,
            **self.extra,
        }

    def to_json(self) -> str:
        """Serializa para string JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def pretty_print(self) -> None:
        """Imprime resumo formatado de latência + custo."""
        print("\n" + "─" * 65)
        print(f"[Observabilidade] Query hash: {self._query_hash()} | Path: {self.path}")
        print(f"{'Estágio':<30} {'Latência (ms)':>15}")
        print("─" * 65)
        for s in self.stages:
            bar = "█" * min(int(s.latency_ms / 100), 30)
            print(f"  {s.name:<28} {s.latency_ms:>12.1f}  {bar}")
        print("─" * 65)
        print(f"  {'TOTAL':<28} {self.total_latency_ms():>12.1f}")
        if self.tokens_input or self.tokens_output:
            print(f"\n  Tokens: {self.tokens_input} entrada + {self.tokens_output} saída")
            print(f"  Custo estimado: USD {self.cost_usd:.6f}")
        if self.recall_at_5 is not None:
            print(f"  Recall@5 (último): {self.recall_at_5:.4f}")
        print("─" * 65)


# ---------------------------------------------------------------------------
# Thread-local: trace ativo no thread atual
# ---------------------------------------------------------------------------

_local = threading.local()


def get_current_trace() -> Optional[PipelineTrace]:
    """
    Retorna o PipelineTrace ativo no thread atual, ou None se não houver.

    Usado pelas funções instrumentadas para registrar suas métricas
    sem precisar receber o trace por parâmetro.
    """
    return getattr(_local, "active_trace", None)


def _set_current_trace(trace: Optional[PipelineTrace]) -> None:
    _local.active_trace = trace


# ---------------------------------------------------------------------------
# Context manager: abre e fecha um trace de pipeline
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def pipeline_trace(query: str, log: bool = True):
    """
    Context manager que cria um PipelineTrace para uma query.

    Uso:
        with pipeline_trace("O que é RAG?") as trace:
            results = hybrid_search(query)
            result  = answer(query)
        # trace agora contém todas as métricas

    Parâmetros:
    - query : pergunta do usuário (armazenada apenas como hash no log).
    - log   : se True, chama log_trace(trace) ao sair do contexto.

    Yields o PipelineTrace para uso dentro do bloco.
    """
    trace = PipelineTrace(query=query)
    _set_current_trace(trace)
    try:
        yield trace
    finally:
        _set_current_trace(None)
        if log:
            log_trace(trace)


# ---------------------------------------------------------------------------
# Decorator @trace_stage
# ---------------------------------------------------------------------------

def trace_stage(stage_name: str):
    """
    Decorator que mede a latência de uma função e registra no trace ativo.

    Uso:
        @trace_stage("retrieval.dense")
        def search_dense(query, top_k):
            ...

    Se não houver trace ativo (função chamada fora de um pipeline_trace),
    o decorator é transparente — não faz nada além de chamar a função.

    Parâmetros extras podem ser passados pela função decorada via
    retorno de um dict com chave "_trace_meta" (removido antes de retornar).
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            trace = get_current_trace()
            t0 = time.time()
            result = func(*args, **kwargs)
            elapsed_ms = (time.time() - t0) * 1000

            if trace is not None:
                trace.record_stage(stage_name, elapsed_ms)

            return result
        wrapper.__name__ = func.__name__
        wrapper.__doc__  = func.__doc__
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_trace(trace: PipelineTrace) -> None:
    """
    Emite o trace como linha JSON no log estruturado (logs/pipeline.jsonl).

    Também imprime um resumo legível no console via logger INFO.
    """
    payload = trace.to_dict()
    _logger.debug(json.dumps(payload, ensure_ascii=False))

    # Resumo legível no console
    stage_summary = " | ".join(
        f"{k}={v:.0f}ms" for k, v in payload["stages"].items() if v > 0
    )
    _logger.info(
        f"hash={payload['query_hash']} path={payload['path']} "
        f"total={payload['total_ms']:.0f}ms cost=${payload['cost_usd']:.5f} "
        f"[{stage_summary}]"
    )


# ---------------------------------------------------------------------------
# Registro de tokens e custo (chamado pelo módulo de geração)
# ---------------------------------------------------------------------------

def record_generation_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
) -> None:
    """
    Registra tokens e custo de geração no trace ativo.

    Chamado por src/generation/answer.py após cada chamada real ao LLM.
    Se não houver trace ativo, é no-op.
    """
    trace = get_current_trace()
    if trace is None:
        return
    trace.tokens_input  += input_tokens
    trace.tokens_output += output_tokens
    cost = estimate_llm_cost(model, input_tokens, output_tokens)
    trace.cost_usd      += cost
    trace.record_stage("generation", latency_ms, model=model, cost_usd=cost)


# ---------------------------------------------------------------------------
# Registro de métricas de roteamento (chamado por routing.py)
# ---------------------------------------------------------------------------

def record_routing_decision(path: str, complexity: float) -> None:
    """
    Registra o path escolhido (fast/full) e a complexidade estimada no trace.

    Chamado por src/retrieval/routing.py.
    """
    trace = get_current_trace()
    if trace is None:
        return
    trace.path = path
    trace.extra["complexity"] = complexity


# ---------------------------------------------------------------------------
# Histórico de Recall@5
# ---------------------------------------------------------------------------

RECALL_HISTORY_FILE = Path("logs/recall_history.jsonl")


def log_recall_to_history(
    recall_at_5: float,
    mrr: float,
    n_questions: int,
    mode: str,
    ms_per_query: float,
) -> None:
    """
    Appenda uma linha ao histórico de Recall@5 (logs/recall_history.jsonl).

    Chamado por evaluate_recall_at_k() toda vez que é executado.
    Permite detectar degradação ao longo do tempo comparando execuções
    históricas.

    Formato da linha:
        {
          "ts": "...",
          "recall_at_5": 0.9697,
          "mrr": 0.7864,
          "n_questions": 33,
          "mode": "hybrid",
          "ms_per_query": 1561.0
        }
    """
    LOG_DIR.mkdir(exist_ok=True)
    entry = {
        "ts":          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "recall_at_5": recall_at_5,
        "mrr":         mrr,
        "n_questions": n_questions,
        "mode":        mode,
        "ms_per_query": ms_per_query,
    }
    with open(RECALL_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    _logger.info(
        f"[Recall@5] recall={recall_at_5:.4f} mrr={mrr:.4f} "
        f"n={n_questions} mode={mode} ms/q={ms_per_query:.0f}"
    )


def load_recall_history() -> List[dict]:
    """
    Carrega o histórico de Recall@5 do arquivo JSONL.

    Retorna lista de dicts ordenada do mais antigo para o mais recente.
    """
    if not RECALL_HISTORY_FILE.exists():
        return []
    entries = []
    with open(RECALL_HISTORY_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def check_recall_drift(
    threshold_drop: float = 0.05,
    window: int = 5,
) -> Optional[str]:
    """
    Verifica se houve queda de Recall@5 acima do threshold nas últimas
    `window` execuções em relação à melhor execução anterior.

    Parâmetros:
    - threshold_drop : queda percentual que dispara alerta (padrão 5%).
    - window         : número de execuções recentes a considerar.

    Retorna string de alerta se houver degradação, None caso contrário.
    """
    history = load_recall_history()
    if len(history) < 2:
        return None

    recent  = [e["recall_at_5"] for e in history[-window:]]
    best    = max(e["recall_at_5"] for e in history[:-window] or history)
    current = recent[-1]

    if best - current >= threshold_drop:
        msg = (
            f"[ALERTA] Recall@5 caiu {best - current:.3f} "
            f"(de {best:.4f} para {current:.4f}) -- "
            f"threshold={threshold_drop}. Verificar pipeline."
        )
        _logger.warning(msg)
        return msg
    return None
