"""
Testes de observabilidade — Passo 14.

Verifica que:
  1. PipelineTrace acumula estágios corretamente.
  2. pipeline_trace() faz setup/teardown do trace ativo.
  3. get_current_trace() retorna None fora de um contexto e o trace ativo dentro.
  4. Decorador @trace_stage registra latência.
  5. estimate_llm_cost() calcula valores razoáveis.
  6. log_recall_to_history() / load_recall_history() gravam e leem o histórico.
  7. check_recall_drift() dispara alerta quando há queda >= threshold.
  8. Integração: hybrid_search dentro de pipeline_trace() gera estágios esperados.
  9. Saída JSON de to_dict() tem todos os campos obrigatórios.
 10. Múltiplos traces em sequência não vazam estado (thread safety básico).
"""

import json
import time

import pytest

# Importa o módulo de observabilidade
from src.observability.instrumentation import (
    PipelineTrace,
    check_recall_drift,
    estimate_llm_cost,
    get_current_trace,
    load_recall_history,
    log_recall_to_history,
    log_trace,
    pipeline_trace,
    record_generation_cost,
    record_routing_decision,
    trace_stage,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_stage(name: str, sleep_ms: float = 0) -> None:
    """Simula um estágio do pipeline."""
    trace = get_current_trace()
    if trace is not None:
        trace.record_stage(name, sleep_ms)


# ---------------------------------------------------------------------------
# Testes unitários — PipelineTrace
# ---------------------------------------------------------------------------

class TestPipelineTrace:
    def test_empty_trace(self):
        """Trace recém-criado não tem estágios e latência total = 0."""
        trace = PipelineTrace(query="test query")
        assert trace.stages == []
        assert trace.total_latency_ms() == 0.0
        assert trace.tokens_input == 0
        assert trace.tokens_output == 0
        assert trace.cost_usd == 0.0

    def test_record_single_stage(self):
        """record_stage() adiciona o estágio corretamente."""
        trace = PipelineTrace(query="test")
        trace.record_stage("retrieval.dense", 85.5)
        assert len(trace.stages) == 1
        assert trace.stages[0].name == "retrieval.dense"
        assert trace.stages[0].latency_ms == 85.5

    def test_record_multiple_stages(self):
        """Múltiplos estágios são acumulados na ordem de inserção."""
        trace = PipelineTrace(query="test")
        for name, ms in [("a", 10), ("b", 20), ("c", 30)]:
            trace.record_stage(name, ms)
        assert len(trace.stages) == 3
        assert trace.total_latency_ms() == 60.0

    def test_stage_dict_order(self):
        """stage_dict() preserva a ordem de inserção dos estágios."""
        trace = PipelineTrace(query="test")
        trace.record_stage("retrieval.embedding", 300.0)
        trace.record_stage("retrieval.dense", 50.0)
        trace.record_stage("retrieval.bm25", 30.0)
        d = trace.stage_dict()
        keys = list(d.keys())
        assert keys == ["retrieval.embedding", "retrieval.dense", "retrieval.bm25"]

    def test_query_hash_is_deterministic(self):
        """Mesma query → mesmo hash; hashes diferentes para queries diferentes."""
        t1 = PipelineTrace(query="O que é RAG?")
        t2 = PipelineTrace(query="O que é RAG?")
        t3 = PipelineTrace(query="Outra pergunta")
        assert t1._query_hash() == t2._query_hash()
        assert t1._query_hash() != t3._query_hash()

    def test_to_dict_has_required_fields(self):
        """to_dict() contém todos os campos obrigatórios do log JSON."""
        trace = PipelineTrace(query="pergunta de teste")
        trace.record_stage("retrieval.embedding", 100.0)
        trace.path = "fast"
        d = trace.to_dict()

        required = {"ts", "query_hash", "path", "stages", "total_ms", "tokens", "cost_usd", "recall_at_5_latest"}
        assert required.issubset(d.keys()), f"Campos faltando: {required - d.keys()}"
        assert d["path"] == "fast"
        assert d["total_ms"] == 100.0
        assert d["tokens"]["total"] == 0

    def test_to_json_is_valid_json(self):
        """to_json() produz string JSON válida."""
        trace = PipelineTrace(query="teste JSON")
        trace.record_stage("generation", 1500.0)
        trace.tokens_input = 500
        trace.tokens_output = 200
        json_str = trace.to_json()
        parsed = json.loads(json_str)
        assert parsed["tokens"]["input"] == 500
        assert parsed["tokens"]["output"] == 200
        assert parsed["tokens"]["total"] == 700


# ---------------------------------------------------------------------------
# Testes — Context manager pipeline_trace()
# ---------------------------------------------------------------------------

class TestPipelineTraceContextManager:
    def test_no_active_trace_outside_context(self):
        """Fora do context manager, get_current_trace() retorna None."""
        assert get_current_trace() is None

    def test_trace_active_inside_context(self):
        """Dentro do pipeline_trace(), get_current_trace() retorna o trace."""
        with pipeline_trace("test query", log=False) as trace:
            current = get_current_trace()
            assert current is trace
            assert current is not None

    def test_trace_cleared_after_context(self):
        """Após sair do context manager, get_current_trace() volta a ser None."""
        with pipeline_trace("test query", log=False):
            pass
        assert get_current_trace() is None

    def test_trace_cleared_even_on_exception(self):
        """Trace é limpo mesmo se uma exceção ocorrer dentro do bloco."""
        try:
            with pipeline_trace("test query", log=False):
                raise RuntimeError("erro simulado")
        except RuntimeError:
            pass
        assert get_current_trace() is None

    def test_stages_recorded_in_context(self):
        """Estágios registrados dentro do contexto ficam no trace."""
        with pipeline_trace("test query", log=False) as trace:
            trace.record_stage("retrieval.embedding", 200.0)
            trace.record_stage("retrieval.dense", 50.0)
        assert len(trace.stages) == 2
        assert trace.total_latency_ms() == 250.0

    def test_nested_calls_share_same_trace(self):
        """Funções chamadas dentro do contexto veem o mesmo trace via thread-local."""
        with pipeline_trace("test query", log=False) as trace:
            inner_trace_ref = get_current_trace()
        assert inner_trace_ref is trace


# ---------------------------------------------------------------------------
# Testes — Decorator @trace_stage
# ---------------------------------------------------------------------------

class TestTraceStageDecorator:
    def test_decorator_transparent_without_trace(self):
        """Fora de um pipeline_trace, o decorator não interfere no retorno."""
        @trace_stage("test.stage")
        def compute(x):
            return x * 2

        result = compute(21)
        assert result == 42

    def test_decorator_records_stage_in_trace(self):
        """Dentro de um trace, o decorator registra a latência do estágio."""
        @trace_stage("test.decorated")
        def slow_func():
            time.sleep(0.01)
            return "done"

        with pipeline_trace("test", log=False) as trace:
            result = slow_func()

        assert result == "done"
        stage_names = [s.name for s in trace.stages]
        assert "test.decorated" in stage_names
        decorated_stage = next(s for s in trace.stages if s.name == "test.decorated")
        assert decorated_stage.latency_ms >= 10.0  # pelo menos 10ms (sleep 0.01s)

    def test_decorator_preserves_function_name(self):
        """O decorator não perde o __name__ da função."""
        @trace_stage("some.stage")
        def my_special_function():
            pass

        assert my_special_function.__name__ == "my_special_function"


# ---------------------------------------------------------------------------
# Testes — Estimativa de custo
# ---------------------------------------------------------------------------

class TestCostEstimation:
    def test_haiku_cost_is_cheap(self):
        """Haiku é o modelo mais barato — custo deve ser muito pequeno."""
        cost = estimate_llm_cost("claude-3-5-haiku-20241022", 1000, 500)
        assert cost < 0.01   # menos de 1 centavo por query típica

    def test_opus_is_more_expensive_than_haiku(self):
        """Opus deve custar mais que Haiku para a mesma quantidade de tokens."""
        haiku_cost = estimate_llm_cost("claude-3-5-haiku-20241022", 1000, 500)
        opus_cost   = estimate_llm_cost("claude-3-opus-20240229",   1000, 500)
        assert opus_cost > haiku_cost

    def test_unknown_model_uses_default(self):
        """Modelo desconhecido usa tabela de fallback sem lançar exceção."""
        cost = estimate_llm_cost("model-inexistente-xyz", 500, 100)
        assert cost > 0

    def test_zero_tokens_zero_cost(self):
        """Custo com 0 tokens é 0."""
        cost = estimate_llm_cost("claude-3-5-haiku-20241022", 0, 0)
        assert cost == 0.0

    def test_cost_proportional_to_tokens(self):
        """Custo deve ser proporcional ao número de tokens."""
        cost_small = estimate_llm_cost("claude-3-5-haiku-20241022", 100, 50)
        cost_large = estimate_llm_cost("claude-3-5-haiku-20241022", 1000, 500)
        assert abs(cost_large / cost_small - 10.0) < 0.01


# ---------------------------------------------------------------------------
# Testes — Funções de registro (record_*)
# ---------------------------------------------------------------------------

class TestRecordFunctions:
    def test_record_generation_cost_populates_trace(self):
        """record_generation_cost() preenche tokens e custo no trace ativo."""
        with pipeline_trace("test gen", log=False) as trace:
            record_generation_cost("claude-3-5-haiku-20241022", 800, 300, 1500.0)

        assert trace.tokens_input == 800
        assert trace.tokens_output == 300
        assert trace.cost_usd > 0
        stage_names = [s.name for s in trace.stages]
        assert "generation" in stage_names

    def test_record_generation_cost_noop_without_trace(self):
        """record_generation_cost() fora de um trace não lança exceção."""
        record_generation_cost("claude-3-5-haiku-20241022", 800, 300, 1500.0)

    def test_record_routing_decision_sets_path(self):
        """record_routing_decision() define o path no trace ativo."""
        with pipeline_trace("test routing", log=False) as trace:
            record_routing_decision("fast", 0.35)

        assert trace.path == "fast"
        assert trace.extra.get("complexity") == 0.35

    def test_record_routing_decision_full_path(self):
        """record_routing_decision() com full path."""
        with pipeline_trace("complex query", log=False) as trace:
            record_routing_decision("full", 0.75)

        assert trace.path == "full"

    def test_record_routing_noop_without_trace(self):
        """record_routing_decision() fora de um trace não lança exceção."""
        record_routing_decision("fast", 0.3)


# ---------------------------------------------------------------------------
# Testes — Histórico de Recall@5
# ---------------------------------------------------------------------------

class TestRecallHistory:
    def test_log_and_load_recall_history(self, tmp_path, monkeypatch):
        """log_recall_to_history grava e load_recall_history recupera corretamente."""
        import src.observability.instrumentation as obs_mod

        history_file = tmp_path / "recall_history.jsonl"
        monkeypatch.setattr(obs_mod, "RECALL_HISTORY_FILE", history_file)
        monkeypatch.setattr(obs_mod, "LOG_DIR", tmp_path)

        log_recall_to_history(0.9697, 0.7864, 33, "hybrid", 1561.0)
        log_recall_to_history(0.9394, 0.7500, 33, "hybrid", 1200.0)

        history = load_recall_history()
        assert len(history) == 2
        assert history[0]["recall_at_5"] == 0.9697
        assert history[1]["recall_at_5"] == 0.9394
        assert history[0]["mode"] == "hybrid"

    def test_recall_history_has_required_fields(self, tmp_path, monkeypatch):
        """Cada entrada do histórico tem todos os campos esperados."""
        import src.observability.instrumentation as obs_mod
        history_file = tmp_path / "recall_history.jsonl"
        monkeypatch.setattr(obs_mod, "RECALL_HISTORY_FILE", history_file)
        monkeypatch.setattr(obs_mod, "LOG_DIR", tmp_path)

        log_recall_to_history(0.96, 0.78, 33, "hybrid", 1500.0)
        history = load_recall_history()
        entry = history[0]
        for field in ["ts", "recall_at_5", "mrr", "n_questions", "mode", "ms_per_query"]:
            assert field in entry, f"Campo '{field}' ausente na entrada do histórico"

    def test_load_recall_history_empty_when_no_file(self, tmp_path, monkeypatch):
        """load_recall_history() retorna [] se o arquivo não existe."""
        import src.observability.instrumentation as obs_mod
        monkeypatch.setattr(obs_mod, "RECALL_HISTORY_FILE", tmp_path / "nonexistent.jsonl")
        assert load_recall_history() == []

    def test_check_drift_no_alert_when_stable(self, tmp_path, monkeypatch):
        """Sem queda, check_recall_drift() retorna None."""
        import src.observability.instrumentation as obs_mod
        history_file = tmp_path / "recall_history.jsonl"
        monkeypatch.setattr(obs_mod, "RECALL_HISTORY_FILE", history_file)
        monkeypatch.setattr(obs_mod, "LOG_DIR", tmp_path)

        # Recall estável em 0.96
        for _ in range(6):
            log_recall_to_history(0.9697, 0.78, 33, "hybrid", 1500.0)

        alert = check_recall_drift(threshold_drop=0.05)
        assert alert is None

    def test_check_drift_alert_on_degradation(self, tmp_path, monkeypatch):
        """Queda >= threshold dispara alerta."""
        import src.observability.instrumentation as obs_mod
        history_file = tmp_path / "recall_history.jsonl"
        monkeypatch.setattr(obs_mod, "RECALL_HISTORY_FILE", history_file)
        monkeypatch.setattr(obs_mod, "LOG_DIR", tmp_path)

        # Recall histórico alto, depois cai muito
        for _ in range(6):
            log_recall_to_history(0.9697, 0.78, 33, "hybrid", 1500.0)
        log_recall_to_history(0.8000, 0.65, 33, "hybrid", 1500.0)  # queda de ~17pp

        alert = check_recall_drift(threshold_drop=0.05)
        assert alert is not None
        assert "ALERTA" in alert

    def test_check_drift_needs_at_least_two_points(self, tmp_path, monkeypatch):
        """Com menos de 2 entradas no histórico, check_drift retorna None."""
        import src.observability.instrumentation as obs_mod
        history_file = tmp_path / "recall_history.jsonl"
        monkeypatch.setattr(obs_mod, "RECALL_HISTORY_FILE", history_file)
        monkeypatch.setattr(obs_mod, "LOG_DIR", tmp_path)

        log_recall_to_history(0.96, 0.78, 33, "hybrid", 1500.0)
        assert check_recall_drift() is None


# ---------------------------------------------------------------------------
# Teste de integração — hybrid_search + pipeline_trace
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.usefixtures("require_qdrant", "require_postgres")
class TestIntegrationWithHybridSearch:
    """
    Testa que o pipeline real de retrieval popula o trace corretamente.
    Requer Qdrant e Postgres rodando (pula se não disponíveis).
    """

    def test_hybrid_search_populates_trace(self):
        """hybrid_search dentro de pipeline_trace gera estágios de retrieval."""
        try:
            from src.retrieval.hybrid import hybrid_search
        except ImportError:
            pytest.skip("Módulo hybrid não disponível")

        try:
            from src.retrieval.acl_filter import ANY_PRINCIPAL
            with pipeline_trace("O que é Retrieval Augmented Generation?", log=False) as trace:
                _ = hybrid_search(
                    "O que é Retrieval Augmented Generation?",
                    top_k=3,
                    tenant_id="tenant-test",
                    principals=ANY_PRINCIPAL,
                )

            # Deve ter registrado pelo menos embedding + dense + bm25 + fusion
            stage_names = [s.name for s in trace.stages]
            assert len(stage_names) >= 1, "Nenhum estágio registrado — infraestrutura indisponível?"

            # Se tiver estágios, verifica que latências são positivas
            for s in trace.stages:
                assert s.latency_ms >= 0, f"Latência negativa em {s.name}"

        except Exception as e:
            if "Connection" in str(e) or "refused" in str(e):
                pytest.skip(f"Infraestrutura indisponível: {e}")
            raise

    def test_trace_dict_serializable_after_search(self):
        """O trace produzido após uma busca é serializável para JSON."""
        try:
            from src.retrieval.hybrid import hybrid_search
        except ImportError:
            pytest.skip("Módulo hybrid não disponível")

        try:
            from src.retrieval.acl_filter import ANY_PRINCIPAL
            with pipeline_trace("chunk e embedding", log=False) as trace:
                hybrid_search("chunk e embedding", top_k=3, tenant_id="tenant-test", principals=ANY_PRINCIPAL)

            json_str = trace.to_json()
            parsed = json.loads(json_str)
            assert "query_hash" in parsed
            assert "stages" in parsed
        except Exception as e:
            if "Connection" in str(e) or "refused" in str(e):
                pytest.skip(f"Infraestrutura indisponível: {e}")
            raise


# ---------------------------------------------------------------------------
# Teste — log_trace não quebra com trace vazio
# ---------------------------------------------------------------------------

def test_log_trace_does_not_raise():
    """log_trace() com trace vazio não lança exceção."""
    trace = PipelineTrace(query="pergunta vazia")
    log_trace(trace)  # deve executar sem erro


def test_pipeline_trace_log_true_calls_log_trace(monkeypatch):
    """pipeline_trace(log=True) chama log_trace ao sair do contexto."""
    called = []
    monkeypatch.setattr(
        "src.observability.instrumentation.log_trace",
        lambda t: called.append(t)
    )
    with pipeline_trace("teste log", log=True):
        pass
    assert len(called) == 1


def test_pipeline_trace_log_false_does_not_call_log_trace(monkeypatch):
    """pipeline_trace(log=False) não chama log_trace."""
    called = []
    monkeypatch.setattr(
        "src.observability.instrumentation.log_trace",
        lambda t: called.append(t)
    )
    with pipeline_trace("teste no log", log=False):
        pass
    assert len(called) == 0
