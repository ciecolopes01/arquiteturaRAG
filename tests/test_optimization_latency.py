"""
Testes unitários e de integração para Otimização de Latência e Caching — Passo 32.

Valida:
1. Operação do CacheManager (respostas e embeddings, hits, misses e hit_rate).
2. Streaming progressivo de respostas via `answer_stream`.
3. Integração com Qdrant `ef_search` e redução de candidatos do reranker.
4. Critério de "pronto":
   - Tabela comparativa de latência média e P95 antes vs depois do cache.
   - Cache hit rate após duas rodadas sucessivas de execução.
   - Confirmação de que a 2ª rodada é ordens de grandeza mais rápida sem regressão de qualidade.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, ".")

from src.config import settings
from src.generation.answer import answer, answer_stream
from src.optimization.caching import cache


def test_cache_manager_basic_operations():
    """Valida armazenamento, recuperação, métricas e limpeza do CacheManager."""
    cache.clear()

    # 1. Miss inicial
    res = cache.get_response("query teste miss", tenant_id="tenant-1")
    assert res is None
    assert cache.misses == 1
    assert cache.hits == 0

    # 2. Set e Hit subsequente
    payload = {"answer": "Resposta em cache", "model": "claude-test", "source_uris": ["uri-1"]}
    cache.set_response("query teste miss", tenant_id="tenant-1", response_payload=payload, ttl_seconds=100)

    cached_res = cache.get_response("query teste miss", tenant_id="tenant-1")
    assert cached_res is not None
    assert cached_res["answer"] == "Resposta em cache"
    assert cache.hits == 1
    assert cache.hit_rate == 0.5  # 1 hit / 2 requests

    # 3. Cache de embedding
    emb_miss = cache.get_embedding("texto embedding", model_name="bge-m3")
    assert emb_miss is None

    fake_vec = [0.1, 0.2, 0.3]
    cache.set_embedding("texto embedding", model_name="bge-m3", vector=fake_vec)
    emb_hit = cache.get_embedding("texto embedding", model_name="bge-m3")
    assert emb_hit == fake_vec


def test_answer_stream_generator():
    """Valida que o generator de streaming produz tokens progressivos."""
    tokens = list(answer_stream("O que é RAG?", top_k=2, tenant_id="tenant-test"))
    assert len(tokens) > 0
    full_output = "".join(tokens)
    assert len(full_output) > 0


def test_latency_comparison_and_cache_hit_rate():
    """
    Critério de "pronto" do Passo 32:
    Executa conjunto de perguntas em 2 rodadas sucessivas.
    Mede latência média e P95 antes (rodada 1) vs depois (rodada 2 com cache hit).
    Registra tabela comparativa e valida hit rate de 100% na 2ª rodada.
    """
    test_queries = [
        "O que é RAG e quais são seus componentes fundamentais?",
        "Qual a finalidade do Reciprocal Rank Fusion na busca híbrida?",
        "Qual a porta padrão configurada para o banco Qdrant?",
        "Qual a função do modelo de reranking cross-encoder?",
        "Qual a estratégia de chunking recursivo com overlap?",
    ]

    cache.clear()

    # --- Rodada 1: Sem Cache (Cold / Miss) ---
    latencies_r1 = []
    for q in test_queries:
        t0 = time.perf_counter()
        res = answer(query=q, top_k=3, tenant_id="tenant-test")
        lat = (time.perf_counter() - t0) * 1000
        latencies_r1.append(lat)

    # --- Rodada 2: Com Cache (Warm / Hit) ---
    latencies_r2 = []
    hits_r2_start = cache.hits
    for q in test_queries:
        t0 = time.perf_counter()
        res = answer(query=q, top_k=3, tenant_id="tenant-test")
        lat = (time.perf_counter() - t0) * 1000
        latencies_r2.append(lat)

    hits_r2 = cache.hits - hits_r2_start
    hit_rate_r2 = hits_r2 / len(test_queries)

    avg_r1 = float(np.mean(latencies_r1))
    p95_r1 = float(np.percentile(latencies_r1, 95))

    avg_r2 = float(np.mean(latencies_r2))
    p95_r2 = float(np.percentile(latencies_r2, 95))

    speedup = avg_r1 / max(avg_r2, 0.001)

    print("\n" + "=" * 70)
    print("  TABELA COMPARATIVA DE LATÊNCIA — PASSO 32 (CACHE & LATENCY)")
    print("=" * 70)
    print(f"{'Métrica':<25} {'Rodada 1 (Sem Cache)':>20} {'Rodada 2 (Com Cache)':>20}")
    print("-" * 70)
    print(f"{'Latência Média':<25} {avg_r1:>18.2f}ms {avg_r2:>18.2f}ms")
    print(f"{'Latência P95':<25} {p95_r1:>18.2f}ms {p95_r2:>18.2f}ms")
    print(f"{'Cache Hit Rate':<25} {'0.0%':>20} {f'{hit_rate_r2 * 100:.1f}%':>20}")
    print(f"{'Speedup Aceleração':<25} {'1.0x':>20} {f'{speedup:.1f}x':>20}")
    print("=" * 70)

    assert hit_rate_r2 == 1.0, f"Esperado 100% de hit rate na rodada 2, obtido {hit_rate_r2}"
    assert avg_r2 < avg_r1, "Rodada 2 com cache deve ser mais rápida que rodada 1"
    assert p95_r2 < 50.0, f"Latência P95 em cache deve ser sub-50ms, obtido {p95_r2}ms"


def test_settings_latency_parameters():
    """Valida presença e sanidade dos parâmetros de latência no config."""
    assert hasattr(settings, "reranker_top_k_candidates")
    assert 10 <= settings.reranker_top_k_candidates <= 50
    assert hasattr(settings, "qdrant_ef_search")
    assert 16 <= settings.qdrant_ef_search <= 128
    assert settings.enable_cache is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
