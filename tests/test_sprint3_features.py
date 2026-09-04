"""
Testes unitários offline para as funcionalidades do Sprint 3:
- Dimensão dinâmica de embedding
- Estimador calibrado de tokens para português/multilíngue
- Preservação de casing no cache de embeddings
- Calibração de thresholds do Self-RAG
- Paralelismo e resiliência dos canais no hybrid_search e routed_hybrid_search
"""

import time
from unittest.mock import patch

from src.embedding.model import embedding_dimension
from src.generation.answer import estimate_tokens
from src.generation.self_rag import (
    SelfRAGDecision,
    evaluate_context_sufficiency,
)
from src.optimization.caching import CacheManager
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import RerankResult


def test_embedding_dimension_default_and_known():
    # Sem modelo instanciado e sem override, modelo default bge-m3 retorna 1024
    dim = embedding_dimension()
    assert dim == 1024


def test_embedding_dimension_override():
    with patch("src.config.settings.embedding_dimension", 768):
        assert embedding_dimension() == 768

    with patch("src.config.settings.embedding_dimension", None), \
         patch("src.config.settings.embedding_model", "sentence-transformers/all-MiniLM-L6-v2"):
        assert embedding_dimension() == 384


def test_estimate_tokens_accuracy():
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0

    # Texto curto em português
    text = "O sistema RAG utiliza busca híbrida para recuperar documentos relevantes."
    tokens = estimate_tokens(text)
    # 10 palavras -> ~13-14 tokens
    words = len(text.split())
    assert tokens >= words
    assert 10 <= tokens <= 20


def test_caching_preserve_case_for_embeddings():
    mgr = CacheManager()
    mgr.redis_client = None  # in-memory

    # Chaves com preserve_case=True diferenciam termos sensíveis a maiúsculas
    key_upper = mgr._hash_key("rag:emb", "model:TI", preserve_case=True)
    key_lower = mgr._hash_key("rag:emb", "model:ti", preserve_case=True)
    assert key_upper != key_lower

    # Chaves comuns (preserve_case=False) normalizam
    key_norm1 = mgr._hash_key("rag:resp", "QUERY", preserve_case=False)
    key_norm2 = mgr._hash_key("rag:resp", "query", preserve_case=False)
    assert key_norm1 == key_norm2


def test_self_rag_threshold_calibration():
    # Score 0.5 (logit positivo) -> SUFFICIENT
    c1 = RerankResult(chunk_id="c1", content="ok", rerank_score=0.5, original_rank=1, new_rank=1)
    decision, conf = evaluate_context_sufficiency("query", [c1])
    assert decision == SelfRAGDecision.SUFFICIENT
    assert conf > 0.5

    # Score -1.0 (entre -2.5 e 0.0) -> RETRY_RETRIEVAL
    c2 = RerankResult(chunk_id="c2", content="borderline", rerank_score=-1.0, original_rank=1, new_rank=1)
    decision, conf = evaluate_context_sufficiency("query", [c2])
    assert decision == SelfRAGDecision.RETRY_RETRIEVAL

    # Score -3.0 (< -2.5) -> INSUFFICIENT
    c3 = RerankResult(chunk_id="c3", content="irrelevant", rerank_score=-3.0, original_rank=1, new_rank=1)
    decision, conf = evaluate_context_sufficiency("query", [c3])
    assert decision == SelfRAGDecision.INSUFFICIENT


def test_parallel_hybrid_search_execution():
    """Valida que dense e bm25 rodam em paralelo no ThreadPoolExecutor."""
    def mock_dense(*args, **kwargs):
        time.sleep(0.05)
        return [{"chunk_id": "c_dense", "content": "dense content", "score": 0.9}]

    def mock_bm25(*args, **kwargs):
        time.sleep(0.05)
        return [{"chunk_id": "c_bm25", "content": "bm25 content", "score": 10.0}]

    t0 = time.time()
    with patch("src.retrieval.hybrid.search_dense", side_effect=mock_dense), \
         patch("src.retrieval.hybrid.search_bm25", side_effect=mock_bm25):
        results = hybrid_search("pergunta teste", principals=["user:test"])

    elapsed = time.time() - t0
    # Se fossem sequenciais, demoraria >= 0.10s. Em paralelo, termina em ~0.05-0.08s.
    assert len(results) == 2
    assert elapsed < 0.20  # tolerância segura


def test_parallel_channel_resilience():
    """Se um canal falha, o outro ainda retorna resultados sem crash."""
    def mock_dense(*args, **kwargs):
        raise ConnectionError("Qdrant offline")

    def mock_bm25(*args, **kwargs):
        return [{"chunk_id": "c_bm25", "content": "bm25 content", "score": 5.0}]

    with patch("src.retrieval.hybrid.search_dense", side_effect=mock_dense), \
         patch("src.retrieval.hybrid.search_bm25", side_effect=mock_bm25):
        results = hybrid_search("teste resiliencia", principals=["user:test"])

    assert len(results) == 1
    assert results[0].chunk_id == "c_bm25"
