"""
Wrapper do modelo de embedding BGE-M3 — Passo 4.

Carrega BAAI/bge-m3 uma única vez como singleton de módulo.
BGE-M3 é multilíngue, open-source (BAAI), e suporta busca densa,
esparsa e colbert num mesmo modelo — usamos apenas dense nesta fase.

Primeira execução baixa o modelo do HuggingFace (~2GB) e salva em cache
local (~/.cache/huggingface). Execuções subsequentes carregam do cache.

Dependências: FlagEmbedding>=1.2.10, torch>=2.0.0
"""

from __future__ import annotations

import logging
import threading
import time
from typing import List

logger = logging.getLogger("rag.embedding")

# Importação lazy para não quebrar outros módulos se FlagEmbedding não
# estiver instalado ainda.
_model = None
_model_name: str = ""
_device: str = ""
_model_lock = threading.Lock()


def _get_model():
    """Carrega o modelo BGE-M3 uma única vez com thread-safety."""
    global _model, _model_name, _device

    from src.config import settings

    # Só recarrega se a configuração mudou (improvável em produção)
    if _model is not None and _model_name == settings.embedding_model:
        return _model

    with _model_lock:
        if _model is not None and _model_name == settings.embedding_model:
            return _model

        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ImportError(
                "FlagEmbedding não instalado. Execute:\n"
                "  pip install FlagEmbedding torch\n"
                "O modelo BAAI/bge-m3 será baixado na primeira execução (~2GB)."
            )

        logger.info(
            "Carregando %s em %s... (primeira vez pode demorar)",
            settings.embedding_model,
            settings.embedding_device,
        )
        t0 = time.time()
        _model = BGEM3FlagModel(
            settings.embedding_model,
            use_fp16=(settings.embedding_device != "cpu"),  # fp16 só faz sentido em GPU
            device=settings.embedding_device,
        )
        _model_name = settings.embedding_model
        _device = settings.embedding_device
        elapsed = time.time() - t0
        logger.info("Modelo %s carregado em %.1fs", settings.embedding_model, elapsed)
        return _model


def embed_texts(texts: List[str], batch_size: int = 12) -> List[List[float]]:
    """
    Gera embeddings densos para uma lista de textos.

    Retorna uma lista de vetores float (dimensão 1024 para BGE-M3).

    Parâmetros:
    - texts: lista de strings a embedar.
    - batch_size: número de textos processados por vez.
      CPU: 8-16 é razoável; GPU: pode aumentar para 32-64.

    Exemplo:
        vectors = embed_texts(["O que é RAG?", "Como funciona o BM25?"])
        # vectors[0] é o vetor da primeira query
    """
    model = _get_model()

    t0 = time.time()
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=512,
        return_dense=True,
        return_sparse=False,   # sparse fica para fase posterior
        return_colbert_vecs=False,
    )
    elapsed = time.time() - t0

    vectors = output["dense_vecs"].tolist()
    per_text = elapsed / max(len(texts), 1) * 1000
    logger.debug(
        "%d texto(s) embedados em %.2fs (%.1fms/texto)",
        len(texts),
        elapsed,
        per_text,
    )
    return vectors


def embed_query(query: str) -> List[float]:
    """
    Gera o embedding de uma query de busca.

    BGE-M3 usa o mesmo modelo para documento e query, mas é boa prática
    deixar a função separada para facilitar eventual prefixo de instrução
    (comum em outros modelos como E5).
    """
    return embed_texts([query])[0]


def embedding_dimension() -> int:
    """
    Retorna a dimensão do vetor gerado pelo modelo.

    Prioridade:
    1. settings.embedding_dimension (se configurado explicitamente).
    2. Dimensão do modelo instanciado (_model.model.config.hidden_size).
    3. Catálogo de dimensões conhecidas para modelos populares.
    4. Fallback padrão 1024.
    """
    from src.config import settings

    if settings.embedding_dimension is not None:
        return settings.embedding_dimension

    global _model, _model_name
    if _model is not None and _model_name == settings.embedding_model:
        try:
            if hasattr(_model, "model") and hasattr(_model.model, "config"):
                return int(_model.model.config.hidden_size)
        except Exception as e:
            logger.debug("Não foi possível inspecionar hidden_size do modelo instanciado: %s", e)

    known_dimensions = {
        "BAAI/bge-m3": 1024,
        "BAAI/bge-large-en-v1.5": 1024,
        "BAAI/bge-large-zh-v1.5": 1024,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-base-zh-v1.5": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }
    return known_dimensions.get(settings.embedding_model, 1024)
