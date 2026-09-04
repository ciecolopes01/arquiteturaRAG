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

import threading
import time
from typing import List

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

    print(
        f"[EmbeddingModel] Carregando {settings.embedding_model} "
        f"em {settings.embedding_device}... (primeira vez pode demorar)"
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
    print(f"[EmbeddingModel] Modelo carregado em {elapsed:.1f}s")
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
    print(
        f"[EmbeddingModel] {len(texts)} texto(s) embedados em "
        f"{elapsed:.2f}s ({per_text:.1f}ms/texto)"
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
    """Retorna a dimensão do vetor gerado pelo modelo (1024 para BGE-M3)."""
    return 1024
