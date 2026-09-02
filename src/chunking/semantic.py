"""
Chunker Semântico — Passo 18.

Estratégia de chunking baseada em significado (não em tamanho fixo):
1. Divide o texto em sentenças.
2. Calcula embeddings de sentenças consecutivas usando BGE-M3.
3. Mede a similaridade de cosseno entre sentenças adjacentes.
4. Identifica pontos de quebra (breakpoint) onde a similaridade semântica
   cai abaixo do limiar (default: similarity_threshold=0.75), indicando
   uma mudança de tópico ou argumento.
5. Respeita limites de segurança:
   - min_tokens (~50 tokens): evita chunks minusculos/fragmentados.
   - max_tokens (~600 tokens): força quebra caso um mesmo tópico se estenda demais.
6. Trata tabelas estruturadas como elementos atômicos integrais.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import List, Optional, Tuple

import numpy as np

from src.chunking.recursive import Chunk, _max_words
from src.embedding.model import embed_texts
from src.parsing.canonical_model import Document, ElementType, Paragraph, Section, Table

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULT_SIMILARITY_THRESHOLD = 0.75
DEFAULT_MAX_TOKENS = 600
DEFAULT_MIN_TOKENS = 50


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> List[str]:
    """
    Divide um texto em sentenças respeitando pontuações comuns (. ! ?).
    """
    # Regex divide em finais de sentença seguidos de espaço ou quebra de linha
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    # Limpa e filtra sentenças vazias
    cleaned = [s.strip() for s in raw_sentences if s.strip()]
    return cleaned if cleaned else ([text.strip()] if text.strip() else [])


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calcula similaridade de cosseno entre dois vetores."""
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _semantic_split_sentences(
    sentences: List[str],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
) -> List[Tuple[str, List[float]]]:
    """
    Agrupa sentenças em blocos semânticos com base na similaridade de cosseno.
    Retorna lista de tuplas (texto_do_bloco, similaridades_das_quebras).
    """
    if not sentences:
        return []
    if len(sentences) == 1:
        return [(sentences[0], [])]

    max_words = _max_words(max_tokens)
    min_words = _max_words(min_tokens)

    # Gera embeddings para todas as sentenças em lote
    embeddings = embed_texts(sentences, batch_size=32)

    chunks: List[Tuple[str, List[float]]] = []
    current_sentences: List[str] = [sentences[0]]
    current_words: int = len(sentences[0].split())
    breakpoints_sims: List[float] = []

    for i in range(len(sentences) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        next_sentence = sentences[i + 1]
        next_words = len(next_sentence.split())

        # Condições de corte:
        # 1. Similaridade caiu abaixo do threshold E já acumulamos min_words
        # 2. OU tamanho acumulado ultrapassa max_words (corte forçado)
        is_semantic_drop = (sim < similarity_threshold) and (current_words >= min_words)
        is_overflow = (current_words + next_words > max_words) and (current_words >= min_words)

        if is_semantic_drop or is_overflow:
            # Emite chunk acumulado
            chunks.append((" ".join(current_sentences), list(breakpoints_sims)))
            current_sentences = [next_sentence]
            current_words = next_words
            breakpoints_sims = [sim]
        else:
            current_sentences.append(next_sentence)
            current_words += next_words
            breakpoints_sims.append(sim)

    if current_sentences:
        chunks.append((" ".join(current_sentences), list(breakpoints_sims)))

    return chunks


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def chunk_semantic(
    document: Document,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    min_tokens: int = DEFAULT_MIN_TOKENS,
) -> List[Chunk]:
    """
    Transforma um Document em chunks usando Semantic Chunking baseado no BGE-M3.

    Parâmetros:
    - document             : documento no Canonical Document Model.
    - similarity_threshold : limiar de similaridade de cosseno (default 0.75).
                             Cortes acontecem onde a similaridade entre sentenças
                             adjacentes for menor que este valor.
    - max_tokens           : teto máximo de tokens por chunk (~600 tokens).
    - min_tokens           : piso mínimo de tokens por chunk (~50 tokens).

    Retorna lista de Chunk com metadados semânticos.
    """
    chunks: List[Chunk] = []
    chunk_index = 0

    current_section_title: Optional[str] = None
    section_sentences: List[str] = []

    def flush_section_sentences() -> None:
        nonlocal chunk_index
        if not section_sentences:
            return

        semantic_blocks = _semantic_split_sentences(
            section_sentences,
            similarity_threshold=similarity_threshold,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
        )
        section_sentences.clear()

        for block_text, sims in semantic_blocks:
            prefix = f"[Seção: {current_section_title}]\n" if current_section_title else ""
            content = prefix + block_text.strip()

            c = Chunk(
                document_id=document.id,
                content=content.strip(),
                tenant_id=document.tenant_id,
                acl=[entry.model_dump() for entry in document.acl],
                metadata={
                    "strategy": "semantic",
                    "similarity_threshold": similarity_threshold,
                    "max_tokens": max_tokens,
                    "min_tokens": min_tokens,
                    "source_uri": document.source_uri,
                    "document_title": document.title,
                    "section_title": current_section_title,
                    "chunk_index": chunk_index,
                    "breakpoint_similarities": [round(s, 4) for s in sims],
                },
            )
            chunks.append(c)
            chunk_index += 1

    for element in document.elements:
        etype = element.element_type

        if etype == ElementType.SECTION:
            assert isinstance(element, Section)
            flush_section_sentences()
            current_section_title = element.title
            if element.content and element.content.strip():
                section_sentences.extend(_split_into_sentences(element.content))

        elif etype == ElementType.PARAGRAPH:
            assert isinstance(element, Paragraph)
            if element.content and element.content.strip():
                section_sentences.extend(_split_into_sentences(element.content))

        elif etype == ElementType.TABLE:
            assert isinstance(element, Table)
            flush_section_sentences()
            # Tabela tratada como bloco atômico estruturado
            tbl_content = element.content or ""
            prefix = f"[Seção: {current_section_title}]\n" if current_section_title else ""
            content = prefix + tbl_content.strip()

            tbl_chunk = Chunk(
                document_id=document.id,
                content=content.strip(),
                tenant_id=document.tenant_id,
                acl=[entry.model_dump() for entry in document.acl],
                metadata={
                    "strategy": "semantic",
                    "source_uri": document.source_uri,
                    "document_title": document.title,
                    "section_title": current_section_title,
                    "chunk_index": chunk_index,
                    "table_headers": element.headers or [],
                    "table_row_count": len(element.rows) if element.rows else 0,
                    "has_table_summary": "## Sumário" in tbl_content,
                },
            )
            chunks.append(tbl_chunk)
            chunk_index += 1

    flush_section_sentences()
    return chunks
