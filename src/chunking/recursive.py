"""
Chunker recursivo — Passo 3.

Transforma um Document (Canonical Document Model) em uma lista de Chunks
prontos para embedding e indexação.

Decisões de design:
- Chunk é um schema Pydantic simples: chunk_id, document_id, content, metadata.
- A unidade de "tamanho" é palavras, não tokens.  Tokens variam por modelo;
  palavras são consistentes e boas o suficiente para esta fase.
  Fator de conversão aproximado: 1 palavra ≈ 1.3 tokens em português.
  max_tokens=512 → max_words ≈ 394.
- Estratégia: recursive split — tenta quebrar por separadores em ordem de
  preferência: '\n\n', '\n', '. ', ', ', ' '.  Isso evita cortar frases no
  meio sempre que possível.
- Overlap: os últimos N words do chunk anterior são pré-pendidos ao próximo.
  Garante que conteúdo na fronteira entre chunks seja recuperável.
- Sections viram um chunk separado com seu título incluído, dando contexto.
- Metadados de cada chunk: document_id, source_uri, tenant_id, element_ids
  de origem, posição (chunk_index dentro do documento), section_title mais
  próxima.
"""

from __future__ import annotations

import re
import uuid
from typing import List, Optional

from pydantic import BaseModel, Field

from src.parsing.canonical_model import Document, ElementType, Paragraph, Section, Table


# ---------------------------------------------------------------------------
# Schema do Chunk
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    """Unidade mínima de texto que será embedded e indexada."""

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    content: str
    metadata: dict = Field(default_factory=dict)

    # Campos de ACL herdados do Document — usados no filtro do Qdrant/Postgres
    tenant_id: Optional[str] = None
    acl: list = Field(default_factory=list)

    model_config = {"frozen": False}

    @property
    def word_count(self) -> int:
        return len(self.content.split())


# ---------------------------------------------------------------------------
# Helpers de split
# ---------------------------------------------------------------------------

SEPARATORS = ["\n\n", "\n", ". ", ", ", " "]
WORDS_PER_TOKEN = 1.3  # fator de conversão aproximado pt-BR


def _max_words(max_tokens: int) -> int:
    return max(1, int(max_tokens / WORDS_PER_TOKEN))


def _split_text(text: str, max_words: int) -> List[str]:
    """
    Divide `text` em pedaços de no máximo `max_words` palavras,
    tentando separadores em ordem de preferência para não cortar frases.

    Retorna lista de strings não-vazias.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text] if text.strip() else []

    # Tenta cada separador em ordem
    for sep in SEPARATORS:
        if sep in text:
            parts = text.split(sep)
            chunks: List[str] = []
            current = ""
            for part in parts:
                candidate = (current + sep + part).strip() if current else part.strip()
                if len(candidate.split()) <= max_words:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    # Se a parte isolada ainda é grande, recursão
                    if len(part.split()) > max_words:
                        chunks.extend(_split_text(part.strip(), max_words))
                        current = ""
                    else:
                        current = part.strip()
            if current:
                chunks.append(current)
            return [c for c in chunks if c.strip()]

    # Fallback: força quebra por palavra
    result = []
    for i in range(0, len(words), max_words):
        result.append(" ".join(words[i : i + max_words]))
    return result


def _get_overlap_prefix(text: str, overlap_words: int) -> str:
    """Retorna as últimas `overlap_words` palavras de `text`."""
    words = text.split()
    if len(words) <= overlap_words:
        return text
    return " ".join(words[-overlap_words:])


# ---------------------------------------------------------------------------
# Chunker principal
# ---------------------------------------------------------------------------

def chunk_recursive(
    document: Document,
    max_tokens: int = 512,
    overlap: int = 64,
) -> List[Chunk]:
    """
    Transforma um Document em uma lista ordenada de Chunks.

    Parâmetros:
    - max_tokens: tamanho máximo de cada chunk em tokens (aprox. 1.3 × palavras).
    - overlap: número de tokens de sobreposição entre chunks consecutivos.

    Fluxo:
    1. Itera pelos elementos do Document em ordem.
    2. Sections: o título é incluído no início do chunk seguinte como contexto.
    3. Tables: cada tabela vira um chunk próprio (não sofre split).
    4. Paragraphs e resto: passa pelo recursive split se necessário.
    5. Overlap é aplicado entre todos os chunks gerados.
    """
    max_words = _max_words(max_tokens)
    overlap_words = _max_words(overlap)

    chunks: List[Chunk] = []
    current_section_title: Optional[str] = None
    last_chunk_text: Optional[str] = None  # para calcular overlap

    def make_chunk(content: str, element_ids: List[str]) -> Chunk:
        return Chunk(
            document_id=document.id,
            content=content.strip(),
            tenant_id=document.tenant_id,
            acl=[entry.model_dump() for entry in document.acl],
            metadata={
                "source_uri": document.source_uri,
                "document_title": document.title,
                "section_title": current_section_title,
                "chunk_index": len(chunks),
                "element_ids": element_ids,
                "max_tokens": max_tokens,
                "overlap": overlap,
            },
        )

    def add_text_chunks(text: str, element_ids: List[str]) -> None:
        """Divide text em sub-chunks com overlap e os adiciona à lista."""
        nonlocal last_chunk_text

        raw_parts = _split_text(text, max_words)
        for part in raw_parts:
            # Prefixa com overlap do chunk anterior
            if last_chunk_text and overlap_words > 0:
                prefix = _get_overlap_prefix(last_chunk_text, overlap_words)
                content = prefix + " " + part if prefix.strip() else part
            else:
                content = part

            chunk = make_chunk(content, element_ids)
            if chunk.content:  # ignora chunks vazios
                chunks.append(chunk)
                last_chunk_text = part  # overlap usa texto original (sem prefixo)

    # Pré-processa: funde Paragraphs muito curtos (≤8 palavras) que parecem
    # headings escapados com o próximo elemento, para evitar chunks minúsculos.
    MIN_STANDALONE_WORDS = 8
    merged_elements = []
    i = 0
    elements = document.elements
    while i < len(elements):
        el = elements[i]
        if (
            el.element_type == ElementType.PARAGRAPH
            and len(el.content.split()) <= MIN_STANDALONE_WORDS
            and i + 1 < len(elements)
            and elements[i + 1].element_type == ElementType.PARAGRAPH
        ):
            # Funde: transforma em Section (usa como contexto) ou prefixa no próximo
            next_el = elements[i + 1]
            fused = Paragraph(
                id=el.id,
                content=el.content + "\n\n" + next_el.content,
                metadata={**el.metadata, "fused_with": next_el.id},
            )
            merged_elements.append(fused)
            i += 2  # pula o próximo que já foi fundido
        else:
            merged_elements.append(el)
            i += 1

    for element in merged_elements:

        etype = element.element_type

        if etype == ElementType.SECTION:
            # Seções atualizam o contexto de seção atual
            assert isinstance(element, Section)
            current_section_title = element.title
            # Se a seção tem conteúdo próprio (raro no TXT), trata como texto
            if element.content and element.content.strip():
                text = f"{element.title}\n\n{element.content}"
                add_text_chunks(text, [element.id])

        elif etype == ElementType.TABLE:
            assert isinstance(element, Table)
            # Tabelas viram um chunk único — não sofrem split nesta fase.
            # Passo 15: o content já inclui markdown + sumário (build_table_content).
            # Se o content da tabela já tem contexto de seção/caption embutido,
            # usamos diretamente; senão adicionamos o prefixo manualmente.
            content = element.content or ""
            if not content.strip():
                # fallback: gera content básico se não vier do table_extractor
                prefix_ctx = f"[Seção: {current_section_title}]\n" if current_section_title else ""
                caption = f"[Tabela: {element.caption}]\n" if element.caption else ""
                content = prefix_ctx + caption + element.content
            chunk = make_chunk(content, [element.id])
            # Metadata enriquecido para tabelas (Passo 15)
            chunk.metadata["table_headers"] = element.headers or []
            chunk.metadata["table_row_count"] = len(element.rows) if element.rows else 0
            chunk.metadata["has_table_summary"] = "## Sumário" in content
            if chunk.content:
                chunks.append(chunk)
                last_chunk_text = element.content

        elif etype == ElementType.PARAGRAPH:
            assert isinstance(element, Paragraph)
            # Inclui seção como contexto no início do parágrafo
            section_ctx = f"[Seção: {current_section_title}] " if current_section_title else ""
            text = section_ctx + element.content
            add_text_chunks(text, [element.id])

        else:
            # Page, Image: ignora nesta fase
            pass

    # Numera os chunks em ordem
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks
