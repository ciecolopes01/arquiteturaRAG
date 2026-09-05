"""
Chunker Parent/Child — Passo 17.

Gera chunks hierárquicos em dois níveis:
1. Chunks Pais (grandes, default 1200 tokens):
   - Preservam contexto amplo (seções inteiras, múltiplos parágrafos correlatos, tabelas completas).
   - Não são enviados para indexação direta de embedding nesta fase, mas servem
     como contexto expandido durante a geração (Context Engine).
2. Chunks Filhos (pequenos, default 300 tokens com 40 tokens de overlap):
   - Derivados de cada pai.
   - Excelentes para busca semântica de alta precisão (dense) e busca lexical (BM25).
   - Cada filho carrega o `parent_chunk_id` e o `parent_content` em seus metadados.

Vantagem arquitetural:
- O retrieval busca pelo filho (alta especificidade de termos e vetor focado).
- O context engine expande para o pai (amplo contexto para o LLM responder sem fragmentação).
- Deduplicação automática: múltiplos filhos do mesmo pai recuperados no top-K
  são condensados em um único bloco de contexto do pai.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.chunking.recursive import (
    Chunk,
    _get_overlap_prefix,
    _max_words,
    _split_text,
)
from src.parsing.canonical_model import (
    AudioSegment,
    Document,
    ElementType,
    Image,
    Paragraph,
    Section,
    Table,
    VideoScene,
)


@dataclass
class ParentChildResult:
    """Resultado da divisão hierárquica Parent/Child de um Document."""

    parents: List[Chunk] = field(default_factory=list)
    children: List[Chunk] = field(default_factory=list)

    @property
    def total_parents(self) -> int:
        return len(self.parents)

    @property
    def total_children(self) -> int:
        return len(self.children)

    def get_parent(self, parent_chunk_id: str) -> Optional[Chunk]:
        """Busca um chunk pai pelo ID."""
        for p in self.parents:
            if p.chunk_id == parent_chunk_id:
                return p
        return None

    def get_children_for_parent(self, parent_chunk_id: str) -> List[Chunk]:
        """Retorna todos os filhos de um determinado pai."""
        return [c for c in self.children if c.parent_chunk_id == parent_chunk_id]


def _build_parent_chunks(
    document: Document,
    parent_max_tokens: int = 1200,
) -> List[Chunk]:
    """
    Agrupa elementos do documento em chunks pais de tamanho até ~parent_max_tokens.
    Respeita seções lógicas e tabelas estruturadas.
    """
    parent_max_words = _max_words(parent_max_tokens)
    parents: List[Chunk] = []

    current_section_title: Optional[str] = None
    current_buffer: List[str] = []
    current_element_ids: List[str] = []

    def flush_parent() -> None:
        if not current_buffer:
            return
        content = "\n\n".join(current_buffer).strip()
        current_buffer.clear()
        elem_ids = list(current_element_ids)
        current_element_ids.clear()

        if not content:
            return

        p_chunk = Chunk(
            document_id=document.id,
            content=content,
            tenant_id=document.tenant_id,
            acl=[entry.model_dump() for entry in document.acl],
            parent_chunk_id=None,
            metadata={
                "is_parent": True,
                "is_child": False,
                "source_uri": document.source_uri,
                "document_title": document.title,
                "section_title": current_section_title,
                "parent_index": len(parents),
                "element_ids": elem_ids,
                "max_tokens": parent_max_tokens,
            },
        )
        parents.append(p_chunk)

    for element in document.elements:
        etype = element.element_type

        if etype == ElementType.SECTION:
            assert isinstance(element, Section)
            # Ao iniciar nova seção importante, se o buffer já tiver conteúdo razoável, faz flush
            if current_buffer and len(" ".join(current_buffer).split()) > parent_max_words // 2:
                flush_parent()
            current_section_title = element.title
            if element.content and element.content.strip():
                sec_text = f"{element.title}\n\n{element.content}"
                current_buffer.append(sec_text)
                current_element_ids.append(element.id)

        elif etype == ElementType.TABLE:
            assert isinstance(element, Table)
            # Tabela gera seu próprio chunk pai se já tiver conteúdo estruturado
            flush_parent()
            table_content = element.content or ""
            if not table_content.strip():
                prefix_ctx = f"[Seção: {current_section_title}]\n" if current_section_title else ""
                caption = f"[Tabela: {element.caption}]\n" if element.caption else ""
                table_content = prefix_ctx + caption + (element.content or "")

            tbl_chunk = Chunk(
                document_id=document.id,
                content=table_content.strip(),
                tenant_id=document.tenant_id,
                acl=[entry.model_dump() for entry in document.acl],
                parent_chunk_id=None,
                metadata={
                    "is_parent": True,
                    "is_child": False,
                    "source_uri": document.source_uri,
                    "document_title": document.title,
                    "section_title": current_section_title,
                    "parent_index": len(parents),
                    "element_ids": [element.id],
                    "table_headers": element.headers or [],
                    "table_row_count": len(element.rows) if element.rows else 0,
                    "has_table_summary": "## Sumário" in table_content,
                },
            )
            parents.append(tbl_chunk)

        elif etype == ElementType.IMAGE:
            assert isinstance(element, Image)
            flush_parent()
            img_content = element.content or ""
            if not img_content.strip():
                parts = []
                if element.visual_description:
                    parts.append(f"## Descrição Visual\n{element.visual_description}")
                if element.ocr_text:
                    parts.append(f"## Texto Extraído (OCR)\n{element.ocr_text}")
                img_content = "\n\n".join(parts)

            sec_ctx = f"[Seção: {current_section_title}]\n" if current_section_title and "Contexto:" not in img_content else ""
            img_full = (sec_ctx + img_content).strip()

            img_chunk = Chunk(
                document_id=document.id,
                content=img_full,
                tenant_id=document.tenant_id,
                acl=[entry.model_dump() for entry in document.acl],
                parent_chunk_id=None,
                metadata={
                    "is_parent": True,
                    "is_child": False,
                    "element_type": "image",
                    "image_id": element.id,
                    "storage_uri": element.storage_uri,
                    "width": element.width,
                    "height": element.height,
                    "ocr_text": element.ocr_text,
                    "visual_description": element.visual_description,
                    "page_number": element.metadata.get("page_number"),
                    "source_uri": document.source_uri,
                    "document_title": document.title,
                    "section_title": current_section_title,
                    "parent_index": len(parents),
                    "element_ids": [element.id],
                },
            )
            parents.append(img_chunk)

        elif etype == ElementType.VIDEO_SCENE:
            assert isinstance(element, VideoScene)
            flush_parent()
            content = element.content or ""
            if not content.strip():
                parts = [f"## Cena {element.scene_index} ({element.start_time_str} - {element.end_time_str})"]
                if element.transcript:
                    parts.append(f"### Transcrição de Áudio\n{element.transcript}")
                if element.visual_description:
                    parts.append(f"### Descrição Visual (Keyframe)\n{element.visual_description}")
                if element.ocr_text:
                    parts.append(f"### Texto na Tela (OCR)\n{element.ocr_text}")
                content = "\n\n".join(parts)

            sec_ctx = f"[Vídeo: {document.title or 'Vídeo'}]\n" if document.title else ""
            video_full = (sec_ctx + content).strip()

            video_chunk = Chunk(
                document_id=document.id,
                content=video_full,
                tenant_id=document.tenant_id,
                acl=[entry.model_dump() for entry in document.acl],
                parent_chunk_id=None,
                metadata={
                    "is_parent": True,
                    "is_child": False,
                    "element_type": "video_scene",
                    "scene_index": element.scene_index,
                    "start_time": element.start_time,
                    "end_time": element.end_time,
                    "start_time_str": element.start_time_str,
                    "end_time_str": element.end_time_str,
                    "transcript": element.transcript,
                    "visual_description": element.visual_description,
                    "ocr_text": element.ocr_text,
                    "source_uri": document.source_uri,
                    "document_title": document.title,
                    "section_title": current_section_title,
                    "parent_index": len(parents),
                    "element_ids": [element.id],
                },
            )
            parents.append(video_chunk)

        elif etype == ElementType.AUDIO_SEGMENT:
            assert isinstance(element, AudioSegment)
            flush_parent()
            content = element.content or ""
            if not content.strip():
                parts = [f"## Segmento de Áudio ({element.start_time_str} - {element.end_time_str})"]
                if element.speaker:
                    parts.append(f"Locutor: {element.speaker}")
                if element.transcript:
                    parts.append(f"Transcrição: {element.transcript}")
                content = "\n\n".join(parts)

            sec_ctx = f"[Áudio: {document.title or 'Áudio'}]\n" if document.title else ""
            audio_full = (sec_ctx + content).strip()

            audio_chunk = Chunk(
                document_id=document.id,
                content=audio_full,
                tenant_id=document.tenant_id,
                acl=[entry.model_dump() for entry in document.acl],
                parent_chunk_id=None,
                metadata={
                    "is_parent": True,
                    "is_child": False,
                    "element_type": "audio_segment",
                    "start_time": element.start_time,
                    "end_time": element.end_time,
                    "start_time_str": element.start_time_str,
                    "end_time_str": element.end_time_str,
                    "speaker": element.speaker,
                    "transcript": element.transcript,
                    "source_uri": document.source_uri,
                    "document_title": document.title,
                    "section_title": current_section_title,
                    "parent_index": len(parents),
                    "element_ids": [element.id],
                },
            )
            parents.append(audio_chunk)

        elif etype == ElementType.PARAGRAPH:
            assert isinstance(element, Paragraph)
            sec_ctx = f"[Seção: {current_section_title}]\n" if current_section_title and not current_buffer else ""
            para_text = sec_ctx + element.content.strip()

            # Testa se cabe no chunk pai atual
            tentative_words = len((" ".join(current_buffer) + " " + para_text).split())
            if tentative_words > parent_max_words and current_buffer:
                flush_parent()

            # Se o próprio parágrafo for maior que parent_max_words, divide recursivamente
            if len(para_text.split()) > parent_max_words:
                sub_parts = _split_text(para_text, parent_max_words)
                for part in sub_parts:
                    current_buffer.append(part)
                    current_element_ids.append(element.id)
                    flush_parent()
            else:
                current_buffer.append(para_text)
                current_element_ids.append(element.id)

    flush_parent()
    return parents


def _build_child_chunks_for_parent(
    parent: Chunk,
    child_max_tokens: int = 300,
    child_overlap: int = 40,
) -> List[Chunk]:
    """
    Subdivide um chunk pai em chunks filhos menores, com sobreposição controlada.
    Cada filho referencia o ID do pai e carrega o conteúdo completo do pai em metadados.
    """
    child_max_words = _max_words(child_max_tokens)
    overlap_words = _max_words(child_overlap)

    children: List[Chunk] = []
    raw_parts = _split_text(parent.content, child_max_words)
    last_text: Optional[str] = None

    for i, part in enumerate(raw_parts):
        if last_text and overlap_words > 0:
            prefix = _get_overlap_prefix(last_text, overlap_words)
            content = prefix + " " + part if prefix.strip() else part
        else:
            content = part

        content = content.strip()
        if not content:
            continue

        child_metadata = dict(parent.metadata)
        child_metadata.update({
            "is_parent": False,
            "is_child": True,
            "parent_chunk_id": parent.chunk_id,
            "parent_content": parent.content,
            "parent_word_count": len(parent.content.split()),
            "child_index": i,
            "child_max_tokens": child_max_tokens,
            "child_overlap": child_overlap,
        })

        child_chunk = Chunk(
            document_id=parent.document_id,
            content=content,
            tenant_id=parent.tenant_id,
            acl=parent.acl,
            parent_chunk_id=parent.chunk_id,
            metadata=child_metadata,
        )
        children.append(child_chunk)
        last_text = part

    return children


def chunk_parent_child(
    document: Document,
    parent_max_tokens: int = 1200,
    child_max_tokens: int = 300,
    child_overlap: int = 40,
) -> ParentChildResult:
    """
    Transforma um Document em uma estrutura hierárquica Parent/Child.

    Parâmetros:
    - document          : documento no Canonical Document Model.
    - parent_max_tokens : tamanho máximo do chunk pai (~1200 tokens para contexto amplo).
    - child_max_tokens  : tamanho máximo do chunk filho (~300 tokens para busca semântica).
    - child_overlap     : sobreposição entre filhos consecutivos (~40 tokens).

    Retorna:
    - ParentChildResult com listas de parents e children devidamente vinculados.
    """
    parents = _build_parent_chunks(document, parent_max_tokens=parent_max_tokens)

    all_children: List[Chunk] = []
    for parent in parents:
        children = _build_child_chunks_for_parent(
            parent,
            child_max_tokens=child_max_tokens,
            child_overlap=child_overlap,
        )
        all_children.extend(children)

    return ParentChildResult(parents=parents, children=all_children)
