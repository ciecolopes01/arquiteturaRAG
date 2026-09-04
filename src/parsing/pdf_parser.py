"""
Parser de PDF — Passo 10 + Tabelas Estruturadas (Passo 15).

Usa pymupdf (fitz) para extrair texto e estrutura de página. Diferente do
parser TXT (Passo 2), que detecta heading por heurística de pontuação,
o PDF real permite detectar heading por **tamanho de fonte**: qualquer
linha cujo tamanho de fonte é notavelmente maior que o corpo do texto vira
uma Section.

Tabelas são detectadas por `page.find_tables()` (PyMuPDF >= 1.23) e
extraídas como objetos `Table` canônicos com markdown + sumário em
linguagem natural (Passo 15). As regiões de tabela são excluídas do
processamento de texto para evitar duplicação.

Decisões de design:
- Um Page (Canonical Document Model) é criado por página do PDF, agrupando
  os element_ids nela contidos — necessário para perguntas do tipo "em que
  página aparece X" (reaproveitado depois no Passo 19/26 para imagem/vídeo).
- Tamanho de fonte do "corpo" = moda (tamanho mais frequente) entre todos
  os spans do documento. Heading = span isolado (linha de 1 bloco) com
  tamanho >= 1.15x o corpo, texto curto (<=120 chars) e sem terminar em
  pontuação de frase.
- Nível do heading: 2 buckets por tamanho relativo ao corpo (>=1.4x -> H1,
  >=1.15x -> H2). Simples e suficiente para o corpus seed desta fase.
- content_hash: SHA-256 dos bytes brutos do PDF.
- Tabelas detectadas têm content = markdown + sumário em linguagem natural
  (build_table_content do table_extractor). As linhas de texto que caem
  dentro da bbox de uma tabela são descartadas do processamento de texto.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import List, Optional

from src.parsing.canonical_model import Document, Image, Page, Paragraph, Section, Table
from src.parsing.image_parser import extract_images_from_pdf_page
from src.parsing.table_extractor import (
    build_table_content,
    extract_tables_from_pdf_page,
    text_line_in_table_bbox,
)

_SENTENCE_END = (".", "!", "?", ":", ";")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _looks_like_heading_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 120:
        return False
    if stripped.endswith(_SENTENCE_END):
        return False
    return True


def parse_pdf(
    path: str | Path,
    tenant_id: Optional[str] = None,
    source_uri: Optional[str] = None,
    extract_images: bool = True,
    upload_storage: bool = True,
) -> Document:
    """
    Lê um arquivo .pdf e produz um Document no Canonical Document Model.

    Fluxo (Passo 15 + Passo 19):
    1. Abre o PDF com pymupdf e extrai spans de texto (com tamanho de fonte
       e bbox) de cada página via get_text("rawdict").
    2. Detecta tabelas por página via `page.find_tables()` e armazena suas
       bboxes para filtrar linhas de texto internas.
    3. Extrai imagens embutidas por página com OCR (Tesseract) e descrição
       visual (Passo 19).
    4. Calcula o tamanho de fonte "corpo" (o mais comum excluindo spans em
       área de tabela).
    5. Para cada linha fora de tabela: heading (Section) se a fonte for
       notavelmente maior que o corpo e o texto for curto; senão vira Paragraph.
    6. Para cada tabela: cria elemento Table com markdown + sumário em
       linguagem natural via build_table_content().
    7. Para cada imagem: cria elemento Image com ocr_text e visual_description.
    8. Um Page é criado por página, agrupando todos os elementos dela.
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError("pymupdf não instalado. Execute: pip install pymupdf")

    path = Path(path)
    raw_bytes = path.read_bytes()
    source = source_uri or str(path)

    pdf = fitz.open(stream=raw_bytes, filetype="pdf")

    # -----------------------------------------------------------------------
    # Passo 1: coleta todas as linhas (texto + tamanho de fonte + bbox)
    # por página, detecta tabelas e extrai imagens por página.
    # -----------------------------------------------------------------------
    pages_data: List[dict] = []  # por página: {"lines": [...], "tables": [...], "images": [...]}
    all_sizes: List[float] = []

    for page_number, page in enumerate(pdf, start=1):
        # Detecta tabelas antes de processar texto
        page_tables, fitz_tbls = extract_tables_from_pdf_page(
            page,
            page_number=page_number,
        )
        table_bboxes = []
        for ft in fitz_tbls:
            try:
                table_bboxes.append(ft.bbox)
            except Exception:
                pass

        # Extrai imagens embutidas (Passo 19)
        page_images = []
        if extract_images:
            page_images = extract_images_from_pdf_page(
                page=page,
                pdf_doc=pdf,
                page_number=page_number,
                upload_storage=upload_storage,
            )

        # Extrai linhas de texto (excluindo bboxes de tabela)
        page_dict = page.get_text("dict")
        lines_out: List[dict] = []
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                size = max(s.get("size", 0.0) for s in spans)
                bbox = line.get("bbox", (0, 0, 0, 0))
                # Descarta linhas dentro de bboxes de tabela
                if text_line_in_table_bbox(bbox, table_bboxes):
                    continue
                lines_out.append({"text": text, "size": size, "bbox": bbox})
                all_sizes.append(round(size, 1))

        pages_data.append({"lines": lines_out, "tables": page_tables, "images": page_images})

    body_size = Counter(all_sizes).most_common(1)[0][0] if all_sizes else 11.0

    # -----------------------------------------------------------------------
    # Passo 2: título do documento
    # -----------------------------------------------------------------------
    meta_title = (pdf.metadata or {}).get("title", "").strip()

    doc = Document(
        source_uri=source,
        title=meta_title or None,
        tenant_id=tenant_id,
        content_hash=_sha256(raw_bytes),
        parser_version="pdf-pymupdf-0.2",
        language="pt",
        metadata={
            "original_filename": path.name,
            "parser": "parse_pdf",
            "page_count": pdf.page_count,
            "body_font_size": body_size,
        },
    )

    pdf.close()

    # -----------------------------------------------------------------------
    # Passo 3: monta elementos Section / Paragraph / Table / Page
    # -----------------------------------------------------------------------
    current_section: Optional[Section] = None
    title_set = bool(doc.title)

    for page_number, page_data in enumerate(pages_data, start=1):
        lines = page_data["lines"]
        page_tables: List[Table] = page_data["tables"]
        page_images: List[Image] = page_data.get("images", [])

        page_element_ids: List[str] = []
        page_text_parts: List[str] = []

        # ---- Helpers locais de flush ----
        buffer: List[str] = []

        def flush_paragraph() -> None:
            if not buffer:
                return
            content = " ".join(buffer).strip()
            buffer.clear()
            if not content:
                return
            section_ctx = f"[Seção: {current_section.title}] " if current_section else ""
            para = Paragraph(
                content=content,
                metadata={
                    "page_number": page_number,
                    "section_id": current_section.id if current_section else None,
                },
            )
            doc.elements.append(para)
            page_element_ids.append(para.id)
            page_text_parts.append(section_ctx + content)

        heading_buffer: List[str] = []
        heading_buffer_size: Optional[float] = None

        def flush_heading() -> None:
            nonlocal current_section, heading_buffer, heading_buffer_size, title_set
            if not heading_buffer:
                return
            text = " ".join(heading_buffer)
            size = heading_buffer_size
            heading_buffer.clear()
            heading_buffer_size = None

            if not title_set:
                doc.title = text
                title_set = True
                return

            level = 1 if size >= body_size * 1.4 else 2
            current_section = Section(
                title=text,
                level=level,
                content="",
                metadata={"page_number": page_number, "font_size": size},
            )
            doc.elements.append(current_section)
            page_element_ids.append(current_section.id)
            page_text_parts.append(text)

        # ---- Processa linhas de texto ----
        for line in lines:
            text = line["text"]
            size = line["size"]
            is_heading = (
                size >= body_size * 1.15
                and _looks_like_heading_text(text)
            )

            if is_heading:
                flush_paragraph()
                if heading_buffer and abs(size - (heading_buffer_size or 0)) > 0.6:
                    flush_heading()
                heading_buffer.append(text)
                heading_buffer_size = size
            else:
                flush_heading()
                buffer.append(text)

        flush_heading()
        flush_paragraph()

        # ---- Adiciona tabelas desta página ao documento ----
        for tbl in page_tables:
            # Atualiza section_title no metadata da tabela
            tbl.metadata["section_title"] = current_section.title if current_section else ""
            # Gera content completo: markdown + sumário em linguagem natural
            tbl.content = build_table_content(tbl, section_title=current_section.title if current_section else None)
            doc.elements.append(tbl)
            page_element_ids.append(tbl.id)
            page_text_parts.append(tbl.content[:200])  # resumo para page.content

        # ---- Adiciona imagens desta página ao documento (Passo 19) ----
        for img in page_images:
            if current_section and not img.metadata.get("section_title"):
                img.metadata["section_title"] = current_section.title
            doc.elements.append(img)
            page_element_ids.append(img.id)
            page_text_parts.append(img.content[:200])  # resumo para page.content

        # ---- Página ----
        page_el = Page(
            page_number=page_number,
            content="\n".join(page_text_parts),
            element_ids=page_element_ids,
            metadata={
                "line_count": len(lines),
                "table_count": len(page_tables),
                "image_count": len(page_images),
            },
        )
        doc.elements.append(page_el)

    return doc
