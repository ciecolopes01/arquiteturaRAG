"""
Parsers simples — Passo 2.

Implementa parse_txt e parse_html: leem um arquivo do disco e produzem
um Document no Canonical Document Model.

Decisões de design:
- TXT: cada bloco de texto separado por linha em branco vira um Paragraph.
  Linhas que parecem títulos (curtas, sem ponto final, seguidas de linha em
  branco) viram Section.
- HTML: usamos BeautifulSoup para extrair texto estruturado.  Cada <h1>-<h6>
  vira Section; <p>, <li> e <td> viram Paragraph.  Tabelas <table> viram Table.
- content_hash: SHA-256 do conteúdo bruto do arquivo.
- storage_uri: opcionalmente salva o arquivo original no MinIO e registra a URI.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List, Optional

from src.parsing.canonical_model import (
    Document,
    ElementType,
    Paragraph,
    Section,
    Table,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][^.!?\n]{0,80}$")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _looks_like_heading(line: str) -> bool:
    """Heurística simples: linha curta, começa com maiúscula, sem ponto final."""
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    if stripped.endswith((".","!","?")):
        return False
    return bool(_HEADING_RE.match(stripped))


# ---------------------------------------------------------------------------
# Parser TXT
# ---------------------------------------------------------------------------

def parse_txt(
    path: str | Path,
    tenant_id: Optional[str] = None,
    source_uri: Optional[str] = None,
) -> Document:
    """
    Lê um arquivo .txt e produz um Document.

    Lógica de parsing:
    - Divide por blocos separados por linha(s) em branco.
    - Primeiro bloco: sempre trata como título do documento (se parecer heading)
      ou como parágrafo introdutório.
    - Blocos subsequentes: heading → Section; resto → Paragraph.
    - O campo Section.children não é preenchido nesta fase (Passo 2 é simples).
    """
    path = Path(path)
    raw_bytes = path.read_bytes()
    raw_text = raw_bytes.decode("utf-8", errors="replace")

    source = source_uri or str(path)
    doc = Document(
        source_uri=source,
        title=None,
        tenant_id=tenant_id,
        content_hash=_sha256(raw_bytes),
        parser_version="txt-simple-0.1",
        language="pt",
        metadata={"original_filename": path.name, "parser": "parse_txt"},
    )

    # Divide em blocos por linha(s) em branco
    blocks: List[str] = re.split(r"\n{2,}", raw_text.strip())

    current_section: Optional[Section] = None
    title_set = False

    for i, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue

        # Bloco com uma única linha — possível heading
        lines = [l for l in block.splitlines() if l.strip()]

        if len(lines) == 1 and _looks_like_heading(lines[0]):
            heading_text = lines[0].strip()

            if not title_set:
                doc.title = heading_text
                title_set = True
                continue  # título do doc não vira elemento

            # Detecta nível pelo contexto: primeiro heading → H1, subsequentes → H2
            level = 1 if current_section is None else 2
            current_section = Section(
                title=heading_text,
                level=level,
                content="",
                metadata={"block_index": i},
            )
            doc.elements.append(current_section)

        else:
            # Bloco com múltiplas linhas ou linha longa → Paragraph
            # Normaliza espaços internos mas preserva quebras de linha
            content = " ".join(lines)

            para = Paragraph(
                content=content,
                metadata={
                    "block_index": i,
                    "section_id": current_section.id if current_section else None,
                },
            )
            doc.elements.append(para)

    return doc


# ---------------------------------------------------------------------------
# Parser HTML
# ---------------------------------------------------------------------------

def parse_html(
    path: str | Path,
    tenant_id: Optional[str] = None,
    source_uri: Optional[str] = None,
) -> Document:
    """
    Lê um arquivo .html e produz um Document usando BeautifulSoup.

    Lógica de parsing:
    - <title> → doc.title
    - <h1>–<h6> → Section (level = número do heading)
    - <p> → Paragraph
    - <li> → Paragraph (com prefixo "• ")
    - <table> → Table (extraindo headers e rows)
    - Elementos aninhados são aplanados (não há tree de Sections nesta fase)
    """
    try:
        from bs4 import BeautifulSoup, NavigableString, Tag
    except ImportError:
        raise ImportError(
            "beautifulsoup4 não instalado. "
            "Execute: pip install beautifulsoup4"
        )

    path = Path(path)
    raw_bytes = path.read_bytes()
    raw_text = raw_bytes.decode("utf-8", errors="replace")

    source = source_uri or str(path)

    soup = BeautifulSoup(raw_text, "html.parser")

    # Título do documento
    title_tag = soup.find("title")
    doc_title = title_tag.get_text(strip=True) if title_tag else path.stem

    doc = Document(
        source_uri=source,
        title=doc_title,
        tenant_id=tenant_id,
        content_hash=_sha256(raw_bytes),
        parser_version="html-simple-0.1",
        language="pt",
        metadata={"original_filename": path.name, "parser": "parse_html"},
    )

    body = soup.find("body") or soup

    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    # Processa elementos em ordem de documento (percorrimento linear do DOM)
    # Rastreamos quais elementos já foram processados (ex: células dentro de <table>)
    processed_ids: set = set()

    for tag in body.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table"],
        recursive=True,
    ):
        tag_id = id(tag)
        if tag_id in processed_ids:
            continue

        name = tag.name

        # --- Heading → Section ---
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(name[1])
            text = _clean(tag.get_text())
            if text:
                sec = Section(
                    title=text,
                    level=level,
                    content="",
                    metadata={"html_tag": name},
                )
                doc.elements.append(sec)

        # --- Parágrafo → Paragraph ---
        elif name == "p":
            text = _clean(tag.get_text())
            if text:
                para = Paragraph(
                    content=text,
                    metadata={"html_tag": "p"},
                )
                doc.elements.append(para)

        # --- Item de lista → Paragraph ---
        elif name == "li":
            text = _clean(tag.get_text())
            if text:
                para = Paragraph(
                    content=f"• {text}",
                    metadata={"html_tag": "li"},
                )
                doc.elements.append(para)

        # --- Tabela → Table ---
        elif name == "table":
            headers: List[str] = []
            rows: List[List[str]] = []

            # Marca todas as células como processadas para evitar re-parse
            for cell in tag.find_all(["th", "td", "tr", "thead", "tbody"]):
                processed_ids.add(id(cell))

            # Extrai headers (<th> na primeira linha ou <thead>)
            thead = tag.find("thead")
            if thead:
                for th in thead.find_all("th"):
                    headers.append(_clean(th.get_text()))

            # Extrai rows (<tr> fora do thead)
            for tr in tag.find_all("tr"):
                cells = tr.find_all(["td", "th"])
                if not cells:
                    continue
                row = [_clean(c.get_text()) for c in cells]
                # Se não havia thead, usa a primeira linha como headers
                if not headers and all(c.name == "th" for c in cells):
                    headers = row
                else:
                    if row:
                        rows.append(row)

            # Monta representação Markdown da tabela
            if headers:
                md = "| " + " | ".join(headers) + " |\n"
                md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                for row in rows:
                    md += "| " + " | ".join(row) + " |\n"
            else:
                md = "\n".join(" | ".join(row) for row in rows)

            tbl = Table(
                content=md.strip(),
                headers=headers,
                rows=rows,
                metadata={"html_tag": "table"},
            )
            doc.elements.append(tbl)
            processed_ids.add(tag_id)

    return doc


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def parse_file(
    path: str | Path,
    tenant_id: Optional[str] = None,
    source_uri: Optional[str] = None,
) -> Document:
    """Despacha para o parser correto com base na extensão do arquivo."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".txt":
        return parse_txt(path, tenant_id=tenant_id, source_uri=source_uri)
    elif ext in (".html", ".htm"):
        return parse_html(path, tenant_id=tenant_id, source_uri=source_uri)
    elif ext == ".pdf":
        from src.parsing.pdf_parser import parse_pdf
        return parse_pdf(path, tenant_id=tenant_id, source_uri=source_uri)
    elif ext == ".docx":
        from src.parsing.docx_parser import parse_docx
        return parse_docx(path, tenant_id=tenant_id, source_uri=source_uri)
    elif ext == ".xlsx":
        from src.parsing.xlsx_parser import parse_xlsx
        return parse_xlsx(path, tenant_id=tenant_id, source_uri=source_uri)
    else:
        raise ValueError(
            f"Extensão '{ext}' não suportada. Use .txt, .html, .pdf, .docx ou .xlsx."
        )
