"""
Extração e representação de tabelas — Passo 15.

Funções:
  table_to_markdown(headers, rows)          — formata como tabela Markdown.
  table_to_natural_language(table)          — sumário determinístico em pt-BR.
  extract_tables_from_pdf_page(page, ...)   — detecta tabelas numa página PyMuPDF.
  build_table_content(table)                — monta content = markdown + sumário.

Design:
- Sumário gerado por template (sem LLM), determinístico e testável.
- Para tabelas pequenas (≤ MAX_SUMMARY_ROWS linhas), cobre todas as linhas.
- Para tabelas grandes, cobre as primeiras MAX_SUMMARY_ROWS e menciona o total.
- Extração via PyMuPDF find_tables() (>= 1.23); fallback puro por texto se
  a tabela não for detectada por grade.
- Cada tabela recebe section_title como contexto quando disponível.

Integrado em:
  - src/parsing/pdf_parser.py   (extrai tabelas por página antes do texto)
  - src/parsing/docx_parser.py  (adiciona sumário às tabelas já extraídas)
  - src/chunking/recursive.py   (metadata extra nos chunks de tabela)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.parsing.canonical_model import Table

# ---------------------------------------------------------------------------
# Constantes de apresentação
# ---------------------------------------------------------------------------

MAX_SUMMARY_ROWS   = 5   # máximo de linhas detalhadas no sumário
MAX_COLUMN_WIDTH   = 40  # trunca células muito largas na exibição de sumário


# ---------------------------------------------------------------------------
# Formatação Markdown
# ---------------------------------------------------------------------------

def table_to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    """
    Converte headers + rows para tabela Markdown padrão.

    Colunas são alinhadas pela largura máxima do conteúdo (com mínimo de 3).
    Células None ou vazias viram '' no markdown.

    Parâmetros:
    - headers : lista de strings de cabeçalho.
    - rows    : lista de listas de strings (cada lista = uma linha de dados).

    Retorna string multiline com a tabela em markdown.
    """
    if not headers:
        return ""

    # Normaliza: garante que todas as linhas tenham o mesmo número de colunas
    n_cols = len(headers)
    norm_rows: List[List[str]] = []
    for row in rows:
        padded = list(row) + [""] * max(0, n_cols - len(row))
        norm_rows.append([str(c or "").strip()[:300] for c in padded[:n_cols]])

    # Largura de cada coluna
    col_widths = [max(3, len(str(h))) for h in headers]
    for row in norm_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells: List[str]) -> str:
        return "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

    header_line = fmt_row([str(h) for h in headers])
    sep_line    = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
    data_lines  = [fmt_row(row) for row in norm_rows]

    return "\n".join([header_line, sep_line] + data_lines)


# ---------------------------------------------------------------------------
# Sumário em linguagem natural (determinístico, sem LLM)
# ---------------------------------------------------------------------------

def _truncate_cell(value: str, max_len: int = MAX_COLUMN_WIDTH) -> str:
    """Trunca uma célula para exibição no sumário."""
    value = str(value or "").strip()
    return value if len(value) <= max_len else value[:max_len - 1] + "…"


def table_to_natural_language(table: Table) -> str:
    """
    Gera um sumário em português a partir dos dados estruturados da tabela.

    Estrutura do sumário gerado:
    1. Cabeçalho: "Esta tabela [sobre <caption>] contém <N> coluna(s) (<nomes>)
       e <M> linha(s) de dados."
    2. Para as primeiras MAX_SUMMARY_ROWS linhas de dados: descreve cada linha
       no formato "Linha <i>: <col1>=<val1>, <col2>=<val2>, ...".
    3. Se há mais linhas, acrescenta "... e mais <X> linha(s) não mostradas."

    Parâmetros:
    - table : objeto Table do Canonical Document Model.

    Retorna string em português com o sumário.
    """
    headers = table.headers or []
    rows    = table.rows or []
    caption = (table.caption or "").strip()
    section = table.metadata.get("section_title", "")

    # --- Cabeçalho do sumário ---
    n_cols = len(headers)
    n_rows = len(rows)

    if n_cols == 0:
        return "Tabela sem colunas identificadas."

    col_names = ", ".join(f'"{h}"' for h in headers[:6])
    if n_cols > 6:
        col_names += f" e mais {n_cols - 6} coluna(s)"

    sobre = f" sobre {caption}" if caption else ""
    ctx   = f" (seção: {section})" if section else ""

    lines: List[str] = [
        f"Esta tabela{sobre}{ctx} contém {n_cols} coluna(s) ({col_names}) "
        f"e {n_rows} linha(s) de dados."
    ]

    # --- Linhas de dados ---
    visible_rows = rows[:MAX_SUMMARY_ROWS]
    for i, row in enumerate(visible_rows, start=1):
        if not headers:
            break
        # Formata pares coluna=valor
        pairs = []
        for h, val in zip(headers, row):
            cell = _truncate_cell(str(val or ""))
            pairs.append(f'{h}={cell}')
        lines.append(f"Linha {i}: " + "; ".join(pairs) + ".")

    remaining = n_rows - len(visible_rows)
    if remaining > 0:
        lines.append(f"... e mais {remaining} linha(s) não detalhadas neste sumário.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Montagem do content do chunk de tabela
# ---------------------------------------------------------------------------

def build_table_content(
    table: Table,
    section_title: Optional[str] = None,
) -> str:
    """
    Monta o campo `content` de um chunk de tabela com DUAS representações:
    1. Markdown estruturado (para busca por estrutura/termos exatos)
    2. Sumário em linguagem natural (para busca semântica por perguntas naturais)

    O conteúdo duplo maximiza a cobertura: uma query como "qual o valor de X
    em 2025" pode encontrar via texto do markdown E via sumário.

    Parâmetros:
    - table         : objeto Table com headers, rows, caption.
    - section_title : título da seção ao redor da tabela (contexto adicional).

    Retorna string pronta para ser usada como content do Chunk.
    """
    parts: List[str] = []

    # Contexto de seção
    if section_title:
        parts.append(f"[Seção: {section_title}]")

    # Caption
    caption = (table.caption or "").strip()
    if caption:
        parts.append(f"[Tabela: {caption}]")

    # Representação Markdown
    if table.headers:
        md = table_to_markdown(table.headers, table.rows or [])
        if md:
            parts.append("\n## Dados estruturados\n")
            parts.append(md)

    # Sumário em linguagem natural
    summary = table_to_natural_language(table)
    if summary:
        parts.append("\n## Sumário\n")
        parts.append(summary)

    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Extração de tabelas de páginas PDF (PyMuPDF)
# ---------------------------------------------------------------------------

def extract_tables_from_pdf_page(
    page,
    page_number: int,
    section_title: Optional[str] = None,
    caption_hint: Optional[str] = None,
) -> Tuple[List[Table], List[object]]:
    """
    Detecta e extrai tabelas de uma página PyMuPDF.

    Usa `page.find_tables()` (PyMuPDF >= 1.23). Para cada tabela encontrada:
    - Extrai `headers` (primeira linha) e `rows` (restante).
    - Gera markdown + sumário em linguagem natural.
    - Retorna objeto `Table` canônico.

    Parâmetros:
    - page         : objeto fitz.Page.
    - page_number  : número da página (1-indexed), para metadata.
    - section_title: título da seção atual (contexto de tabela).
    - caption_hint : texto próximo à tabela usado como caption (opcional).

    Retorna:
    - tables    : lista de objetos Table do Canonical Document Model.
    - fitz_tbls : lista de objetos fitz.table.Table (para remoção de bbox do texto).
    """
    tables: List[Table] = []
    fitz_tbls: List[object] = []

    try:
        tbl_finder = page.find_tables()
        found_tables = tbl_finder.tables if hasattr(tbl_finder, "tables") else []
    except Exception:
        return tables, fitz_tbls

    for t_idx, fitz_tbl in enumerate(found_tables):
        try:
            raw_data = fitz_tbl.extract()
        except Exception:
            continue

        if not raw_data or len(raw_data) < 2:
            # Tabela com menos de 2 linhas (só cabeçalho ou vazia) — ignora
            continue

        # Filtra células None → string vazia; remove linhas completamente vazias
        cleaned: List[List[str]] = []
        for row in raw_data:
            cleaned_row = [str(cell or "").strip() for cell in row]
            if any(c for c in cleaned_row):  # pelo menos uma célula não-vazia
                cleaned.append(cleaned_row)

        if len(cleaned) < 2:
            continue

        headers = cleaned[0]
        rows    = cleaned[1:]

        # Garante que headers não sejam todos vazios
        if not any(headers):
            headers = [f"Col{i+1}" for i in range(len(headers))]

        # Caption: preferência ao hint externo, depois a um padrão
        caption = caption_hint or f"Tabela {t_idx + 1} (pág. {page_number})"

        # Monta a tabela canônica
        tbl_section = section_title or ""
        md = table_to_markdown(headers, rows)
        tbl = Table(
            headers=headers,
            rows=rows,
            caption=caption,
            content=md,  # content inicial = markdown (será expandido pelo chunker)
            metadata={
                "page_number": page_number,
                "table_index": t_idx,
                "section_title": tbl_section,
                "row_count": len(rows),
                "col_count": len(headers),
                "parser": "pymupdf_find_tables",
            },
        )
        tables.append(tbl)
        fitz_tbls.append(fitz_tbl)

    return tables, fitz_tbls


# ---------------------------------------------------------------------------
# Utilidade: bbox de interseção (para filtrar texto dentro de tabelas)
# ---------------------------------------------------------------------------

def _bbox_overlap(bbox_a, bbox_b, tolerance: float = 2.0) -> bool:
    """
    Retorna True se os dois bboxes (x0,y0,x1,y1) se sobrepõem.
    Tolerance evita falsos positivos por bordas compartilhadas.
    """
    ax0, ay0, ax1, ay1 = bbox_a
    bx0, by0, bx1, by1 = bbox_b
    return (
        ax0 < bx1 - tolerance
        and ax1 > bx0 + tolerance
        and ay0 < by1 - tolerance
        and ay1 > by0 + tolerance
    )


def text_line_in_table_bbox(
    line_bbox: tuple,
    table_bboxes: List[tuple],
    tolerance: float = 2.0,
) -> bool:
    """
    Verifica se uma linha de texto está dentro de alguma bbox de tabela.

    Usado pelo pdf_parser para excluir linhas de texto que já foram
    capturadas como parte de uma tabela, evitando conteúdo duplicado.

    Parâmetros:
    - line_bbox   : (x0, y0, x1, y1) da linha de texto.
    - table_bboxes: lista de bboxes das tabelas detectadas na página.
    - tolerance   : margem de pixels para considerar sobreposição.

    Retorna True se a linha estiver dentro de alguma tabela.
    """
    for tbbox in table_bboxes:
        if _bbox_overlap(line_bbox, tbbox, tolerance):
            return True
    return False
