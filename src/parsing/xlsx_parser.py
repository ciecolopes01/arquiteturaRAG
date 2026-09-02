"""
Parser de XLSX — Passo 16.

Lê planilhas nativas do Excel (.xlsx) e produz um Document no Canonical
Document Model com objetos Table estruturados.

Decisões de design:
- Usa `openpyxl` com `data_only=True` para obter os valores calculados das
  fórmulas em vez do texto de fórmulas como `=SUM(...)`.
- Trata células mescladas (`merged_cells`): propaga o valor da célula top-left
  para toda a região mesclada em memória, garantindo integridade de colunas e linhas.
- Suporta cabeçalhos de múltiplas linhas: quando identificado cabeçalho hierárquico
  (ex: categoria na linha 1 e subcategoria na linha 2), combina os nomes em
  "Categoria - Subcategoria".
- Descarta abas vazias ou ocultas (`sheet_state != 'visible'`).
- Divide tabelas múltiplas na mesma aba quando separadas por linhas em branco.
- Reaproveita `build_table_content` de `src/parsing/table_extractor.py` (Passo 15)
  para gerar representação dual (Markdown estruturado + sumário determinístico em pt-BR).
- Gera metadados completos em cada Table (`sheet_name`, `table_index`, `row_count`,
  `col_count`, `section_title`).
- content_hash: SHA-256 dos bytes brutos do arquivo .xlsx.
"""

from __future__ import annotations

import datetime
import hashlib
from pathlib import Path
from typing import Any, List, Optional, Tuple

from src.parsing.canonical_model import Document, Section, Table
from src.parsing.table_extractor import build_table_content


def _sha256(content: bytes) -> str:
    """Calcula hash SHA-256 dos bytes do arquivo."""
    return hashlib.sha256(content).hexdigest()


def _format_cell_value(val: Any) -> str:
    """
    Formata o valor de uma célula para representação textual limpa.
    - None -> ''
    - float com valor inteiro -> '10' em vez de '10.0'
    - float com casas decimais -> string numérica limpa
    - datetime / date -> formato ISO 'YYYY-MM-DD'
    - boolean / int -> string direta
    """
    if val is None:
        return ""
    if isinstance(val, bool):
        return "Sim" if val else "Não"
    if isinstance(val, (datetime.datetime, datetime.date)):
        if isinstance(val, datetime.datetime) and (val.hour or val.minute or val.second):
            return val.strftime("%Y-%m-%d %H:%M:%S")
        return val.strftime("%Y-%m-%d")
    if isinstance(val, float):
        if val.is_integer():
            return str(int(val))
        return f"{val:.4f}".rstrip("0").rstrip(".")
    return str(val).strip()


def _resolve_sheet_grid(sheet) -> List[List[str]]:
    """
    Carrega todos os dados da aba para uma matriz 2D em memória, resolvendo
    automaticamente as células mescladas ao propagar o valor da célula top-left
    para todas as células do intervalo.
    """
    max_row = sheet.max_row or 0
    max_col = sheet.max_column or 0

    if max_row == 0 or max_col == 0:
        return []

    # Inicializa matriz com None
    grid: List[List[Any]] = [[None for _ in range(max_col)] for _ in range(max_row)]

    # Preenche com os valores das células
    for r in range(1, max_row + 1):
        for c in range(1, max_col + 1):
            cell = sheet.cell(row=r, column=c)
            grid[r - 1][c - 1] = cell.value

    # Propaga valores de células mescladas
    for rng in sheet.merged_cells.ranges:
        min_r, min_c = rng.min_row, rng.min_col
        max_r, max_c = rng.max_row, rng.max_col
        top_left_val = sheet.cell(row=min_r, column=min_c).value

        for r in range(min_r, min_r + (max_r - min_r + 1)):
            for c in range(min_c, min_c + (max_c - min_c + 1)):
                if 1 <= r <= max_row and 1 <= c <= max_col:
                    grid[r - 1][c - 1] = top_left_val

    # Converte tudo para string limpa
    str_grid: List[List[str]] = []
    for row in grid:
        str_grid.append([_format_cell_value(val) for val in row])

    return str_grid


def _is_row_empty(row: List[str]) -> bool:
    """Verifica se uma linha é totalmente vazia."""
    return not any(cell.strip() for cell in row)


def _split_grid_into_blocks(grid: List[List[str]]) -> List[List[List[str]]]:
    """
    Divide a grade em blocos de dados quando há linhas totalmente em branco
    separando tabelas diferentes na mesma aba.
    """
    blocks: List[List[List[str]]] = []
    current_block: List[List[str]] = []

    for row in grid:
        if _is_row_empty(row):
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            current_block.append(row)

    if current_block:
        blocks.append(current_block)

    return blocks


def _is_numeric_str(val: str) -> bool:
    """Verifica se a string representa um número ou valor monetário."""
    cleaned = val.replace("$", "").replace("R$", "").replace("%", "").replace(",", ".").strip()
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def _extract_table_from_block(
    block: List[List[str]],
) -> Optional[Tuple[List[str], List[List[str]], Optional[str]]]:
    """
    Processa um bloco contíguo de linhas, identificando título/caption,
    cabeçalhos (simples ou multi-linha) e linhas de dados.

    Retorna (headers, rows, caption) ou None se não houver dados válidos.
    """
    if not block:
        return None

    caption: Optional[str] = None
    start_idx = 0

    # 1. Se a primeira linha tiver apenas 1 célula preenchida e a segunda linha tiver várias,
    # a primeira linha é o título/caption da tabela.
    first_row_non_empty = [c for c in block[0] if c]
    if len(first_row_non_empty) == 1 and len(block) > 1:
        second_row_non_empty = [c for c in block[1] if c]
        if len(second_row_non_empty) > 1:
            caption = first_row_non_empty[0]
            start_idx = 1

    remaining = block[start_idx:]
    if not remaining:
        return None

    # Corta colunas totalmente vazias à direita
    max_filled_col = 0
    for row in remaining:
        for col_idx, cell in enumerate(row):
            if cell.strip():
                max_filled_col = max(max_filled_col, col_idx)

    trimmed_block = [row[: max_filled_col + 1] for row in remaining]
    if not trimmed_block or max_filled_col < 0:
        return None

    # 2. Identificação de cabeçalho multi-linha vs simples
    # Se a linha 0 e a linha 1 são ambas não-numéricas, e a linha 2 existe e tem números,
    # combinamos as linhas 0 e 1 como cabeçalho composto.
    has_multiline_header = False
    if len(trimmed_block) >= 3:
        row0_num = sum(1 for c in trimmed_block[0] if _is_numeric_str(c))
        row1_num = sum(1 for c in trimmed_block[1] if _is_numeric_str(c))
        row2_num = sum(1 for c in trimmed_block[2] if _is_numeric_str(c))

        if row0_num == 0 and row1_num == 0 and row2_num > 0:
            has_multiline_header = True

    if has_multiline_header:
        raw_headers: List[str] = []
        for c0, c1 in zip(trimmed_block[0], trimmed_block[1]):
            c0_clean = c0.strip()
            c1_clean = c1.strip()
            if c0_clean and c1_clean and c0_clean != c1_clean:
                raw_headers.append(f"{c0_clean} - {c1_clean}")
            elif c1_clean:
                raw_headers.append(c1_clean)
            elif c0_clean:
                raw_headers.append(c0_clean)
            else:
                raw_headers.append("")
        data_rows = trimmed_block[2:]
    else:
        raw_headers = [c.strip() for c in trimmed_block[0]]
        data_rows = trimmed_block[1:]

    # Normaliza cabeçalhos vazios e deduplica
    headers: List[str] = []
    seen_headers = set()
    for col_idx, h in enumerate(raw_headers):
        clean_h = h if h else f"Coluna_{col_idx + 1}"
        orig_h = clean_h
        counter = 2
        while clean_h in seen_headers:
            clean_h = f"{orig_h}_{counter}"
            counter += 1
        seen_headers.add(clean_h)
        headers.append(clean_h)

    # Filtra linhas de dados totalmente vazias
    rows = [r for r in data_rows if any(c.strip() for c in r)]

    if not rows:
        return None

    return headers, rows, caption


def parse_xlsx(
    path: str | Path,
    tenant_id: Optional[str] = None,
    source_uri: Optional[str] = None,
) -> Document:
    """
    Lê um arquivo .xlsx e produz um Document no Canonical Document Model.

    Cada aba visível e não-vazia do Excel é convertida em um ou mais objetos
    `Table` estruturados, enriquecidos com Markdown e sumário em linguagem natural.

    Parâmetros:
    - path: caminho local para o arquivo .xlsx.
    - tenant_id: identificador de tenant (opcional).
    - source_uri: URI de origem (default: str(path)).

    Retorna:
    - Document estruturado com elementos Section e Table.
    """
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl não instalado. Execute: pip install openpyxl")

    path = Path(path)
    raw_bytes = path.read_bytes()
    source = source_uri or str(path)

    wb = openpyxl.load_workbook(str(path), data_only=True)

    core_title = ""
    if wb.properties and wb.properties.title:
        core_title = str(wb.properties.title).strip()

    doc_title = core_title or path.stem.replace("_", " ").title()

    doc = Document(
        source_uri=source,
        title=doc_title,
        tenant_id=tenant_id,
        content_hash=_sha256(raw_bytes),
        parser_version="xlsx-openpyxl-0.1",
        language="pt",
        metadata={
            "original_filename": path.name,
            "parser": "parse_xlsx",
            "sheet_count": len(wb.worksheets),
        },
    )

    table_counter = 0

    for sheet in wb.worksheets:
        # Descarta abas ocultas
        if sheet.sheet_state != "visible":
            continue

        sheet_title = str(sheet.title).strip()
        grid = _resolve_sheet_grid(sheet)

        if not grid:
            continue

        # Divide por blocos de tabelas
        blocks = _split_grid_into_blocks(grid)
        if not blocks:
            continue

        sheet_section = Section(
            title=f"Aba: {sheet_title}",
            level=1,
            content="",
            metadata={"sheet_name": sheet_title},
        )
        sheet_has_tables = False

        for block in blocks:
            extracted = _extract_table_from_block(block)
            if not extracted:
                continue

            headers, rows, caption = extracted
            table_counter += 1

            table_caption = caption or f"Tabela da aba {sheet_title}"
            section_title = f"Aba: {sheet_title}"

            tbl = Table(
                headers=headers,
                rows=rows,
                caption=table_caption,
                metadata={
                    "sheet_name": sheet_title,
                    "table_index": table_counter,
                    "row_count": len(rows),
                    "col_count": len(headers),
                    "section_title": section_title,
                },
            )

            # Gera content com Markdown + Sumário em linguagem natural (Passo 15)
            tbl.content = build_table_content(tbl, section_title=section_title)

            if not sheet_has_tables:
                doc.elements.append(sheet_section)
                sheet_has_tables = True

            sheet_section.children.append(tbl.id)
            doc.elements.append(tbl)

    return doc
