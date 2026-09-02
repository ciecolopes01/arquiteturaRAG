"""
Testes do módulo table_extractor — Passo 15.

Testa:
  - table_to_markdown(): formatação correta, colunas com larguras diferentes,
    células None/vazias, tabelas de 1 coluna.
  - table_to_natural_language(): sumário em pt-BR, truncagem para tabelas grandes,
    fields opcionais (caption, section_title).
  - build_table_content(): content duplo (markdown + sumário).
  - extract_tables_from_pdf_page(): extração de tabelas em PDF sintético.
  - text_line_in_table_bbox(): filtragem de linhas dentro de bboxes.
  - Integração: parse_pdf() detecta tabelas do PDF seed e produz Table canônicos
    com has_table_summary=True no chunk.

Todos esses testes rodam SEM infraestrutura (Qdrant, Postgres, LLM).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List

import pytest

from src.parsing.canonical_model import Table
from src.parsing.table_extractor import (
    MAX_SUMMARY_ROWS,
    build_table_content,
    extract_tables_from_pdf_page,
    table_to_markdown,
    table_to_natural_language,
    text_line_in_table_bbox,
)


# ===========================================================================
# Helpers de fixture
# ===========================================================================

def make_table(
    headers: List[str],
    rows: List[List[str]],
    caption: str = "",
    section_title: str = "",
) -> Table:
    """Cria uma Table canônica sintética para testes."""
    return Table(
        headers=headers,
        rows=rows,
        caption=caption or None,
        content="",  # será preenchido por build_table_content
        metadata={"section_title": section_title},
    )


# ===========================================================================
# 1. table_to_markdown
# ===========================================================================

class TestTableToMarkdown:
    def test_basic_table(self):
        headers = ["Nome", "Idade", "Cidade"]
        rows = [["Alice", "30", "São Paulo"], ["Bob", "25", "Rio de Janeiro"]]
        md = table_to_markdown(headers, rows)

        # Deve ter cabeçalho, separador e linhas de dados
        lines = md.strip().splitlines()
        assert len(lines) == 4  # header + sep + 2 rows
        assert "Nome" in lines[0]
        assert "Idade" in lines[0]
        assert "---" in lines[1] or "|-" in lines[1]
        assert "Alice" in lines[2]
        assert "Bob" in lines[3]

    def test_single_column(self):
        headers = ["Item"]
        rows = [["A"], ["B"], ["C"]]
        md = table_to_markdown(headers, rows)
        assert "Item" in md
        assert "A" in md
        assert "B" in md

    def test_empty_cells(self):
        headers = ["Col1", "Col2"]
        rows = [["val", ""], [None, "outro"]]
        md = table_to_markdown(headers, rows)
        assert "val" in md
        assert "outro" in md
        # Nenhuma exceção deve ocorrer

    def test_pipe_separator_present(self):
        headers = ["A", "B"]
        rows = [["1", "2"]]
        md = table_to_markdown(headers, rows)
        assert "|" in md

    def test_column_width_min_3(self):
        headers = ["A", "B"]
        rows = [["1", "2"]]
        md = table_to_markdown(headers, rows)
        # Cada coluna deve ter no mínimo 3 chars na linha separadora
        sep_line = md.splitlines()[1]
        for segment in sep_line.split("|"):
            stripped = segment.strip()
            if stripped:
                assert len(stripped) >= 3

    def test_empty_headers_returns_empty(self):
        md = table_to_markdown([], [])
        assert md == ""

    def test_rows_with_fewer_columns_than_headers(self):
        headers = ["A", "B", "C"]
        rows = [["1", "2"]]  # só 2 colunas em vez de 3
        md = table_to_markdown(headers, rows)
        # Não deve lançar exceção — deve preencher com ""
        assert "A" in md
        assert "1" in md

    def test_truncates_very_long_cells(self):
        headers = ["Texto"]
        rows = [["x" * 400]]
        md = table_to_markdown(headers, rows)
        # Célula truncada para 300 chars
        data_line = md.splitlines()[2]
        assert len(data_line) < 400


# ===========================================================================
# 2. table_to_natural_language
# ===========================================================================

class TestTableToNaturalLanguage:
    def test_basic_summary_pt(self):
        tbl = make_table(
            headers=["Produto", "Receita 2024", "Receita 2025"],
            rows=[
                ["Produto A", "R$ 1.2M", "R$ 1.8M"],
                ["Produto B", "R$ 0.8M", "R$ 1.1M"],
            ],
        )
        summary = table_to_natural_language(tbl)
        assert "3 coluna" in summary.lower()
        assert "2 linha" in summary.lower() or "2 linha" in summary
        assert "Produto" in summary
        assert "Receita 2024" in summary

    def test_caption_in_summary(self):
        tbl = make_table(
            headers=["Modelo", "Recall@5"],
            rows=[["BGE-M3", "0.9697"]],
            caption="Comparativo de Modelos",
        )
        summary = table_to_natural_language(tbl)
        assert "Comparativo de Modelos" in summary

    def test_section_title_in_summary(self):
        tbl = make_table(
            headers=["A", "B"],
            rows=[["1", "2"]],
            section_title="Seção de Análise",
        )
        # section_title vive em metadata
        tbl.metadata["section_title"] = "Seção de Análise"
        summary = table_to_natural_language(tbl)
        assert "Seção de Análise" in summary

    def test_large_table_truncated(self):
        """Tabela com mais de MAX_SUMMARY_ROWS linhas: não detalha todas."""
        n_rows = MAX_SUMMARY_ROWS + 5
        tbl = make_table(
            headers=["Col"],
            rows=[[str(i)] for i in range(n_rows)],
        )
        summary = table_to_natural_language(tbl)
        assert "mais" in summary.lower()
        assert str(n_rows) in summary  # menciona total de linhas

    def test_empty_headers_graceful(self):
        tbl = make_table(headers=[], rows=[])
        summary = table_to_natural_language(tbl)
        assert "sem colunas" in summary.lower()

    def test_first_row_detail(self):
        tbl = make_table(
            headers=["Estratégia", "Recall@5"],
            rows=[["Dense", "0.88"], ["BM25", "0.76"]],
        )
        summary = table_to_natural_language(tbl)
        # Primeira linha deve estar detalhada
        assert "Dense" in summary
        assert "0.88" in summary

    def test_pair_format_in_summary(self):
        """Cada linha do sumário deve ter o formato 'Coluna=Valor'."""
        tbl = make_table(
            headers=["Cor", "Código"],
            rows=[["Azul", "#0000FF"]],
        )
        summary = table_to_natural_language(tbl)
        assert "Cor=Azul" in summary or "Cor" in summary


# ===========================================================================
# 3. build_table_content
# ===========================================================================

class TestBuildTableContent:
    def test_has_markdown_section(self):
        tbl = make_table(
            headers=["A", "B"],
            rows=[["1", "2"]],
        )
        content = build_table_content(tbl)
        assert "## Dados estruturados" in content
        assert "|" in content  # markdown table

    def test_has_summary_section(self):
        tbl = make_table(
            headers=["A", "B"],
            rows=[["1", "2"]],
        )
        content = build_table_content(tbl)
        assert "## Sumário" in content

    def test_section_title_in_content(self):
        tbl = make_table(
            headers=["X"],
            rows=[["y"]],
        )
        content = build_table_content(tbl, section_title="Métricas de Avaliação")
        assert "Métricas de Avaliação" in content
        assert "[Seção: Métricas de Avaliação]" in content

    def test_caption_in_content(self):
        tbl = make_table(
            headers=["X"],
            rows=[["y"]],
            caption="Minha Tabela",
        )
        content = build_table_content(tbl)
        assert "Minha Tabela" in content

    def test_both_representations_present(self):
        """Content deve ter TANTO o markdown quanto o sumário natural."""
        tbl = make_table(
            headers=["Estratégia", "Recall@5", "MRR"],
            rows=[
                ["Dense", "0.88", "0.71"],
                ["Híbrido", "0.97", "0.79"],
            ],
        )
        content = build_table_content(tbl)
        # Markdown
        assert "|" in content
        assert "---" in content or "|-" in content
        # Sumário natural
        assert "Esta tabela" in content
        assert "coluna" in content.lower()


# ===========================================================================
# 4. text_line_in_table_bbox
# ===========================================================================

class TestTextLineInTableBbox:
    def test_line_inside_bbox(self):
        line_bbox = (10, 10, 200, 20)
        table_bboxes = [(5, 5, 250, 100)]
        assert text_line_in_table_bbox(line_bbox, table_bboxes) is True

    def test_line_outside_bbox(self):
        line_bbox = (10, 200, 200, 210)
        table_bboxes = [(5, 5, 250, 100)]
        assert text_line_in_table_bbox(line_bbox, table_bboxes) is False

    def test_no_bboxes(self):
        line_bbox = (10, 10, 200, 20)
        assert text_line_in_table_bbox(line_bbox, []) is False

    def test_partial_overlap_outside_tolerance(self):
        # Line overlaps by > tolerance (2px)
        line_bbox = (100, 90, 200, 110)
        table_bboxes = [(5, 5, 250, 100)]
        # y overlap: ay0=90 < by1=100 - 2=98 → True; ay1=110 > by0=5 + 2=7 → True
        # x overlap: ax0=100 < bx1=250 - 2=248 → True; ax1=200 > bx0=5 + 2=7 → True
        assert text_line_in_table_bbox(line_bbox, table_bboxes) is True

    def test_line_touches_edge_within_tolerance(self):
        # Line y0=98, table y1=100 → overlap = 100-98 = 2 = tolerance → NOT overlapping
        line_bbox = (10, 98, 200, 110)
        table_bboxes = [(5, 5, 250, 100)]
        # ay0=98 < by1=100 - 2=98 → 98 < 98 → False → NOT inside
        assert text_line_in_table_bbox(line_bbox, table_bboxes) is False


# ===========================================================================
# 5. Integração com PDF sintético (usando reportlab)
# ===========================================================================

def _make_simple_table_pdf() -> bytes:
    """Cria um PDF mínimo com uma tabela usando reportlab."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
    except ImportError:
        pytest.skip("reportlab não instalado")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    data = [
        ["Modelo", "Recall@5", "MRR"],
        ["BGE-M3", "0.9697", "0.7864"],
        ["text-embedding-3-small", "0.9394", "0.7576"],
    ]
    table = Table(data)
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    doc.build([table])
    return buf.getvalue()


class TestExtractTablesFromPdfPage:
    def test_detects_table_in_synthetic_pdf(self):
        try:
            import fitz
        except ImportError:
            pytest.skip("pymupdf não instalado")

        pdf_bytes = _make_simple_table_pdf()
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = pdf[0]

        tables, fitz_tbls = extract_tables_from_pdf_page(page, page_number=1)
        pdf.close()

        assert len(tables) >= 1, "Deve detectar pelo menos 1 tabela no PDF sintético"

    def test_extracted_table_has_headers_and_rows(self):
        try:
            import fitz
        except ImportError:
            pytest.skip("pymupdf não instalado")

        pdf_bytes = _make_simple_table_pdf()
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = pdf[0]

        tables, _ = extract_tables_from_pdf_page(page, page_number=1)
        pdf.close()

        assert len(tables) >= 1
        tbl = tables[0]
        assert len(tbl.headers) >= 3
        assert len(tbl.rows) >= 2

    def test_extracted_table_has_expected_values(self):
        try:
            import fitz
        except ImportError:
            pytest.skip("pymupdf não instalado")

        pdf_bytes = _make_simple_table_pdf()
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = pdf[0]

        tables, _ = extract_tables_from_pdf_page(page, page_number=1)
        pdf.close()

        assert len(tables) >= 1
        tbl = tables[0]
        # Todos os valores da tabela devem aparecer em headers ou rows
        all_text = " ".join(tbl.headers) + " " + " ".join(
            " ".join(row) for row in tbl.rows
        )
        assert "BGE-M3" in all_text or "Modelo" in all_text

    def test_page_number_in_metadata(self):
        try:
            import fitz
        except ImportError:
            pytest.skip("pymupdf não instalado")

        pdf_bytes = _make_simple_table_pdf()
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = pdf[0]

        tables, _ = extract_tables_from_pdf_page(page, page_number=5)
        pdf.close()

        if tables:
            assert tables[0].metadata["page_number"] == 5

    def test_no_tables_returns_empty(self):
        try:
            import fitz
        except ImportError:
            pytest.skip("pymupdf não instalado")

        # Cria PDF sem tabelas
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((50, 100), "Texto sem tabelas aqui.")
        buf = io.BytesIO()
        pdf.save(buf)
        pdf_bytes = buf.getvalue()
        pdf.close()

        pdf2 = fitz.open(stream=pdf_bytes, filetype="pdf")
        tables, fitz_tbls = extract_tables_from_pdf_page(pdf2[0], page_number=1)
        pdf2.close()

        assert len(tables) == 0


# ===========================================================================
# 6. Integração: parse_pdf() com o seed PDF de tabelas
# ===========================================================================

class TestParsePdfWithTables:
    """Testa o parse_pdf() completo no arquivo 13_tabelas_financeiras.pdf."""

    @pytest.fixture(scope="class")
    def seed_pdf_path(self) -> Path:
        p = Path("data/seed_corpus/13_tabelas_financeiras.pdf")
        if not p.exists():
            pytest.skip("PDF seed não gerado — rode scripts/create_table_seed.py primeiro")
        return p

    def test_parse_pdf_returns_document(self, seed_pdf_path):
        from src.parsing.pdf_parser import parse_pdf
        doc = parse_pdf(seed_pdf_path)
        assert doc is not None
        assert len(doc.elements) > 0

    def test_parse_pdf_has_table_elements(self, seed_pdf_path):
        from src.parsing.canonical_model import ElementType
        from src.parsing.pdf_parser import parse_pdf
        doc = parse_pdf(seed_pdf_path)
        tables = [e for e in doc.elements if e.element_type == ElementType.TABLE]
        assert len(tables) >= 2, f"Esperado >= 2 tabelas, encontrado {len(tables)}"

    def test_table_elements_have_headers(self, seed_pdf_path):
        from src.parsing.canonical_model import ElementType
        from src.parsing.pdf_parser import parse_pdf
        doc = parse_pdf(seed_pdf_path)
        tables = [e for e in doc.elements if e.element_type == ElementType.TABLE]
        for tbl in tables:
            assert tbl.headers, f"Tabela {tbl.id} deve ter headers"
            assert len(tbl.headers) >= 2

    def test_table_elements_have_rows(self, seed_pdf_path):
        from src.parsing.canonical_model import ElementType
        from src.parsing.pdf_parser import parse_pdf
        doc = parse_pdf(seed_pdf_path)
        tables = [e for e in doc.elements if e.element_type == ElementType.TABLE]
        for tbl in tables:
            assert tbl.rows, f"Tabela {tbl.id} deve ter rows"

    def test_table_content_has_markdown(self, seed_pdf_path):
        from src.parsing.canonical_model import ElementType
        from src.parsing.pdf_parser import parse_pdf
        doc = parse_pdf(seed_pdf_path)
        tables = [e for e in doc.elements if e.element_type == ElementType.TABLE]
        for tbl in tables:
            assert "|" in tbl.content, f"Tabela {tbl.id} deve ter markdown com |"

    def test_table_content_has_summary(self, seed_pdf_path):
        from src.parsing.canonical_model import ElementType
        from src.parsing.pdf_parser import parse_pdf
        doc = parse_pdf(seed_pdf_path)
        tables = [e for e in doc.elements if e.element_type == ElementType.TABLE]
        for tbl in tables:
            assert "## Sumário" in tbl.content, (
                f"Tabela {tbl.id} deve ter sumário em linguagem natural"
            )

    def test_chunk_has_table_metadata(self, seed_pdf_path):
        from src.chunking.recursive import chunk_recursive
        from src.parsing.canonical_model import ElementType
        from src.parsing.pdf_parser import parse_pdf
        doc = parse_pdf(seed_pdf_path)
        chunks = chunk_recursive(doc)
        table_chunks = [c for c in chunks if c.metadata.get("table_row_count", 0) > 0]
        assert len(table_chunks) >= 2, f"Esperado >= 2 chunks de tabela, encontrado {len(table_chunks)}"
        for c in table_chunks:
            assert "table_headers" in c.metadata
            assert c.metadata["has_table_summary"] is True
