"""
Testes unitários do parser de XLSX — Passo 16.

Testa:
1. Parsing básico de planilha simples com cabeçalho e dados.
2. Propagação correta de valores em células mescladas (merged cells).
3. Combinação de cabeçalhos de múltiplas linhas.
4. Descarte de abas vazias ou ocultas.
5. Divisão de múltiplas tabelas separadas por linhas em branco na mesma aba.
6. Formatação limpa de dados (datas, números, floats, booleanos).
7. Integração com o Canonical Document Model e serialização JSON.
8. Integração com chunk_recursive: chunks com metadados de tabela e sumário.
9. Dispatcher parse_file() reconhecendo arquivos .xlsx.
10. Validação ponta a ponta contra as planilhas seed geradas no corpus.

Rodar com:
    python tests/test_xlsx_parser.py
    # ou
    pytest tests/test_xlsx_parser.py
"""

from __future__ import annotations

import datetime
import io
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, ".")

import pytest

from src.chunking.recursive import chunk_recursive
from src.parsing.canonical_model import Document, ElementType, Table
from src.parsing.simple_parsers import parse_file
from src.parsing.xlsx_parser import (
    _extract_table_from_block,
    _format_cell_value,
    _resolve_sheet_grid,
    parse_xlsx,
)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Helpers de criação de planilhas de teste (in-memory)
# ---------------------------------------------------------------------------

def create_basic_xlsx(tmp_path: Path) -> Path:
    """Cria uma planilha simples para testes."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Dados"

    ws.append(["Produto", "Preco", "Estoque"])
    ws.append(["Notebook", 4500.50, 12])
    ws.append(["Mouse", 120.00, 45])
    ws.append(["Teclado", 250.00, 30])

    file_path = tmp_path / "basico.xlsx"
    wb.save(str(file_path))
    return file_path


def create_merged_cells_xlsx(tmp_path: Path) -> Path:
    """Cria uma planilha com células mescladas e cabeçalho multilinhas."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Relatorio"

    # Linha 1: categorias mescladas
    ws.cell(row=1, column=1, value="Região")
    ws.cell(row=1, column=2, value="Vendas 2025")
    ws.merge_cells(start_row=1, start_column=2, end_row=1, end_column=3)

    # Linha 2: subcolunas
    ws.cell(row=2, column=1, value="Nome")
    ws.cell(row=2, column=2, value="Q1")
    ws.cell(row=2, column=3, value="Q2")

    # Linhas de dados
    ws.cell(row=3, column=1, value="Sudeste")
    ws.cell(row=3, column=2, value=150000)
    ws.cell(row=3, column=3, value=180000)

    ws.cell(row=4, column=1, value="Sul")
    ws.cell(row=4, column=2, value=95000)
    ws.cell(row=4, column=3, value=110000)

    file_path = tmp_path / "mesclado.xlsx"
    wb.save(str(file_path))
    return file_path


def create_multi_table_xlsx(tmp_path: Path) -> Path:
    """Cria uma planilha com 2 tabelas separadas por linhas vazias e uma aba vazia."""
    import openpyxl

    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "Tabelas_Multiplas"

    # Tabela 1
    ws1.append(["Parametro", "Valor"])
    ws1.append(["Batch Size", 32])
    ws1.append(["Learning Rate", 0.001])

    # Linhas em branco
    ws1.append([])
    ws1.append([])

    # Tabela 2
    ws1.append(["Metrica", "Score"])
    ws1.append(["Accuracy", 0.945])
    ws1.append(["F1-Score", 0.932])

    # Aba 2: vazia
    ws2 = wb.create_sheet(title="Aba_Vazia")

    file_path = tmp_path / "multi_tabela.xlsx"
    wb.save(str(file_path))
    return file_path


@pytest.fixture
def basic_xlsx_path(tmp_path: Path) -> Path:
    return create_basic_xlsx(tmp_path)


@pytest.fixture
def merged_cells_xlsx_path(tmp_path: Path) -> Path:
    return create_merged_cells_xlsx(tmp_path)


@pytest.fixture
def multi_table_xlsx_path(tmp_path: Path) -> Path:
    return create_multi_table_xlsx(tmp_path)


# ---------------------------------------------------------------------------
# 1. Testes de helpers e formatação
# ---------------------------------------------------------------------------

class TestXlsxFormatting:
    def test_format_cell_values(self):
        assert _format_cell_value(None) == ""
        assert _format_cell_value(True) == "Sim"
        assert _format_cell_value(False) == "Não"
        assert _format_cell_value(10.0) == "10"
        assert _format_cell_value(10.5) == "10.5"
        assert _format_cell_value(100) == "100"
        assert _format_cell_value(datetime.date(2026, 9, 2)) == "2026-09-02"
        assert _format_cell_value("  Texto com espaços  ") == "Texto com espaços"


# ---------------------------------------------------------------------------
# 2. Testes de parsing e resolução de mesclagem
# ---------------------------------------------------------------------------

class TestXlsxParser:
    def test_parse_basic_xlsx(self, basic_xlsx_path: Path):
        doc = parse_xlsx(basic_xlsx_path)

        assert isinstance(doc, Document)
        assert doc.source_uri == str(basic_xlsx_path)
        assert doc.content_hash
        assert doc.metadata["parser"] == "parse_xlsx"

        # Deve conter 1 Section (aba) e 1 Table
        sections = [e for e in doc.elements if e.element_type == ElementType.SECTION]
        tables = [e for e in doc.elements if e.element_type == ElementType.TABLE]

        assert len(sections) == 1
        assert "Aba: Dados" in sections[0].title

        assert len(tables) == 1
        table = tables[0]
        assert isinstance(table, Table)
        assert table.headers == ["Produto", "Preco", "Estoque"]
        assert len(table.rows) == 3
        assert table.rows[0] == ["Notebook", "4500.5", "12"]
        assert table.metadata["sheet_name"] == "Dados"
        assert table.metadata["row_count"] == 3
        assert table.metadata["col_count"] == 3

        # Conteúdo gerado deve ter Markdown e Sumário (Passo 15)
        assert "| Produto" in table.content
        assert "## Sumário" in table.content
        assert "Notebook" in table.content

    def test_merged_cells_and_multiline_headers(self, merged_cells_xlsx_path: Path):
        doc = parse_xlsx(merged_cells_xlsx_path)

        tables = [e for e in doc.elements if e.element_type == ElementType.TABLE]
        assert len(tables) == 1
        table = tables[0]

        # Cabeçalhos compostos esperados
        assert "Região - Nome" in table.headers or "Nome" in table.headers
        assert "Vendas 2025 - Q1" in table.headers
        assert "Vendas 2025 - Q2" in table.headers

        # Linhas de dados sem None/vazios no valor mesclado
        assert len(table.rows) == 2
        assert table.rows[0][0] == "Sudeste"
        assert table.rows[0][1] == "150000"
        assert table.rows[0][2] == "180000"

    def test_multi_table_and_empty_sheet(self, multi_table_xlsx_path: Path):
        doc = parse_xlsx(multi_table_xlsx_path)

        tables = [e for e in doc.elements if e.element_type == ElementType.TABLE]
        # Deve ter extraído 2 tabelas da primeira aba
        assert len(tables) == 2

        t1, t2 = tables[0], tables[1]
        assert t1.headers == ["Parametro", "Valor"]
        assert t1.rows[0] == ["Batch Size", "32"]

        assert t2.headers == ["Metrica", "Score"]
        assert t2.rows[0] == ["Accuracy", "0.945"]

        # A aba vazia "Aba_Vazia" NÃO deve gerar elementos de tabela
        sheet_names = [t.metadata["sheet_name"] for t in tables]
        assert "Aba_Vazia" not in sheet_names

    def test_dispatcher_integration(self, basic_xlsx_path: Path):
        """Valida que parse_file despacha corretamente para parse_xlsx."""
        doc = parse_file(basic_xlsx_path, tenant_id="tenant-alpha")
        assert doc is not None
        assert doc.tenant_id == "tenant-alpha"
        assert any(e.element_type == ElementType.TABLE for e in doc.elements)

    def test_chunking_integration(self, basic_xlsx_path: Path):
        """Valida que Document de XLSX gera chunks estruturados com metadados."""
        doc = parse_xlsx(basic_xlsx_path)
        chunks = chunk_recursive(doc)

        assert len(chunks) >= 1
        table_chunks = [c for c in chunks if c.metadata.get("table_row_count", 0) > 0]
        assert len(table_chunks) == 1

        tc = table_chunks[0]
        assert tc.metadata["table_headers"] == ["Produto", "Preco", "Estoque"]
        assert tc.metadata["table_row_count"] == 3
        assert tc.metadata["has_table_summary"] is True
        assert "| Notebook" in tc.content
        assert "## Sumário" in tc.content


# ---------------------------------------------------------------------------
# 3. Teste ponta a ponta com seed corpus
# ---------------------------------------------------------------------------

class TestSeedCorpusXlsx:
    def test_seed_corpus_files(self):
        from scripts.create_xlsx_seed import create_xlsx_files

        # Garante que os arquivos existam
        create_xlsx_files()

        f1 = Path("data/seed_corpus/14_metricas_modelos_llm.xlsx")
        f2 = Path("data/seed_corpus/15_inventario_hardware_rag.xlsx")

        assert f1.exists(), f"{f1} deve existir"
        assert f2.exists(), f"{f2} deve existir"

        # 1. Valida 14_metricas_modelos_llm.xlsx
        doc1 = parse_file(f1)
        assert doc1.title is not None
        tables1 = [e for e in doc1.elements if e.element_type == ElementType.TABLE]
        assert len(tables1) == 2, "Deve conter 2 tabelas (Benchmarks e Custos)"

        # Spot-check dados dos modelos
        all_content1 = "\n".join(t.content for t in tables1)
        assert "Claude 3.5 Sonnet" in all_content1
        assert "GPT-4o" in all_content1
        assert "DeepSeek V2.5" in all_content1

        # 2. Valida 15_inventario_hardware_rag.xlsx
        doc2 = parse_file(f2)
        tables2 = [e for e in doc2.elements if e.element_type == ElementType.TABLE]
        assert len(tables2) == 2, "Deve conter 2 tabelas (GPU e Storage, ignorando Rascunho_Vazio)"

        all_content2 = "\n".join(t.content for t in tables2)
        assert "NVIDIA A100 SXM4" in all_content2
        assert "Qdrant Vetorial" in all_content2


# ---------------------------------------------------------------------------
# Runner autônomo (executável sem pytest)
# ---------------------------------------------------------------------------

def main() -> None:
    print("Testando parser de XLSX (Passo 16)...\n")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        test_format = TestXlsxFormatting()
        test_format.test_format_cell_values()
        print("  [OK] TestXlsxFormatting.test_format_cell_values")

        test_parser = TestXlsxParser()

        f_basic = create_basic_xlsx(tmp_path)
        test_parser.test_parse_basic_xlsx(f_basic)
        print("  [OK] TestXlsxParser.test_parse_basic_xlsx")

        f_merged = create_merged_cells_xlsx(tmp_path)
        test_parser.test_merged_cells_and_multiline_headers(f_merged)
        print("  [OK] TestXlsxParser.test_merged_cells_and_multiline_headers")

        f_multi = create_multi_table_xlsx(tmp_path)
        test_parser.test_multi_table_and_empty_sheet(f_multi)
        print("  [OK] TestXlsxParser.test_multi_table_and_empty_sheet")

        test_parser.test_dispatcher_integration(f_basic)
        print("  [OK] TestXlsxParser.test_dispatcher_integration")

        test_parser.test_chunking_integration(f_basic)
        print("  [OK] TestXlsxParser.test_chunking_integration")

        test_seed = TestSeedCorpusXlsx()
        test_seed.test_seed_corpus_files()
        print("  [OK] TestSeedCorpusXlsx.test_seed_corpus_files")

    print("\nTodos os 7 testes do Passo 16 passaram com sucesso!")
    sys.exit(0)


if __name__ == "__main__":
    main()
