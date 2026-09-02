"""
Script: create_xlsx_seed.py — Passo 16.

Gera duas planilhas Excel (.xlsx) sintéticas e representativas em data/seed_corpus/:
  1. 14_metricas_modelos_llm.xlsx
     - Aba 1: Benchmarks_Modelos (cabeçalho de múltiplas linhas / células mescladas)
     - Aba 2: Custos_Operacionais (projeções financeiras)
  2. 15_inventario_hardware_rag.xlsx
     - Aba 1: Servidores_GPU (recursos computacionais)
     - Aba 2: Storage_Indices (capacidade e latência de storage)
     - Aba 3: Rascunho_Vazio (aba sem dados para validar descarte)

Uso:
    python scripts/create_xlsx_seed.py
"""

from pathlib import Path


def create_xlsx_files() -> None:
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    except ImportError:
        raise ImportError("openpyxl não instalado. Execute: pip install openpyxl")

    corpus_dir = Path("data/seed_corpus")
    corpus_dir.mkdir(parents=True, exist_ok=True)

    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    sub_fill = PatternFill(start_color="2F5597", end_color="2F5597", fill_type="solid")
    align_center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    align_left = Alignment(horizontal="left", vertical="center")

    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    # ---------------------------------------------------------------------------
    # 1. 14_metricas_modelos_llm.xlsx
    # ---------------------------------------------------------------------------
    wb1 = openpyxl.Workbook()
    wb1.properties.title = "Benchmarks e Custos de Modelos de Linguagem para RAG"

    # Aba 1: Benchmarks_Modelos (Cabeçalho multilinhas mesclado)
    ws1_1 = wb1.active
    ws1_1.title = "Benchmarks_Modelos"

    # Linha 1 do cabeçalho
    ws1_1.cell(row=1, column=1, value="Identificação")
    ws1_1.cell(row=1, column=2, value="Benchmarks")
    ws1_1.cell(row=1, column=4, value="Especificações Técnicas")
    ws1_1.cell(row=1, column=6, value="Preços por 1M Tokens (USD)")

    # Mesclagens de categorias na linha 1
    ws1_1.merge_cells(start_row=1, start_column=2, end_row=1, end_column=3)
    ws1_1.merge_cells(start_row=1, start_column=4, end_row=1, end_column=5)
    ws1_1.merge_cells(start_row=1, start_column=6, end_row=1, end_column=7)

    # Linha 2 do cabeçalho
    subheaders = [
        "Nome do Modelo",
        "MMLU Score (%)",
        "GSM8K Score (%)",
        "Context Window (k)",
        "Tipo Arquitetura",
        "Input USD",
        "Output USD",
    ]
    for col_idx, sh in enumerate(subheaders, start=1):
        ws1_1.cell(row=2, column=col_idx, value=sh)

    # Estiliza cabeçalhos
    for r in (1, 2):
        for c in range(1, 8):
            cell = ws1_1.cell(row=r, column=c)
            cell.font = header_font
            cell.fill = header_fill if r == 1 else sub_fill
            cell.alignment = align_center

    # Dados da aba 1
    models_data = [
        ["Claude 3.5 Sonnet", 88.7, 96.4, 200, "MoE/Denso", 3.00, 15.00],
        ["GPT-4o", 88.2, 95.8, 128, "MoE", 2.50, 10.00],
        ["Llama 3.1 70B", 86.0, 95.1, 128, "Denso", 0.80, 0.80],
        ["DeepSeek V2.5", 87.2, 94.8, 128, "MoE", 0.14, 0.28],
        ["Gemini 1.5 Pro", 85.9, 91.7, 2048, "MoE", 3.50, 10.50],
    ]

    for row_idx, row_values in enumerate(models_data, start=3):
        for col_idx, val in enumerate(row_values, start=1):
            cell = ws1_1.cell(row=row_idx, column=col_idx, value=val)
            cell.border = thin_border
            cell.alignment = align_left if col_idx == 1 or col_idx == 5 else align_center

    # Aba 2: Custos_Operacionais
    ws1_2 = wb1.create_sheet(title="Custos_Operacionais")
    cost_headers = [
        "Faixa de Volume (Queries/Mês)",
        "Modelo Principal",
        "Custo LLM Estimado (USD)",
        "Custo Embedding (USD)",
        "Custo Total Mensal (USD)",
    ]
    for col_idx, h in enumerate(cost_headers, start=1):
        cell = ws1_2.cell(row=1, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = align_center

    cost_data = [
        ["10000", "Llama 3.1 70B", 16.00, 0.50, 16.50],
        ["50000", "DeepSeek V2.5", 32.00, 2.50, 34.50],
        ["200000", "GPT-4o", 375.00, 10.00, 385.00],
        ["1000000", "Claude 3.5 Sonnet", 2700.00, 50.00, 2750.00],
    ]

    for row_idx, row_values in enumerate(cost_data, start=2):
        for col_idx, val in enumerate(row_values, start=1):
            cell = ws1_2.cell(row=row_idx, column=col_idx, value=val)
            cell.border = thin_border
            cell.alignment = align_left if col_idx <= 2 else align_center

    file1_path = corpus_dir / "14_metricas_modelos_llm.xlsx"
    wb1.save(str(file1_path))
    print(f"[OK] Criado: {file1_path}")

    # ---------------------------------------------------------------------------
    # 2. 15_inventario_hardware_rag.xlsx
    # ---------------------------------------------------------------------------
    wb2 = openpyxl.Workbook()
    wb2.properties.title = "Inventário de Hardware e Infraestrutura do Pipeline RAG"

    # Aba 1: Servidores_GPU
    ws2_1 = wb2.active
    ws2_1.title = "Servidores_GPU"

    gpu_headers = [
        "Hostname",
        "Modelo da GPU",
        "Quantidade GPUs",
        "VRAM Total (GB)",
        "Consumo TDP (Watts)",
        "Throughput Estimado (QPS)",
    ]
    for col_idx, h in enumerate(gpu_headers, start=1):
        cell = ws2_1.cell(row=1, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = align_center

    gpu_data = [
        ["srv-rag-gpu-01", "NVIDIA A100 SXM4", 8, 640, 3200, 450],
        ["srv-rag-gpu-02", "NVIDIA H100 PCIe", 4, 320, 1400, 680],
        ["srv-rag-gpu-03", "NVIDIA L40S", 4, 192, 1400, 290],
        ["srv-rag-gpu-04", "NVIDIA RTX 4090", 2, 48, 900, 110],
    ]

    for row_idx, row_values in enumerate(gpu_data, start=2):
        for col_idx, val in enumerate(row_values, start=1):
            cell = ws2_1.cell(row=row_idx, column=col_idx, value=val)
            cell.border = thin_border
            cell.alignment = align_left if col_idx <= 2 else align_center

    # Aba 2: Storage_Indices
    ws2_2 = wb2.create_sheet(title="Storage_Indices")
    storage_headers = [
        "Serviço Destino",
        "Tipo de Mídia",
        "Capacidade Útil (TB)",
        "IOPS Mínimo",
        "Latência Média Alvo (ms)",
    ]
    for col_idx, h in enumerate(storage_headers, start=1):
        cell = ws2_2.cell(row=1, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = align_center

    storage_data = [
        ["Qdrant Vetorial", "NVMe PCIe Gen4", 2.0, 50000, 1.5],
        ["PostgreSQL BM25", "SSD SAS Enterprise", 4.0, 20000, 3.0],
        ["MinIO Object Store", "HDD SATA Pool", 20.0, 5000, 15.0],
    ]

    for row_idx, row_values in enumerate(storage_data, start=2):
        for col_idx, val in enumerate(row_values, start=1):
            cell = ws2_2.cell(row=row_idx, column=col_idx, value=val)
            cell.border = thin_border
            cell.alignment = align_left if col_idx <= 2 else align_center

    # Aba 3: Rascunho_Vazio (intencionalmente vazia para testar descarte pelo parser)
    ws2_3 = wb2.create_sheet(title="Rascunho_Vazio")
    # Deixar sem dados

    file2_path = corpus_dir / "15_inventario_hardware_rag.xlsx"
    wb2.save(str(file2_path))
    print(f"[OK] Criado: {file2_path}")


if __name__ == "__main__":
    create_xlsx_files()
