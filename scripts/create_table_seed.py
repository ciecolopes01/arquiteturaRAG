"""
Script: create_table_seed.py — Passo 15.

Gera o arquivo data/seed_corpus/13_tabelas_financeiras.pdf com tabelas
reais usando reportlab. O PDF contém:
  1. Tabela de metricas de performance RAG (Recall@5, MRR, latencia)
     por combinacao de estrategia de chunking e retrieval.
  2. Tabela comparativa de modelos de embedding (dimensao, velocidade,
     Recall@5 no corpus seed, custo por 1M tokens).
  3. Tabela de evolucao do Recall@5 por passo do projeto.

Nota: texto em ASCII para compatibilidade com Helvetica/PyMuPDF sem
problemas de encoding na extracao por find_tables().

Uso:
    python scripts/create_table_seed.py
"""

from pathlib import Path


def create_pdf() -> Path:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        raise ImportError("reportlab nao instalado. Execute: pip install reportlab")

    output_path = Path(__file__).parent.parent / "data" / "seed_corpus" / "13_tabelas_financeiras.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    elements = []

    # -----------------------------------------------------------------------
    # Cabecalho do documento
    # -----------------------------------------------------------------------
    elements.append(Paragraph("Analise Comparativa de Estrategias RAG", styles["Title"]))
    elements.append(Spacer(1, 0.4 * cm))
    elements.append(Paragraph(
        "Este documento apresenta metricas de avaliacao offline de diferentes "
        "configuracoes do pipeline RAG, incluindo estrategias de chunking, "
        "modelos de embedding e configuracoes de retrieval.",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.6 * cm))

    # -----------------------------------------------------------------------
    # Secao 1: Metricas de performance por estrategia
    # -----------------------------------------------------------------------
    elements.append(Paragraph("1. Metricas de Performance por Estrategia de Retrieval", styles["Heading1"]))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        "A tabela abaixo apresenta os resultados de avaliacao offline (golden dataset, "
        "33 perguntas) para diferentes combinacoes de estrategia de retrieval. "
        "Latencia medida com modelo BGE-M3 em CPU.",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.4 * cm))

    # Dados da tabela 1
    perf_headers = ["Estrategia", "Recall@5", "MRR", "Latencia (ms)", "Custo/Query (USD)"]
    perf_rows = [
        ["Dense (BGE-M3)", "0.8788", "0.7121", "520", "0.00000"],
        ["BM25 (Postgres)", "0.7576", "0.6364", "18", "0.00000"],
        ["Hibrido RRF (sem reranker)", "0.9394", "0.7576", "540", "0.00000"],
        ["Hibrido RRF (fast path)", "0.9697", "0.7864", "560", "0.00000"],
        ["Hibrido RRF (full path + reranker)", "0.9697", "0.8182", "1320", "0.00000"],
    ]

    table1_data = [perf_headers] + perf_rows
    table1 = Table(table1_data, repeatRows=1)
    table1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(table1)
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        "Nota: O full path com reranker tem MRR 4% superior ao fast path, "
        "mas latencia 2.4x maior. O fast path e o padrao para queries simples.",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.8 * cm))

    # -----------------------------------------------------------------------
    # Secao 2: Comparativo de modelos de embedding
    # -----------------------------------------------------------------------
    elements.append(Paragraph("2. Comparativo de Modelos de Embedding", styles["Heading1"]))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        "Comparacao de modelos de embedding candidatos para uso no pipeline RAG. "
        "Recall@5 medido no corpus seed de 12 documentos, 33 perguntas. "
        "Velocidade medida em CPU (Intel i7, single thread).",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.4 * cm))

    # Dados da tabela 2
    emb_headers = ["Modelo", "Dimensao", "Recall@5", "Velocidade (ms/doc)", "Parametros", "Custo (USD/1M tokens)"]
    emb_rows = [
        ["BAAI/bge-m3", "1024", "0.9697", "1565", "568M", "0.00 (local)"],
        ["text-embedding-3-small", "1536", "0.9394", "45", "n/a", "0.020"],
        ["text-embedding-3-large", "3072", "0.9697", "62", "n/a", "0.130"],
        ["BAAI/bge-small-en-v1.5", "384", "0.8485", "210", "33M", "0.00 (local)"],
        ["sentence-transformers/all-MiniLM-L6-v2", "384", "0.7879", "180", "22M", "0.00 (local)"],
    ]

    table2_data = [emb_headers] + emb_rows
    table2 = Table(table2_data, repeatRows=1)
    table2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eaf2ff")]),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(table2)
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        "O modelo BGE-M3 mantem Recall@5 de 0.9697 com custo zero (local), "
        "equivalente ao text-embedding-3-large da OpenAI a $0.13/1M tokens. "
        "Para migracao de modelo, manter collection paralela por 30 dias.",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.8 * cm))

    # -----------------------------------------------------------------------
    # Secao 3: Evolucao do Recall@5 ao longo dos passos do projeto
    # -----------------------------------------------------------------------
    elements.append(Paragraph("3. Evolucao do Recall@5 ao Longo dos Passos do Projeto", styles["Heading1"]))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        "Esta tabela mostra como o Recall@5 evoluiu a cada passo de melhoria "
        "implementado no pipeline RAG, permitindo rastrear o impacto de cada "
        "mudanca na qualidade de retrieval.",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 0.4 * cm))

    evo_headers = ["Passo", "Descricao", "Recall@5", "MRR", "Delta Recall"]
    evo_rows = [
        ["Passo 6", "Retrieval Hibrido RRF (baseline)", "0.9394", "0.7576", "--"],
        ["Passo 7", "Golden dataset (33 perguntas)", "0.9394", "0.7576", "+0.00"],
        ["Passo 9", "Reranker BGE adicionado", "0.9697", "0.8182", "+0.03"],
        ["Passo 13", "Fast/Full path routing", "0.9697", "0.7864", "0.00"],
        ["Passo 14", "Observabilidade (sem mudanca de pipeline)", "0.9697", "0.7864", "0.00"],
    ]

    table3_data = [evo_headers] + evo_rows
    table3 = Table(table3_data, repeatRows=1)
    table3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#145a32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 1), (1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eafaf1")]),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        # Destaca linha do Passo 9 (maior ganho)
        ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#d5f5e3")),
    ]))
    elements.append(table3)

    # Monta o PDF
    doc.build(elements)
    return output_path


if __name__ == "__main__":
    output = create_pdf()
    print(f"PDF gerado: {output}")
    print(f"Tamanho: {output.stat().st_size:,} bytes")
