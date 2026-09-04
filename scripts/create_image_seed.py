"""
Script: create_image_seed.py — Passo 19.

Gera 4 documentos no seed_corpus contendo imagens e diagramas reais embutidos:
1. data/seed_corpus/16_arquitetura_multimodal_rag.pdf (PDF com 2 páginas e 2 diagramas)
2. data/seed_corpus/17_metricas_desempenho_visao.pdf (PDF com 1 página e gráfico técnico)
3. data/seed_corpus/18_fluxo_processamento_imagens.docx (DOCX com 1 infográfico)
4. data/seed_corpus/19_benchmark_extracao_visual.pdf (PDF de 2 páginas com comparativo visual na página 2)

As imagens são desenhadas em alta resolução com Pillow (linhas, caixas, setas, e texto
nítido) para que o Tesseract OCR consiga extrair o texto perfeitamente.
"""

from __future__ import annotations

import io
from pathlib import Path

import docx
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
)
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "seed_corpus"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _get_font(size: int = 18):
    """Tenta carregar fonte padrão do sistema ou fonte bitmap do Pillow."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Geradores de Imagens Sintéticas com Alto Contraste para OCR
# ---------------------------------------------------------------------------

def create_diagram_multimodal_rag() -> bytes:
    """Diagrama da Arquitetura Multimodal RAG."""
    w, h = 900, 420
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    d = ImageDraw.Draw(img)

    title_font = _get_font(22)
    body_font = _get_font(16)
    sub_font = _get_font(13)

    # Borda externa
    d.rectangle([(4, 4), (w - 4, h - 4)], outline=(40, 60, 100), width=3)

    # Cabeçalho
    d.rectangle([(4, 4), (w - 4, 50)], fill=(40, 60, 100))
    d.text((20, 14), "DIAGRAMA: PIPELINE MULTIMODAL RAG - V1.1", fill=(255, 255, 255), font=title_font)

    # Caixas dos componentes
    boxes = [
        (40, 90, 220, 220, "1. INGESTAO", "Extracao de PDF e DOCX\nDetecao de Figuras e Bboxes\nFiltro de Icones Minimos", (230, 240, 255)),
        (260, 90, 440, 220, "2. EXTRACAO DUAL", "Tesseract 5.4 (OCR Local)\nClaude 3.5 Sonnet (Visao)\nObject Storage MinIO", (255, 245, 230)),
        (480, 90, 660, 220, "3. VECTOR ENGINE", "Modelo BGE-M3 Multimodal\nChunk Unificado de Imagem\nIndice Qdrant Collection", (235, 255, 235)),
        (700, 90, 860, 220, "4. RETRIEVAL", "Busca Hibrida RRF\nReranking Cross-Encoder\nRecall Multimodal: 96.2%", (255, 235, 245)),
    ]

    for x1, y1, x2, y2, title, desc, bg in boxes:
        d.rectangle([(x1, y1), (x2, y2)], fill=bg, outline=(50, 50, 50), width=2)
        d.text((x1 + 10, y1 + 10), title, fill=(20, 20, 20), font=body_font)
        d.text((x1 + 10, y1 + 45), desc, fill=(40, 40, 40), font=sub_font)

    # Setas conectoras
    for start_x in (220, 440, 660):
        d.line([(start_x, 155), (start_x + 40, 155)], fill=(40, 60, 100), width=4)
        d.polygon([(start_x + 40, 155), (start_x + 30, 148), (start_x + 30, 162)], fill=(40, 60, 100))

    # Rodapé informativo
    d.rectangle([(40, 260), (860, 390)], fill=(245, 245, 245), outline=(180, 180, 180), width=1)
    d.text((55, 275), "METRICAS CHAVE DO COMPONENTE DE IMAGEM:", fill=(30, 30, 30), font=body_font)
    d.text((55, 310), "- OCR Local Tesseract: latencia media de 180ms por imagem sem custo de API", fill=(50, 50, 50), font=sub_font)
    d.text((55, 335), "- Claude Vision: descricao contextual semantica para diagramas e fluxos", fill=(50, 50, 50), font=sub_font)
    d.text((55, 360), "- Schema Canonical Model: Image com ocr_text, visual_description e storage_uri", fill=(50, 50, 50), font=sub_font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_diagram_indexacao_fluxo() -> bytes:
    """Diagrama de fluxo de indexação de imagens."""
    w, h = 850, 360
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    d = ImageDraw.Draw(img)

    title_font = _get_font(20)
    body_font = _get_font(15)
    sub_font = _get_font(13)

    d.rectangle([(4, 4), (w - 4, h - 4)], outline=(30, 120, 80), width=3)
    d.rectangle([(4, 4), (w - 4, 45)], fill=(30, 120, 80))
    d.text((20, 12), "FLUXO DE INDEXACAO DE IMAGENS E BUSCA HIBRIDA", fill=(255, 255, 255), font=title_font)

    steps = [
        (30, 75, 250, 180, "A. Imagem Extraida", "Objeto Image Canonico\nMetadados: pagina e dimensao", (240, 250, 245)),
        (310, 75, 540, 180, "B. Chunk Unificado", "Texto OCR + Descricao Visual\n[Texto na Imagem] + [Descricao]", (245, 245, 255)),
        (600, 75, 820, 180, "C. Indices de Busca", "BM25 no Postgres\nVetores BGE-M3 no Qdrant", (255, 250, 240)),
    ]

    for x1, y1, x2, y2, title, desc, bg in steps:
        d.rectangle([(x1, y1), (x2, y2)], fill=bg, outline=(40, 40, 40), width=2)
        d.text((x1 + 10, y1 + 10), title, fill=(20, 20, 20), font=body_font)
        d.text((x1 + 10, y1 + 45), desc, fill=(40, 40, 40), font=sub_font)

    for start_x in (250, 540):
        d.line([(start_x, 125), (start_x + 60, 125)], fill=(30, 120, 80), width=4)
        d.polygon([(start_x + 60, 125), (start_x + 50, 118), (start_x + 50, 132)], fill=(30, 120, 80))

    d.rectangle([(30, 220), (820, 330)], fill=(250, 250, 250), outline=(200, 200, 200), width=1)
    d.text((50, 235), "REGRA DE RECUPERACAO:", fill=(30, 30, 30), font=body_font)
    d.text((50, 265), "Perguntas pontuais sobre texto na figura sao atendidas pelo canal BM25.", fill=(60, 60, 60), font=sub_font)
    d.text((50, 290), "Perguntas conceituais sobre o grafico sao atendidas pelo vetor semantico BGE-M3.", fill=(60, 60, 60), font=sub_font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_chart_ocr_accuracy() -> bytes:
    """Gráfico de Acurácia de OCR vs Resolução DPI."""
    w, h = 850, 420
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    d = ImageDraw.Draw(img)

    title_font = _get_font(20)
    body_font = _get_font(15)
    sub_font = _get_font(13)

    d.rectangle([(4, 4), (w - 4, h - 4)], outline=(120, 40, 40), width=3)
    d.rectangle([(4, 4), (w - 4, 45)], fill=(120, 40, 40))
    d.text((20, 12), "GRAFICO: ACURACIA DO OCR TESSERACT VS RESOLUCAO (DPI)", fill=(255, 255, 255), font=title_font)

    # Eixos
    origin_x, origin_y = 120, 310
    end_x, end_y = 780, 80

    d.line([(origin_x, origin_y), (end_x, origin_y)], fill=(50, 50, 50), width=2)
    d.line([(origin_x, origin_y), (origin_x, end_y)], fill=(50, 50, 50), width=2)

    # Barras de dados
    bars = [
        ("72 DPI", 80, 180, "52.4% Acuracia (Baixa precisao)", (220, 100, 100)),
        ("150 DPI", 180, 320, "81.5% Acuracia (Aceitavel)", (240, 180, 90)),
        ("300 DPI", 270, 480, "98.4% Acuracia (Ponto Otimo)", (90, 190, 110)),
        ("600 DPI", 275, 640, "98.9% Acuracia (Latencia Alta)", (100, 140, 220)),
    ]

    for label, bar_h, center_x, caption, color in bars:
        top_y = origin_y - bar_h
        d.rectangle([(center_x - 40, top_y), (center_x + 40, origin_y)], fill=color, outline=(40, 40, 40), width=2)
        d.text((center_x - 30, origin_y + 10), label, fill=(20, 20, 20), font=body_font)
        d.text((center_x - 65, top_y - 25), caption.split()[0], fill=(20, 20, 20), font=body_font)

    d.text((origin_x + 20, end_y + 10), "Conclusao Técnica: 300 DPI oferece precisao de 98.4% com latencia equilibrada.", fill=(40, 40, 40), font=sub_font)
    d.text((origin_x + 20, end_y + 35), "Em 72 DPI ha perda critica de caracteres tecnicos em tabelas e graficos.", fill=(40, 40, 40), font=sub_font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_infographic_comparativo_visao() -> bytes:
    """Infográfico Comparativo: OCR Tradicional vs Visão Multimodal."""
    w, h = 850, 400
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    d = ImageDraw.Draw(img)

    title_font = _get_font(20)
    body_font = _get_font(15)
    sub_font = _get_font(13)

    d.rectangle([(4, 4), (w - 4, h - 4)], outline=(70, 40, 110), width=3)
    d.rectangle([(4, 4), (w - 4, 45)], fill=(70, 40, 110))
    d.text((20, 12), "INFOGRAFICO: OCR TRADICIONAL VS VISAO MULTIMODAL", fill=(255, 255, 255), font=title_font)

    # Painel Esquerdo
    d.rectangle([(40, 70), (410, 310)], fill=(250, 245, 255), outline=(120, 100, 160), width=2)
    d.text((60, 85), "CANAL 1: OCR TRADICIONAL (TESSERACT)", fill=(60, 30, 100), font=body_font)
    d.text((60, 125), "• Foco exclusivo em caracteres de texto visiveis", fill=(40, 40, 40), font=sub_font)
    d.text((60, 155), "• Ignora relacoes espaciais e formas geometricas", fill=(40, 40, 40), font=sub_font)
    d.text((60, 185), "• Custo zero de API e execucao local rapida", fill=(40, 40, 40), font=sub_font)
    d.text((60, 215), "• Indispensavel para recuperar siglas, codigos e labels", fill=(40, 40, 40), font=sub_font)
    d.text((60, 255), "Taxa de Sucesso em Textos Tecnicos: 97.1%", fill=(80, 40, 120), font=body_font)

    # Painel Direito
    d.rectangle([(440, 70), (810, 310)], fill=(245, 250, 255), outline=(90, 120, 180), width=2)
    d.text((460, 85), "CANAL 2: VISAO MULTIMODAL (CLAUDE)", fill=(30, 60, 120), font=body_font)
    d.text((460, 125), "• Compreende significado semantico de diagramas", fill=(40, 40, 40), font=sub_font)
    d.text((460, 155), "• Interpreta eixos, tendencias e barras de graficos", fill=(40, 40, 40), font=sub_font)
    d.text((460, 185), "• Gera sumario estruturado para busca vetorial", fill=(40, 40, 40), font=sub_font)
    d.text((460, 215), "• Permite responder 'o que este grafico conclui?'", fill=(40, 40, 40), font=sub_font)
    d.text((460, 255), "Recall Semantico Isolado: 94.5%", fill=(30, 60, 140), font=body_font)

    # Conclusão Unificada
    d.rectangle([(40, 330), (810, 380)], fill=(240, 255, 245), outline=(60, 150, 90), width=1)
    d.text((60, 345), "RESULTADO COMBINADO (PASSO 19): Recall@5 de 0.982 com fusao OCR + Visao", fill=(20, 100, 40), font=body_font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Geração dos Documentos Seed
# ---------------------------------------------------------------------------

def generate_doc16_pdf() -> Path:
    """Gera 16_arquitetura_multimodal_rag.pdf com 2 páginas e 2 diagramas."""
    target = OUTPUT_DIR / "16_arquitetura_multimodal_rag.pdf"
    doc = SimpleDocTemplate(
        str(target),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    styles = getSampleStyleSheet()

    story = []

    # Página 1
    story.append(Paragraph("Arquitetura do Pipeline RAG Multimodal", styles["Title"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "A integracao de componentes visuais transforma o RAG corporativo ao permitir "
        "a indexacao e busca de graficos, diagramas de blocos e esquemas de engenharia. "
        "O diagrama a seguir ilustra a topologia completa do pipeline:",
        styles["Normal"],
    ))
    story.append(Spacer(1, 0.4 * cm))

    raw1 = create_diagram_multimodal_rag()
    img1_stream = io.BytesIO(raw1)
    story.append(RLImage(img1_stream, width=17 * cm, height=8 * cm))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Figura 1: Topologia do pipeline multimodal integrando Tesseract OCR, Claude Vision e Qdrant.",
        styles["Italic"],
    ))

    story.append(PageBreak())

    # Página 2
    story.append(Paragraph("Indexacao e Estrategia de Busca de Figuras", styles["Heading1"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Na segunda fase do processamento, as imagens extraidas recebem metadados de pagina "
        "e secao de origem. O fluxo de transformacao e vetorizacao e demonstrado a seguir:",
        styles["Normal"],
    ))
    story.append(Spacer(1, 0.4 * cm))

    raw2 = create_diagram_indexacao_fluxo()
    img2_stream = io.BytesIO(raw2)
    story.append(RLImage(img2_stream, width=17 * cm, height=7.2 * cm))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Figura 2: Fluxo de indexacao com enriquecimento semantico e busca hibrida.",
        styles["Italic"],
    ))

    doc.build(story)
    print(f"[OK] Gerado: {target}")
    return target


def generate_doc17_pdf() -> Path:
    """Gera 17_metricas_desempenho_visao.pdf com 1 página e gráfico técnico."""
    target = OUTPUT_DIR / "17_metricas_desempenho_visao.pdf"
    doc = SimpleDocTemplate(
        str(target),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    styles = getSampleStyleSheet()

    story = []
    story.append(Paragraph("Relatorio de Desempenho e Acuracia de Visao", styles["Title"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Este relatorio consolida os testes de estresse de resolucao realizados com "
        "o motor Tesseract OCR v5.4 acoplado ao pipeline de ingestao. Abaixo apresentamos "
        "a curva de acuracia por resolucao grafica:",
        styles["Normal"],
    ))
    story.append(Spacer(1, 0.4 * cm))

    raw = create_chart_ocr_accuracy()
    img_stream = io.BytesIO(raw)
    story.append(RLImage(img_stream, width=17 * cm, height=8.4 * cm))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Figura 1: Acuracia do OCR em funcao da resolucao em DPI para documentos digitalizados.",
        styles["Italic"],
    ))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Conclusao: A taxa de 300 DPI e recomendada formalmente pela equipe de arquitetura "
        "para garantir 98.4% de acuracia sem sobrecarregar o tempo de inferencia.",
        styles["Normal"],
    ))

    doc.build(story)
    print(f"[OK] Gerado: {target}")
    return target


def generate_doc18_docx() -> Path:
    """Gera 18_fluxo_processamento_imagens.docx com texto e imagem embutida."""
    target = OUTPUT_DIR / "18_fluxo_processamento_imagens.docx"
    doc = docx.Document()

    doc.add_heading("Guia de Processamento de Imagens e Documentos Multimodais", level=0)

    doc.add_heading("1. Visao Geral da Extracao de Imagens", level=1)
    doc.add_paragraph(
        "Arquivos em formato Word (DOCX) frequentemente trazem diagramas de arquitetura "
        "e fluxogramas incorporados diretamente no corpo do texto. O pipeline do Passo 19 "
        "extrai essas midias e executa o reconhecimento textual e conceitual."
    )

    doc.add_heading("2. Detalhamento do Pipeline de Visao", level=1)
    doc.add_paragraph(
        "O diagrama a seguir sintetiza os componentes responsaveis pelo processamento "
        "de imagens dentro da plataforma:"
    )

    raw_img = create_diagram_multimodal_rag()
    img_stream = io.BytesIO(raw_img)
    doc.add_picture(img_stream, width=docx.shared.Inches(6.0))

    doc.add_paragraph("Figura 1: Pipeline de Visao e OCR para Documentos Corporativos.")

    doc.add_heading("3. Principais Diretrizes", level=2)
    doc.add_paragraph("• O Tesseract 5.4 e executado com os idiomas portugues e ingles.")
    doc.add_paragraph("• Claude 3.5 Sonnet gera sumarios com analise semantica visual.")
    doc.add_paragraph("• Os binarios de imagem sao preservados no MinIO em rag-documents.")

    doc.save(str(target))
    print(f"[OK] Gerado: {target}")
    return target


def generate_doc19_pdf() -> Path:
    """Gera 19_benchmark_extracao_visual.pdf com 2 páginas e comparativo na página 2."""
    target = OUTPUT_DIR / "19_benchmark_extracao_visual.pdf"
    doc = SimpleDocTemplate(
        str(target),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    styles = getSampleStyleSheet()

    story = []

    # Página 1
    story.append(Paragraph("Benchmark Comparativo de Extracao Visual", styles["Title"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "A avaliacao de efetividade de extracao em documentos com conteudos ricos "
        "exige a comparacao rigorosa entre abordagens baseadas exclusivamente em OCR "
        "versus sistemas que combinam OCR com modelos de linguagem e visao (VLM).",
        styles["Normal"],
    ))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Na pagina seguinte deste relatorio, apresentamos o quadro comparativo "
        "que fundamentou a decisao do Passo 19 de adotar ambos os canais de forma unificada.",
        styles["Normal"],
    ))

    story.append(PageBreak())

    # Página 2
    story.append(Paragraph("Quadro Comparativo: OCR vs Visao Multimodal", styles["Heading1"]))
    story.append(Spacer(1, 0.3 * cm))

    raw = create_infographic_comparativo_visao()
    img_stream = io.BytesIO(raw)
    story.append(RLImage(img_stream, width=17 * cm, height=8.0 * cm))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Figura 1: Quadro comparativo de capacidades tecnicas entre OCR puro e Visao Multimodal.",
        styles["Italic"],
    ))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "Nota: O resultado combinado alcancou Recall@5 de 0.982 na bateria de testes do corpus seed.",
        styles["Normal"],
    ))

    doc.build(story)
    print(f"[OK] Gerado: {target}")
    return target


def main() -> None:
    print("Gerando documentos seed com imagens e graficos (Passo 19)...")
    generate_doc16_pdf()
    generate_doc17_pdf()
    generate_doc18_docx()
    generate_doc19_pdf()
    print("Todos os 4 documentos seed multimodais foram gerados com sucesso!")


if __name__ == "__main__":
    main()
