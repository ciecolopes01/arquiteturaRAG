"""
Gerador de GIF Animado do Fluxo RAG Multimodal (v5 Réplica Fiel do HTML).
Inclui:
- Header com cronômetro em tempo real
- Barra de timeline de progresso por etapa
- Canvas com simulação visual de partículas, trilho de dados e 5 hubs (1..5)
- Grid dos 5 Cards com estado ativo em destaque e barras pulsantes
- Painel de Detalhes inferior com descrição completa e badges/tags arredondadas
"""

import math
import os
from PIL import Image, ImageDraw, ImageFont

# ── Dimensões Canvas ──────────────────────────────────────────────────────────
WIDTH  = 1350
HEIGHT = 800
FPS    = 12                        # 12 fps
DURATION_SEC  = 5
TOTAL_FRAMES  = FPS * DURATION_SEC # 60 frames

# ── Paleta de Cores (Dark Cyberpunk) ─────────────────────────────────────────
BG_COLOR    = (9,   12,  18)
PANEL_BG    = (20,  25,  34)
CARD_BG     = (18,  23,  32)
CARD_ACTIVE = (26,  34,  48)
BORDER_COL  = (42,  49,  60)
TEXT_WHITE  = (240, 246, 252)
TEXT_MUTED  = (130, 140, 155)
CYAN        = (56,  189, 248)
ORANGE      = (251, 146, 60)
PURPLE      = (168, 85,  247)
GREEN       = (52,  211, 153)
YELLOW      = (250, 204, 21)

# ── Dados dos Estágios ───────────────────────────────────────────────────────
STAGES = [
    {
        "title":    "1. Parse Multimodal & Modelo Canônico",
        "short":    "Parse",
        "time":     "0.00s – 1.00s",
        "color":    CYAN,
        "subtitle": "Unificação no CanonicalDocument (SHA-256 + ACL)",
        "desc":     "PDF: layout por tamanho de fonte (H1/H2), tabelas via find_tables(), imagens embutidas com OCR+Vision. XLSX: células mescladas, cabeçalhos multi-linha, múltiplas tabelas. DOCX/TXT/HTML: seções, parágrafos e tabelas. Áudio: Whisper com segmentação por pausas. Vídeo: PySceneDetect, keyframe (OpenCV), Whisper word-level, OCR (Tesseract), Vision (Claude 3.5).",
        "card_items": [
            "• PDF Layout-aware (PyMuPDF)",
            "• XLSX Tabelas (openpyxl)",
            "• DOCX / TXT / HTML Estruturado",
            "• IMG OCR + Vision",
            "• Audio Whisper (faster-whisper)",
            "• Video Keyframes + Whisper + OCR",
        ],
        "tags": [
            "PDF Layout+Img+Tab", "XLSX Merged+Multi-aba", "DOCX / TXT / HTML",
            "OCR Tesseract", "Whisper faster-whisper", "PySceneDetect Cenas",
            "Claude Vision", "MinIO Keyframes", "CanonicalDocument"
        ],
    },
    {
        "title":    "2. Chunking Engine Estratégico",
        "short":    "Chunking",
        "time":     "1.00s – 2.00s",
        "color":    ORANGE,
        "subtitle": "Divisão Hierárquica Estratégica",
        "desc":     "Parent-Child hierárquico — pais de ~1200 tokens preservam contexto amplo, filhos de ~300 tokens com overlap de 40 tokens são indexados para busca precisa. Chunks de imagem, vídeo e áudio recebem tratamento específico preservando metadados de timestamps e storage_uri.",
        "card_items": [
            "• Parent-Child 1200 / 300 tk",
            "• Recursivo UUIDv5 det.",
            "• Semântico Cosseno",
            "• Overlap 40 – 64 tokens",
            "• Multimodal Sync chunks",
        ],
        "tags": [
            "Parent-Child 1200/300tk", "UUIDv5 Determinístico", "Semantic Cosine Split",
            "Overlap 40-64 tokens", "Multimodal Sync Chunks", "parent_content Metadado"
        ],
    },
    {
        "title":    "3. Ingestão Segura & Governança (ACL)",
        "short":    "Ingestão",
        "time":     "2.00s – 3.00s",
        "color":    GREEN,
        "subtitle": "Segurança Fail-Closed Granular",
        "desc":     "Tagging estrito de tenant_id e principals de ACL em cada chunk antes da indexação. Arquitetura fail-closed: chunk sem ACL é recusado. Reindexação incremental via document_registry (Postgres): calcula SHA-256 do arquivo e reindexa só o diff.",
        "card_items": [
            "• Tenant ID Tagging isolado",
            "• ACL Pré-filtro Fail-Closed",
            "• SHA-256 Content Hash",
            "• Doc Registry Postgres",
            "• Δ-Reindex Incremental",
        ],
        "tags": [
            "Tenant ID Isolation", "ACL Granular Fail-Closed", "SHA-256 Content Hash",
            "Document Registry PG", "Δ-Reindex Incremental", "UUID5 doc_id Estável"
        ],
    },
    {
        "title":    "4. Transformação & Indexação Tri-Canal",
        "short":    "Transform.",
        "time":     "3.00s – 4.00s",
        "color":    PURPLE,
        "subtitle": "3 Índices em Paralelo (Tri-Canal)",
        "desc":     "Geração em paralelo de três representações: (1) Vetor Denso BGE-M3 (1024 dims) indexado no Qdrant; (2) Vetor Esparso BM25/tsvector no PostgreSQL; (3) Grafo de conhecimento Neo4j / NetworkX com entidades e relações extraídas por NER + Regex semânticos. Dual-Write atômico garante consistência.",
        "card_items": [
            "• BGE-M3 Qdrant Dense",
            "• BM25 Postgres tsvector",
            "• Graph RAG Neo4j/NetworkX",
            "• NER Extração Regex",
            "• Cache Dual-Write Atômico",
        ],
        "tags": [
            "Qdrant BGE-M3 Dense", "PostgreSQL BM25 tsvector", "Neo4j / NetworkX Graph",
            "NER + Relações Regex", "Embedding Cache", "Dual-Write Atômico"
        ],
    },
    {
        "title":    "5. Busca Híbrida, RRF Ponderado & Rerank",
        "short":    "Busca RRF",
        "time":     "4.00s – 5.00s",
        "color":    YELLOW,
        "subtitle": "RRF Ponderado + Reranking Cross-Encoder",
        "desc":     "Query Router classifica a intenção em 5 categorias determinísticas. HyDE e Query Decomposition expandem a query. Busca simultânea em 3 canais com pré-filtro ACL obrigatório. Fusão via RRF Ponderado (k=60). Reranking com BGE-Reranker-v2-m3 e validação Self-RAG.",
        "card_items": [
            "• Query Router 5 intenções",
            "• HyDE + Decomposição",
            "• RRF Dense+BM25+Graph",
            "• BGE-Reranker-v2-m3",
            "• Self-RAG Quality Gate",
        ],
        "tags": [
            "Query Router 5 Intenções", "HyDE Hipotético", "Query Decomposition",
            "Dense + BM25 + Graph RRF", "ACL Pré-Filtrada", "RRF k=60 Ponderado",
            "BGE-Reranker-v2-m3", "Self-RAG Evaluator"
        ],
    },
]

# ── Fontes ───────────────────────────────────────────────────────────────────
def try_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf", "calibri.ttf"]
    for name in names:
        path = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", name)
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()

FONT_TITLE = try_font(20, bold=True)
FONT_H2    = try_font(15, bold=True)
FONT_REG   = try_font(13)
FONT_SM    = try_font(11)
FONT_MONO  = try_font(10)

def text_w(draw: ImageDraw.Draw, text: str, font) -> int:
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0]

def draw_text_lines(draw, x, y, text, font, fill, max_w, line_h):
    words = text.split()
    line  = ""
    for word in words:
        candidate = (line + " " + word).strip()
        if text_w(draw, candidate, font) <= max_w:
            line = candidate
        else:
            if line:
                draw.text((x, y), line, fill=fill, font=font)
                y += line_h
            line = word
    if line:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_h
    return y

# ── Partículas Persistentes para a Animação ────────────────────────────────────
PARTICLES = []
for i in range(35):
    PARTICLES.append({
        "x": (i * 38) % 1250 + 40,
        "y": 115 + (i * 17) % 100,
        "speed": 1.5 + (i % 3) * 0.8,
        "size": 2 + (i % 3),
        "color_idx": i % 5
    })

# ── Componentes de Desenho ───────────────────────────────────────────────────

def draw_header(draw: ImageDraw.Draw, current_time: float, active_idx: int) -> None:
    draw.rectangle([(0, 0), (WIDTH, 52)], fill=(16, 20, 28))
    draw.text((20, 14), "ARQUITETURA RAG MULTIMODAL — FLUXO COMPLETO (5s)", fill=TEXT_WHITE, font=FONT_TITLE)
    
    stage_label = f"ETAPA {active_idx + 1}/5: {STAGES[active_idx]['short'].upper()}"
    sw = text_w(draw, stage_label, FONT_H2)
    draw.text((WIDTH - sw - 170, 17), stage_label, fill=STAGES[active_idx]["color"], font=FONT_H2)
    
    time_str = f"{current_time:04.2f}s / 5.00s"
    draw.text((WIDTH - 150, 17), time_str, fill=YELLOW, font=FONT_H2)
    draw.line([(0, 52), (WIDTH, 52)], fill=BORDER_COL, width=1)


def draw_timeline(img: Image.Image, draw: ImageDraw.Draw, progress: float, active_idx: int) -> None:
    y0, h = 62, 10
    start_x, end_x = 34, WIDTH - 34
    total_w = end_x - start_x
    
    # Fundo do trilho
    draw.rounded_rectangle([(start_x, y0), (end_x, y0 + h)], radius=5, fill=(30, 41, 59))
    
    # Progresso com Gradiente 3 Cores (Cyan -> Purple -> Amber) idêntico ao HTML Tailwind
    fill_w = max(10, int(total_w * progress))
    for x_i in range(fill_w):
        curr_x = start_x + x_i
        t_grad = x_i / float(total_w)
        if t_grad < 0.5:
            ratio = t_grad / 0.5
            r = int(6 + (168 - 6) * ratio)
            g = int(182 + (85 - 182) * ratio)
            b = int(212 + (247 - 212) * ratio)
        else:
            ratio = (t_grad - 0.5) / 0.5
            r = int(168 + (251 - 168) * ratio)
            g = int(85 + (191 - 85) * ratio)
            b = int(247 + (36 - 247) * ratio)
        draw.line([(curr_x, y0), (curr_x, y0 + h)], fill=(r, g, b))
    
    # Rótulos dos tempos nas etapas
    labels = ["0s: Parse", "1s: Chunking", "2s: Ingestão/ACL", "3s: Transformação", "4s-5s: Busca Híbrida"]
    step_w = total_w / 5
    for i in range(5):
        cx = start_x + i * step_w + step_w / 2
        col = STAGES[i]["color"] if i == active_idx else (TEXT_WHITE if i < active_idx else TEXT_MUTED)
        draw.text((cx - text_w(draw, labels[i], FONT_SM) / 2, y0 + 14), labels[i], fill=col, font=FONT_SM)


def draw_canvas_simulation(draw: ImageDraw.Draw, current_time: float, active_idx: int, frame_idx: int) -> None:
    box_x0, box_y0 = 34, 96
    box_x1, box_y1 = WIDTH - 34, 236
    
    # Fundo do container do canvas
    draw.rounded_rectangle([(box_x0, box_y0), (box_x1, box_y1)], radius=10, fill=(11, 15, 23), outline=BORDER_COL, width=1)
    
    # Etiqueta de aviso superior
    draw.rounded_rectangle([(box_x0 + 12, box_y0 + 10), (box_x0 + 280, box_y0 + 28)], radius=4, fill=(20, 26, 36), outline=BORDER_COL)
    draw.text((box_x0 + 18, box_y0 + 13), "Simulação Visual de Partículas em Alta Velocidade", fill=TEXT_MUTED, font=FONT_MONO)
    
    # Linha horizontal central do fluxo
    rail_y = 175
    rail_x0 = box_x0 + 120
    rail_x1 = box_x1 - 120
    rail_w  = rail_x1 - rail_x0
    
    draw.line([(rail_x0, rail_y), (rail_x1, rail_y)], fill=(45, 55, 72), width=4)
    
    # Pulso/Feixe luminoso percorrendo a linha
    beam_x = rail_x0 + (current_time / DURATION_SEC) * rail_w
    draw.ellipse([(beam_x - 18, rail_y - 18), (beam_x + 18, rail_y + 18)], fill=STAGES[active_idx]["color"])
    
    # 5 Hubs / Estações Numéricas (1 a 5)
    step_w = rail_w / 4
    hub_labels = ["Parse", "Chunking", "Ingestão", "Transform.", "Busca RRF"]
    
    for i in range(5):
        cx = rail_x0 + i * step_w
        cy = rail_y
        is_current = (i == active_idx)
        is_passed  = (i < active_idx)
        stage_col  = STAGES[i]["color"]
        
        r = 20 if is_current else 14
        bg_col = stage_col if is_current else ((50, 60, 75) if is_passed else (22, 28, 38))
        border_col = TEXT_WHITE if is_current else (stage_col if is_passed else (60, 70, 85))
        
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=bg_col, outline=border_col, width=2)
        
        num_str = str(i + 1)
        num_w = text_w(draw, num_str, FONT_H2 if is_current else FONT_SM)
        draw.text((cx - num_w / 2, cy - (10 if is_current else 7)), num_str, fill=(TEXT_WHITE if not is_current else (15, 20, 28)), font=FONT_H2 if is_current else FONT_SM)
        
        lbl = hub_labels[i]
        lw = text_w(draw, lbl, FONT_SM)
        draw.text((cx - lw / 2, cy + 24), lbl, fill=stage_col if is_current else TEXT_MUTED, font=FONT_SM)
        
    # Partículas voadoras
    for p in PARTICLES:
        p["x"] += p["speed"]
        if p["x"] > (box_x1 - 10):
            p["x"] = box_x0 + 10
        py = p["y"]
        px = p["x"]
        if box_y0 + 35 < py < box_y1 - 15:
            col = STAGES[p["color_idx"]]["color"]
            draw.ellipse([(px - p["size"], py - p["size"]), (px + p["size"], py + p["size"])], fill=col)


def draw_cards(draw: ImageDraw.Draw, active_idx: int, frame_idx: int) -> None:
    card_w = 245
    card_h = 265
    spacing = 14
    start_x = (WIDTH - (5 * card_w + 4 * spacing)) // 2
    top_y   = 250
    
    for i, stage in enumerate(STAGES):
        x = start_x + i * (card_w + spacing)
        is_active = (i == active_idx)
        
        bg     = CARD_ACTIVE if is_active else CARD_BG
        border = stage["color"] if is_active else BORDER_COL
        bw     = 2 if is_active else 1
        
        # Fundo do card
        draw.rounded_rectangle([(x, top_y), (x + card_w, top_y + card_h)], radius=9, fill=bg, outline=border, width=bw)
        
        # Cabeçalho do card
        hdr_h = 36
        hdr_fill = (34, 44, 60) if is_active else (24, 29, 38)
        draw.rounded_rectangle([(x, top_y), (x + card_w, top_y + hdr_h)], radius=9, fill=hdr_fill, outline=border, width=bw)
        
        # Título do Card
        t_color = stage["color"] if is_active else TEXT_MUTED
        title_str = stage["title"].split("&")[0].split("—")[0].strip()
        draw.text((x + 10, top_y + 8), title_str, fill=t_color, font=FONT_SM)
        
        # Janela de Tempo
        draw.text((x + 10, top_y + hdr_h + 6), stage["time"], fill=YELLOW if is_active else TEXT_MUTED, font=FONT_SM)
        
        # Subtítulo
        draw.text((x + 10, top_y + hdr_h + 22), stage["subtitle"], fill=TEXT_MUTED, font=FONT_MONO)
        
        # Itens em bullet points
        y_item = top_y + hdr_h + 40
        item_col = TEXT_WHITE if is_active else (140, 150, 165)
        for item in stage["card_items"]:
            y_item = draw_text_lines(draw, x + 10, y_item, item, FONT_SM, item_col, max_w=card_w - 20, line_h=18)
            y_item += 3
            
        # Barra pulsante no card ativo
        if is_active:
            pulse = (frame_idx % 12) / 12.0
            pb_y  = top_y + card_h - 14
            draw.rounded_rectangle([(x + 10, pb_y), (x + card_w - 10, pb_y + 6)], radius=3, fill=(38, 48, 66))
            fw = int((card_w - 20) * pulse)
            if fw > 0:
                draw.rounded_rectangle([(x + 10, pb_y), (x + 10 + fw, pb_y + 6)], radius=3, fill=stage["color"])


def draw_detail_panel(draw: ImageDraw.Draw, active_idx: int, frame_idx: int) -> None:
    stage = STAGES[active_idx]
    col   = stage["color"]
    bot_y = 530
    bot_h = 250
    
    # Container do painel
    draw.rounded_rectangle([(34, bot_y), (WIDTH - 34, bot_y + bot_h)], radius=10, fill=(16, 22, 32), outline=col, width=2)
    draw.rounded_rectangle([(34, bot_y), (44, bot_y + bot_h)], radius=4, fill=col)
    
    # Cabeçalho do Painel
    badge_str = f"ETAPA {active_idx + 1} / 5"
    draw.text((56, bot_y + 12), badge_str, fill=col, font=FONT_SM)
    
    draw.text((56, bot_y + 30), stage["title"], fill=TEXT_WHITE, font=FONT_H2)
    draw.text((WIDTH - 160, bot_y + 30), stage["time"], fill=TEXT_MUTED, font=FONT_MONO)
    
    # Linha separadora
    draw.line([(56, bot_y + 54), (WIDTH - 50, bot_y + 54)], fill=BORDER_COL, width=1)
    
    # Descrição completa (quebra automática em várias linhas)
    draw_text_lines(draw, 56, bot_y + 64, stage["desc"], FONT_REG, (200, 210, 225), max_w=WIDTH - 120, line_h=20)
    
    # Badges / Tags arredondadas na parte inferior do painel
    tag_x = 56
    tag_y = bot_y + 185
    for tag in stage["tags"]:
        tw = text_w(draw, tag, FONT_MONO) + 24
        if tag_x + tw > (WIDTH - 60):
            tag_x = 56
            tag_y += 28
            
        draw.rounded_rectangle([(tag_x, tag_y), (tag_x + tw, tag_y + 22)], radius=6, fill=(26, 34, 48), outline=(48, 58, 75))
        draw.ellipse([(tag_x + 6, tag_y + 8), (tag_x + 12, tag_y + 14)], fill=col)
        draw.text((tag_x + 18, tag_y + 4), tag, fill=TEXT_WHITE, font=FONT_MONO)
        tag_x += tw + 8


def generate_gif(output_path: str) -> None:
    print(f"Gerando {TOTAL_FRAMES} frames da réplica fiel ({WIDTH}×{HEIGHT}px, {FPS}fps)…")
    frames = []

    for frame_idx in range(TOTAL_FRAMES):
        current_time = (frame_idx / TOTAL_FRAMES) * DURATION_SEC
        progress     = frame_idx / float(TOTAL_FRAMES)
        active_idx   = min(int(current_time), 4)

        img  = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(img)

        draw_header(draw, current_time, active_idx)
        draw_timeline(img, draw, progress, active_idx)
        draw_canvas_simulation(draw, current_time, active_idx, frame_idx)
        draw_cards(draw, active_idx, frame_idx)
        draw_detail_panel(draw, active_idx, frame_idx)

        frames.append(img)

    print(f"Salvando GIF em: {output_path}")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    size_kb = os.path.getsize(output_path) // 1024
    print(f"GIF gerado com sucesso! ({size_kb} KB)")


if __name__ == "__main__":
    os.makedirs("docs", exist_ok=True)
    generate_gif(os.path.join("docs", "rag_flow_5s.gif"))
