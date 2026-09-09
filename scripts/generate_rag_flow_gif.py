"""
Gerador de GIF Animado de 5 segundos do Fluxo RAG Multimodal.
Etapas: Parse -> Chunking -> Ingestão/ACL -> Transformação -> Busca Híbrida/RRF.

Correções v2:
- Canvas 1200×680 (era 840×480): espaço suficiente para todo o texto
- Cards 210px de largura (era 152px): sem truncamento
- Fontes maiores: 20/15/13/11pt (era 18/14/12/10pt)
- Sem truncamento de itens (era item[:20])
- Painel inferior com layout de 3 colunas
- draw_text_lines(): quebra de linha automática por largura máxima
"""

import math
import os
from PIL import Image, ImageDraw, ImageFont

# ── Dimensões ────────────────────────────────────────────────────────────────
WIDTH  = 1200
HEIGHT = 580
FPS    = 12                        # 12 fps → arquivo menor
DURATION_SEC  = 5
TOTAL_FRAMES  = FPS * DURATION_SEC # 60 frames

# ── Paleta de Cores (Dark Cyberpunk) ─────────────────────────────────────────
BG_COLOR    = (13,  17,  23)
PANEL_BG    = (22,  27,  34)
BORDER_COL  = (48,  54,  61)
TEXT_WHITE  = (240, 246, 252)
TEXT_MUTED  = (120, 130, 140)
CYAN        = (56,  189, 248)
ORANGE      = (251, 146, 60)
PURPLE      = (168, 85,  247)
GREEN       = (52,  211, 153)
YELLOW      = (250, 204, 21)
BLUE        = (96,  165, 250)

# ── Dados dos Estágios ───────────────────────────────────────────────────────
STAGES = [
    {
        "title":    "1. PARSE MULTIMODAL",
        "short":    "Parse",
        "time":     "0.0s – 1.0s",
        "color":    CYAN,
        "subtitle": "CanonicalDocument Unificado",
        "items": [
            "• PDF  Layout + Img + Tabelas",
            "• XLSX Merged + Multi-aba",
            "• DOCX / TXT / HTML",
            "• IMG  OCR + Vision + MinIO",
            "• Audio Whisper + Pausas",
            "• Video Cenas + Whisper + OCR",
        ],
    },
    {
        "title":    "2. CHUNKING ENGINE",
        "short":    "Chunking",
        "time":     "1.0s – 2.0s",
        "color":    ORANGE,
        "subtitle": "Divisao Hierarquica Estrategica",
        "items": [
            "• Parent-Child  1200 / 300 tokens",
            "• Recursivo UUIDv5 Deterministico",
            "• Semantico por Cosseno",
            "• Overlap  40 – 64 tokens",
            "• Multimodal Sync Chunks",
        ],
    },
    {
        "title":    "3. INGESTAO & ACL",
        "short":    "Ingestao",
        "time":     "2.0s – 3.0s",
        "color":    GREEN,
        "subtitle": "Seguranca Fail-Closed Granular",
        "items": [
            "• Tenant ID + ACL Principals",
            "• Pre-filtro Fail-Closed",
            "• SHA-256 Content Hash",
            "• Doc Registry (Postgres)",
            "• Delta-Reindex Incremental",
        ],
    },
    {
        "title":    "4. TRANSFORMACAO",
        "short":    "Transform",
        "time":     "3.0s – 4.0s",
        "color":    PURPLE,
        "subtitle": "3 Indices em Paralelo",
        "items": [
            "• BGE-M3 -> Qdrant Dense",
            "• BM25   -> Postgres tsvector",
            "• Graph RAG Neo4j / NetworkX",
            "• NER + Relacoes por Regex",
            "• Embedding Cache + Dual-Write",
        ],
    },
    {
        "title":    "5. BUSCA HIBRIDA",
        "short":    "Busca RRF",
        "time":     "4.0s – 5.0s",
        "color":    YELLOW,
        "subtitle": "RRF Ponderado + Rerank",
        "items": [
            "• Query Router  5 intencoes",
            "• HyDE + Query Decomposition",
            "• RRF Dense + BM25 + Graph k=60",
            "• BGE-Reranker-v2-m3",
            "• Self-RAG Evaluator",
        ],
    },
]

# ── Layout de Cards ───────────────────────────────────────────────────────────
CARD_W   = 210
CARD_H   = 205   # 6 itens × 16px + header(36) + tempo(22) + barra(14) + margem
SPACING  =  10
TOTAL_CARDS_W = 5 * CARD_W + 4 * SPACING   # 1090
START_X  = (WIDTH - TOTAL_CARDS_W) // 2     # 55

TOP_Y    = 110   # topo dos cards
BOT_Y    = TOP_Y + CARD_H + 15  # 330 — painel inferior logo abaixo dos cards
BOT_H    = 215   # altura do painel inferior

# ── Fontes ───────────────────────────────────────────────────────────────────
def try_font(size: int) -> ImageFont.FreeTypeFont:
    """Tenta carregar Segoe UI, Arial ou Consolas; fallback para bitmap."""
    for name in ["segoeui.ttf", "arial.ttf", "consola.ttf", "calibri.ttf"]:
        path = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", name)
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()

FONT_H1  = try_font(20)   # título do header
FONT_H2  = try_font(15)   # título dos cards
FONT_REG = try_font(13)   # corpo
FONT_SM  = try_font(11)   # items e labels pequenos

# ── Helpers de Texto ──────────────────────────────────────────────────────────
def text_w(draw: ImageDraw.Draw, text: str, font) -> int:
    """Largura em pixels de 'text' com a font dada."""
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0]

def draw_text_lines(draw, x, y, text, font, fill, max_w, line_h):
    """
    Escreve texto com quebra automática de linha quando ultrapassa max_w.
    Retorna o y final.
    """
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

# ── Seções de Desenho ─────────────────────────────────────────────────────────

def draw_header(draw: ImageDraw.Draw, current_time: float, active_idx: int) -> None:
    # Fundo do cabeçalho
    draw.rectangle([(0, 0), (WIDTH, 52)], fill=(18, 22, 28))
    draw.text((20, 14), "ARQUITETURA RAG MULTIMODAL — FLUXO COMPLETO (5s)",
              fill=TEXT_WHITE, font=FONT_H1)
    stage_label = f"ETAPA {active_idx + 1}/5: {STAGES[active_idx]['title']}"
    sw = text_w(draw, stage_label, FONT_REG)
    draw.text((WIDTH - sw - 150, 18), stage_label,
              fill=STAGES[active_idx]["color"], font=FONT_REG)
    time_str = f"{current_time:04.2f}s"
    draw.text((WIDTH - 90, 14), time_str, fill=YELLOW, font=FONT_H2)
    draw.line([(0, 52), (WIDTH, 52)], fill=BORDER_COL, width=1)


def draw_progress(draw: ImageDraw.Draw, progress: float, active_idx: int) -> None:
    y0, h = 62, 9
    draw.rounded_rectangle([(START_X, y0), (WIDTH - START_X, y0 + h)],
                            radius=4, fill=(33, 38, 45))
    fill_w = max(12, int((WIDTH - 2 * START_X) * progress))
    draw.rounded_rectangle([(START_X, y0), (START_X + fill_w, y0 + h)],
                            radius=4, fill=STAGES[active_idx]["color"])

    step_w = (WIDTH - 2 * START_X) / 5
    for i, stage in enumerate(STAGES):
        cx  = START_X + i * step_w + step_w / 2
        col = stage["color"] if i <= active_idx else (55, 62, 72)
        r   = 7 if i == active_idx else 5
        draw.ellipse([(cx - r, y0 - 1), (cx + r, y0 + h + 1)], fill=col)
        label_col = TEXT_WHITE if i == active_idx else TEXT_MUTED
        lw = text_w(draw, stage["short"], FONT_SM)
        draw.text((cx - lw / 2, y0 + h + 5), stage["short"],
                  fill=label_col, font=FONT_SM)


def draw_cards(draw: ImageDraw.Draw, active_idx: int, frame_idx: int) -> None:
    for i, stage in enumerate(STAGES):
        x         = START_X + i * (CARD_W + SPACING)
        is_active = i == active_idx

        bg     = (28, 36, 50) if is_active else PANEL_BG
        border = stage["color"] if is_active else BORDER_COL
        bw     = 2 if is_active else 1

        # Corpo do card
        draw.rounded_rectangle([(x, TOP_Y), (x + CARD_W, TOP_Y + CARD_H)],
                                radius=9, fill=bg, outline=border, width=bw)

        # Cabeçalho do card
        hdr_h = 36
        hdr_fill = (35, 44, 58) if is_active else (24, 29, 38)
        draw.rounded_rectangle([(x, TOP_Y), (x + CARD_W, TOP_Y + hdr_h)],
                                radius=9, fill=hdr_fill, outline=border, width=bw)

        # Título (duas linhas se necessário)
        t_color = stage["color"] if is_active else TEXT_MUTED
        title   = stage["title"]
        # Divide em linha 1 (ex "1. PARSE") e linha 2 (ex "MULTIMODAL")
        parts = title.split(" ", 2)
        line1 = " ".join(parts[:2])
        line2 = parts[2] if len(parts) > 2 else ""
        draw.text((x + 8, TOP_Y + 5), line1, fill=t_color, font=FONT_SM)
        if line2:
            draw.text((x + 8, TOP_Y + 19), line2, fill=t_color, font=FONT_SM)

        # Faixa de tempo
        draw.text((x + 8, TOP_Y + hdr_h + 4), stage["time"],
                  fill=YELLOW if is_active else TEXT_MUTED, font=FONT_SM)

        # Itens
        y_item = TOP_Y + hdr_h + 22
        item_col = TEXT_WHITE if is_active else (105, 115, 128)
        for item in stage["items"]:
            draw_text_lines(draw, x + 8, y_item, item,
                            FONT_SM, item_col,
                            max_w=CARD_W - 16, line_h=16)
            y_item += 16
            if y_item > TOP_Y + CARD_H - 22:
                break   # segurança: nunca sai do card

        # Mini barra de progresso pulsante no card ativo
        if is_active:
            pulse    = (frame_idx % 12) / 12.0
            pb_y     = TOP_Y + CARD_H - 14
            draw.rounded_rectangle([(x + 8, pb_y), (x + CARD_W - 8, pb_y + 6)],
                                   radius=3, fill=(38, 48, 66))
            fw = int((CARD_W - 16) * pulse)
            if fw > 0:
                draw.rounded_rectangle([(x + 8, pb_y), (x + 8 + fw, pb_y + 6)],
                                       radius=3, fill=stage["color"])


def draw_detail_panel(draw: ImageDraw.Draw, active_idx: int, frame_idx: int) -> None:
    stage = STAGES[active_idx]
    col   = stage["color"]

    # Borda e fundo
    draw.rounded_rectangle([(20, BOT_Y), (WIDTH - 20, BOT_Y + BOT_H)],
                            radius=9, fill=(16, 22, 32), outline=col, width=2)
    # Barra lateral esquerda colorida
    draw.rounded_rectangle([(20, BOT_Y), (30, BOT_Y + BOT_H)],
                            radius=4, fill=col)

    # Cabeçalho do painel
    header_txt = f"EXECUTANDO:  {stage['title']}    ({stage['time']})"
    draw.text((42, BOT_Y + 10), header_txt, fill=col, font=FONT_H2)
    draw.text((42, BOT_Y + 30), stage["subtitle"], fill=TEXT_MUTED, font=FONT_REG)

    # Linha separadora
    draw.line([(42, BOT_Y + 50), (WIDTH - 30, BOT_Y + 50)],
              fill=BORDER_COL, width=1)

    # Items em 3 colunas
    items     = stage["items"]
    n         = len(items)
    col_count = 3
    col_w     = (WIDTH - 80) // col_count   # ~360px each
    col_xs    = [42 + c * col_w for c in range(col_count)]

    for idx, it in enumerate(items):
        col_i = idx % col_count
        row_i = idx // col_count
        ix    = col_xs[col_i]
        iy    = BOT_Y + 58 + row_i * 22
        draw.text((ix, iy), it, fill=TEXT_WHITE, font=FONT_REG)

    # Partículas animadas no canto
    phase = (frame_idx % 12) / 12.0
    for p in range(6):
        px = WIDTH - 90 + int(math.sin(phase * 6.28 + p * 1.05) * 22)
        py = BOT_Y + 70 + p * 22
        r  = 5 - (p % 2)
        draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=col)


# ── Geração dos Frames ────────────────────────────────────────────────────────

def generate_gif(output_path: str) -> None:
    print(f"Gerando {TOTAL_FRAMES} frames  ({WIDTH}×{HEIGHT}px, {FPS}fps)…")
    frames = []

    for frame_idx in range(TOTAL_FRAMES):
        current_time = (frame_idx / TOTAL_FRAMES) * DURATION_SEC
        progress     = frame_idx / float(TOTAL_FRAMES)
        active_idx   = min(int(current_time), 4)

        img  = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(img)

        draw_header(draw, current_time, active_idx)
        draw_progress(draw, progress, active_idx)
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
