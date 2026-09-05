"""
Gerador de GIF Animado de 5 segundos do Fluxo RAG Multimodal.
Etapas: Parse -> Chunking -> Ingestão/ACL -> Transformação -> Busca Híbrida/RRF.
"""

import math
import os
from PIL import Image, ImageDraw, ImageFont

WIDTH = 840
HEIGHT = 480
FPS = 15
DURATION_SEC = 5
TOTAL_FRAMES = FPS * DURATION_SEC  # 75 frames

# Cores do tema (Dark Cyberpunk / Tech Data Platform)
BG_COLOR = (13, 17, 23)
PANEL_BG = (22, 27, 34)
BORDER_COLOR = (48, 54, 61)
TEXT_WHITE = (240, 246, 252)
TEXT_MUTED = (139, 148, 158)
CYAN = (56, 189, 248)
ORANGE = (251, 146, 60)
PURPLE = (168, 85, 247)
GREEN = (52, 211, 153)
YELLOW = (250, 204, 21)
BLUE = (96, 165, 250)

STAGES = [
    {
        "title": "1. PARSE MULTIMODAL",
        "time": "0.0s - 1.0s",
        "color": CYAN,
        "subtitle": "Unificação para CanonicalDocument",
        "items": ["• PDF (Layout-aware)", "• XLSX (Tabelas/Planilhas)", "• IMG (OCR + Vision)", "• Áudio (Whisper)", "• Vídeo (Keyframes + Áudio)"]
    },
    {
        "title": "2. CHUNKING ENGINE",
        "time": "1.0s - 2.0s",
        "color": ORANGE,
        "subtitle": "Estratégias de Divisão de Contexto",
        "items": ["• Parent-Child (Hierárquico)", "• Recursivo UUIDv5 Determinístico", "• Semântico (Similaridade Cosseno)", "• Multi-Vector (Resumos/Keywords)", "• Multimodal Sincronizado"]
    },
    {
        "title": "3. INGESTÃO & GOVERNANÇA",
        "time": "2.0s - 3.0s",
        "color": GREEN,
        "subtitle": "Segurança Granular Fail-Closed",
        "items": ["• Tenant ID & Principals Tagging", "• Filtro ACL Pré-Indexação", "• Deduplicação & Content Hash", "• Grafo de Dependências Versionado", "• Zero-Bypass de Permissões"]
    },
    {
        "title": "4. TRANSFORMAÇÃO & ÍNDICES",
        "time": "3.0s - 4.0s",
        "color": PURPLE,
        "subtitle": "Geração Tri-Canal em Paralelo",
        "items": ["• BGE-M3 Denso (1024d) -> Qdrant", "• BM25 + tsvector -> PostgreSQL", "• Entidades & Triplas -> Neo4j Graph", "• Cache de Embeddings & Versionamento", "• Dual-Write Indexing"]
    },
    {
        "title": "5. BUSCA HÍBRIDA & RRF",
        "time": "4.0s - 5.0s",
        "color": YELLOW,
        "subtitle": "Recuperação Tri-Canal + Rerank",
        "items": ["• Soft Query Router (Fast/Full Path)", "• Busca Tri-Canal com Pré-Filtro ACL", "• Fusão RRF: Σ wi / (k + ranki)", "• Cross-Encoder BGE-Reranker-v2", "• Self-RAG Context Evaluator"]
    }
]

def try_load_font(size):
    try:
        for font_name in ["segoeui.ttf", "arial.ttf", "consola.ttf"]:
            font_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", font_name)
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
    except Exception:
        pass
    return ImageFont.load_default()

FONT_TITLE = try_load_font(18)
FONT_BOLD = try_load_font(14)
FONT_REGULAR = try_load_font(12)
FONT_SMALL = try_load_font(10)

def draw_header(draw, current_time, active_idx):
    draw.rectangle([(0, 0), (WIDTH, 48)], fill=(18, 22, 28))
    draw.text((20, 12), "ARQUITETURA RAG MULTIMODAL — FLUXO COMPLETO (5s)", fill=TEXT_WHITE, font=FONT_TITLE)
    time_str = f"TIMELINE: {current_time:04.2f}s / 5.00s"
    draw.text((WIDTH - 210, 14), time_str, fill=YELLOW, font=FONT_BOLD)
    draw.line([(0, 48), (WIDTH, 48)], fill=BORDER_COLOR, width=1)

def draw_progress_bar(draw, progress, active_idx):
    y_bar = 60
    bar_h = 8
    draw.rounded_rectangle([(20, y_bar), (WIDTH - 20, y_bar + bar_h)], radius=4, fill=(33, 38, 45))
    fill_w = max(10, int((WIDTH - 40) * progress))
    current_color = STAGES[active_idx]["color"]
    draw.rounded_rectangle([(20, y_bar), (20 + fill_w, y_bar + bar_h)], radius=4, fill=current_color)

    step_w = (WIDTH - 40) / 5
    for i, stage in enumerate(STAGES):
        cx = 20 + i * step_w + step_w / 2
        is_active = (i == active_idx)
        is_past = (i < active_idx)
        badge_color = stage["color"] if (is_active or is_past) else (60, 68, 78)
        
        draw.ellipse([(cx - 6, y_bar - 2), (cx + 6, y_bar + bar_h + 2)], fill=badge_color)
        
        label = stage["title"].split(". ")[1].split(" ")[0]
        label_color = TEXT_WHITE if is_active else (TEXT_MUTED if not is_past else (200, 205, 215))
        draw.text((cx - 24, y_bar + 14), label, fill=label_color, font=FONT_SMALL)

def draw_stage_cards(draw, active_idx, frame_idx):
    top_y = 95
    card_w = 152
    card_h = 235
    spacing = 8
    start_x = 20

    for i, stage in enumerate(STAGES):
        x = start_x + i * (card_w + spacing)
        is_active = (i == active_idx)
        
        if is_active:
            bg = (28, 36, 48)
            border = stage["color"]
            border_w = 2
        else:
            bg = PANEL_BG
            border = BORDER_COLOR
            border_w = 1

        draw.rounded_rectangle([(x, top_y), (x + card_w, top_y + card_h)], radius=8, fill=bg, outline=border, width=border_w)
        draw.rectangle([(x, top_y), (x + card_w, top_y + 32)], fill=(35, 43, 56) if is_active else (25, 30, 38))
        draw.text((x + 8, top_y + 8), stage["title"][:14], fill=stage["color"] if is_active else TEXT_MUTED, font=FONT_BOLD)
        draw.text((x + 8, top_y + 38), stage["time"], fill=YELLOW if is_active else TEXT_MUTED, font=FONT_SMALL)
        
        y_text = top_y + 56
        for item in stage["items"]:
            item_text = item if len(item) < 22 else item[:20] + ".."
            item_color = TEXT_WHITE if is_active else (110, 118, 129)
            draw.text((x + 8, y_text), item_text, fill=item_color, font=FONT_SMALL)
            y_text += 22

        if is_active:
            pulse = (frame_idx % 15) / 15.0
            pulse_y = top_y + card_h - 14
            draw.rounded_rectangle([(x + 8, pulse_y), (x + card_w - 8, pulse_y + 6)], radius=3, fill=(40, 50, 70))
            active_fill_w = int((card_w - 16) * pulse)
            draw.rounded_rectangle([(x + 8, pulse_y), (x + 8 + active_fill_w, pulse_y + 6)], radius=3, fill=stage["color"])

def draw_bottom_active_details(draw, active_idx, frame_idx):
    bot_y = 345
    bot_h = 120
    stage = STAGES[active_idx]
    
    draw.rounded_rectangle([(20, bot_y), (WIDTH - 20, bot_y + bot_h)], radius=8, fill=(18, 24, 34), outline=stage["color"], width=2)
    draw.rectangle([(20, bot_y), (28, bot_y + bot_h)], fill=stage["color"])
    
    draw.text((40, bot_y + 10), f"EXECUTANDO AGORA: {stage['title']}  ({stage['time']})", fill=stage["color"], font=FONT_BOLD)
    draw.text((40, bot_y + 30), stage["subtitle"], fill=TEXT_MUTED, font=FONT_REGULAR)
    
    col1_x = 40
    col2_x = 430
    
    items = stage["items"]
    for idx, it in enumerate(items[:3]):
        draw.text((col1_x, bot_y + 54 + idx * 20), it, fill=TEXT_WHITE, font=FONT_REGULAR)
    for idx, it in enumerate(items[3:]):
        draw.text((col2_x, bot_y + 54 + idx * 20), it, fill=TEXT_WHITE, font=FONT_REGULAR)
        
    phase = (frame_idx % 15) / 15.0
    for p in range(5):
        px = 750 + int(math.sin(phase * 6.28 + p) * 20) + p * 8
        py = bot_y + 25 + p * 15
        draw.ellipse([(px - 4, py - 4), (px + 4, py + 4)], fill=stage["color"])

def generate_gif(output_path):
    print(f"Gerando {TOTAL_FRAMES} frames...")
    frames = []
    
    for frame_idx in range(TOTAL_FRAMES):
        current_time = (frame_idx / TOTAL_FRAMES) * DURATION_SEC
        progress = frame_idx / float(TOTAL_FRAMES)
        active_idx = min(int(current_time), 4)
        
        img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(img)
        
        draw_header(draw, current_time, active_idx)
        draw_progress_bar(draw, progress, active_idx)
        draw_stage_cards(draw, active_idx, frame_idx)
        draw_bottom_active_details(draw, active_idx, frame_idx)
        
        frames.append(img)
        
    print(f"Salvando GIF em: {output_path}")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True
    )
    print("GIF gerado com sucesso!")

if __name__ == "__main__":
    os.makedirs("docs", exist_ok=True)
    out_file = os.path.join("docs", "rag_flow_5s.gif")
    generate_gif(out_file)
