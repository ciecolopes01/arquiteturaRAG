"""
Script: generate_seed_videos_step26.py — Passo 26.

Gera 2 arquivos de vídeo sintéticos no corpus seed:
1. data/seed_corpus/20_visao_computacional_e_multimodal.mp4 (2 cenas: Ingestão Multimodal e Detecção de Cenas)
2. data/seed_corpus/21_monitoramento_video_rag.mp4 (2 cenas: Painel de Latência e Alerta de Degradação de Recall)

Cada vídeo possui:
- Cortes de cena distintos com transição brusca de cores para detecção via PySceneDetect.
- Texto visual renderizado em alto contraste via Pillow para extração perfeita via OCR (Tesseract).
- Faixa de áudio sintetizada sincronizada com cada cena para transcrição via Whisper.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SEED_DIR = Path(__file__).parent.parent / "data" / "seed_corpus"
SEED_DIR.mkdir(parents=True, exist_ok=True)


def _get_font(size: int = 24):
    """Obtém fonte TrueType ou default."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


def create_frame_image(
    title: str,
    subtitle: str,
    bg_color: tuple = (245, 247, 250),
    header_color: tuple = (30, 64, 175),
    width: int = 800,
    height: int = 450,
) -> np.ndarray:
    """Gera um frame visual com layout limpo e texto de alto contraste para OCR."""
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    title_font = _get_font(32)
    sub_font = _get_font(20)

    # Banner superior
    draw.rectangle([(0, 0), (width, 80)], fill=header_color)
    draw.text((30, 22), title, font=title_font, fill=(255, 255, 255))

    # Caixa de conteúdo central
    box_rect = [(50, 130), (width - 50, height - 70)]
    draw.rectangle(box_rect, fill=(255, 255, 255), outline=header_color, width=3)

    # Texto explicativo central
    draw.text((80, 180), subtitle, font=sub_font, fill=(15, 23, 42))
    draw.text(
        (80, 240),
        "Sistema RAG Multimodal — Retrieval Engineering Platform",
        font=sub_font,
        fill=(71, 85, 105),
    )

    # Converte de PIL (RGB) para numpy (BGR para OpenCV)
    rgb_arr = np.array(img)
    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    return bgr_arr


def synthesize_audio_tts(text: str, output_wav: Path) -> bool:
    """Gera áudio WAV via sintetizador de voz do Windows (System.Speech)."""
    # Escapa aspas para PowerShell
    safe_text = text.replace("'", " ").replace('"', " ")
    ps_script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile('{output_wav.as_posix()}')
$synth.Speak('{safe_text}')
$synth.Dispose()
"""
    try:
        res = subprocess.run(
            ["powershell", "-Command", ps_script],
            capture_output=True,
            text=True,
            check=False,
        )
        return res.returncode == 0 and output_wav.exists() and output_wav.stat().st_size > 0
    except Exception as e:
        print(f"[generate_seed_videos] Erro no TTS: {e}")
        return False


def build_scene_video(
    scenes_spec: list[dict],
    output_mp4: Path,
    width: int = 800,
    height: int = 450,
    fps: int = 20,
):
    """
    Constrói um vídeo com múltiplas cenas e áudio narrado sincronizado.
    Cada scene_spec possui:
    - title: str
    - subtitle: str
    - bg_color: tuple
    - header_color: tuple
    - duration_sec: float
    - speech: str
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="seed_vid_"))
    try:
        # 1. Renderiza faixa de vídeo
        silent_video_path = temp_dir / "silent.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(silent_video_path), fourcc, fps, (width, height))

        audio_wav_files = []
        for idx, spec in enumerate(scenes_spec):
            frame_bgr = create_frame_image(
                title=spec["title"],
                subtitle=spec["subtitle"],
                bg_color=spec.get("bg_color", (245, 247, 250)),
                header_color=spec.get("header_color", (30, 64, 175)),
                width=width,
                height=height,
            )
            frames_count = int(spec["duration_sec"] * fps)
            for _ in range(frames_count):
                writer.write(frame_bgr)

            # Gera áudio da cena
            scene_wav = temp_dir / f"scene_{idx:02d}.wav"
            ok = synthesize_audio_tts(spec["speech"], scene_wav)
            if ok:
                # Ajusta a duração do áudio com ffmpeg (adiciona silêncio para bater com duration_sec)
                padded_wav = temp_dir / f"padded_{idx:02d}.wav"
                dur = spec["duration_sec"]
                cmd_pad = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(scene_wav),
                    "-af",
                    f"apad=whole_dur={dur}",
                    "-t",
                    str(dur),
                    str(padded_wav),
                ]
                subprocess.run(cmd_pad, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                if padded_wav.exists():
                    audio_wav_files.append(padded_wav)
                else:
                    audio_wav_files.append(scene_wav)
            else:
                # Gera silêncio se TTS falhar
                silence_wav = temp_dir / f"silence_{idx:02d}.wav"
                dur = spec["duration_sec"]
                cmd_sil = [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    f"anullsrc=r=16000:cl=mono",
                    "-t",
                    str(dur),
                    str(silence_wav),
                ]
                subprocess.run(cmd_sil, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                audio_wav_files.append(silence_wav)

        writer.release()

        # 2. Concatena os arquivos de áudio
        concat_txt = temp_dir / "concat_list.txt"
        with open(concat_txt, "w", encoding="utf-8") as f:
            for wav_file in audio_wav_files:
                f.write(f"file '{wav_file.as_posix()}'\n")

        merged_audio = temp_dir / "merged_audio.wav"
        cmd_concat = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_txt),
            "-c",
            "copy",
            str(merged_audio),
        ]
        subprocess.run(cmd_concat, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

        # 3. Faz o multiplex final com ffmpeg em H.264 + AAC
        cmd_mux = [
            "ffmpeg",
            "-y",
            "-i",
            str(silent_video_path),
            "-i",
            str(merged_audio),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            str(output_mp4),
        ]
        subprocess.run(cmd_mux, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"[generate_seed_videos] Gerado com sucesso: {output_mp4} ({output_mp4.stat().st_size} bytes)")
    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def generate_all_seed_videos():
    """Gera os vídeos de teste especificados no Passo 26."""
    # Vídeo 1: Visão Computacional e Multimodal
    v1_path = SEED_DIR / "20_visao_computacional_e_multimodal.mp4"
    v1_scenes = [
        {
            "title": "FASE 1: INGESTAO MULTIMODAL",
            "subtitle": "Pipeline de extracao e preparacao de dados visuais",
            "bg_color": (219, 234, 254),      # Azul claro suave
            "header_color": (30, 58, 138),    # Azul marinho
            "duration_sec": 4.0,
            "speech": "Nesta primeira fase apresentamos a ingestao de dados visuais e extracao de quadros.",
        },
        {
            "title": "FASE 2: DETECCAO DE CENAS E OCR",
            "subtitle": "Segmentacao temporal com keyframes e leitura de texto na tela",
            "bg_color": (254, 243, 199),      # Amarelo/âmbar claro
            "header_color": (180, 83, 9),     # Âmbar escuro
            "duration_sec": 4.0,
            "speech": "Na segunda fase realizamos a deteccao de cenas com extracao de keyframes e reconhecimento optico de caracteres.",
        },
    ]
    build_scene_video(v1_scenes, v1_path)

    # Vídeo 2: Monitoramento e Alertas
    v2_path = SEED_DIR / "21_monitoramento_video_rag.mp4"
    v2_scenes = [
        {
            "title": "PAINEL DE LATENCIA DE RETRIEVAL",
            "subtitle": "Medicao em tempo real do tempo de resposta dos canais de busca",
            "bg_color": (254, 226, 226),      # Vermelho claro
            "header_color": (153, 27, 27),    # Carmesim escuro
            "duration_sec": 4.0,
            "speech": "O painel de monitoramento de latencia avalia o tempo de execucao dos indices vetorial e esparso.",
        },
        {
            "title": "ALERTA DE DEGRADACAO DE RECALL",
            "subtitle": "Notificacao automatica quando a metrica de recall cai mais de 5 por cento",
            "bg_color": (237, 233, 254),      # Lavanda/roxo claro
            "header_color": (107, 33, 168),   # Roxo profundo
            "duration_sec": 4.0,
            "speech": "Alertas automaticos disparam quando o recall sofre degradacao superior a cinco pontos percentuais.",
        },
    ]
    build_scene_video(v2_scenes, v2_path)


if __name__ == "__main__":
    generate_all_seed_videos()
