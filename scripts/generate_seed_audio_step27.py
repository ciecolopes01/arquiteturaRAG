"""
Script: generate_seed_audio_step27.py — Passo 27.

Gera 2 arquivos de áudio sintéticos com fala natural e pausas no seed_corpus:
1. data/seed_corpus/22_podcast_governanca_rag.mp3 (2 trechos: Governança/Multi-tenant e ACL Pre-filter)
2. data/seed_corpus/23_reuniao_arquitetura_reranking.mp3 (2 trechos: Modelo BGE Reranker e Meta de Latência)

Cada arquivo possui:
- Conteúdo falado gerado via sintetizador de voz do Windows (System.Speech).
- Pausas explícitas entre ideias para testar a segmentação natural por pausas do faster-whisper.
- Codificação final em MP3 com taxa de amostragem padronizada via ffmpeg.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

SEED_DIR = Path(__file__).parent.parent / "data" / "seed_corpus"
SEED_DIR.mkdir(parents=True, exist_ok=True)


def synthesize_audio_tts(text: str, output_wav: Path) -> bool:
    """Gera áudio WAV via sintetizador de voz do Windows (System.Speech)."""
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
        print(f"[generate_seed_audio] Erro no TTS: {e}")
        return False


def build_audio_file(
    segments_text: list[str],
    output_mp3: Path,
    pause_sec: float = 1.5,
):
    """
    Constrói um arquivo MP3 contendo múltiplos trechos de fala separados por pausas de silêncio.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="seed_audio_"))
    try:
        part_wavs = []
        silence_wav = temp_dir / "silence.wav"

        # Gera áudio de silêncio de pause_sec segundos
        cmd_silence = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r=16000:cl=mono",
            "-t",
            str(pause_sec),
            str(silence_wav),
        ]
        subprocess.run(cmd_silence, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        for idx, text in enumerate(segments_text):
            seg_wav = temp_dir / f"seg_{idx:02d}.wav"
            ok = synthesize_audio_tts(text, seg_wav)
            if ok:
                part_wavs.append(seg_wav)
            if idx < len(segments_text) - 1:
                part_wavs.append(silence_wav)

        # Concatena via ffmpeg
        concat_txt = temp_dir / "concat_list.txt"
        with open(concat_txt, "w", encoding="utf-8") as f:
            for w in part_wavs:
                f.write(f"file '{w.as_posix()}'\n")

        # Converte para MP3 final
        cmd_mp3 = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_txt),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "128k",
            "-ar",
            "16000",
            str(output_mp3),
        ]
        subprocess.run(cmd_mp3, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"[generate_seed_audio] Gerado com sucesso: {output_mp3} ({output_mp3.stat().st_size} bytes)")
    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)


def generate_all_seed_audio():
    """Gera os arquivos de áudio de teste especificados no Passo 27."""
    # Áudio 1: Podcast de Governança
    a1_path = SEED_DIR / "22_podcast_governanca_rag.mp3"
    a1_segments = [
        "Bem-vindos ao podcast sobre governanca de dados no ecossistema RAG. Hoje discutimos o isolamento multi-tenant e politicas de controle de acesso nos indices vetoriais.",
        "A validacao das listas de controle de acesso deve ocorrer obrigatoriamente no estagio de pre-filtro para impedir vazamento de informacoes confidenciais.",
    ]
    build_audio_file(a1_segments, a1_path, pause_sec=1.5)

    # Áudio 2: Reunião Técnica de Reranking
    a2_path = SEED_DIR / "23_reuniao_arquitetura_reranking.mp3"
    a2_segments = [
        "Iniciando o alinhamento tecnico sobre o modelo de reranking. Os testes com BGE Reranker demonstraram um aumento significativo no recall do top cinco.",
        "Entretanto devemos assegurar que a latencia adicional do reranker nao ultrapasse a meta de quatrocentos milissegundos por consulta.",
    ]
    build_audio_file(a2_segments, a2_path, pause_sec=1.5)


if __name__ == "__main__":
    generate_all_seed_audio()
