"""
Parser de Áudio (Transcrição via Whisper com Segmentação por Pausas e Timestamps) — Passo 27.

Transforma arquivos de áudio (.mp3, .wav, .m4a, .flac, .ogg, .aac) em uma sequência de elementos
AudioSegment do Canonical Document Model, onde cada segmento contém:
1. Timestamps de início e término (segundos e string formatada HH:MM:SS.mmm).
2. Transcrição completa e pontuada do trecho via Whisper (faster-whisper).
3. Segmentação natural baseada em pausas de fala com suporte a agrupamento configurável.
4. Representação textual estruturada pronta para indexação vetorial e busca léxica.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.parsing.canonical_model import AudioSegment, Document, ElementType
from src.parsing.video_parser import format_timestamp, get_whisper_model


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def convert_audio_to_wav(
    input_path: str | Path,
    output_wav_path: Optional[str | Path] = None,
) -> Optional[Path]:
    """
    Converte arquivo de áudio de qualquer formato para WAV 16kHz mono via ffmpeg.
    Se já for WAV 16kHz mono ou se a conversão direta não for requerida, pode ser usado
    como pré-processamento para garantir compatibilidade máxima.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {input_path}")

    if output_wav_path is None:
        fd, temp_wav = tempfile.mkstemp(suffix=".wav", prefix="audio_conv_")
        os.close(fd)
        out_path = Path(temp_wav)
    else:
        out_path = Path(output_wav_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(out_path),
    ]

    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        else:
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            return None
    except Exception as e:
        print(f"[audio_parser] Aviso ao converter áudio com ffmpeg: {e}")
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        return None


def transcribe_audio_segments(
    audio_path: str | Path,
    model_size: str = "tiny",
    min_segment_duration_sec: float = 2.0,
    max_segment_duration_sec: float = 30.0,
) -> List[Dict[str, Any]]:
    """
    Executa transcrição sobre o arquivo de áudio utilizando faster-whisper.
    Whisper divide nativamente por pausas e pontuações da fala.
    Agrupa microsegmentos menores que min_segment_duration_sec para evitar
    chunks com uma única palavra isolada.
    """
    path_str = str(audio_path)
    if not os.path.exists(path_str):
        return []

    try:
        model = get_whisper_model(model_size=model_size)
        segments_gen, info = model.transcribe(
            path_str,
            beam_size=1,
            word_timestamps=False,
        )

        raw_segments = []
        for s in segments_gen:
            text = s.text.strip()
            if text:
                raw_segments.append(
                    {
                        "start": round(float(s.start), 3),
                        "end": round(float(s.end), 3),
                        "text": text,
                    }
                )

        if not raw_segments:
            return []

        # Agrupamento opcional de microsegmentos muito curtos
        grouped: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None

        for seg in raw_segments:
            if current is None:
                current = dict(seg)
                continue

            dur = current["end"] - current["start"]
            cand_dur = seg["end"] - current["start"]

            # Se o segmento atual ainda é curto (< min_segment_duration_sec)
            # e adicionar o próximo não excede max_segment_duration_sec, funde
            if dur < min_segment_duration_sec and cand_dur <= max_segment_duration_sec:
                current["end"] = seg["end"]
                current["text"] = f"{current['text']} {seg['text']}".strip()
            else:
                grouped.append(current)
                current = dict(seg)

        if current:
            grouped.append(current)

        return grouped
    except Exception as e:
        print(f"[audio_parser] Erro ao transcrever áudio com faster-whisper: {e}")
        return []


def parse_audio(
    path: str | Path,
    tenant_id: Optional[str] = None,
    source_uri: Optional[str] = None,
    whisper_model_size: str = "tiny",
    min_segment_sec: float = 2.0,
    max_segment_sec: float = 30.0,
) -> Document:
    """
    Lê um arquivo de áudio e produz um Document do Canonical Document Model
    composto por elementos AudioSegment com transcrição e timestamps.

    Passos:
    1. Calcula content_hash (SHA-256 do arquivo original).
    2. Transcreve com faster-whisper obtendo segmentos temporais naturais.
    3. Constrói elementos AudioSegment para cada trecho com timestamps formatados.
    4. Retorna Document serializável pronto para chunking e indexação.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {path}")

    content_hash = _sha256_file(path)
    title = path.stem.replace("_", " ").title()

    # Executa transcrição (Whisper aceita mp3, wav, flac etc. via PyAV)
    segments_data = transcribe_audio_segments(
        path,
        model_size=whisper_model_size,
        min_segment_duration_sec=min_segment_sec,
        max_segment_duration_sec=max_segment_sec,
    )

    # Fallback caso PyAV não consiga decodificar o container diretamente:
    if not segments_data:
        conv_wav = convert_audio_to_wav(path)
        if conv_wav:
            try:
                segments_data = transcribe_audio_segments(
                    conv_wav,
                    model_size=whisper_model_size,
                    min_segment_duration_sec=min_segment_sec,
                    max_segment_duration_sec=max_segment_sec,
                )
            finally:
                conv_wav.unlink(missing_ok=True)

    elements: List[AudioSegment] = []
    doc_id = str(uuid.uuid4())

    for idx, seg in enumerate(segments_data):
        start_sec = seg["start"]
        end_sec = seg["end"]
        start_str = format_timestamp(start_sec)
        end_str = format_timestamp(end_sec)
        transcript = seg["text"]

        content = (
            f"## Trecho {idx + 1} ({start_str} - {end_str})\n"
            f"{transcript}"
        )

        audio_el = AudioSegment(
            element_type=ElementType.AUDIO_SEGMENT,
            segment_index=idx + 1,
            start_time=start_sec,
            end_time=end_sec,
            start_time_str=start_str,
            end_time_str=end_str,
            transcript=transcript,
            content=content,
            speaker=seg.get("speaker"),
            metadata={
                "audio_file": path.name,
                "duration_seconds": round(end_sec - start_sec, 3),
            },
        )
        elements.append(audio_el)

    return Document(
        id=doc_id,
        source_uri=source_uri or str(path),
        title=title,
        elements=elements,
        tenant_id=tenant_id,
        content_hash=content_hash,
        parser_version="0.27.0",
        metadata={
            "media_type": "audio",
            "segment_count": len(elements),
            "total_segments": len(elements),
        },
    )
