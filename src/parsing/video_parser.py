"""
Parser de Vídeo (Detecção de Cenas + Transcrição com Timestamps + OCR + Visão) — Passo 26.

Transforma arquivos de vídeo (.mp4, .avi, .mov, etc.) em uma sequência de elementos
VideoScene do Canonical Document Model, onde cada cena contém:
1. Timestamps de início e fim (segundos e string formatada).
2. Transcrição de áudio via Whisper (faster-whisper) sincronizada com o intervalo temporal.
3. Descrição visual do keyframe representativo da cena (Claude Vision ou fallback estruturado).
4. Texto na tela extraído via OCR no keyframe (Tesseract).
5. Conteúdo textual combinado pronto para embedding multimodal e busca com timestamps.
"""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import tempfile
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from scenedetect import ContentDetector, detect

from src.config import settings
from src.parsing.canonical_model import Document, ElementType, VideoScene
from src.parsing.image_parser import describe_image_visual, extract_text_from_image

logger = logging.getLogger("rag_demo")

# ---------------------------------------------------------------------------
# Singleton / Cache para o modelo Whisper
# ---------------------------------------------------------------------------

_WHISPER_MODEL = None
_CURRENT_MODEL_SIZE: Optional[str] = None


def get_whisper_model(model_size: str = "tiny"):
    """Retorna instância singleton do WhisperModel para evitar múltiplos downloads/inits."""
    global _WHISPER_MODEL, _CURRENT_MODEL_SIZE
    if _WHISPER_MODEL is None or _CURRENT_MODEL_SIZE != model_size:
        from faster_whisper import WhisperModel

        _WHISPER_MODEL = WhisperModel(model_size, device="cpu", compute_type="int8")
        _CURRENT_MODEL_SIZE = model_size
    return _WHISPER_MODEL


# ---------------------------------------------------------------------------
# Formatador de Timestamp
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    """Formata segundos em HH:MM:SS.mmm."""
    if seconds < 0:
        seconds = 0.0
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Extração de Áudio via ffmpeg
# ---------------------------------------------------------------------------

def extract_audio_from_video(
    video_path: str | Path,
    output_wav_path: Optional[str | Path] = None,
) -> Optional[Path]:
    """
    Extrai o áudio do arquivo de vídeo como WAV 16kHz mono utilizando ffmpeg.
    Retorna o Path do arquivo de áudio gerado, ou None se o vídeo não possuir áudio.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

    if output_wav_path is None:
        fd, temp_wav = tempfile.mkstemp(suffix=".wav", prefix="audio_ext_")
        os.close(fd)
        out_path = Path(temp_wav)
    else:
        out_path = Path(output_wav_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
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
        print(f"[video_parser] Aviso ao extrair áudio com ffmpeg: {e}")
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        return None


# ---------------------------------------------------------------------------
# Transcrição de Áudio com Timestamps (Whisper)
# ---------------------------------------------------------------------------

def transcribe_audio_with_timestamps(
    audio_path: str | Path,
    model_size: str = "tiny",
) -> List[Dict[str, Any]]:
    """
    Transcreve um arquivo de áudio utilizando faster-whisper.
    Retorna lista de segmentos com start, end e text.
    """
    path_str = str(audio_path)
    if not os.path.exists(path_str):
        return []

    try:
        model = get_whisper_model(model_size=model_size)
        segments_gen, info = model.transcribe(
            path_str,
            beam_size=1,
            word_timestamps=True,
        )
        results = []
        for s in segments_gen:
            text = s.text.strip()
            words = []
            if hasattr(s, "words") and s.words:
                for w in s.words:
                    words.append(
                        {
                            "start": round(float(w.start), 3),
                            "end": round(float(w.end), 3),
                            "word": w.word,
                        }
                    )
            if text:
                logger.info(f"  • Áudio [{format_timestamp(s.start)} -> {format_timestamp(s.end)}]: {text[:80]}")
                results.append(
                    {
                        "start": round(float(s.start), 3),
                        "end": round(float(s.end), 3),
                        "text": text,
                        "words": words,
                    }
                )
        return results
    except Exception as e:
        print(f"[video_parser] Erro na transcrição com faster-whisper: {e}")
        return []


# ---------------------------------------------------------------------------
# Detecção de Cenas (PySceneDetect)
# ---------------------------------------------------------------------------

def detect_video_scenes(
    video_path: str | Path,
    threshold: float = 15.0,
    min_scene_len_sec: float = 1.0,
) -> List[Tuple[float, float]]:
    """
    Executa corte de cenas no vídeo utilizando PySceneDetect.
    Retorna lista de tuplas (start_seconds, end_seconds).
    Se nenhum corte for detectado, retorna uma única cena englobando todo o vídeo.
    """
    video_path_str = str(video_path)
    scenes = []

    try:
        detected = detect(video_path_str, ContentDetector(threshold=threshold))
        for scene in detected:
            start_sec = getattr(scene[0], "seconds", None)
            if start_sec is None:
                start_sec = scene[0].get_seconds()

            end_sec = getattr(scene[1], "seconds", None)
            if end_sec is None:
                end_sec = scene[1].get_seconds()

            if (end_sec - start_sec) >= 0.1:
                scenes.append((round(float(start_sec), 3), round(float(end_sec), 3)))
    except Exception as e:
        print(f"[video_parser] Aviso ao detectar cenas com scenedetect: {e}")

    # Fallback se nenhuma cena foi detectada ou ocorreu erro
    if not scenes:
        cap = cv2.VideoCapture(video_path_str)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()

        duration = round(frame_count / fps, 3) if fps > 0 else 0.0
        if duration <= 0.0:
            duration = 1.0
        scenes.append((0.0, duration))

    return scenes


# ---------------------------------------------------------------------------
# Extração de Keyframe
# ---------------------------------------------------------------------------

def extract_keyframe_bytes(
    video_path: str | Path,
    timestamp_sec: float,
) -> Optional[bytes]:
    """
    Extrai o frame no timestamp solicitado como bytes JPEG.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec * 1000.0))
    ret, frame = cap.read()

    if not ret:
        # Tenta o primeiro frame como fallback
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    cap.release()

    if not ret or frame is None:
        return None

    success, buffer = cv2.imencode(".jpg", frame)
    if success:
        return buffer.tobytes()
    return None


# ---------------------------------------------------------------------------
# Upload Opcional para MinIO
# ---------------------------------------------------------------------------

_MINIO_AVAILABLE: Optional[bool] = None

def _is_minio_reachable() -> bool:
    """Verifica se o servidor MinIO está acessível via socket rápido (0.15s) com cache."""
    global _MINIO_AVAILABLE
    if _MINIO_AVAILABLE is not None:
        return _MINIO_AVAILABLE

    import socket
    endpoint = settings.minio_endpoint or "localhost:9000"
    try:
        if ":" in endpoint:
            host, port_str = endpoint.split(":", 1)
            port = int(port_str)
        else:
            host, port = endpoint, 9000
    except Exception:
        host, port = "localhost", 9000

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.15)
    try:
        sock.connect((host, port))
        sock.close()
        _MINIO_AVAILABLE = True
    except Exception:
        _MINIO_AVAILABLE = False
    return _MINIO_AVAILABLE


def _upload_keyframe(keyframe_bytes: bytes, object_name: str) -> Optional[str]:
    """Salva keyframe no MinIO se disponível."""
    if not _is_minio_reachable():
        return None

    try:
        from minio import Minio

        endpoint = settings.minio_endpoint
        access_key = settings.minio_access_key
        secret_key = settings.minio_secret_key
        bucket = settings.minio_bucket

        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=settings.minio_secure,
        )
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)

        client.put_object(
            bucket,
            object_name,
            io.BytesIO(keyframe_bytes),
            len(keyframe_bytes),
            content_type="image/jpeg",
        )
        return f"minio://{bucket}/{object_name}"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Parser Principal de Vídeo
# ---------------------------------------------------------------------------

def parse_video(
    path: str | Path,
    tenant_id: Optional[str] = None,
    source_uri: Optional[str] = None,
    scene_threshold: float = 15.0,
    whisper_model_size: str = "tiny",
) -> Document:
    """
    Lê um arquivo de vídeo e produz um Document do Canonical Document Model
    composto por elementos VideoScene.

    Passos:
    1. Extração de áudio via ffmpeg e transcrição com timestamps via faster-whisper.
    2. Detecção de cenas via PySceneDetect.
    3. Para cada cena:
       - Extração de keyframe representativo (ponto médio temporal).
       - Execução de OCR local (Tesseract) sobre o keyframe.
       - Geração de descrição visual (Claude Vision ou fallback).
       - Associação dos trechos de transcrição de áudio do intervalo da cena.
       - Construção do elemento VideoScene com conteúdo combinado e timestamps.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de vídeo não encontrado: {path}")

    content_hash = _sha256_file(path)
    title = path.stem.replace("_", " ").title()
    size_mb = round(path.stat().st_size / (1024 * 1024), 2)
    logger.info(f"Iniciando pipeline de vídeo: '{path.name}' ({size_mb} MB)")

    # 1. Extração e Transcrição de Áudio
    logger.info("Etapa 1/3: Extraindo áudio e transcrevendo com timestamps via Whisper...")
    audio_path = extract_audio_from_video(path)
    transcription_segments: List[Dict[str, Any]] = []
    if audio_path:
        try:
            transcription_segments = transcribe_audio_with_timestamps(
                audio_path,
                model_size=whisper_model_size,
            )
        finally:
            audio_path.unlink(missing_ok=True)
    else:
        # Se ffmpeg não extraiu áudio, tenta transcrição direta se suportado
        try:
            transcription_segments = transcribe_audio_with_timestamps(
                path,
                model_size=whisper_model_size,
            )
        except Exception:
            transcription_segments = []

    logger.info(f"Transcrição concluída: {len(transcription_segments)} segmentos de fala detectados.")

    # 2. Detecção de Cenas
    logger.info("Etapa 2/3: Detectando cortes de cena no vídeo via PySceneDetect...")
    raw_scenes = detect_video_scenes(path, threshold=scene_threshold)
    logger.info(f"Detecção de cenas concluída: {len(raw_scenes)} cenas identificadas.")

    # 3. Construção dos elementos VideoScene
    logger.info(f"Etapa 3/3: Processando keyframes e OCR para {len(raw_scenes)} cenas...")
    elements: List[VideoScene] = []
    video_doc_id = str(uuid.uuid4())

    for idx, (start_sec, end_sec) in enumerate(raw_scenes):
        start_str = format_timestamp(start_sec)
        end_str = format_timestamp(end_sec)
        logger.info(f"  • Processando cena {idx + 1}/{len(raw_scenes)} [{start_str} - {end_str}]...")

        # Midpoint temporal para extrair o keyframe
        mid_sec = (start_sec + end_sec) / 2.0
        keyframe_bytes = extract_keyframe_bytes(path, mid_sec)

        ocr_text = ""
        visual_desc = ""
        storage_uri = None

        if keyframe_bytes:
            ocr_text = extract_text_from_image(keyframe_bytes)
            visual_desc = describe_image_visual(
                keyframe_bytes,
                context_hint=f"Cena {idx + 1} ({start_str} - {end_str}) do vídeo {title}",
            )
            storage_obj_name = f"keyframes/{content_hash[:12]}_scene_{idx:03d}.jpg"
            storage_uri = _upload_keyframe(keyframe_bytes, storage_obj_name)

        # Transcrição associada à janela temporal da cena
        matching_words = []
        matching_texts = []
        has_words = any(seg.get("words") for seg in transcription_segments)

        if has_words:
            for seg in transcription_segments:
                for w in seg.get("words", []):
                    w_mid = (w["start"] + w["end"]) / 2.0
                    if start_sec <= w_mid < end_sec:
                        matching_words.append(w["word"])
            scene_transcript = "".join(matching_words).strip()
            scene_transcript = " ".join(scene_transcript.split())
        else:
            for seg in transcription_segments:
                s_start = seg["start"]
                s_end = seg["end"]
                if s_start < end_sec and s_end > start_sec:
                    matching_texts.append(seg["text"])
            scene_transcript = " ".join(matching_texts).strip()

        # Monta a representação textual combinada multimodal
        content_lines = [
            f"## Cena {idx + 1} ({start_str} - {end_str})",
        ]
        if scene_transcript:
            content_lines.append(f"### Transcrição de Áudio\n{scene_transcript}")
        if visual_desc:
            content_lines.append(f"### Descrição Visual (Keyframe)\n{visual_desc}")
        if ocr_text:
            content_lines.append(f"### Texto na Tela (OCR)\n{ocr_text}")

        combined_content = "\n\n".join(content_lines).strip()

        scene_element = VideoScene(
            element_type=ElementType.VIDEO_SCENE,
            scene_index=idx + 1,
            start_time=start_sec,
            end_time=end_sec,
            start_time_str=start_str,
            end_time_str=end_str,
            transcript=scene_transcript,
            visual_description=visual_desc,
            ocr_text=ocr_text,
            content=combined_content,
            keyframe_storage_uri=storage_uri,
            metadata={
                "video_file": path.name,
                "midpoint_second": mid_sec,
            },
        )
        elements.append(scene_element)

    return Document(
        id=video_doc_id,
        source_uri=source_uri or str(path),
        title=title,
        elements=elements,
        tenant_id=tenant_id,
        content_hash=content_hash,
        parser_version="0.26.0",
        metadata={
            "media_type": "video",
            "scene_count": len(elements),
            "total_scenes": len(elements),
        },
    )
