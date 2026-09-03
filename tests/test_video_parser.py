"""
Testes unitários e de integração para o Parser de Vídeo — Passo 26.

Valida:
1. Extração estruturada de cenas (VideoScene) com PySceneDetect.
2. Transcrição de áudio com timestamps sincronizados via faster-whisper.
3. Extração de keyframe, OCR (Tesseract) e descrição visual.
4. Serialização e desserialização JSON no Canonical Document Model.
5. Chunking recursivo multimodal com metadados temporais preservados.
6. Recuperação baseada em perguntas temporais ("em que momento aparece X" -> timestamp correto).
7. Despacho transparente via simple_parsers.parse_file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from src.chunking.recursive import chunk_recursive
from src.parsing.canonical_model import Document, ElementType, VideoScene
from src.parsing.simple_parsers import parse_file
from src.parsing.video_parser import format_timestamp, parse_video

SEED_DIR = Path("data/seed_corpus")
VIDEO_1 = SEED_DIR / "20_visao_computacional_e_multimodal.mp4"
VIDEO_2 = SEED_DIR / "21_monitoramento_video_rag.mp4"


def test_timestamp_formatting():
    """Valida o helper de formatação temporal HH:MM:SS.mmm."""
    assert format_timestamp(0.0) == "00:00:00.000"
    assert format_timestamp(4.5) == "00:00:04.500"
    assert format_timestamp(65.123) == "00:01:05.123"
    assert format_timestamp(3661.0) == "01:01:01.000"


def test_video_parser_extracts_scenes_and_fields():
    """Valida que parse_video extrai cenas com todos os metadados e atributos preenchidos."""
    assert VIDEO_1.exists(), f"Seed video não encontrado: {VIDEO_1}"

    doc = parse_video(VIDEO_1)

    assert doc.id is not None
    assert doc.title == "20 Visao Computacional E Multimodal"
    assert len(doc.elements) >= 2

    for el in doc.elements:
        assert isinstance(el, VideoScene)
        assert el.element_type == ElementType.VIDEO_SCENE
        assert el.scene_index >= 1
        assert el.start_time < el.end_time
        assert ":" in el.start_time_str
        assert ":" in el.end_time_str
        assert len(el.visual_description) > 0
        assert len(el.content) > 0

    # Validações pontuais da cena 1 e 2
    scene_1 = doc.elements[0]
    scene_2 = doc.elements[1]

    assert "INGESTAO" in scene_1.ocr_text.upper() or "INGESTAO" in scene_1.transcript.upper()
    assert "DETECCAO" in scene_2.ocr_text.upper() or "CENAS" in scene_2.transcript.upper()


def test_video_document_serialization():
    """Valida que o Document com VideoScene é serializável e desserializável em JSON."""
    doc = parse_video(VIDEO_1)
    json_str = doc.to_json()
    assert isinstance(json_str, str)
    assert "video_scene" in json_str

    reloaded = Document.from_json(json_str)
    assert reloaded.id == doc.id
    assert len(reloaded.elements) == len(doc.elements)

    reloaded_scene = reloaded.elements[0]
    assert isinstance(reloaded_scene, VideoScene)
    assert reloaded_scene.scene_index == doc.elements[0].scene_index
    assert reloaded_scene.start_time == doc.elements[0].start_time
    assert reloaded_scene.end_time == doc.elements[0].end_time


def test_video_multimodal_chunking():
    """Valida que chunk_recursive produz chunks com metadados temporais de vídeo."""
    doc = parse_video(VIDEO_1)
    chunks = chunk_recursive(doc)

    assert len(chunks) == len(doc.elements)

    for chunk in chunks:
        assert chunk.metadata["element_type"] == "video_scene"
        assert "start_time" in chunk.metadata
        assert "end_time" in chunk.metadata
        assert "start_time_str" in chunk.metadata
        assert "end_time_str" in chunk.metadata
        assert "scene_index" in chunk.metadata
        assert chunk.metadata["start_time"] < chunk.metadata["end_time"]
        assert "Cena" in chunk.content


def test_parse_file_video_dispatcher():
    """Valida que simple_parsers.parse_file roteia arquivos de vídeo (.mp4)."""
    doc = parse_file(VIDEO_2)
    assert isinstance(doc, Document)
    assert len(doc.elements) >= 2
    assert all(isinstance(el, VideoScene) for el in doc.elements)


def test_video_timestamp_retrieval_query():
    """
    Critério de "pronto" do Passo 26:
    Pergunta do tipo 'em que momento aparece X' retorna a cena certa com timestamp.
    """
    doc1 = parse_video(VIDEO_1)
    doc2 = parse_video(VIDEO_2)

    chunks1 = chunk_recursive(doc1)
    chunks2 = chunk_recursive(doc2)
    all_chunks = chunks1 + chunks2

    def rank_chunks(query: str, chunks_list: list) -> list:
        # Busca léxica com scoring por frequência de termos relevantes
        stopwords = {"em", "que", "momento", "aparece", "e", "o", "a", "os", "as", "de", "do", "da"}
        q_tokens = [w.lower().strip(".,!?:;()") for w in query.split() if w.lower().strip(".,!?:;()") not in stopwords and len(w) > 2]
        scores = []
        for c in chunks_list:
            c_lower = c.content.lower()
            score = sum(c_lower.count(t) * 2 for t in q_tokens)
            scores.append((score, c))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [c for score, c in scores if score > 0]

    # Consulta 1: Detecção de cenas e OCR
    results_1 = rank_chunks("em que momento aparece deteccao de cenas e OCR", all_chunks)
    assert len(results_1) > 0
    top_chunk_1 = results_1[0]
    assert top_chunk_1.metadata["element_type"] == "video_scene"
    assert top_chunk_1.metadata["scene_index"] == 2
    assert top_chunk_1.metadata["start_time_str"] == "00:00:04.000"
    assert top_chunk_1.metadata["end_time_str"] == "00:00:08.000"
    assert "DETECCAO DE CENAS" in top_chunk_1.content.upper()

    # Consulta 2: Painel de latência
    results_2 = rank_chunks("em que momento aparece o painel de latencia de retrieval", all_chunks)
    assert len(results_2) > 0
    top_chunk_2 = results_2[0]
    assert top_chunk_2.metadata["element_type"] == "video_scene"
    assert top_chunk_2.metadata["scene_index"] == 1
    assert top_chunk_2.metadata["start_time_str"] == "00:00:00.000"
    assert top_chunk_2.metadata["end_time_str"] == "00:00:04.000"
    assert "LATENCIA" in top_chunk_2.content.upper()

    # Consulta 3: Alerta de degradação de recall
    results_3 = rank_chunks("em que momento e apresentado o alerta de degradacao de recall", all_chunks)
    assert len(results_3) > 0
    top_chunk_3 = results_3[0]
    assert top_chunk_3.metadata["element_type"] == "video_scene"
    assert top_chunk_3.metadata["scene_index"] == 2
    assert top_chunk_3.metadata["start_time_str"] == "00:00:04.000"
    assert top_chunk_3.metadata["end_time_str"] == "00:00:08.000"
    assert "DEGRADACAO DE RECALL" in top_chunk_3.content.upper()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
