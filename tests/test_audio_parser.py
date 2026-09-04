"""
Testes unitários e de integração para o Parser de Áudio — Passo 27.

Valida:
1. Extração estruturada de segmentos de áudio (AudioSegment) via faster-whisper.
2. Segmentação baseada em pausas naturais e pontuação de fala com timestamps precisos.
3. Serialização e desserialização JSON no Canonical Document Model.
4. Chunking recursivo com metadados temporais de áudio preservados.
5. Despacho transparente de extensões de áudio (.mp3, .wav) via simple_parsers.parse_file.
6. Recuperação sobre conteúdo de áudio do corpus seed ("pergunta retorna o segmento certo").
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from src.chunking.recursive import chunk_recursive
from src.parsing.audio_parser import parse_audio
from src.parsing.canonical_model import AudioSegment, Document, ElementType
from src.parsing.simple_parsers import parse_file

SEED_DIR = Path("data/seed_corpus")
AUDIO_1 = SEED_DIR / "22_podcast_governanca_rag.mp3"
AUDIO_2 = SEED_DIR / "23_reuniao_arquitetura_reranking.mp3"


def test_audio_parser_extracts_segments():
    """Valida que parse_audio extrai segmentos com atributos temporais e transcrição."""
    assert AUDIO_1.exists(), f"Seed audio não encontrado: {AUDIO_1}"

    doc = parse_audio(AUDIO_1)

    assert doc.id is not None
    assert doc.title == "22 Podcast Governanca Rag"
    assert len(doc.elements) >= 2

    for el in doc.elements:
        assert isinstance(el, AudioSegment)
        assert el.element_type == ElementType.AUDIO_SEGMENT
        assert el.segment_index >= 1
        assert el.start_time < el.end_time
        assert ":" in el.start_time_str
        assert ":" in el.end_time_str
        assert len(el.transcript) > 0
        assert len(el.content) > 0


def test_audio_document_serialization():
    """Valida que o Document com AudioSegment é serializável e desserializável em JSON."""
    doc = parse_audio(AUDIO_1)
    json_str = doc.to_json()
    assert isinstance(json_str, str)
    assert "audio_segment" in json_str

    reloaded = Document.from_json(json_str)
    assert reloaded.id == doc.id
    assert len(reloaded.elements) == len(doc.elements)

    reloaded_seg = reloaded.elements[0]
    assert isinstance(reloaded_seg, AudioSegment)
    assert reloaded_seg.segment_index == doc.elements[0].segment_index
    assert reloaded_seg.start_time == doc.elements[0].start_time
    assert reloaded_seg.end_time == doc.elements[0].end_time


def test_audio_multimodal_chunking():
    """Valida que chunk_recursive produz chunks com metadados temporais de áudio."""
    doc = parse_audio(AUDIO_1)
    chunks = chunk_recursive(doc)

    assert len(chunks) == len(doc.elements)

    for chunk in chunks:
        assert chunk.metadata["element_type"] == "audio_segment"
        assert "start_time" in chunk.metadata
        assert "end_time" in chunk.metadata
        assert "start_time_str" in chunk.metadata
        assert "end_time_str" in chunk.metadata
        assert "segment_index" in chunk.metadata
        assert chunk.metadata["start_time"] < chunk.metadata["end_time"]
        assert "Trecho" in chunk.content


def test_parse_file_audio_dispatcher():
    """Valida que simple_parsers.parse_file roteia arquivos de áudio (.mp3)."""
    doc = parse_file(AUDIO_2)
    assert isinstance(doc, Document)
    assert len(doc.elements) >= 2
    assert all(isinstance(el, AudioSegment) for el in doc.elements)


def test_audio_content_retrieval_query():
    """
    Critério de "pronto" do Passo 27:
    Pergunta sobre o conteúdo de um áudio específico do corpus seed retorna o segmento certo.
    """
    doc1 = parse_audio(AUDIO_1)
    doc2 = parse_audio(AUDIO_2)

    chunks1 = chunk_recursive(doc1)
    chunks2 = chunk_recursive(doc2)
    all_chunks = chunks1 + chunks2

    def rank_chunks(query: str, chunks_list: list) -> list:
        stopwords = {"qual", "o", "a", "os", "as", "de", "do", "da", "em", "no", "na", "que", "para", "por", "e", "um", "uma"}
        q_tokens = [w.lower().strip(".,!?:;()") for w in query.split() if w.lower().strip(".,!?:;()") not in stopwords and len(w) > 2]
        scores = []
        for c in chunks_list:
            c_lower = c.content.lower()
            score = sum(c_lower.count(t) * 2 for t in q_tokens)
            scores.append((score, c))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [c for score, c in scores if score > 0]

    # Consulta 1: Meta de latência de 400 milissegundos para o reranker
    results_1 = rank_chunks("qual a meta de latencia de 400 milissegundos para o reranker", all_chunks)
    assert len(results_1) > 0
    top_1 = results_1[0]
    assert top_1.metadata["element_type"] == "audio_segment"
    assert "Reuniao Arquitetura Reranking" in top_1.metadata["document_title"]
    assert "400" in top_1.content or "lat" in top_1.content.lower()

    # Consulta 2: Validação de controle de acesso (ACL) e pré-filtro
    results_2 = rank_chunks("validacao de controle de acesso pre-filtro para impedir vazamento", all_chunks)
    assert len(results_2) > 0
    top_2 = results_2[0]
    assert top_2.metadata["element_type"] == "audio_segment"
    assert "Podcast Governanca Rag" in top_2.metadata["document_title"]
    assert "controle de acesso" in top_2.content.lower() or "confidenciais" in top_2.content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
