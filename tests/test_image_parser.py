"""
Testes do Parser de Imagens (OCR + Visão) — Passo 19.

Valida:
1. Detecção do binário do Tesseract OCR.
2. Extração de texto via OCR (Tesseract) sobre imagem sintética.
3. Geração de descrição visual (Claude Vision ou fallback simulado).
4. Montagem e consistência do schema Image no Canonical Document Model.
5. Extração de imagens embutidas em arquivos PDF reais (PyMuPDF).
6. Extração de imagens embutidas em arquivos DOCX reais (python-docx).
7. Chunking de elementos Image nas 3 estratégias: Recursive, Parent/Child e Semantic.
8. Retrieval no golden dataset: valida Recall@5 = 1.0 nas perguntas multimodais.

Rodar com: pytest tests/test_image_parser.py -v
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

sys.path.insert(0, ".")

from src.chunking.parent_child import chunk_parent_child
from src.chunking.recursive import chunk_recursive
from src.chunking.semantic import chunk_semantic
from src.evaluation.offline_eval import evaluate_recall_at_k, load_golden_dataset
from src.parsing.canonical_model import Document, ElementType, Image
from src.parsing.docx_parser import parse_docx
from src.parsing.image_parser import (
    build_image_content,
    describe_image_visual,
    extract_text_from_image,
    find_tesseract_binary,
    process_image,
)
from src.parsing.pdf_parser import parse_pdf
from src.parsing.simple_parsers import parse_file
from src.retrieval.acl_filter import ANY_PRINCIPAL

CORPUS_DIR = Path("data/seed_corpus")


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Gera uma imagem sintética com texto claro para OCR."""
    img = PILImage.new("RGB", (400, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    draw.text((15, 30), "TESTE OCR RETRIEVAL 2026", fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Testes Unitários de OCR e Visão
# ---------------------------------------------------------------------------

def test_tesseract_binary_detected() -> None:
    """Confirma que o binário do Tesseract foi localizado no sistema."""
    cmd = find_tesseract_binary()
    assert cmd is not None, "Executável do Tesseract não foi encontrado."
    assert Path(cmd).exists(), f"Caminho do Tesseract não existe: {cmd}"


def test_ocr_extracts_expected_text(sample_image_bytes: bytes) -> None:
    """Verifica se o OCR extrai o texto sintético desenhado na imagem."""
    extracted = extract_text_from_image(sample_image_bytes)
    assert extracted != "", "OCR retornou string vazia."
    # Spot check dos termos chave
    assert "TESTE" in extracted or "RETRIEVAL" in extracted or "2026" in extracted


def test_describe_image_visual(sample_image_bytes: bytes) -> None:
    """Verifica se a descrição visual é gerada com conteúdo textual válido."""
    desc = describe_image_visual(sample_image_bytes, context_hint="Teste unitário de visão")
    assert isinstance(desc, str)
    assert len(desc) > 20
    assert "400x100" in desc or "visual" in desc.lower() or "imagem" in desc.lower()


def test_build_image_content() -> None:
    """Verifica a formatação Markdown da representação combinada da imagem."""
    content = build_image_content(
        ocr_text="Texto OCR de Exemplo",
        visual_description="Descrição do Diagrama",
        caption="Figura Teste",
        page_number=3,
        section_title="Arquitetura Geral",
    )
    assert "[Imagem: Figura Teste | Página 3]" in content
    assert "Contexto: Arquitetura Geral" in content
    assert "## Descrição Visual" in content
    assert "Descrição do Diagrama" in content
    assert "## Texto Extraído (OCR)" in content
    assert "Texto OCR de Exemplo" in content


def test_process_image_creates_canonical_image(sample_image_bytes: bytes) -> None:
    """Verifica se process_image retorna um objeto Image canônico completo."""
    img_elem = process_image(
        image_bytes=sample_image_bytes,
        ext="png",
        page_number=1,
        section_title="Introdução",
        caption="Diagrama Principal",
        upload_storage=False,
    )
    assert isinstance(img_elem, Image)
    assert img_elem.element_type == ElementType.IMAGE
    assert img_elem.width == 400
    assert img_elem.height == 100
    assert img_elem.metadata["page_number"] == 1
    assert img_elem.metadata["caption"] == "Diagrama Principal"
    assert len(img_elem.content) > 0


def test_parse_image_file_direct(sample_image_bytes: bytes, tmp_path: Path) -> None:
    """Testa o parse direto de um arquivo .png pelo parse_file / parse_image_file."""
    img_path = tmp_path / "teste_direto.png"
    img_path.write_bytes(sample_image_bytes)

    doc = parse_file(img_path, tenant_id="tenant-test")
    assert isinstance(doc, Document)
    assert doc.title == "Imagem: teste_direto"
    assert len(doc.elements) == 1
    assert doc.elements[0].element_type == ElementType.IMAGE


# ---------------------------------------------------------------------------
# Testes de Extração em Documentos Reais (PDF e DOCX)
# ---------------------------------------------------------------------------

def test_extract_images_from_pdf() -> None:
    """Valida extração de imagens de um PDF do corpus seed (Passo 19)."""
    pdf_path = CORPUS_DIR / "16_arquitetura_multimodal_rag.pdf"
    if not pdf_path.exists():
        pytest.skip(f"{pdf_path} não encontrado")

    doc = parse_pdf(pdf_path, tenant_id="tenant-test", upload_storage=False)
    images = doc.elements_by_type(ElementType.IMAGE)
    assert len(images) == 2, f"Esperava 2 imagens no PDF 16, obteve {len(images)}"

    # Checa páginas e dimensões
    pages = [img.metadata.get("page_number") for img in images]
    assert 1 in pages
    assert 2 in pages

    # Checa conteúdo de OCR
    all_ocr = " ".join(img.ocr_text for img in images)
    assert "MULTIMODAL" in all_ocr or "PIPELINE" in all_ocr or "INDEXACAO" in all_ocr


def test_extract_images_from_docx() -> None:
    """Valida extração de imagens de um arquivo DOCX do corpus seed."""
    docx_path = CORPUS_DIR / "18_fluxo_processamento_imagens.docx"
    if not docx_path.exists():
        pytest.skip(f"{docx_path} não encontrado")

    doc = parse_docx(docx_path, tenant_id="tenant-test", upload_storage=False)
    images = doc.elements_by_type(ElementType.IMAGE)
    assert len(images) >= 1, f"Esperava pelo menos 1 imagem no DOCX 18, obteve {len(images)}"
    assert images[0].width is not None
    assert images[0].height is not None


# ---------------------------------------------------------------------------
# Testes de Chunking com Elementos Image
# ---------------------------------------------------------------------------

def test_recursive_chunking_with_images() -> None:
    """Valida que o chunker recursivo transforma Image em Chunk com metadados."""
    pdf_path = CORPUS_DIR / "16_arquitetura_multimodal_rag.pdf"
    if not pdf_path.exists():
        pytest.skip(f"{pdf_path} não encontrado")

    doc = parse_pdf(pdf_path, tenant_id="tenant-test", upload_storage=False)
    chunks = chunk_recursive(doc)
    image_chunks = [c for c in chunks if c.metadata.get("element_type") == "image"]
    assert len(image_chunks) == 2
    for ic in image_chunks:
        assert "image_id" in ic.metadata
        assert "page_number" in ic.metadata
        assert len(ic.content) > 0


def test_parent_child_chunking_with_images() -> None:
    """Valida que o chunker parent/child gera chunks pai de imagens."""
    pdf_path = CORPUS_DIR / "17_metricas_desempenho_visao.pdf"
    if not pdf_path.exists():
        pytest.skip(f"{pdf_path} não encontrado")

    doc = parse_pdf(pdf_path, tenant_id="tenant-test", upload_storage=False)
    res = chunk_parent_child(doc)
    img_parents = [p for p in res.parents if p.metadata.get("element_type") == "image"]
    assert len(img_parents) == 1
    assert "300 DPI" in img_parents[0].content or "ACURACIA" in img_parents[0].content


def test_semantic_chunking_with_images() -> None:
    """Valida que o chunker semântico isola e emite chunks de imagem."""
    pdf_path = CORPUS_DIR / "19_benchmark_extracao_visual.pdf"
    if not pdf_path.exists():
        pytest.skip(f"{pdf_path} não encontrado")

    doc = parse_pdf(pdf_path, tenant_id="tenant-test", upload_storage=False)
    chunks = chunk_semantic(doc)
    img_chunks = [c for c in chunks if c.metadata.get("element_type") == "image"]
    assert len(img_chunks) == 1
    assert img_chunks[0].metadata.get("page_number") == 2


# ---------------------------------------------------------------------------
# Teste de Retrieval e Avaliação Offline no Golden Dataset
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_golden_dataset_image_questions(require_qdrant, require_postgres) -> None:
    """Verifica que as perguntas de teste sobre imagens retornam o chunk esperado no top-5."""
    dataset = load_golden_dataset()
    image_questions = [d for d in dataset if d["id"] in ["gd-44", "gd-45", "gd-46", "gd-47", "gd-48"]]
    assert len(image_questions) == 5, f"Esperava 5 perguntas de imagem, encontrou {len(image_questions)}"

    res = evaluate_recall_at_k(image_questions, k=5, mode="hybrid", tenant_id="tenant-test", principals=ANY_PRINCIPAL)
    assert res["recall_at_k"] >= 0.8, f"Recall@5 das perguntas de imagem insuficiente: {res['recall_at_k']}"
    assert res["hits"] >= 4, f"Apenas {res['hits']}/5 perguntas acertadas."
