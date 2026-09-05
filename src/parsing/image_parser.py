"""
Parser de Imagens (OCR + Descrição Visual) — Passo 19.

Trata imagem como um elemento de primeira classe do Canonical Document Model.
Combina:
1. OCR Local: extração de texto via Tesseract OCR (open source, sem custo de API).
2. Descrição Visual: resumo do conteúdo gráfico/diagramático via API da Anthropic
   (Claude com suporte a visão). Quando a API key não está configurada, utiliza
   um fallback simulado determinístico estruturado para manter testes offline.
3. Object Storage: salva imagens brutas no MinIO (bucket configurado) e registra
   a URI canônica correspondente.
4. Extratores: extração de imagens embutidas em páginas de PDF (PyMuPDF) e em
   documentos DOCX (python-docx).
"""

from __future__ import annotations

import base64
import io
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image as PILImage

from src.config import settings
from src.parsing.canonical_model import Document, Image

# ---------------------------------------------------------------------------
# Configuração e detecção do Tesseract OCR
# ---------------------------------------------------------------------------

_TESSERACT_INITIALIZED = False
_TESSERACT_AVAILABLE = False


def find_tesseract_binary() -> Optional[str]:
    """Localiza o executável do Tesseract no ambiente ou caminhos comuns."""
    global _TESSERACT_INITIALIZED, _TESSERACT_AVAILABLE

    # 1. Configuração explícita
    if settings.tesseract_cmd and os.path.exists(settings.tesseract_cmd):
        return settings.tesseract_cmd

    # 2. Variável de ambiente
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and os.path.exists(env_cmd):
        return env_cmd

    # 3. PATH do sistema
    which_cmd = shutil.which("tesseract")
    if which_cmd:
        return which_cmd

    # 4. Caminhos comuns no Windows
    win_paths = [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for p in win_paths:
        if os.path.exists(p):
            return p

    # 5. Caminhos comuns no Linux / macOS
    nix_paths = [
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
    ]
    for p in nix_paths:
        if os.path.exists(p):
            return p

    return None


def _init_tesseract() -> bool:
    """Configura pytesseract com o caminho correto se disponível."""
    global _TESSERACT_INITIALIZED, _TESSERACT_AVAILABLE
    if _TESSERACT_INITIALIZED:
        return _TESSERACT_AVAILABLE

    _TESSERACT_INITIALIZED = True
    try:
        import pytesseract

        cmd = find_tesseract_binary()
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
            _TESSERACT_AVAILABLE = True
        else:
            _TESSERACT_AVAILABLE = False
    except ImportError:
        _TESSERACT_AVAILABLE = False

    return _TESSERACT_AVAILABLE


# ---------------------------------------------------------------------------
# OCR (Tesseract)
# ---------------------------------------------------------------------------

def extract_text_from_image(
    image_input: Union[bytes, PILImage.Image],
    lang: str = "por+eng",
) -> str:
    """
    Executa OCR sobre a imagem utilizando Tesseract via pytesseract.

    Parâmetros:
    - image_input: bytes da imagem ou instância PIL.Image.
    - lang: idiomas para o Tesseract (default: 'por+eng').

    Retorna string com o texto extraído, ou string vazia em caso de falha/ausência.
    """
    if not _init_tesseract():
        return ""

    try:
        import pytesseract

        if isinstance(image_input, bytes):
            img = PILImage.open(io.BytesIO(image_input))
        else:
            img = image_input

        # Converte para RGB se necessário (ex: RGBA ou paleta)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        try:
            text = pytesseract.image_to_string(img, lang=lang)
        except Exception:
            # Fallback se 'por' não estiver instalado no pacote de idiomas
            try:
                text = pytesseract.image_to_string(img, lang="eng")
            except Exception:
                text = pytesseract.image_to_string(img)

        return text.strip()
    except Exception as e:
        print(f"[image_parser] Aviso ao executar OCR: {e}")
        return ""


# ---------------------------------------------------------------------------
# Descrição Visual (Anthropic Claude Vision / Fallback)
# ---------------------------------------------------------------------------

DEFAULT_VISION_PROMPT = (
    "Descreva objetiva e detalhadamente o conteúdo visual desta imagem, "
    "destacando: tipo de figura (gráfico, diagrama de arquitetura, fluxo, fotografia), "
    "título, legendas, eixos, valores numéricos centrais e conclusões ilustradas. "
    "Mantenha o foco em detalhes técnicos úteis para busca e recuperação (RAG)."
)


_CLAUDE_VISION_DISABLED = False

def describe_image_visual(
    image_bytes: bytes,
    prompt: Optional[str] = None,
    media_type: str = "image/png",
    context_hint: Optional[str] = None,
) -> str:
    """
    Gera descrição visual semântica de uma imagem utilizando Claude 3.5 Vision.
    Se a chave não estiver configurada ou falhar por crédito/rede, gera descrição
    estruturada com base nas propriedades da imagem e metadados.
    """
    global _CLAUDE_VISION_DISABLED
    if settings.anthropic_api_key and not _CLAUDE_VISION_DISABLED:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            b64_data = base64.b64encode(image_bytes).decode("utf-8")

            valid_types = {"image/jpeg", "image/png", "image/gif", "image/webp"}
            mt = media_type.lower()
            if mt == "image/jpg":
                mt = "image/jpeg"
            if mt not in valid_types:
                mt = "image/png"

            user_prompt = prompt or DEFAULT_VISION_PROMPT
            if context_hint:
                user_prompt += f"\nContexto do documento: {context_hint}"

            response = client.messages.create(
                model=getattr(settings, "anthropic_vision_model", "claude-3-5-sonnet-20241022"),
                max_tokens=600,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mt,
                                    "data": b64_data,
                                },
                            },
                            {"type": "text", "text": user_prompt},
                        ],
                    }
                ],
            )
            text_blocks = [b.text for b in response.content if hasattr(b, "text")]
            return "\n".join(text_blocks).strip()
        except Exception as e:
            err_str = str(e)
            if "credit balance is too low" in err_str or "400" in err_str or "401" in err_str:
                _CLAUDE_VISION_DISABLED = True
                print("[image_parser] Anthropic API sem créditos ou indisponível. Desativando chamadas remotas de visão e ativando fallback estruturado ultra-rápido.")
            else:
                print(f"[image_parser] Erro ao chamar Claude Vision: {e}. Usando descrição simulada.")

    # --- Fallback simulado estruturado para ambiente offline/dev ---
    try:
        img = PILImage.open(io.BytesIO(image_bytes))
        width, height = img.size
        mode = img.mode
    except Exception:
        width, height, mode = 0, 0, "UNKNOWN"

    desc_parts = [
        f"Imagem técnica ({width}x{height} pixels, modo {mode}).",
    ]
    if context_hint:
        desc_parts.append(f"Associada à seção/contexto: '{context_hint}'.")
    desc_parts.append(
        "Apresenta esquema visual, diagrama arquitetural ou gráfico ilustrativo "
        "relevante para a documentação técnica do sistema."
    )
    return " ".join(desc_parts)


# ---------------------------------------------------------------------------
# Object Storage (MinIO)
# ---------------------------------------------------------------------------

def upload_image_to_storage(
    image_bytes: bytes,
    image_id: str,
    ext: str = "png",
) -> Optional[str]:
    """
    Salva os bytes da imagem no MinIO sob o caminho: images/{image_id}.{ext}.
    Retorna a URI canônica minio://... ou None em caso de falha.
    """
    import socket
    endpoint = settings.minio_endpoint or "localhost:9000"
    try:
        host, port_str = endpoint.split(":", 1) if ":" in endpoint else (endpoint, "9000")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.15)
        sock.connect((host, int(port_str)))
        sock.close()
    except Exception:
        return None

    try:
        from minio import Minio

        client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        if not client.bucket_exists(settings.minio_bucket):
            client.make_bucket(settings.minio_bucket)

        clean_ext = ext.lstrip(".").lower() or "png"
        object_name = f"images/{image_id}.{clean_ext}"
        media_type = f"image/{clean_ext}" if clean_ext != "jpg" else "image/jpeg"

        client.put_object(
            bucket_name=settings.minio_bucket,
            object_name=object_name,
            data=io.BytesIO(image_bytes),
            length=len(image_bytes),
            content_type=media_type,
        )
        return f"minio://{settings.minio_bucket}/{object_name}"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Montagem do Conteúdo Canônico da Imagem
# ---------------------------------------------------------------------------

def build_image_content(
    ocr_text: str,
    visual_description: str,
    caption: Optional[str] = None,
    page_number: Optional[int] = None,
    section_title: Optional[str] = None,
) -> str:
    """
    Constrói a representação textual combinada do elemento Image para fins
    de indexação lexical (BM25) e embedding semântico (BGE-M3).
    """
    parts = []

    header = "[Imagem"
    if caption:
        header += f": {caption.strip()}"
    if page_number is not None:
        header += f" | Página {page_number}"
    header += "]"
    parts.append(header)

    if section_title:
        parts.append(f"Contexto: {section_title.strip()}")

    if visual_description and visual_description.strip():
        parts.append(f"## Descrição Visual\n{visual_description.strip()}")

    if ocr_text and ocr_text.strip():
        parts.append(f"## Texto Extraído (OCR)\n{ocr_text.strip()}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Processamento Completo de uma Imagem
# ---------------------------------------------------------------------------

def process_image(
    image_bytes: bytes,
    ext: str = "png",
    page_number: Optional[int] = None,
    section_title: Optional[str] = None,
    caption: Optional[str] = None,
    upload_storage: bool = True,
    metadata_extra: Optional[Dict[str, Any]] = None,
) -> Image:
    """
    Executa o fluxo completo do Passo 19 sobre bytes de imagem:
    1. Calcula dimensões via Pillow.
    2. Roda OCR local (Tesseract).
    3. Chama modelo de visão (Claude Vision ou fallback simulado).
    4. Sobe para MinIO se habilitado.
    5. Popula e retorna o schema Image canônico.
    """
    image_id = str(uuid.uuid4())

    width: Optional[int] = None
    height: Optional[int] = None
    try:
        with PILImage.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
    except Exception as e:
        print(f"[image_parser] Aviso ao abrir imagem com Pillow: {e}")

    # 1. OCR local
    ocr_text = extract_text_from_image(image_bytes)

    # 2. Descrição visual
    context_hint = f"Página {page_number}, Seção '{section_title}'" if (page_number and section_title) else section_title
    media_type = f"image/{ext.lstrip('.').lower()}"
    if media_type == "image/jpg":
        media_type = "image/jpeg"
    visual_description = describe_image_visual(
        image_bytes=image_bytes,
        media_type=media_type,
        context_hint=context_hint,
    )

    # 3. Object Storage
    storage_uri = None
    if upload_storage:
        storage_uri = upload_image_to_storage(image_bytes, image_id=image_id, ext=ext)

    # 4. Conteúdo textual combinado
    content = build_image_content(
        ocr_text=ocr_text,
        visual_description=visual_description,
        caption=caption,
        page_number=page_number,
        section_title=section_title,
    )

    metadata: Dict[str, Any] = {
        "page_number": page_number,
        "section_title": section_title,
        "format": ext.lstrip(".").lower(),
        "byte_size": len(image_bytes),
    }
    if caption:
        metadata["caption"] = caption
    if metadata_extra:
        metadata.update(metadata_extra)

    return Image(
        id=image_id,
        content=content,
        storage_uri=storage_uri,
        width=width,
        height=height,
        ocr_text=ocr_text,
        visual_description=visual_description,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Extratores de Imagens em PDF e DOCX
# ---------------------------------------------------------------------------

MIN_IMAGE_DIMENSION = 50


def extract_images_from_pdf_page(
    page: Any,
    pdf_doc: Any,
    page_number: int,
    section_title: Optional[str] = None,
    upload_storage: bool = True,
) -> List[Image]:
    """
    Extrai imagens embutidas de uma página de PDF via PyMuPDF (fitz).
    Filtra pequenas decorações (<50x50) e processa com OCR + Visão.
    """
    extracted: List[Image] = []
    try:
        images_info = page.get_images(full=True)
    except Exception as e:
        print(f"[image_parser] Erro ao obter imagens da página {page_number}: {e}")
        return []

    seen_xrefs = set()
    for img_idx, img_info in enumerate(images_info):
        xref = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        try:
            base_image = pdf_doc.extract_image(xref)
            if not base_image:
                continue

            img_bytes = base_image.get("image", b"")
            img_ext = base_image.get("ext", "png")
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)

            # Ignora ícones pequenos
            if (width and width < MIN_IMAGE_DIMENSION) or (height and height < MIN_IMAGE_DIMENSION):
                continue

            image_element = process_image(
                image_bytes=img_bytes,
                ext=img_ext,
                page_number=page_number,
                section_title=section_title,
                caption=f"Figura {img_idx + 1} (página {page_number})",
                upload_storage=upload_storage,
                metadata_extra={"xref": xref, "pdf_image_index": img_idx},
            )
            extracted.append(image_element)
        except Exception as e:
            print(f"[image_parser] Erro ao processar imagem xref={xref} pág {page_number}: {e}")

    return extracted


def extract_images_from_docx(
    word_doc_or_path: Any,
    section_title: Optional[str] = None,
    upload_storage: bool = True,
) -> List[Image]:
    """
    Extrai imagens embutidas de um documento DOCX (python-docx).
    Inspeciona os relacionamentos de partes de imagem do documento.
    """
    extracted: List[Image] = []

    try:
        import docx

        if isinstance(word_doc_or_path, (str, Path)):
            word_doc = docx.Document(str(word_doc_or_path))
        else:
            word_doc = word_doc_or_path

        part = word_doc.part
        if not hasattr(part, "rels"):
            return []

        img_counter = 0
        for rel_id, rel in part.rels.items():
            reltype = getattr(rel, "reltype", "") or ""
            target_ref = getattr(rel, "target_ref", "") or ""
            if "image" in reltype.lower() or "image" in target_ref.lower():
                try:
                    target_part = rel.target_part
                    img_bytes = target_part.blob
                    partname = target_part.partname
                    ext = Path(partname).suffix.lstrip(".") or "png"

                    with PILImage.open(io.BytesIO(img_bytes)) as pimg:
                        w, h = pimg.size
                        if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
                            continue

                    img_counter += 1
                    img_elem = process_image(
                        image_bytes=img_bytes,
                        ext=ext,
                        section_title=section_title,
                        caption=f"Figura DOCX {img_counter}",
                        upload_storage=upload_storage,
                        metadata_extra={"docx_rel_id": rel_id, "docx_partname": partname},
                    )
                    extracted.append(img_elem)
                except Exception as e:
                    print(f"[image_parser] Erro ao processar imagem DOCX rel={rel_id}: {e}")

    except Exception as e:
        print(f"[image_parser] Erro ao extrair imagens do DOCX: {e}")

    return extracted


# ---------------------------------------------------------------------------
# Parser Direto para Arquivos de Imagem Isolados (.png, .jpg, etc.)
# ---------------------------------------------------------------------------

def parse_image_file(
    path: Union[str, Path],
    tenant_id: Optional[str] = None,
    source_uri: Optional[str] = None,
) -> Document:
    """
    Lê um arquivo de imagem diretamente e produz um Document canônico
    contendo um único elemento Image processado.
    """
    import hashlib

    path = Path(path)
    raw_bytes = path.read_bytes()
    source = source_uri or str(path)
    ext = path.suffix.lstrip(".") or "png"

    img_elem = process_image(
        image_bytes=raw_bytes,
        ext=ext,
        caption=f"Imagem {path.name}",
        metadata_extra={"filename": path.name},
    )

    doc = Document(
        source_uri=source,
        title=f"Imagem: {path.stem}",
        tenant_id=tenant_id,
        content_hash=hashlib.sha256(raw_bytes).hexdigest(),
        parser_version="image-parser-0.1",
        language="pt",
        metadata={
            "original_filename": path.name,
            "parser": "parse_image_file",
            "width": img_elem.width,
            "height": img_elem.height,
        },
    )
    doc.elements.append(img_elem)
    return doc
