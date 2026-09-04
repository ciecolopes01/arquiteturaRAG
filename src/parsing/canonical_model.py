"""
Canonical Document Model — Passo 1.

Representa qualquer documento (TXT, HTML, PDF, DOCX…) como um grafo de
elementos tipados.  Todo parser downstream produz instâncias desses schemas;
todo pipeline downstream consome apenas esses schemas.

Decisões de design:
- Pydantic v2 para validação e serialização JSON automática.
- Cada elemento tem um `id` UUID estável gerado na criação.
- `metadata` é um dict livre — parsers adicionam campos específicos sem
  alterar o schema central.
- `tenant_id` e `acl` vivem no nível do Document; elementos herdam via
  contexto no pipeline (não precisam ser repetidos em cada chunk).
- `dependency_graph_id` reservado para o grafo de reindexação incremental
  (Passo 12) — por enquanto é opcional e None.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Helpers — ID determinístico de documentos
# ---------------------------------------------------------------------------

# Namespace raiz para documentos deste projeto (UUIDv5 fixo, nunca muda)
DOC_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "arquiteturaRAG.documents")


def document_id_for(source_uri: str, content_hash: str = "") -> str:
    """
    Gera um ID estável UUIDv5 para um documento a partir de source_uri e
    content_hash.

    Por que UUIDv5 e não uuid4()?
    - UUIDv5 é determinístico: o mesmo (source_uri, content_hash) sempre
      produz o mesmo ID.
    - Isso permite reindexações incrementais sem criar pontos duplicados no
      Qdrant e garante que os expected_chunk_ids do golden dataset continuem
      válidos entre execuções.

    Se content_hash for vazio (ex.: durante testes sem leitura real do arquivo),
    o ID ainda é estável dado o source_uri — suficiente para identificar
    documentos sem conteúdo ainda computado.
    """
    seed = f"{source_uri}:{content_hash}" if content_hash else source_uri
    return str(uuid.uuid5(DOC_NAMESPACE, seed))


def _new_id() -> str:
    """
    Fallback para elementos que não são Document (Paragraph, Section, Table…).
    Esses elementos não têm source_uri próprio, então uuid4 é aceitável.
    Documentos (classe Document) usam document_id_for() via model_validator.
    """
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Tipos de elemento
# ---------------------------------------------------------------------------

class ElementType(str, Enum):
    PARAGRAPH = "paragraph"
    SECTION = "section"
    TABLE = "table"
    IMAGE = "image"
    PAGE = "page"
    VIDEO_SCENE = "video_scene"
    AUDIO_SEGMENT = "audio_segment"


# ---------------------------------------------------------------------------
# Elementos atômicos
# ---------------------------------------------------------------------------

class Paragraph(BaseModel):
    """Bloco de texto contínuo — o elemento mais comum."""

    id: str = Field(default_factory=_new_id)
    element_type: ElementType = ElementType.PARAGRAPH
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dependency_graph_id: Optional[str] = None

    model_config = {"frozen": False}


class Section(BaseModel):
    """Seção lógica do documento (ex: capítulo, heading H2)."""

    id: str = Field(default_factory=_new_id)
    element_type: ElementType = ElementType.SECTION
    title: str
    level: int = 1                        # 1 = H1, 2 = H2, …
    content: str = ""                     # texto do cabeçalho em si, se houver
    children: List[str] = Field(default_factory=list)  # IDs dos elementos filhos
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dependency_graph_id: Optional[str] = None


class Table(BaseModel):
    """Tabela — nesta fase guardada como texto Markdown; V2 adiciona estrutura."""

    id: str = Field(default_factory=_new_id)
    element_type: ElementType = ElementType.TABLE
    content: str = ""                     # representação texto/markdown da tabela
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    caption: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dependency_graph_id: Optional[str] = None


class Image(BaseModel):
    """Imagem — Passo 19: contém OCR local e descrição visual."""

    id: str = Field(default_factory=_new_id)
    element_type: ElementType = ElementType.IMAGE
    content: str = ""                     # representação textual combinada (ocr + visual_description)
    storage_uri: Optional[str] = None    # URI no MinIO / S3
    width: Optional[int] = None
    height: Optional[int] = None
    ocr_text: str = ""                   # Texto extraído via OCR (Passo 19)
    visual_description: str = ""         # Descrição visual gerada por LLM (Passo 19)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dependency_graph_id: Optional[str] = None


class Page(BaseModel):
    """Página física — relevante para PDF; agrupa elementos de uma página."""

    id: str = Field(default_factory=_new_id)
    element_type: ElementType = ElementType.PAGE
    page_number: int
    content: str = ""                     # texto bruto da página concatenado
    element_ids: List[str] = Field(default_factory=list)  # IDs dos elementos da página
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dependency_graph_id: Optional[str] = None


class VideoScene(BaseModel):
    """Cena de vídeo com transcrição de áudio, descrição visual do keyframe e OCR — Passo 26."""

    id: str = Field(default_factory=_new_id)
    element_type: ElementType = ElementType.VIDEO_SCENE
    scene_index: int = 0
    start_time: float = 0.0              # segundos de início
    end_time: float = 0.0                # segundos de término
    start_time_str: str = "00:00:00"     # timestamp formatado HH:MM:SS ou MM:SS
    end_time_str: str = "00:00:00"       # timestamp formatado HH:MM:SS ou MM:SS
    transcript: str = ""                 # transcrição do áudio correspondente à cena
    visual_description: str = ""         # descrição visual do keyframe (Claude Vision ou fallback)
    ocr_text: str = ""                   # texto na tela extraído via OCR no keyframe
    content: str = ""                    # representação textual combinada multimodal
    keyframe_storage_uri: Optional[str] = None # URI do keyframe no MinIO / storage
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dependency_graph_id: Optional[str] = None

    model_config = {"frozen": False}


class AudioSegment(BaseModel):
    """Segmento de áudio com transcrição e timestamps — Passo 27."""

    id: str = Field(default_factory=_new_id)
    element_type: ElementType = ElementType.AUDIO_SEGMENT
    segment_index: int = 0
    start_time: float = 0.0              # segundos de início
    end_time: float = 0.0                # segundos de término
    start_time_str: str = "00:00:00"     # timestamp formatado HH:MM:SS ou MM:SS
    end_time_str: str = "00:00:00"       # timestamp formatado HH:MM:SS ou MM:SS
    transcript: str = ""                 # transcrição do trecho via Whisper
    content: str = ""                    # texto estruturado para chunking e busca
    speaker: Optional[str] = None        # orador identificado ou None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dependency_graph_id: Optional[str] = None

    model_config = {"frozen": False}


# Union de todos os tipos de elemento
DocumentElement = Paragraph | Section | Table | Image | Page | VideoScene | AudioSegment


# ---------------------------------------------------------------------------
# ACL
# ---------------------------------------------------------------------------

class ACLEntry(BaseModel):
    """Entrada de controle de acesso.

    Exemplo:
        ACLEntry(principal="grupo:engenharia", permission="read")
        ACLEntry(principal="user:joao@empresa.com", permission="read")
    """

    principal: str        # "user:<id>", "grupo:<nome>", "role:<nome>"
    permission: str = "read"   # "read" | "write" | "admin"


# ---------------------------------------------------------------------------
# Document — raiz do modelo
# ---------------------------------------------------------------------------

class Document(BaseModel):
    """
    Representa um documento completo no Canonical Document Model.

    Campos obrigatórios:
    - source_uri: de onde o documento foi lido (path local, URI MinIO, URL)
    - elements: lista ordenada de elementos do documento

    Campos de ACL (preenchidos desde o ingestion):
    - tenant_id: identifica o tenant dono do documento
    - acl: lista de entradas de controle de acesso

    Campos de rastreabilidade:
    - content_hash: SHA-256 do conteúdo bruto (para reindexação incremental)
    - parser_version: versão do parser que gerou este Document
    """

    id: str = Field(default="", description="ID determinístico gerado via document_id_for()")
    source_uri: str
    title: Optional[str] = None

    # Elementos ordenados do documento
    elements: List[DocumentElement] = Field(default_factory=list)

    # ACL — obrigatório em produção; opcionais nesta fase dev para facilitar testes
    tenant_id: Optional[str] = None
    acl: List[ACLEntry] = Field(default_factory=list)

    # Rastreabilidade
    content_hash: Optional[str] = None   # SHA-256 do conteúdo bruto
    parser_version: str = "0.1.0"
    language: Optional[str] = None       # ISO 639-1: "pt", "en", …

    # Metadados livres (autor, data de criação, tags, etc.)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}

    @model_validator(mode="after")
    def _compute_deterministic_id(self) -> "Document":
        """
        Gera o ID do documento de forma determinística a partir de source_uri
        e content_hash usando UUIDv5.

        Por que aqui e não no default_factory?
        - default_factory roda antes de os outros campos serem preenchidos,
          então não teria acesso a source_uri nem content_hash.
        - O model_validator 'after' roda com o modelo já construído.

        Se o ID já foi fornecido explicitamente (ex.: carregando do JSON),
        ele é preservado — sem sobrescrever.
        """
        if not self.id:
            self.id = document_id_for(
                self.source_uri,
                self.content_hash or "",
            )
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_full_text(self) -> str:
        """Concatena o conteúdo de todos os elementos em texto plano."""
        parts: List[str] = []
        for el in self.elements:
            if isinstance(el, Section):
                parts.append(el.title)
                if el.content:
                    parts.append(el.content)
            elif hasattr(el, "content") and el.content:
                parts.append(el.content)
        return "\n\n".join(parts)

    def elements_by_type(self, element_type: ElementType) -> List[DocumentElement]:
        """Retorna todos os elementos de um tipo específico."""
        return [el for el in self.elements if el.element_type == element_type]

    def to_json(self) -> str:
        """Serializa o Document para JSON (UTF-8, com indentação)."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Document":
        """Desserializa um Document a partir de JSON."""
        return cls.model_validate_json(json_str)
