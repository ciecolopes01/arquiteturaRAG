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

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_id() -> str:
    """Gera um ID UUID4 como string."""
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


# Union de todos os tipos de elemento
DocumentElement = Paragraph | Section | Table | Image | Page


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

    id: str = Field(default_factory=_new_id)
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
