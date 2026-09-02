"""
Reindexação incremental — Passo 12.

Ao editar um documento, apenas os chunks/embeddings/índices derivados dele
são recalculados — não o corpus inteiro. Isso resolve, de forma correta,
um problema documentado no Passo 10: `chunk_id` (src/chunking/recursive.py)
é um `uuid.uuid4()` aleatório a cada chunking, então rodar um reindex
completo (`recreate_collection=True`) trocava os IDs de TUDO, invalidando
o golden dataset inteiro mesmo quando só um documento havia mudado.

Como isso é resolvido aqui:
- `document_id` passa a ser DETERMINÍSTICO por `source_uri`
  (`uuid5(NAMESPACE, source_uri)`), então o mesmo arquivo sempre produz o
  mesmo document_id, não importa quantas vezes for reparseado. `chunk_id`
  continua aleatório (não precisa ser estável — só document_id precisa,
  para servir de chave do grafo de dependências), mas isso deixa de importar
  porque a invalidação agora é feita por document_id, não por chunk_id.
- Uma tabela `document_registry` (Postgres) guarda, por `source_uri`:
  document_id, tenant_id e content_hash da última indexação bem-sucedida —
  o "grafo de dependências" é simplesmente:
      document_registry.content_hash  (detecta mudança)
      bm25_chunks.document_id         (aresta document_id -> chunk_ids)
      chunk_id == Qdrant point id     (aresta chunk_id -> embedding, trivial)
- Fluxo por arquivo:
    1. Parseia o arquivo, calcula content_hash.
    2. Busca o registro anterior por source_uri.
    3. Hash igual -> nada a fazer (status "unchanged").
    4. Hash diferente (ou nunca indexado) -> remove os chunks antigos
       DAQUELE document_id (Qdrant + Postgres), re-chunka e reindexa
       apenas os chunks novos, atualiza o registro.

Funções principais:
    incremental_reindex_file(path, tenant_id)   -> dict (1 arquivo)
    incremental_reindex_corpus(dir, tenant_id)  -> list[dict] (vários arquivos)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional

from src.chunking.recursive import chunk_recursive
from src.embedding.embed_index import embed_and_index
from src.embedding.embed_index import delete_document as _qdrant_delete_document
from src.parsing.simple_parsers import parse_file
from src.retrieval.bm25_index import create_bm25_table, index_bm25, _get_conn
from src.retrieval.bm25_index import delete_document as _bm25_delete_document

# Namespace fixo — garante que uuid5(NAMESPACE, source_uri) seja o mesmo
# document_id sempre, em qualquer máquina/execução.
_DOC_ID_NAMESPACE = uuid.UUID("6b7f7a6e-6e2a-4a2e-9f2a-1f6a5a2c9e10")

REGISTRY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS document_registry (
    source_uri      TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL,
    tenant_id       TEXT NOT NULL DEFAULT '',
    content_hash    TEXT NOT NULL,
    parser_version  TEXT,
    chunk_count     INT NOT NULL DEFAULT 0,
    indexed_at      TIMESTAMPTZ DEFAULT NOW()
);
"""

SUPPORTED_EXTENSIONS = {".txt", ".html", ".htm", ".pdf", ".docx", ".xlsx"}


def stable_document_id(source_uri: str) -> str:
    """document_id determinístico por source_uri — mesma URI, mesmo ID sempre."""
    return str(uuid.uuid5(_DOC_ID_NAMESPACE, source_uri))


def _ensure_registry_table() -> None:
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(REGISTRY_TABLE_SQL)
    conn.commit()


def _get_registry_entry(source_uri: str) -> Optional[dict]:
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT document_id, content_hash, chunk_count FROM document_registry "
            "WHERE source_uri = %s;",
            (source_uri,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {"document_id": row[0], "content_hash": row[1], "chunk_count": row[2]}


def _upsert_registry_entry(
    source_uri: str,
    document_id: str,
    tenant_id: str,
    content_hash: str,
    parser_version: str,
    chunk_count: int,
) -> None:
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO document_registry
                (source_uri, document_id, tenant_id, content_hash, parser_version, chunk_count, indexed_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (source_uri) DO UPDATE SET
                document_id    = EXCLUDED.document_id,
                tenant_id      = EXCLUDED.tenant_id,
                content_hash   = EXCLUDED.content_hash,
                parser_version = EXCLUDED.parser_version,
                chunk_count    = EXCLUDED.chunk_count,
                indexed_at     = NOW();
            """,
            (source_uri, document_id, tenant_id, content_hash, parser_version, chunk_count),
        )
    conn.commit()


def incremental_reindex_file(
    path: str | Path,
    tenant_id: Optional[str] = None,
    max_tokens: int = 512,
    overlap: int = 64,
) -> dict:
    """
    Reindexa um único arquivo de forma incremental.

    Retorna um dict com:
      - status         : "unchanged" | "indexed" (primeira vez) | "reindexed" (mudou)
      - source_uri, document_id, content_hash
      - chunks_removed, chunks_added
    """
    _ensure_registry_table()

    path = Path(path)
    source_uri = str(path)

    doc = parse_file(path, tenant_id=tenant_id, source_uri=source_uri)
    # document_id determinístico por source_uri — ver docstring do módulo.
    doc.id = stable_document_id(source_uri)

    previous = _get_registry_entry(source_uri)

    if previous is not None and previous["content_hash"] == doc.content_hash:
        return {
            "status": "unchanged",
            "source_uri": source_uri,
            "document_id": doc.id,
            "content_hash": doc.content_hash,
            "chunks_removed": 0,
            "chunks_added": 0,
        }

    chunks_removed = 0
    if previous is not None:
        # Documento já existia com outro conteúdo — invalida SÓ os chunks dele.
        chunks_removed_qdrant = _qdrant_delete_document(doc.id)
        chunks_removed_pg = _bm25_delete_document(doc.id)
        chunks_removed = max(chunks_removed_qdrant, chunks_removed_pg)

    chunks = chunk_recursive(doc, max_tokens=max_tokens, overlap=overlap)
    if chunks:
        embed_and_index(chunks, recreate_collection=False)
        index_bm25(chunks, clear_first=False)

    _upsert_registry_entry(
        source_uri=source_uri,
        document_id=doc.id,
        tenant_id=tenant_id or "",
        content_hash=doc.content_hash,
        parser_version=doc.parser_version,
        chunk_count=len(chunks),
    )

    return {
        "status": "indexed" if previous is None else "reindexed",
        "source_uri": source_uri,
        "document_id": doc.id,
        "content_hash": doc.content_hash,
        "chunks_removed": chunks_removed,
        "chunks_added": len(chunks),
    }


def incremental_reindex_corpus(
    directory: str | Path,
    tenant_id: Optional[str] = None,
) -> List[dict]:
    """
    Varre um diretório e reindexa incrementalmente cada arquivo suportado.

    Documentos sem mudança de conteúdo custam apenas 1 parse + 1 SELECT —
    nenhum chunk é tocado. Seguro para rodar repetidamente (idempotente).
    """
    directory = Path(directory)
    files = sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return [incremental_reindex_file(f, tenant_id=tenant_id) for f in files]
