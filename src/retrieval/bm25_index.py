"""
Índice BM25 via Postgres tsvector — Passo 5.

Indexa os mesmos chunks que estão no Qdrant em uma tabela Postgres com
coluna tsvector, possibilitando busca full-text (BM25-like) via
to_tsquery / plainto_tsquery.

Por que Postgres e não OpenSearch?
- O runbook define Postgres como infra base (já rodando desde o Passo 0).
- Postgres tsvector é suficiente para V1 — ranking BM25 real (Okapi)
  pode entrar depois via extensão pg_bm25 (ParadeDB) se necessário.
- Adia OpenSearch/Elasticsearch para quando o volume justificar.

Schema da tabela `bm25_chunks`:
  chunk_id        TEXT PRIMARY KEY
  document_id     TEXT NOT NULL
  tenant_id       TEXT NOT NULL DEFAULT ''
  content         TEXT NOT NULL
  content_tsv     TSVECTOR          -- coluna indexada
  source_uri      TEXT
  document_title  TEXT
  section_title   TEXT
  chunk_index     INT
  created_at      TIMESTAMPTZ DEFAULT NOW()

Funções:
  create_bm25_table()              -- cria tabela e índice GIN
  index_bm25(chunks)               -- insere/atualiza chunks na tabela
  search_bm25(query, top_k, ...)   -- busca full-text retornando lista de dicts
  clear_bm25_table()               -- limpa a tabela (para re-indexação)
"""

from __future__ import annotations

import time
from typing import List, Optional

# Observabilidade (Passo 14)
try:
    from src.observability.instrumentation import get_current_trace
except ImportError:
    def get_current_trace():
        return None

import psycopg2
import psycopg2.extras
from psycopg2 import errors as pg_errors

from src.chunking.recursive import Chunk
from src.config import settings

# ---------------------------------------------------------------------------
# Conexão (singleton simples — sem pool nesta fase)
# ---------------------------------------------------------------------------

_conn = None


def _get_conn():
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(settings.postgres_url)
        _conn.autocommit = False
    return _conn


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS bm25_chunks (
    chunk_id        TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL,
    tenant_id       TEXT NOT NULL DEFAULT '',
    content         TEXT NOT NULL,
    content_tsv     TSVECTOR,
    source_uri      TEXT,
    document_title  TEXT,
    section_title   TEXT,
    chunk_index     INT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Índice GIN para busca full-text eficiente
CREATE INDEX IF NOT EXISTS bm25_chunks_tsv_idx
    ON bm25_chunks USING GIN (content_tsv);

-- Índice no tenant_id para filtro eficiente
CREATE INDEX IF NOT EXISTS bm25_chunks_tenant_idx
    ON bm25_chunks (tenant_id);

-- Passo 11: ACL — migração idempotente para tabelas já existentes (a tabela
-- pode ter sido criada em uma versão anterior do schema, sem esta coluna).
ALTER TABLE bm25_chunks
    ADD COLUMN IF NOT EXISTS acl_principals TEXT[] NOT NULL DEFAULT '{}';

CREATE INDEX IF NOT EXISTS bm25_chunks_acl_idx
    ON bm25_chunks USING GIN (acl_principals);

-- Passo 17: Parent/Child — suporte a chunk pai e conteúdo expandido
ALTER TABLE bm25_chunks
    ADD COLUMN IF NOT EXISTS parent_chunk_id TEXT;
ALTER TABLE bm25_chunks
    ADD COLUMN IF NOT EXISTS parent_content TEXT;
"""

UPSERT_SQL = """
INSERT INTO bm25_chunks
    (chunk_id, document_id, tenant_id, content, content_tsv,
     source_uri, document_title, section_title, chunk_index, acl_principals,
     parent_chunk_id, parent_content)
VALUES
    (%(chunk_id)s, %(document_id)s, %(tenant_id)s, %(content)s,
     to_tsvector('portuguese', %(content)s),
     %(source_uri)s, %(document_title)s, %(section_title)s, %(chunk_index)s,
     %(acl_principals)s, %(parent_chunk_id)s, %(parent_content)s)
ON CONFLICT (chunk_id) DO UPDATE SET
    content         = EXCLUDED.content,
    content_tsv     = EXCLUDED.content_tsv,
    document_title  = EXCLUDED.document_title,
    section_title   = EXCLUDED.section_title,
    chunk_index     = EXCLUDED.chunk_index,
    acl_principals  = EXCLUDED.acl_principals,
    parent_chunk_id = EXCLUDED.parent_chunk_id,
    parent_content  = EXCLUDED.parent_content;
"""

# Filtro de ACL (Passo 11): "principals = '{}'" (nenhum principal informado)
# pula o filtro — mantém compatibilidade com chamadas antigas sem ACL.
# Caso contrário, exige overlap (&&) entre os principals do usuário e o
# acl_principals do chunk — pré-filtro, roda ANTES do ranking BM25.
SEARCH_SQL = """
SELECT
    chunk_id,
    document_id,
    tenant_id,
    content,
    source_uri,
    document_title,
    section_title,
    chunk_index,
    parent_chunk_id,
    parent_content,
    ts_rank_cd(content_tsv, query) AS score
FROM bm25_chunks,
     plainto_tsquery('portuguese', %(query_text)s) AS query
WHERE content_tsv @@ query
  AND (%(tenant_id)s = '' OR tenant_id = %(tenant_id)s)
  AND (%(principals)s = '{}'::text[] OR acl_principals && %(principals)s)
ORDER BY score DESC
LIMIT %(top_k)s;
"""

SEARCH_SQL_SIMPLE = """
SELECT
    chunk_id,
    document_id,
    tenant_id,
    content,
    source_uri,
    document_title,
    section_title,
    chunk_index,
    parent_chunk_id,
    parent_content,
    ts_rank_cd(content_tsv, query) AS score
FROM bm25_chunks,
     to_tsquery('portuguese', %(query_text)s) AS query
WHERE content_tsv @@ query
  AND (%(tenant_id)s = '' OR tenant_id = %(tenant_id)s)
  AND (%(principals)s = '{}'::text[] OR acl_principals && %(principals)s)
ORDER BY score DESC
LIMIT %(top_k)s;
"""


# ---------------------------------------------------------------------------
# Funções públicas
# ---------------------------------------------------------------------------

def create_bm25_table() -> None:
    """Cria a tabela bm25_chunks e os índices GIN se ainda não existirem."""
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    print("[BM25] Tabela e índices criados (ou já existiam).")


def clear_bm25_table() -> None:
    """Remove todos os registros da tabela (para re-indexação limpa)."""
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM bm25_chunks;")
    conn.commit()
    print("[BM25] Tabela limpa.")


def index_bm25(chunks: List[Chunk], clear_first: bool = False) -> int:
    """
    Insere ou atualiza os chunks na tabela bm25_chunks.

    Parâmetros:
    - chunks      : lista de Chunk (do chunker recursivo).
    - clear_first : se True, limpa a tabela antes de inserir.

    Retorna o número de chunks inseridos.
    """
    create_bm25_table()

    if clear_first:
        clear_bm25_table()

    conn = _get_conn()
    records = [
        {
            "chunk_id":      c.chunk_id,
            "document_id":   c.document_id,
            "tenant_id":     c.tenant_id or "",
            "content":       c.content,
            "source_uri":    c.metadata.get("source_uri", ""),
            "document_title":c.metadata.get("document_title", ""),
            "section_title": c.metadata.get("section_title") or "",
            "chunk_index":   c.metadata.get("chunk_index", 0),
            "acl_principals": [e.get("principal") for e in c.acl if e.get("principal")],
            "parent_chunk_id": c.parent_chunk_id or c.metadata.get("parent_chunk_id"),
            "parent_content": c.metadata.get("parent_content"),
        }
        for c in chunks
    ]

    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, UPSERT_SQL, records, page_size=50)
    conn.commit()

    print(f"[BM25] {len(chunks)} chunk(s) indexado(s) no Postgres.")
    return len(chunks)


def search_bm25(
    query: str,
    top_k: int = 10,
    tenant_id: Optional[str] = None,
    principals: Optional[List[str]] = None,
    use_plain: bool = True,
) -> List[dict]:
    """
    Busca full-text via tsvector no Postgres.

    Parâmetros:
    - query      : texto da pergunta (linguagem natural).
    - top_k      : número de resultados.
    - tenant_id  : filtro de tenant (pré-filtro, na própria query SQL).
    - principals : se fornecido (Passo 11), exige overlap com acl_principals
                   do chunk — mesmo pré-filtro (na query SQL, antes do
                   ranking) usado no canal dense.
    - use_plain  : True usa plainto_tsquery (recomendado para queries naturais);
                   False usa to_tsquery (para termos exatos com operadores).

    Retorna lista de dicts com rank score normalizado.
    """
    from src.retrieval.acl_filter import resolve_principals
    principals = resolve_principals(principals, tenant_id=tenant_id)

    conn = _get_conn()
    sql = SEARCH_SQL if use_plain else SEARCH_SQL_SIMPLE

    params = {
        "query_text": query,
        "tenant_id": tenant_id or "",
        "principals": principals or [],
        "top_k": top_k,
    }

    _t0 = time.time()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        try:
            cur.execute(sql, params)
            rows = cur.fetchall()
        except pg_errors.SyntaxError as e:
            conn.rollback()
            # Fallback: plainto_tsquery sempre funciona; to_tsquery pode falhar
            # com termos que contêm caracteres especiais
            if not use_plain:
                return search_bm25(
                    query, top_k=top_k, tenant_id=tenant_id,
                    principals=principals, use_plain=True,
                )
            raise e
        except psycopg2.Error as e:
            conn.rollback()
            raise e
    _bm25_ms = (time.time() - _t0) * 1000
    trace = get_current_trace()
    if trace is not None:
        trace.record_stage("retrieval.bm25", _bm25_ms, top_k=top_k)

    if not rows and use_plain:
        import re
        words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2 and w not in {"qual", "quais", "onde", "como", "quando", "para", "pela", "pelo", "que", "com", "dos", "das"}]
        if words:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                try:
                    or_terms = " | ".join(words)
                    cur.execute(SEARCH_SQL_SIMPLE, {
                        "query_text": or_terms,
                        "tenant_id": tenant_id or "",
                        "principals": principals or [],
                        "top_k": top_k,
                    })
                    rows = cur.fetchall()
                except Exception:
                    conn.rollback()

    if not rows:
        return []

    # Normaliza score para [0, 1] (ts_rank_cd já é pequeno — multiplica por 10)
    max_score = max(float(r["score"]) for r in rows) or 1.0
    return [
        {
            "chunk_id":        r["chunk_id"],
            "score":           round(float(r["score"]) / max_score, 4),
            "raw_score":       round(float(r["score"]), 6),
            "content":         r["content"],
            "source_uri":      r["source_uri"],
            "document_title":  r["document_title"],
            "section_title":   r["section_title"],
            "parent_chunk_id": r.get("parent_chunk_id"),
            "parent_content":  r.get("parent_content"),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Invalidação seletiva (Passo 12)
# ---------------------------------------------------------------------------

def delete_document(document_id: str) -> int:
    """
    Remove do Postgres todos os chunks de um document_id.

    Usado pela reindexação incremental: ao detectar que um documento mudou,
    seus chunks antigos são removidos antes de indexar os novos — nenhum
    outro documento é tocado.

    Retorna o número de chunks removidos.
    """
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM bm25_chunks WHERE document_id = %s;", (document_id,))
        n = cur.rowcount
    conn.commit()
    return n
