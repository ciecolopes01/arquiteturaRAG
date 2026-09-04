"""
Backfill do document_registry (Passo 12) para o corpus seed real.

Os documentos já indexados nos Passos 4-11 usam um `document_id` aleatório
(gerado no parse original de cada um). O Passo 12 introduz um `document_id`
determinístico por source_uri (uuid5), necessário para a reindexação
incremental saber "quais chunks pertencem a este arquivo" de forma estável.

Este script faz a migração SEM tocar em chunk_id/vetores/conteúdo:
1. Para cada arquivo do corpus seed, calcula o novo document_id determinístico.
2. Descobre o(s) document_id(s) atualmente em uso para aquele source_uri.
3. Faz um patch de payload (Qdrant set_payload) e um UPDATE (Postgres) trocando
   só o campo document_id dos chunks existentes — mesma técnica usada no
   backfill de acl_principals do Passo 11.
4. Registra a entrada em document_registry com o content_hash atual.

Depois deste backfill, `incremental_reindex_corpus("data/seed_corpus")`
reporta "unchanged" para os 12 arquivos (nada é reprocessado) e passa a
ser a forma seguro de reindexar o corpus real dali em diante — evita o
risco (documentado no Passo 10) de rodar tests/test_embedding.py ou
tests/test_bm25.py com recreate_collection=True/clear_first=True, que
troca os chunk_ids de TUDO e invalida o golden dataset inteiro.

Rodar com: python scripts/backfill_document_registry.py
"""

import sys
from pathlib import Path

sys.path.insert(0, ".")

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.config import settings
from src.ingestion.incremental_reindex import (
    _ensure_registry_table,
    _upsert_registry_entry,
    stable_document_id,
)
from src.parsing.simple_parsers import parse_file
from src.retrieval.bm25_index import _get_conn, create_bm25_table

CORPUS_DIR = Path("data/seed_corpus")
TENANT_ID = "tenant-test"


def main() -> None:
    create_bm25_table()
    _ensure_registry_table()

    conn = _get_conn()
    qc = QdrantClient(url=settings.qdrant_url)

    files = sorted(
        list(CORPUS_DIR.glob("*.txt")) + list(CORPUS_DIR.glob("*.html"))
        + list(CORPUS_DIR.glob("*.pdf")) + list(CORPUS_DIR.glob("*.docx"))
    )

    for f in files:
        source_uri = str(f)
        doc = parse_file(f, tenant_id=TENANT_ID, source_uri=source_uri)
        new_doc_id = stable_document_id(source_uri)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT document_id FROM bm25_chunks WHERE source_uri = %s;",
                (source_uri,),
            )
            old_ids = [r[0] for r in cur.fetchall()]

        if not old_ids:
            print(f"[SKIP] {f.name}: nenhum chunk indexado ainda (rode a indexação primeiro)")
            continue

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE bm25_chunks SET document_id = %s WHERE source_uri = %s;",
                (new_doc_id, source_uri),
            )
            n_pg = cur.rowcount
        conn.commit()

        for old_id in old_ids:
            qc.set_payload(
                collection_name=settings.qdrant_collection,
                payload={"document_id": new_doc_id},
                points=Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=old_id))]),
            )

        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM bm25_chunks WHERE document_id = %s;", (new_doc_id,)
            )
            chunk_count = cur.fetchone()[0]

        _upsert_registry_entry(
            source_uri=source_uri,
            document_id=new_doc_id,
            tenant_id=TENANT_ID,
            content_hash=doc.content_hash,
            parser_version=doc.parser_version,
            chunk_count=chunk_count,
        )

        print(f"[OK] {f.name}: {len(old_ids)} document_id(s) antigo(s) -> {new_doc_id[:8]}... ({n_pg} chunks Postgres)")

    print(f"\nBackfill concluído: {len(files)} documento(s) registrados em document_registry.")


if __name__ == "__main__":
    main()
