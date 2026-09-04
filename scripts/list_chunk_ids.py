"""
Script para gerar os chunk_ids do corpus seed indexado no Qdrant.
Roda uma vez para descobrir os IDs reais e montar o golden dataset.
"""
import sys

sys.path.insert(0, ".")

from qdrant_client import QdrantClient

from src.config import settings

client = QdrantClient(url=settings.qdrant_url)

# Lista todos os pontos da coleção com payload
offset = None
all_points = []
while True:
    result, offset = client.scroll(
        collection_name=settings.qdrant_collection,
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    all_points.extend(result)
    if offset is None:
        break

print(f"Total de chunks no Qdrant: {len(all_points)}\n")
for p in sorted(all_points, key=lambda x: (x.payload.get("source_uri",""), x.payload.get("chunk_index",0))):
    src = p.payload.get("source_uri","").split("\\")[-1]
    idx = p.payload.get("chunk_index", "?")
    sec = (p.payload.get("section_title") or "")[:50]
    snippet = p.payload.get("content","")[:80].replace("\n"," ")
    print(f"[{src}][{idx:>2}] {p.id}")
    print(f"         sec={sec}")
    print(f"         '{snippet}'")
    print()
