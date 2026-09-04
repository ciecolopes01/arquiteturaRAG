"""
Script para reindexação incremental de todo o seed_corpus.
"""
import sys

sys.path.insert(0, ".")

from src.ingestion.incremental_reindex import incremental_reindex_corpus

if __name__ == "__main__":
    print("Iniciando reindexação incremental do seed_corpus...")
    results = incremental_reindex_corpus(directory="data/seed_corpus", tenant_id="tenant-test")
    print(f"\nTotal de documentos processados: {len(results)}")
    for r in results:
        name = r["source_uri"].replace("\\", "/").split("/")[-1]
        print(f"  {name:<38} status={r['status']:<10} added={r['chunks_added']} removed={r['chunks_removed']}")
