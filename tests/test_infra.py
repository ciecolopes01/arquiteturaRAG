"""
Teste de infraestrutura — Passo 0.

Confirma que os 3 serviços (Qdrant, Postgres, MinIO) estão de pé e
acessíveis antes de qualquer código de pipeline ser escrito.

Rodar com: python tests/test_infra.py
(depois do `docker-compose up -d` e do venv com requirements.txt instalado)
"""

import sys

sys.path.insert(0, ".")  # permite rodar da raiz do projeto

from src.config import settings


def test_qdrant() -> bool:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url)
        client.get_collections()
        print("[OK] Qdrant respondeu em", settings.qdrant_url)
        return True
    except Exception as e:
        print("[FALHOU] Qdrant:", e)
        print("  -> verifique se o container 'rag_qdrant' está rodando: docker ps")
        return False


def test_postgres() -> bool:
    try:
        import psycopg2

        conn = psycopg2.connect(settings.postgres_url)
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        cur.fetchone()
        cur.close()
        conn.close()
        print("[OK] Postgres respondeu em", settings.postgres_url.split("@")[-1])
        return True
    except Exception as e:
        print("[FALHOU] Postgres:", e)
        print("  -> verifique se o container 'rag_postgres' está rodando e saudável: docker ps")
        return False


def test_minio() -> bool:
    try:
        from minio import Minio

        client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        found = client.bucket_exists(settings.minio_bucket)
        if not found:
            client.make_bucket(settings.minio_bucket)
            print(f"[OK] MinIO respondeu — bucket '{settings.minio_bucket}' criado agora")
        else:
            print(f"[OK] MinIO respondeu — bucket '{settings.minio_bucket}' já existe")
        return True
    except Exception as e:
        print("[FALHOU] MinIO:", e)
        print("  -> verifique se o container 'rag_minio' está rodando: docker ps")
        return False


def main() -> None:
    print("Testando infraestrutura local...\n")
    results = {
        "Qdrant": test_qdrant(),
        "Postgres": test_postgres(),
        "MinIO": test_minio(),
    }

    print("\nResumo:")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FALHOU'}")

    if all(results.values()):
        print("\nPasso 0 concluído — infraestrutura pronta.")
        sys.exit(0)
    else:
        print("\nPasso 0 incompleto — corrija os serviços acima antes de seguir para o Passo 1.")
        sys.exit(1)


if __name__ == "__main__":
    main()
