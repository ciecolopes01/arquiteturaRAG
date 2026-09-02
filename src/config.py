"""
Configuração central do projeto.

Carrega variáveis do .env e valida na subida da aplicação — se faltar
alguma variável obrigatória, falha aqui, e não no meio de um pipeline
de ingestão em produção.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    anthropic_api_key: str = ""

    # Embedding local (BGE-M3)
    embedding_device: str = "cpu"          # "cpu" | "cuda"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_chunks"

    # Postgres
    postgres_url: str = "postgresql://rag:rag_dev_password@localhost:5432/rag_platform"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin_dev"
    minio_bucket: str = "rag-documents"
    minio_secure: bool = False


settings = Settings()
