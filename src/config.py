"""
Configuração central do projeto.

Carrega variáveis do .env e valida na subida da aplicação — se faltar
alguma variável obrigatória, falha aqui, e não no meio de um pipeline
de ingestão em produção.
"""

from typing import Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Ambiente
    environment: str = "development"       # "development" | "production" | "test"

    # LLM & Visão
    anthropic_api_key: str = ""
    anthropic_vision_model: str = "claude-3-5-sonnet-20241022"

    # OCR (Passo 19)
    tesseract_cmd: str = ""

    # Embedding local (BGE-M3)
    embedding_device: str = "cpu"          # "cpu" | "cuda"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: Optional[int] = None  # None = inferido dinamicamente do modelo
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

    # Neo4j (Passo 28)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "rag_neo4j_password"

    # Redis (Passo 32)
    redis_url: str = "redis://localhost:6379/0"

    # Auto-Tuning de Pesos de Fusão (Passo 30)
    fusion_tuned_weights: dict = {
        "dense": 1.2,
        "bm25": 1.0,
        "graph": 0.8,
    }

    # Otimização de Latência e Cache (Passo 32)
    reranker_top_k_candidates: int = 20
    qdrant_ef_search: int = 64
    enable_cache: bool = True
    cache_response_ttl: int = 3600
    cache_embedding_ttl: int = 86400

    # Self-RAG Thresholds (Passo 29)
    self_rag_confidence_threshold: float = 0.0  # Logit >= 0.0 (~50% sigmoid) -> SUFFICIENT
    self_rag_retry_threshold: float = -2.5       # Logit >= -2.5 (~7.5% sigmoid) -> RETRY

    @model_validator(mode="after")
    def validate_production_credentials(self) -> "Settings":
        if self.environment.lower() == "production":
            insecure_defaults = [
                ("postgres_url", "rag_dev_password"),
                ("minio_secret_key", "minioadmin_dev"),
                ("neo4j_password", "rag_neo4j_password"),
            ]
            for field_name, default_val in insecure_defaults:
                val = getattr(self, field_name, "")
                if default_val in val:
                    raise ValueError(
                        f"Configuração insegura detectada em produção: o campo '{field_name}' "
                        f"contém a credencial padrão de desenvolvimento '{default_val}'. "
                        "Defina credenciais seguras no arquivo .env ou variáveis de ambiente."
                    )
        return self


settings = Settings()
