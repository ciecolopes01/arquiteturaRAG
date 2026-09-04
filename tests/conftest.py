"""
Configuração global de testes e fixtures de disponibilidade de infraestrutura.
"""

import socket

import pytest

from src.config import settings


def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


@pytest.fixture(scope="session")
def qdrant_available() -> bool:
    """Verifica se o Qdrant está respondendo na porta configurada."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(settings.qdrant_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6333
        return _is_port_open(host, port)
    except Exception:
        return False


@pytest.fixture(scope="session")
def postgres_available() -> bool:
    """Verifica se o Postgres está respondendo na porta configurada."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(settings.postgres_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5432
        return _is_port_open(host, port)
    except Exception:
        return False


@pytest.fixture(scope="session")
def neo4j_available() -> bool:
    """Verifica se o Neo4j está respondendo na porta configurada."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(settings.neo4j_uri)
        host = parsed.hostname or "localhost"
        port = parsed.port or 7687
        return _is_port_open(host, port)
    except Exception:
        return False


@pytest.fixture
def require_qdrant(qdrant_available):
    if not qdrant_available:
        pytest.skip("Qdrant não está disponível localmente")


@pytest.fixture
def require_postgres(postgres_available):
    if not postgres_available:
        pytest.skip("Postgres não está disponível localmente")


@pytest.fixture
def require_neo4j(neo4j_available):
    if not neo4j_available:
        pytest.skip("Neo4j não está disponível localmente")
