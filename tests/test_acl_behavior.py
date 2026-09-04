"""
Testes comportamentais de ACL e Segurança (Passo 7 e Sprint 2).

Valida em tempo de execução:
1. Comportamento fail-closed: chamadas a buscas com tenant_id sem principals explícitos lançam ValueError.
2. Suporte ao sentinela ANY_PRINCIPAL para fluxos administrativos e benchmarks.
3. Deny-by-default no canal de grafo (chunks sem ACL ou com tenant divergente são rejeitados).
4. Sanitização de metadados em add_entity (chaves reservadas não sobrescrevem atributos).
5. Compatibilidade de index_chunks com ACL em formato de lista de strings e de dicionários.
6. Bloqueio no boot de configurações com senhas padrão quando ENVIRONMENT=production.

Todos os testes são 100% offline e não requerem serviços vivos.
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")

from src.chunking.recursive import Chunk
from src.config import Settings
from src.generation.answer import answer, answer_stream
from src.retrieval.acl_filter import ANY_PRINCIPAL, resolve_principals
from src.retrieval.bm25_index import extract_acl_principals, search_bm25
from src.retrieval.graph_search import (
    Entity,
    KnowledgeGraphIndex,
    search_graph,
)
from src.retrieval.hybrid import hybrid_search
from src.retrieval.query_router import routed_hybrid_search

# ---------------------------------------------------------------------------
# 1. Testes de Fail-Closed nas Fronteiras Públicas de Busca
# ---------------------------------------------------------------------------

class TestFailClosedSearchBoundaries:
    """Valida que todas as funções públicas rejeitam tenant_id sem principals."""

    def test_resolve_principals_fail_closed_by_default(self):
        with pytest.raises(ValueError, match="requer 'principals' expl"):
            resolve_principals(None, tenant_id="tenant-alpha")

    def test_resolve_principals_any_principal_allowed(self):
        resolved = resolve_principals(ANY_PRINCIPAL, tenant_id="tenant-alpha")
        assert resolved is None

    def test_resolve_principals_explicit_list_allowed(self):
        resolved = resolve_principals(["user:alice"], tenant_id="tenant-alpha")
        assert resolved == ["user:alice"]

    def test_hybrid_search_without_principals_raises(self):
        with pytest.raises(ValueError, match="requer 'principals' expl"):
            hybrid_search("query de teste", tenant_id="tenant-alpha", principals=None)

    def test_search_bm25_without_principals_raises(self):
        with pytest.raises(ValueError, match="requer 'principals' expl"):
            search_bm25("query de teste", tenant_id="tenant-alpha", principals=None)

    def test_search_graph_without_principals_raises(self):
        with pytest.raises(ValueError, match="requer 'principals' expl"):
            search_graph("query de teste", tenant_id="tenant-alpha", principals=None)

    def test_routed_hybrid_search_without_principals_raises(self):
        with pytest.raises(ValueError, match="requer 'principals' expl"):
            routed_hybrid_search("query de teste", tenant_id="tenant-alpha", principals=None)

    def test_answer_without_principals_raises(self):
        with pytest.raises(ValueError, match="requer 'principals' expl"):
            answer("query de teste", tenant_id="tenant-alpha", principals=None)

    def test_answer_stream_without_principals_raises(self):
        with pytest.raises(ValueError, match="requer 'principals' expl"):
            list(answer_stream("query de teste", tenant_id="tenant-alpha", principals=None))


# ---------------------------------------------------------------------------
# 2. Testes de Grafo: Deny-By-Default, Sanitização e Formatos de ACL
# ---------------------------------------------------------------------------

class TestGraphSecurityAndRobustness:
    """Valida segurança, sanitização e tolerância a formatos no grafo."""

    def test_add_entity_sanitizes_reserved_metadata(self):
        """Metadados arbitrários não podem sobrescrever name, entity_type ou chunk_id."""
        idx = KnowledgeGraphIndex()
        idx.neo4j_driver = None  # garante in-memory NetworkX

        malicious_entity = Entity(
            name="RAG",
            entity_type="CONCEPT",
            chunk_id="chk-legit",
            metadata={
                "name": "OVERRIDDEN",
                "entity_type": "OVERRIDDEN_TYPE",
                "chunk_id": "OVERRIDDEN_CHUNK",
                "domain": "retrieval",
            },
        )
        idx.add_entity(malicious_entity)

        node_data = idx.graph.nodes["RAG"]
        assert node_data["entity_type"] == "CONCEPT"
        assert node_data["chunk_id"] == "chk-legit"
        assert node_data["domain"] == "retrieval"
        assert "name" not in node_data  # nó é indexado por chave 'RAG'

    def test_index_chunks_accepts_strings_and_dicts_in_acl(self):
        """Valida que index_chunks suporta tanto lista de strings quanto lista de dicts."""
        idx = KnowledgeGraphIndex()
        idx.neo4j_driver = None

        chunks = [
            Chunk(
                chunk_id="chk-str-acl",
                document_id="doc-1",
                content="RAG e busca vetorial integrada ao Postgres.",
                tenant_id="tenant-test",
                acl=["user:alice", "group:eng"],
            ),
            Chunk(
                chunk_id="chk-dict-acl",
                document_id="doc-2",
                content="BM25 e Cross-Encoder para otimização de ranking.",
                tenant_id="tenant-test",
                acl=[{"principal": "user:bob"}, {"principal": "group:ai"}],
            ),
        ]

        # Não deve lançar AttributeError: 'str' object has no attribute 'get'
        idx.index_chunks(chunks)

        meta_1 = idx.chunk_metadata["chk-str-acl"]
        assert meta_1["acl_principals"] == ["user:alice", "group:eng"]

        meta_2 = idx.chunk_metadata["chk-dict-acl"]
        assert meta_2["acl_principals"] == ["user:bob", "group:ai"]

    def test_graph_search_deny_by_default(self):
        """Valida que chunks sem ACL ou de tenant incorreto nunca vazam no grafo."""
        idx = KnowledgeGraphIndex()
        idx.neo4j_driver = None

        chunks = [
            Chunk(
                chunk_id="chk-allowed",
                document_id="doc-1",
                content="Recuperação lexical e busca de textos com bm25.",
                tenant_id="tenant-alpha",
                acl=["user:alice"],
            ),
            Chunk(
                chunk_id="chk-wrong-tenant",
                document_id="doc-2",
                content="Recuperação lexical confidencial com bm25 em outro tenant.",
                tenant_id="tenant-bravo",
                acl=["user:alice"],
            ),
            Chunk(
                chunk_id="chk-no-acl",
                document_id="doc-3",
                content="Recuperação lexical com bm25 sem permissões explicitadas.",
                tenant_id="tenant-alpha",
                acl=[],
            ),
        ]
        idx.index_chunks(chunks)

        # Alice no tenant-alpha busca "bm25"
        results = idx.search_graph(
            query="Como funciona a busca com bm25?",
            top_k=5,
            tenant_id="tenant-alpha",
            principals=["user:alice"],
        )

        result_ids = [r["chunk_id"] for r in results]
        assert "chk-allowed" in result_ids
        assert "chk-wrong-tenant" not in result_ids, "Vazamento inter-tenant no canal de grafo!"
        assert "chk-no-acl" not in result_ids, "Vazamento fail-open de chunk sem ACL no canal de grafo!"


# ---------------------------------------------------------------------------
# 3. Testes do Extrator Seguro de ACL do BM25
# ---------------------------------------------------------------------------

class TestAclExtraction:
    def test_extract_acl_principals_various_formats(self):
        assert extract_acl_principals(None) == []
        assert extract_acl_principals("not-a-list") == []
        assert extract_acl_principals(["p1", "p2"]) == ["p1", "p2"]
        assert extract_acl_principals([{"principal": "p1"}, {"principal": "p2"}]) == ["p1", "p2"]
        assert extract_acl_principals(["p1", {"principal": "p2"}, None, {}]) == ["p1", "p2"]


# ---------------------------------------------------------------------------
# 4. Testes de Bloqueio de Credenciais em Produção
# ---------------------------------------------------------------------------

class TestProductionCredentialsValidation:
    def test_production_rejects_default_dev_passwords(self):
        """Em ambiente de produção, credenciais padrão de dev impedem o boot."""
        with pytest.raises(ValueError, match="Configuração insegura detectada em produção"):
            Settings(
                environment="production",
                postgres_url="postgresql://rag:rag_dev_password@prod-db:5432/rag",
            )

    def test_production_accepts_secure_custom_passwords(self):
        """Em ambiente de produção, credenciais customizadas e seguras são aceitas."""
        cfg = Settings(
            environment="production",
            postgres_url="postgresql://rag:pAssw0rd_Super_Segur4@prod-db:5432/rag",
            minio_secret_key="ChaveSecretaProducao987654",
            neo4j_password="Neo4j_Strong_Prod_Passw0rd",
        )
        assert cfg.environment == "production"
