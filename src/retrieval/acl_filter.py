"""
Filtro de ACL/Tenant — Passo 11.

Especificação seguida (definida em data/seed_corpus/07_acl_tenancy.html,
parte do próprio corpus seed): um chunk só é visível para um usuário se

    1. tenant_id do chunk == tenant do usuário; E
    2. pelo menos uma entrada do acl do chunk corresponde a um principal
       do usuário (o próprio user_id, ou um dos grupos/roles dele).

Os dois filtros rodam ANTES do ranking (pré-filtro no Qdrant e na query SQL
do Postgres, não pós-filtro em memória) — filtrar depois é tratado como
risco de segurança: o modelo teria acesso ao chunk mesmo que o resultado
fosse descartado.

Critério de qualidade: zero falsos negativos de segurança. Um usuário sem
permissão nunca deve ver um chunk, mesmo com query adversarial muito
específica (ver tests/test_acl.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.retrieval.hybrid import HybridResult, hybrid_search


@dataclass
class UserContext:
    """
    Contexto de autorização de um usuário — construído pela camada de auth
    da aplicação (fora de escopo aqui) e repassado ao retrieval.

    - tenant_id : tenant/organização do usuário. Filtro primário e obrigatório.
    - user_id   : identificador do usuário (vira o principal "user:<user_id>").
    - groups    : grupos do usuário (cada um vira "grupo:<nome>").
    - roles     : roles do usuário (cada um vira "role:<nome>").
    """

    tenant_id: str
    user_id: str
    groups: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)

    def principals(self) -> List[str]:
        """
        Lista de principals que o Qdrant/Postgres usam para o filtro de ACL —
        mesmo formato usado em ACLEntry.principal (canonical_model.py):
        "user:<id>", "grupo:<nome>", "role:<nome>".
        """
        p = [f"user:{self.user_id}"]
        p += [f"grupo:{g}" for g in self.groups]
        p += [f"role:{r}" for r in self.roles]
        return p


def secure_search(
    query: str,
    user_context: UserContext,
    top_k: int = 5,
    dense_candidates: int = 20,
    bm25_candidates: int = 20,
) -> List[HybridResult]:
    """
    Ponto de entrada único para retrieval com ACL aplicado.

    Não é um wrapper que filtra resultados depois — apenas repassa
    tenant_id e principals para hybrid_search(), que por sua vez repassa
    para search_dense()/search_bm25(), onde o filtro entra na própria
    query (Qdrant Filter / SQL WHERE) antes do ranking.
    """
    return hybrid_search(
        query,
        top_k=top_k,
        tenant_id=user_context.tenant_id,
        principals=user_context.principals(),
        dense_candidates=dense_candidates,
        bm25_candidates=bm25_candidates,
    )
