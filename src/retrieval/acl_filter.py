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

from dataclasses import dataclass, field
from typing import Any, List, Optional

from src.retrieval.hybrid import HybridResult, hybrid_search


class _AnyPrincipal:
    """Sentinela explícita para busca administrativa/benchmark sem filtro de ACL."""
    __slots__ = ()

    def __repr__(self) -> str:
        return "<ANY_PRINCIPAL>"


ANY_PRINCIPAL = _AnyPrincipal()


def resolve_principals(
    principals: Any,
    tenant_id: Optional[str] = None,
    allow_unrestricted: bool = False,
) -> Optional[List[str]]:
    """
    Resolve e valida a lista de principals para prevenir comportamento fail-open involuntário.

    - Se `principals` for `ANY_PRINCIPAL`: retorna None (acesso deliberadamente irrestrito).
    - Se `principals` for fornecido: retorna cópia em lista.
    - Se `principals` for None e `allow_unrestricted=False` (padrão): levanta ValueError
      se `tenant_id` estiver definido — evita busca sem filtro de ACL por omissão acidental.

    Para busca administrativa intencional sem restrição de ACL, passe `ANY_PRINCIPAL`
    explicitamente. Para contextos sem tenant (ex.: ingestion, benchmark sem multi-tenant),
    chame sem `tenant_id` ou com `allow_unrestricted=True`.
    """
    if principals is ANY_PRINCIPAL:
        return None
    if principals is None:
        if tenant_id and not allow_unrestricted:
            raise ValueError(
                f"Busca para o tenant '{tenant_id}' requer 'principals' explícito. "
                "Para busca administrativa intencional sem restrição de ACL, passe ANY_PRINCIPAL."
            )
        return None
    return list(principals)



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
