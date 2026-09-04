"""
Teste de regressão de segurança via AST (Abstract Syntax Tree).
Garante que todas as funções de busca, recuperação, geração e avaliação
do projeto declarem explicitamente 'tenant_id' e 'principals' em suas assinaturas.
"""

import ast
from pathlib import Path

import pytest

from src.retrieval.acl_filter import ANY_PRINCIPAL, resolve_principals

# Mapeamento de arquivo relativo -> funções públicas que devem exigir tenant_id e principals
RETRIEVAL_FUNCTIONS = {
    "src/embedding/embed_index.py": ["search_dense"],
    "src/retrieval/bm25_index.py": ["search_bm25"],
    "src/retrieval/hybrid.py": ["hybrid_search"],
    "src/retrieval/graph_search.py": ["search_graph"],
    "src/retrieval/query_router.py": ["routed_hybrid_search"],
    "src/retrieval/hyde.py": ["hyde_search", "hyde_hybrid_search"],
    "src/retrieval/decomposition.py": ["decomposed_search"],
    "src/representation/multi_vector.py": ["search_multi_vector", "multi_vector_hybrid_search"],
    "src/generation/answer.py": ["answer", "answer_stream"],
    "src/generation/self_rag.py": ["adaptive_answer"],
    "src/evaluation/offline_eval.py": ["_retrieve", "evaluate_recall_at_k"],
}


def _get_function_params(file_path: Path, func_name: str) -> list[str]:
    """Lê a AST do arquivo e retorna a lista de nomes de parâmetros da função."""
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(file_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return [arg.arg for arg in node.args.args]
    raise ValueError(f"Função '{func_name}' não encontrada no arquivo {file_path}")


@pytest.mark.security
@pytest.mark.parametrize(
    "rel_path,func_name",
    [(path, func) for path, funcs in RETRIEVAL_FUNCTIONS.items() for func in funcs],
)
def test_function_has_tenant_and_principals_params(rel_path: str, func_name: str):
    root_dir = Path(__file__).parent.parent
    file_path = root_dir / rel_path
    assert file_path.exists(), f"Arquivo {file_path} não encontrado"

    params = _get_function_params(file_path, func_name)
    assert "tenant_id" in params, (
        f"Função '{func_name}' em '{rel_path}' NÃO possui parâmetro 'tenant_id'. "
        f"Parâmetros encontrados: {params}"
    )
    assert "principals" in params, (
        f"Função '{func_name}' em '{rel_path}' NÃO possui parâmetro 'principals'. "
        f"Parâmetros encontrados: {params}"
    )


@pytest.mark.security
def test_acl_sentinel_and_resolution():
    """Testa se o sentinela ANY_PRINCIPAL e resolve_principals funcionam corretamente."""
    # ANY_PRINCIPAL representa acesso público deliberado
    assert repr(ANY_PRINCIPAL) == "<ANY_PRINCIPAL>"

    # Lista normal
    res1 = resolve_principals(["user:alice", "group:devs"])
    assert res1 == ["user:alice", "group:devs"]

    # None com allow_unrestricted=True retorna None
    res2 = resolve_principals(None, allow_unrestricted=True)
    assert res2 is None

    # ANY_PRINCIPAL explícito resolve para None (sem filtro)
    res3 = resolve_principals(ANY_PRINCIPAL)
    assert res3 is None

    # String única vira lista
    res4 = resolve_principals(["user:bob"])
    assert res4 == ["user:bob"]

    # Fail-closed quando tenant_id é passado sem principals e allow_unrestricted=False
    with pytest.raises(ValueError, match="requer 'principals' explícito"):
        resolve_principals(None, tenant_id="tenant_123", allow_unrestricted=False)

