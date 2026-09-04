"""
Query Router Completo — Passo 21.

Classifica a intenção da query e aplica routing suave (ajuste dinâmico de pesos
por canal no RRF), garantindo que nenhum canal seja eliminado (min_weight_floor).

Intenções suportadas:
1. FACT_LOOKUP: buscas pontuais, termos específicos, códigos, siglas -> favorece BM25.
2. TABLE_LOOKUP: perguntas quantitativas, financeiras, tabelas, planilhas -> favorece BM25.
3. SEMANTIC_EXPLANATION: explicações conceituais, comparações, "por que", "como funciona" -> favorece Dense.
4. MULTIMODAL: buscas direcionadas a figuras, diagramas, gráficos e fluxogramas -> favorece Dense/Multimodal.

Fórmula RRF Ponderada (§5 da arquitetura revisada):
    score(c) = Σ_i  w_i / (k + rank_i(c))
    onde w_i >= min_weight_floor
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.embedding.embed_index import search_dense
from src.retrieval.bm25_index import search_bm25
from src.retrieval.hybrid import RRF_K, HybridResult, reciprocal_rank_fusion

logger = logging.getLogger("rag.query_router")


# Observabilidade (Passo 14)
try:
    from src.observability.instrumentation import get_current_trace
except ImportError:
    def get_current_trace():
        return None


class QueryIntent(str, Enum):
    FACT_LOOKUP = "fact_lookup"
    TABLE_LOOKUP = "table_lookup"
    SEMANTIC_EXPLANATION = "semantic_explanation"
    MULTIMODAL = "multimodal"
    RELATIONAL = "relational"


# Piso mínimo para qualquer canal (routing suave: nenhum canal cai a zero)
MIN_WEIGHT_FLOOR = 0.10

# Pesos canônicos por intenção [dense, bm25, graph]
DEFAULT_BOOSTS: Dict[QueryIntent, Dict[str, float]] = {
    QueryIntent.FACT_LOOKUP: {
        "dense": 0.8,
        "bm25": 1.2,
        "graph": 0.2,
    },
    QueryIntent.TABLE_LOOKUP: {
        "dense": 0.7,
        "bm25": 1.3,
        "graph": 0.2,
    },
    QueryIntent.SEMANTIC_EXPLANATION: {
        "dense": 1.4,
        "bm25": 0.6,
        "graph": 0.3,
    },
    QueryIntent.MULTIMODAL: {
        "dense": 1.3,
        "bm25": 0.7,
        "graph": 0.2,
    },
    QueryIntent.RELATIONAL: {
        "dense": 0.8,
        "bm25": 0.8,
        "graph": 1.4,
    },
}


# Padrões regex para classificação rápida e determinística de intenção
_TABLE_PATTERNS = re.compile(
    r"\b(quant[oa]s?|qual\s+o\s+valor|pre[çc]o|custo|or[çc]amento|faturamento|receita|"
    r"mrr|arr|ebitda|lucro|churn|dau|mau|tabela|planilha|xlsx|coluna|cota[çc][ãa]o|"
    r"saldo|despesa|hardware|servidor|ram|ssd|vram|usd|r\$|\$)\b",
    re.IGNORECASE,
)

_MULTIMODAL_PATTERNS = re.compile(
    r"\b(em\s+que\s+p[áa]gina\s+aparece|gr[áa]fico|diagrama|figura|imagem|fluxograma|"
    r"infogr[áa]fico|esquema\s+visual|desenho|arquitetura\s+visual|eixo|dpi|resolu[çc][ãa]o)\b",
    re.IGNORECASE,
)

_SEMANTIC_PATTERNS = re.compile(
    r"\b(explique|como\s+funciona|por\s+que|qual\s+o\s+conceito|o\s+que\s+[ée]|vis[ãa]o\s+geral|"
    r"diferen[çc]a\s+entre|vantagen?s?|desvantagen?s?|benef[íi]cios?|princ[íi]pios?|trade-offs?|"
    r"estrat[ée]gia|quando\s+usar|compare|comparad[oa])\b",
    re.IGNORECASE,
)

_RELATIONAL_PATTERNS = re.compile(
    r"\b(relac[aã]o|relacionad[oa]s?|conectad[oa]s?|entidades?|quem\s+comprou|quem\s+integra|"
    r"quais\s+documentos\s+mencionam|interliga|grafo|vizinh[oa]s?|ligad[oa]s?|arquitetura\s+relacional|"
    r"depend[eê]ncia|interconex[aã]o)\b",
    re.IGNORECASE,
)


def classify_intent(query: str) -> QueryIntent:
    """
    Classifica a intenção de busca da query em uma das categorias canônicas.
    Execução instantânea por expressões regulares sem latência de inferência de LLM.
    """
    q = query.strip().lower()

    # 1. Padrões relacionais / grafo (Passo 28)
    if _RELATIONAL_PATTERNS.search(q):
        return QueryIntent.RELATIONAL

    # 2. Padrões multimodais (figuras, gráficos, páginas de imagem)
    if _MULTIMODAL_PATTERNS.search(q):
        return QueryIntent.MULTIMODAL

    # 3. Padrões tabulares e quantitativos
    if _TABLE_PATTERNS.search(q):
        return QueryIntent.TABLE_LOOKUP

    # 4. Padrões explicativos e conceituais
    if _SEMANTIC_PATTERNS.search(q):
        return QueryIntent.SEMANTIC_EXPLANATION

    # 5. Padrão factual / exato por default
    return QueryIntent.FACT_LOOKUP


def get_channel_weights(
    intent: QueryIntent,
    boosts: Optional[Dict[QueryIntent, Dict[str, float]]] = None,
    min_weight_floor: float = MIN_WEIGHT_FLOOR,
) -> Dict[str, float]:
    """
    Retorna os pesos dos canais para a intenção fornecida, aplicando
    o piso mínimo (min_weight_floor) para garantir routing suave.
    """
    table = boosts or DEFAULT_BOOSTS
    raw_weights = table.get(intent, {"dense": 1.0, "bm25": 1.0})

    # Aplica o piso mínimo
    return {
        channel: max(weight, min_weight_floor)
        for channel, weight in raw_weights.items()
    }


def weighted_rrf(
    dense_hits: List[Any],
    bm25_hits: List[Any],
    weights: Dict[str, float],
    graph_hits: Optional[List[Any]] = None,
    k: int = RRF_K,
    top_k: int = 5,
) -> List[HybridResult]:
    """
    Executa fusão RRF com pesos diferenciados por canal:
        score(chunk) = Σ_i  w_i / (k + rank_i(chunk))

    Reutiliza internamente a implementação canônica de hybrid.py.
    """
    ranked_lists: Dict[str, List[Dict[str, Any]]] = {
        "dense": dense_hits or [],
        "bm25": bm25_hits or [],
    }
    if graph_hits:
        ranked_lists["graph"] = graph_hits

    return reciprocal_rank_fusion(
        ranked_lists=ranked_lists,
        top_k=top_k,
        k=k,
        weights=weights,
    )



def routed_hybrid_search(
    query: str,
    top_k: int = 5,
    tenant_id: Optional[str] = None,
    principals: Optional[List[str]] = None,
    boosts: Optional[Dict[QueryIntent, Dict[str, float]]] = None,
    candidate_pool_size: int = 20,
    use_graph: bool = True,
) -> Tuple[List[HybridResult], QueryIntent, Dict[str, float]]:
    """
    Executa busca híbrida com Query Router:
    1. Classifica a intenção da query.
    2. Obtém pesos ajustados para os canais (routing suave com min_weight_floor).
    3. Recupera candidatos do Qdrant, Postgres e opcionalmente Neo4j/Graph (com pré-filtro de tenant e ACL).
    4. Funde com RRF ponderado.
    5. Retorna resultados, intenção detectada e pesos aplicados.
    """
    t0 = time.perf_counter()
    from src.retrieval.acl_filter import resolve_principals
    principals = resolve_principals(principals, tenant_id=tenant_id)

    intent = classify_intent(query)
    weights = get_channel_weights(intent, boosts=boosts)

    dense_hits: List[dict] = []
    bm25_hits: List[dict] = []
    graph_hits: List[dict] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f_dense = executor.submit(
            search_dense,
            query=query,
            top_k=candidate_pool_size,
            tenant_id=tenant_id,
            principals=principals,
        )
        f_bm25 = executor.submit(
            search_bm25,
            query=query,
            top_k=candidate_pool_size,
            tenant_id=tenant_id,
            principals=principals,
        )
        f_graph = None
        if use_graph:
            try:
                from src.retrieval.graph_search import search_graph
                f_graph = executor.submit(
                    search_graph,
                    query=query,
                    top_k=candidate_pool_size,
                    tenant_id=tenant_id,
                    principals=principals,
                )
            except Exception as e:
                logger.warning("Falha ao agendar busca em grafo (%s)", e)

        try:
            dense_hits = f_dense.result()
        except Exception as e:
            logger.warning("Busca densa falhou (%s); prosseguindo com demais canais", e)
            dense_hits = []

        try:
            bm25_hits = f_bm25.result()
        except Exception as e:
            logger.warning("Busca BM25 falhou (%s); prosseguindo com demais canais", e)
            bm25_hits = []

        if f_graph is not None:
            try:
                graph_hits = f_graph.result()
            except Exception as e:
                logger.warning("Busca em grafo falhou (%s); prosseguindo com dense + bm25", e)
                graph_hits = []
                trace = get_current_trace()
                if trace:
                    trace.metadata["graph_channel_degraded"] = True


    fused = weighted_rrf(
        dense_hits=dense_hits,
        bm25_hits=bm25_hits,
        graph_hits=graph_hits,
        weights=weights,
        top_k=top_k,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    trace = get_current_trace()
    if trace:
        trace.metadata["router_intent"] = intent.value
        trace.metadata["router_weights"] = weights
        trace.metadata["router_latency_ms"] = elapsed_ms

    return fused, intent, weights
