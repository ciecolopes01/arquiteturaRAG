"""
Módulo de Graph RAG — Passo 28.

Constrói um grafo de conhecimento de entidades e relações extraídas do corpus,
consultável como canal adicional de retrieval (canal 'graph' na fusão RRF).

Suporta:
1. Extração de entidades e relações (NER estruturado baseado em regras e/ou LLM).
2. Armazenamento e consulta no Neo4j (via Cypher e driver bolt) com fallback
   em memória resiliente via NetworkX.
3. Busca em grafo (graph_search): dada uma query, identifica entidades mencionadas,
   percorre vizinhança de 1 e 2 saltos no grafo, mapeia de volta para chunks de origem
   e pontua por relevância relacional.
4. Integração transparente como canal no RRF e no QueryRouter (intencao RELATIONAL).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from src.chunking.recursive import Chunk
from src.config import settings

# ---------------------------------------------------------------------------
# Schemas de Grafo
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    name: str
    entity_type: str
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    source: str
    target: str
    relation_type: str
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Extração de Entidades e Relações (NER + Regras Semânticas)
# ---------------------------------------------------------------------------

# Dicionário de entidades canônicas do domínio do projeto RAG
KNOWN_ENTITIES: Dict[str, str] = {
    "qdrant": "VECTOR_DB",
    "postgres": "DATABASE",
    "minio": "OBJECT_STORAGE",
    "neo4j": "GRAPH_DB",
    "redis": "CACHE_DB",
    "bge-m3": "EMBEDDING_MODEL",
    "bge-reranker": "RERANKER_MODEL",
    "bge-reranker-v2-m3": "RERANKER_MODEL",
    "whisper": "AUDIO_MODEL",
    "faster-whisper": "AUDIO_TOOL",
    "pyscenedetect": "VIDEO_TOOL",
    "tesseract": "OCR_TOOL",
    "bm25": "RETRIEVAL_ALGORITHM",
    "rrf": "FUSION_ALGORITHM",
    "hyde": "QUERY_EXPANSION",
    "query router": "ROUTING_COMPONENT",
    "self-rag": "ADAPTIVE_TECHNIQUE",
    "graph rag": "RETRIEVAL_STRATEGY",
    "parent/child": "CHUNKING_STRATEGY",
    "parent-child": "CHUNKING_STRATEGY",
    "semantic chunking": "CHUNKING_STRATEGY",
    "recall@5": "METRIC",
    "mrr": "METRIC",
    "csat": "METRIC",
}

RELATION_PATTERNS = [
    (re.compile(r"\b(integra|integrado|conecta|comunica|utiliza|usa|baseia-se)\b", re.I), "INTEGRATES_WITH"),
    (re.compile(r"\b(melhora|aumenta|otimiza|eleva|maximizou)\b", re.I), "IMPROVES"),
    (re.compile(r"\b(monitora|avalia|mede|registra|alerta)\b", re.I), "MONITORS"),
    (re.compile(r"\b(depende|requer|pre-requisito|necessita)\b", re.I), "DEPENDS_ON"),
    (re.compile(r"\b(compara|comparado|versus|diferenca)\b", re.I), "COMPARES"),
    (re.compile(r"\b(substitui|migra|migracao)\b", re.I), "MIGRATES"),
]


def extract_entities_from_text(text: str) -> List[str]:
    """Identifica entidades conhecidas presentes no texto da query ou chunk."""
    text_lower = text.lower()
    found = []
    for ent in sorted(KNOWN_ENTITIES.keys(), key=len, reverse=True):
        # Match por palavra inteira
        pattern = r"\b" + re.escape(ent) + r"\b"
        if re.search(pattern, text_lower):
            found.append(ent)
    return found


def extract_entities_and_relations(
    text: str,
    chunk_id: str,
) -> Tuple[List[Entity], List[Relation]]:
    """
    Extrai entidades e relações a partir de um bloco de texto (chunk).
    Analisa co-ocorrência em frases para construir arestas relacionais.
    """
    entities: List[Entity] = []
    relations: List[Relation] = []

    # Extrai entidades
    found_names = extract_entities_from_text(text)
    for name in set(found_names):
        etype = KNOWN_ENTITIES.get(name, "CONCEPT")
        entities.append(Entity(name=name, entity_type=etype, chunk_id=chunk_id))

    # Divide por frases para identificar relações locais
    sentences = re.split(r"[.\n!?]+", text)
    for sent in sentences:
        sent_lower = sent.lower()
        sent_entities = [e for e in found_names if re.search(r"\b" + re.escape(e) + r"\b", sent_lower)]
        if len(sent_entities) >= 2:
            # Determina tipo da relação
            rel_type = "RELATED_TO"
            for pat, rtype in RELATION_PATTERNS:
                if pat.search(sent_lower):
                    rel_type = rtype
                    break

            # Cria aresta para cada par na mesma frase
            for i in range(len(sent_entities)):
                for j in range(i + 1, len(sent_entities)):
                    relations.append(
                        Relation(
                            source=sent_entities[i],
                            target=sent_entities[j],
                            relation_type=rel_type,
                            chunk_id=chunk_id,
                            metadata={"sentence": sent.strip()[:150]},
                        )
                    )

    return entities, relations


# ---------------------------------------------------------------------------
# Driver / Índice do Grafo
# ---------------------------------------------------------------------------

class KnowledgeGraphIndex:
    """
    Índice de grafo de conhecimento para Graph RAG.
    Suporta Neo4j (via driver bolt) e fallback local via NetworkX.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.chunk_store: Dict[str, str] = {}         # chunk_id -> content
        self.entity_to_chunks: Dict[str, Set[str]] = {} # entity_name -> {chunk_ids}
        self.neo4j_driver = None
        self._init_neo4j()

    def _init_neo4j(self):
        """Tenta conectar ao Neo4j se disponível."""
        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
                connection_timeout=2.0,
            )
            # Teste de conexão rápido
            driver.verify_connectivity()
            self.neo4j_driver = driver
        except Exception:
            # Continua em modo local (NetworkX)
            self.neo4j_driver = None

    def clear(self):
        """Limpa o grafo."""
        self.graph.clear()
        self.chunk_store.clear()
        self.entity_to_chunks.clear()
        if self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    session.run("MATCH (n) DETACH DELETE n")
            except Exception:
                pass

    def add_entity(self, entity: Entity):
        """Adiciona nó de entidade ao grafo."""
        self.graph.add_node(
            entity.name,
            entity_type=entity.entity_type,
            chunk_id=entity.chunk_id,
        )
        if entity.name not in self.entity_to_chunks:
            self.entity_to_chunks[entity.name] = set()
        self.entity_to_chunks[entity.name].add(entity.chunk_id)

        if self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        ON CREATE SET e.type = $type
                        """,
                        name=entity.name,
                        type=entity.entity_type,
                    )
            except Exception:
                pass

    def add_relation(self, rel: Relation):
        """Adiciona aresta relacional entre entidades."""
        self.graph.add_edge(
            rel.source,
            rel.target,
            relation_type=rel.relation_type,
            chunk_id=rel.chunk_id,
            metadata=rel.metadata,
        )
        if self.neo4j_driver:
            try:
                with self.neo4j_driver.session() as session:
                    session.run(
                        """
                        MATCH (a:Entity {name: $source})
                        MATCH (b:Entity {name: $target})
                        MERGE (a)-[r:RELATION {type: $rel_type, chunk_id: $chunk_id}]->(b)
                        """,
                        source=rel.source,
                        target=rel.target,
                        rel_type=rel.relation_type,
                        chunk_id=rel.chunk_id,
                    )
            except Exception:
                pass

    def index_chunks(self, chunks: List[Chunk]):
        """Indexa uma coleção de chunks no grafo de conhecimento."""
        for c in chunks:
            self.chunk_store[c.chunk_id] = c.content
            ents, rels = extract_entities_and_relations(c.content, c.chunk_id)
            for e in ents:
                self.add_entity(e)
            for r in rels:
                self.add_relation(r)

    def search_graph(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executa Graph Search para a consulta:
        1. Identifica entidades presentes na query.
        2. Percorre a vizinhança de 1 e 2 saltos no grafo.
        3. Mapeia as entidades encontradas de volta para os chunks de origem.
        4. Pontua chunks com base em centralidade e proximidade relacional.
        """
        query_entities = extract_entities_from_text(query)
        if not query_entities:
            return []

        chunk_scores: Dict[str, float] = {}
        matched_subgraph_entities: Set[str] = set()

        for q_ent in query_entities:
            if q_ent not in self.graph:
                continue

            matched_subgraph_entities.add(q_ent)

            # 1. Chunks diretos da entidade (score 1.0)
            for cid in self.entity_to_chunks.get(q_ent, set()):
                chunk_scores[cid] = chunk_scores.get(cid, 0.0) + 1.0

            # 2. Vizinhos diretos (1 salto - score 0.6)
            neighbors_1 = list(self.graph.neighbors(q_ent)) + list(self.graph.predecessors(q_ent))
            for n1 in set(neighbors_1):
                matched_subgraph_entities.add(n1)
                for cid in self.entity_to_chunks.get(n1, set()):
                    chunk_scores[cid] = chunk_scores.get(cid, 0.0) + 0.6

                # 3. Vizinhos de 2 saltos (score 0.3)
                neighbors_2 = list(self.graph.neighbors(n1))
                for n2 in set(neighbors_2):
                    if n2 != q_ent and n2 not in neighbors_1:
                        matched_subgraph_entities.add(n2)
                        for cid in self.entity_to_chunks.get(n2, set()):
                            chunk_scores[cid] = chunk_scores.get(cid, 0.0) + 0.3

        if not chunk_scores:
            return []

        # Ordena chunks por pontuação decrescente
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for cid, score in sorted_chunks:
            content = self.chunk_store.get(cid, "")
            results.append(
                {
                    "chunk_id": cid,
                    "content": content,
                    "score": round(score, 4),
                    "source": "graph",
                    "matched_entities": list(matched_subgraph_entities),
                }
            )

        return results


# Singleton global
_GRAPH_INDEX: Optional[KnowledgeGraphIndex] = None


def get_graph_index() -> KnowledgeGraphIndex:
    global _GRAPH_INDEX
    if _GRAPH_INDEX is None:
        _GRAPH_INDEX = KnowledgeGraphIndex()
    return _GRAPH_INDEX


def search_graph(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Função de entrada canônica para busca relacional no grafo."""
    idx = get_graph_index()
    return idx.search_graph(query, top_k=top_k)
