# ADR 0004: Retrieval Multicanal Paralelo com ThreadPoolExecutor

## Status
Aceito

## Data
2026-09-04

## Contexto
O pipeline de busca híbrida (`hybrid_search`) e roteada (`routed_hybrid_search`) combina múltiplos canais de recuperação:
1. Canal Denso (Qdrant via REST/gRPC)
2. Canal Esparso (PostgreSQL BM25 via SQL/GIN)
3. Canal Relacional (Neo4j Cypher / NetworkX)

Na implementação inicial, as chamadas eram executadas de forma estritamente sequencial. A latência total do retrieval era a soma aritmética das latências de rede e processamento de cada backend:
`Latência_Total = Latência_Dense + Latência_BM25 + Latência_Graph + Latência_Fusão`

## Decisão
1. Utilizar `concurrent.futures.ThreadPoolExecutor(max_workers=3)` para disparar as consultas aos canais de busca concorrentemente.
2. Cada canal roda em sua respectiva thread de I/O, aproveitando o tempo de espera de rede/banco dos demais canais.
3. Isolar a resolução de resultados com `Future.result()` dentro de blocos `try/except` individuais por canal.
4. Se um canal sofrer timeout ou exceção de conexão (ex: cluster Neo4j indisponível), a falha é registrada nos logs/telemetria (`graph_channel_degraded = True`) e os canais restantes concluem a fusão RRF sem derrubar a consulta do usuário.

## Consequências
- **Positivas:** Redução média de 40% a 60% na latência do estágio de retrieval.
- **Positivas:** Alta resiliência: tolerância a falhas parciais em nós de banco sem indisponibilizar a API de busca.
- **Neutras:** Utiliza threads leves do sistema operacional especializadas em operações bloqueantes de I/O em rede.
