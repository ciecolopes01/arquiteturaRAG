# ADR 0001: Aplicação de ACL Fail-Closed no Multi-Tenancy

## Status
Aceito

## Data
2026-09-04

## Contexto
Em uma plataforma de RAG multi-tenant, documentos e chunks pertencem a tenants específicos e possuem listas de controle de acesso (ACL com `principals`).
No design original, a ausência de `principals` (`principals=None`) era interpretada como "sem restrição de ACL", retornando todos os documentos daquele tenant. Se um desenvolvedor ou middleware upstream cometesse um equívoco e esquecesse de extrair ou repassar o usuário autenticado, a busca abria todo o acervo do tenant (comportamento *fail-open*).

Adicionalmente, no canal de grafo (`graph_search.py`), chunks com metadados vazios de ACL passavam para qualquer usuário, enquanto nos canais Dense (Qdrant `MatchAny`) e BM25 (Postgres `&&`) chunks sem ACL eram estritamente rejeitados.

## Decisão
1. Inverter o comportamento padrão em `resolve_principals(principals, tenant_id, allow_unrestricted=False)` para **fail-closed**.
2. Qualquer chamada a endpoints públicos de busca (`search_dense`, `search_bm25`, `search_graph`, `hybrid_search`, `routed_hybrid_search`, `answer`, `answer_stream`) que informe `tenant_id` DEVE fornecer:
   - Uma lista explícita de `principals` (`["user:123", "role:admin"]`), OU
   - O sentinela explícito `ANY_PRINCIPAL` para casos onde o acesso irrestrito do tenant é deliberado (ex: rotinas administrativas, benchmarks ou indexadores).
3. Se `principals is None` ao consultar um tenant, levantar `ValueError` imediatamente antes de disparar qualquer query ao banco.
4. No canal de grafo, alinhar para *deny-by-default*: chunks sem ACL ou pertencentes a outro tenant são sumariamente descartados do ranking, e entidades retornadas em `matched_entities` são restritas estritamente aos chunks autorizados.

## Consequências
- **Positivas:** Eliminação completa de vazamento acidental de dados entre usuários ou tenants por omissão de parâmetros.
- **Positivas:** Comportamento uniforme de segurança em todos os 3 canais de retrieval (Dense, BM25 e Graph).
- **Negativas/Atenção:** Chamadores internos ou scripts de manutenção que acessam tenants precisam passar `principals=ANY_PRINCIPAL` conscientemente.
