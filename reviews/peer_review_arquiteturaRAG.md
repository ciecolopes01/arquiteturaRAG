# Peer Review — `ciecolopes01/arquiteturaRAG`

**Revisor:** Abacus AI Agent
**Data da revisão:** 03/09/2026
**Commit analisado:** `3cb602f` — *refactor: fix security ACL cache isolation, query router fusion, stream and pytest coverage* (2026-09-03)
**Branch:** `master` · **Histórico:** 19 commits ("Passo 15" → "Passo 32")

> Escopo: revisão estática completa do código-fonte, dependências, testes, documentação e artefatos versionados.
> **Nada foi executado em runtime** — os serviços do `docker-compose.yml` (Qdrant, Postgres, MinIO, Neo4j, Redis) não foram
> subidos e as dependências pesadas (`torch`, `FlagEmbedding`, `scenedetect`) não foram instaladas. Onde há afirmação sobre
> comportamento, ela deriva de leitura de código, análise de AST, `ruff` e coleta do `pytest` — nunca de execução real.

---

### Sumário executivo

`arquiteturaRAG` é uma plataforma de RAG (Retrieval-Augmented Generation) multi-tenant, construída de forma incremental em
32 passos, com **18.548 linhas de Python** distribuídas em 91 arquivos (9.807 LOC em `src/`, 6.448 LOC em `tests/`).
A amplitude funcional é notável: 9 parsers de documento, 3 estratégias de chunking, retrieval híbrido tri-canal
(denso + BM25 + grafo), fusão RRF com pesos auto-calibráveis, reranking cross-encoder, HyDE, decomposição de query,
multi-vector, Self-RAG, cache em duas camadas, observabilidade com custo em USD e uma plataforma de benchmark comparável.

O projeto tem **fundações arquiteturais sólidas** — separação de camadas limpa, decisões de design documentadas com racional
explícito, e pré-filtro de ACL executado na query (não em memória), que é a escolha correta. O commit mais recente também
demonstra maturidade: **9 dos 13 achados do review anterior** (`reviews/review_2026-09-04.md`) foram efetivamente corrigidos.

Dito isso, a revisão identificou **problemas que impedem a promoção do sistema a produção no estado atual**:

1. 🔴 **Bypass de ACL no canal de grafo** — resultados do grafo entram na resposta final sem qualquer filtro de tenant ou permissão.
2. 🔴 **O modelo de ACL é fail-open** — esquecer de passar `principals` libera todo o corpus, em vez de negar acesso.
3. 🔴 **O Self-RAG descarta os chunks que validou** e refaz o retrieval do zero, invalidando a própria premissa do padrão.
4. 🔴 **O relatório de benchmark emite conclusões hardcoded que os dados refutam** (Recall@5 = 0.0000 em todas as 5 configurações).
5. 🟠 **Zero automação de qualidade** — sem CI, sem `conftest.py`, sem `pyproject.toml`; 30 arquivos de teste dependem de `sys.path.insert(0, ".")`.
6. 🟠 **`networkx` e `pytest` ausentes de `requirements.txt`** — instalação limpa quebra na importação e nos testes.

A boa notícia: nenhum desses problemas é estrutural. São correções pontuais e bem localizadas sobre uma base que já está certa.

#### Placar por severidade

| Severidade | Qtd. | Natureza |
| :--- | :---: | :--- |
| 🔴 Crítico | 6 | Vazamento de dados entre tenants, ACL fail-open, bug lógico no Self-RAG, relatório falso, dataset inválido, deps quebradas |
| 🟠 Alto | 9 | Ausência de CI, Postgres sem pool, upsert monolítico, perda de contexto pai, streaming sem cache/telemetria, Neo4j write-only |
| 🟡 Médio | 12 | 90 `print()` vs 6 loggers, 11 `except: pass`, thresholds mágicos, tabela de custos defasada, artefatos de runtime versionados |
| 🟢 Baixo | 8 | 140 issues de lint (excl. E501), 14% das funções sem anotação de retorno, 27% sem docstring, README limitado |

#### Nota por dimensão

| Dimensão | Nota | Comentário |
| :--- | :---: | :--- |
| Arquitetura e design | **8,5 / 10** | Camadas bem separadas, decisões justificadas, extensível |
| Qualidade do código | **6,5 / 10** | Legível e bem comentado, mas com débito de lint e `print()` em vez de logging |
| Tratamento de erros | **5,5 / 10** | Fallbacks resilientes, porém 11 exceções silenciadas |
| Segurança | **4,0 / 10** | Pré-filtro correto, mas com bypass real no canal de grafo e modelo fail-open |
| Testes | **5,0 / 10** | Volume alto (141 funções), mas sem CI, sem mocks e dependentes de infra viva |
| Documentação | **6,0 / 10** | Docstrings de módulo excelentes; README cobre só o "Passo 0" |
| Dependências | **6,0 / 10** | Bem organizado por camada, mas com 2 pacotes faltando e sem separação dev/prod |
| Performance | **5,0 / 10** | Cache e prompt caching bons; sem pool, sem paralelismo, upsert monolítico |
| **Média ponderada** | **≈ 5,8 / 10** | Protótipo avançado de alta qualidade; ainda não é *production-ready* |

---

### 1. Visão geral da arquitetura e design patterns

#### 1.1 Estrutura de diretórios

```
arquiteturaRAG/
├── .agents/rules/concisao.md          # convenções de comunicação do projeto
├── .env.example                       # 42 linhas — cobre todos os serviços
├── docker-compose.yml                 # qdrant · postgres:16 · minio · neo4j:5 · redis:7
├── rag.py / rag.bat                   # CLI: answer | benchmark | eval
├── requirements.txt                   # 24 dependências efetivas
├── README.md                          # 58 linhas — apenas setup do ambiente
├── arquitetura-rag-revisada.md        # 19 KB — documento de arquitetura
├── runbook-construcao-rag.md          # 78 KB (+ .bak de 93 KB versionado)
├── benchmarks/{configs,results}/      # 5 configs YAML + relatórios gerados
├── data/golden_dataset.json           # 48 perguntas de avaliação
├── logs/*.jsonl                       # artefatos de runtime versionados
├── reviews/review_2026-09-04.md       # auto-review anterior
├── scripts/                           # 12 utilitários operacionais
├── src/                               # 45 módulos em 10 pacotes
│   ├── parsing/        chunking/      embedding/
│   ├── retrieval/      representation/ generation/
│   ├── evaluation/     observability/ optimization/  ingestion/
└── tests/                             # 33 arquivos · 141 funções · 21 classes
```

#### 1.2 Fluxo do pipeline

```
                      ┌──────────────── ingestão ────────────────┐
  documento  ──▶  parsing (9 parsers) ──▶ canonical_model ──▶ chunking (3 estratégias)
                                                                    │
                                                                    ▼
                                              embedding (BGE-M3) ──▶ Qdrant + Postgres FTS + Neo4j
                      ┌──────────────── consulta ────────────────┐
  query ──▶ query_router ──▶ ┌─ dense_search  (Qdrant, ACL na query)
                             ├─ search_bm25   (Postgres tsvector, ACL no WHERE)
                             └─ search_graph  (networkx em memória)  ⚠️ SEM ACL
                                        │
                                        ▼
                            weighted_rrf (pesos auto-calibrados)
                                        ▼
                            rerank (cross-encoder BGE)
                                        ▼
                    self_rag: SUFFICIENT / RETRY / INSUFFICIENT
                                        ▼
                  answer() ──▶ Claude (prompt caching) ──▶ resposta + citações
                                        │
                     cache (Redis → memória) · trace · custo USD · recall history
```

#### 1.3 Padrões de projeto identificados

| Padrão | Onde | Avaliação |
| :--- | :--- | :--- |
| **Layered architecture** | `src/` dividido por estágio do pipeline | ✅ Excelente — cada pacote tem responsabilidade única |
| **Strategy** | 3 chunkers (`recursive`, `semantic`, `parent_child`), 5 configs de benchmark | ✅ Bem aplicado |
| **Facade** | `search_graph()` módulo-nível envolvendo `KnowledgeGraphIndex.search_graph()` (`graph_search.py:329`) | ✅ Correto |
| **Singleton (lazy)** | `_get_conn()`, `_get_client()`, modelos de embedding/rerank | ⚠️ Sem thread-safety nem pool (§9.1) |
| **Chain of Responsibility** | Fallbacks de retrieval: `plainto_tsquery` → `to_tsquery` OR-terms | ⚠️ Silencia erros reais (§6.2) |
| **Decorator** | Instrumentação de estágio em `observability/instrumentation.py` | ✅ Boa separação de concerns |
| **Cache-aside** | `optimization/caching.py`, Redis com fallback para LRU em memória | ✅ Resiliente; ⚠️ não escreve no caminho de streaming (§4.5) |
| **Settings object** | `pydantic-settings` em `src/config.py` | ✅ Idiomático |

#### 1.4 Stack

Python (`from __future__ import annotations` consistente) · Qdrant (vetorial) · PostgreSQL 16 (BM25 via `tsvector` +
`ts_rank_cd`) · Neo4j 5 (grafo) · Redis 7 (cache) · MinIO (objetos) · BGE-M3 (embeddings) · BGE reranker (cross-encoder) ·
Anthropic Claude (geração) · `pydantic-settings` · `psycopg2`.

---

### 2. Pontos fortes

Estes são méritos reais e devem ser preservados em qualquer refatoração.

**2.1 Separação de responsabilidades exemplar.**
Cada pacote de `src/` mapeia um estágio do pipeline, e camadas transversais (`observability`, `optimization`, `evaluation`)
estão isoladas das camadas de domínio. É possível trocar o vector store ou o LLM tocando em um único pacote.

**2.2 Docstrings que explicam *o porquê*, não o *o quê*.**
Este é o ponto mais forte do código. Exemplos: o cabeçalho de `bm25_index.py` justifica a escolha de Postgres FTS sobre
OpenSearch; `incremental_reindex.py` documenta o racional do `document_id` determinístico via `uuid5`; `acl_filter.py`
explica por que o filtro roda **na query** e não em pós-processamento. Documentação de decisão arquitetural embutida no
código é raro e valioso — é o que permite que outra pessoa mantenha o projeto.

**2.3 Pré-filtro de ACL na query (decisão de segurança correta).**
No Qdrant, via `Filter(must=[FieldCondition(key="acl_principals", match=MatchAny(any=principals))])`
(`embed_index.py:246-249`); no Postgres, via cláusula `WHERE ... acl_principals && %(principals)s` (`bm25_index.py:150`).
Filtrar antes do ranking evita tanto o vazamento por pós-filtro quanto a degradação de recall causada por descartar
resultados após o top-k. A ressalva de fail-open está em §3.3 — mas o mecanismo base está certo.

**2.4 Observabilidade acima da média para um projeto solo.**
Trace por estágio, custo estimado em USD por query, `logs/pipeline.jsonl` estruturado, histórico de Recall@5 com detecção
de drift (`check_recall_drift`). Ponto extra: o sistema loga `query_hash` em vez da query crua — decisão consciente de
privacidade que muitos times de produção não tomam.

**2.5 Cache com isolamento correto e degradação graciosa.**
`_response_key()` inclui os `principals` ordenados na chave, eliminando o vazamento cross-usuário apontado no review
anterior. O fallback Redis → LRU em memória permite operar sem Redis disponível.

**2.6 Prompt caching implementado.**
Uso de `cache_control: {"type": "ephemeral"}` no system prompt — otimização de custo que raramente aparece em projetos
deste porte.

**2.7 Modo degradado sem API key.**
`_simulated_answer()` permite exercitar o pipeline inteiro sem chave da Anthropic. Excelente para desenvolvimento e CI.
(Ver §4.7 para o risco associado.)

**2.8 Migração de schema idempotente.**
`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` em `bm25_index.py` — o schema evolui sem quebrar bases existentes.

**2.9 Cultura de auto-review com follow-through comprovado.**
O diretório `reviews/` não é decorativo: 9 dos 13 achados do review de 2026-09-04 foram corrigidos no código, verificados
individualmente nesta revisão (tabela em §11). Isso é sinal forte de disciplina de engenharia.

**2.10 Cobertura funcional ampla e coerente.**
9 parsers (PDF, DOCX, XLSX, imagem com OCR + visão, vídeo, áudio, tabelas, HTML, TXT), 3 chunkers, 3 canais de retrieval,
HyDE, decomposição, multi-vector, Graph RAG, Self-RAG, auto-tuning de pesos RRF, benchmark comparável e feedback online.
Poucos projetos de RAG cobrem tanto terreno com organização tão consistente.

---

### 3. Segurança

#### 3.1 🔴 CRÍTICO — Bypass total de ACL e de tenant no canal de grafo

**Arquivos:** `src/retrieval/graph_search.py:256,329` · `src/retrieval/query_router.py:222-235`

A função de entrada do canal de grafo não recebe `tenant_id` nem `principals`:

```python
# src/retrieval/graph_search.py:329
def search_graph(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Função de entrada canônica para busca relacional no grafo."""
    idx = get_graph_index()
    return idx.search_graph(query, top_k=top_k)
```

E é invocada exatamente dentro do caminho que *é* ACL-aware:

```python
# src/retrieval/query_router.py:222-235
graph_hits = []
if use_graph:
    try:
        from src.retrieval.graph_search import search_graph
        graph_hits = search_graph(query=query, top_k=candidate_pool_size)   # ← sem tenant, sem principals
    except Exception as e:
        logger.warning("Busca em grafo falhou (%s); prosseguindo com dense + bm25", e)
        graph_hits = []

fused = weighted_rrf(
    dense_hits=dense_hits,     # filtrado por ACL ✅
    bm25_hits=bm25_hits,       # filtrado por ACL ✅
    graph_hits=graph_hits,     # NÃO filtrado ❌ — entra na fusão com peso igual
    ...
)
```

**Impacto.** Todo o esforço de isolamento multi-tenant das outras duas rotas é anulado: chunks de qualquer tenant e de
qualquer nível de permissão podem ser fundidos no ranking final e enviados ao LLM como contexto da resposta. Em um sistema
multi-tenant isto é vazamento de dados entre clientes. O mesmo caminho é usado por `src/evaluation/benchmark.py:128`.

**Correção sugerida.** Propagar a identidade até o grafo e filtrar na travessia:

```python
# graph_search.py
def search_graph(
    query: str,
    top_k: int = 5,
    tenant_id: Optional[str] = None,
    principals: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    idx = get_graph_index()
    return idx.search_graph(query, top_k=top_k, tenant_id=tenant_id, principals=principals)
```

Dentro de `KnowledgeGraphIndex.search_graph`, ao mapear entidades → chunks, descartar todo chunk cujo `tenant_id` divirja
ou cujo `acl_principals` não intersecte `principals`. Enquanto a correção não é feita, **desative o canal de grafo por
padrão** (`use_graph=False`) para qualquer chamada com `tenant_id` definido.

#### 3.2 🔴 CRÍTICO — Rotas alternativas de retrieval sem propagação de `principals`

Um levantamento por AST das assinaturas de todas as funções de busca mostrou que o parâmetro `principals` só chega a
parte do sistema. As rotas abaixo recebem `tenant_id` (isolamento de cliente) mas **não** `principals` (permissão dentro
do cliente):

| Função | Arquivo | `tenant_id` | `principals` |
| :--- | :--- | :---: | :---: |
| `hyde_search()` / `hyde_hybrid_search()` | `src/retrieval/hyde.py` | ✅ | ❌ |
| `decomposed_search()` | `src/retrieval/decomposition.py` | ✅ | ❌ |
| `search_multi_vector()` / `multi_vector_hybrid_search()` | `src/representation/multi_vector.py` | ✅ | ❌ |
| `execute_pipeline()` | `src/evaluation/benchmark.py` | ✅ | ❌ |
| `_retrieve()` / `evaluate_recall_at_k()` | `src/evaluation/offline_eval.py` | ✅ | ❌ |
| `side_by_side_eval()` | `src/embedding/migration.py` | ✅ | ❌ |
| `search_graph()` | `src/retrieval/graph_search.py` | ❌ | ❌ |

**Impacto.** Um usuário do tenant correto, mas sem permissão sobre um documento restrito, recebe esse documento se a query
for roteada por HyDE, decomposição ou multi-vector. O isolamento entre clientes se mantém; o controle de acesso **dentro**
do cliente não.

**Correção sugerida.** Adicionar `principals: Optional[List[str]] = None` a todas as assinaturas acima e repassar até o
`dense_search`/`search_bm25` subjacente. Em seguida, criar um teste de regressão que percorra o AST e falhe se alguma
função cujo nome case com `r"(search|retrieve|answer)"` em `src/retrieval/`, `src/representation/` e `src/generation/`
não declarar ambos os parâmetros — isso impede a reintrodução do problema.

#### 3.3 🔴 CRÍTICO — O modelo de ACL é *fail-open*

O filtro só é aplicado quando `principals` é informado e não-vazio. No Qdrant:

```python
# src/embedding/embed_index.py:246
if principals:
    conditions.append(FieldCondition(key="acl_principals", match=MatchAny(any=principals)))
```

No Postgres:

```sql
-- src/retrieval/bm25_index.py:150
AND (%(principals)s = '{}'::text[] OR acl_principals && %(principals)s)
```

Em ambos, `principals = None` ou `[]` significa **"sem restrição"**, e não **"sem acesso"**. Qualquer chamador novo que
esqueça o parâmetro — e há sete deles hoje, conforme §3.2 — obtém acesso irrestrito ao corpus, silenciosamente e sem
erro. Segurança que depende de o desenvolvedor lembrar de um argumento opcional é segurança que eventualmente falha.

**Correção sugerida.** Inverter o default para fail-closed com opt-out explícito:

```python
# src/retrieval/acl_filter.py
class _AnyPrincipal:
    """Sentinela explícita para busca administrativa sem filtro de ACL."""
    __slots__ = ()

ANY_PRINCIPAL = _AnyPrincipal()

def resolve_principals(principals) -> list[str] | None:
    if principals is ANY_PRINCIPAL:
        return None                      # sem filtro — escolha consciente e auditável
    if not principals:
        raise ValueError(
            "principals é obrigatório. Use ANY_PRINCIPAL explicitamente para busca administrativa."
        )
    return list(principals)
```

Toda função de busca chama `resolve_principals()` no início. O código administrativo/benchmark passa `ANY_PRINCIPAL`, o
que deixa a decisão visível em code review e em `grep`. O esquecimento passa a ser um erro, não um vazamento.

#### 3.4 🟡 MÉDIO — Credenciais fracas hardcoded como default

```python
# src/config.py
anthropic_api_key: str = ""
postgres_url: str = "postgresql://rag:rag_dev_password@localhost:5432/rag_platform"   # :32
minio_access_key: str = "minioadmin"                                                  # :36
minio_secret_key: str = "minioadmin_dev"                                              # :37
neo4j_password:  str = "rag_neo4j_password"                                           # :44
```

As mesmas senhas estão espelhadas em `docker-compose.yml`. Em desenvolvimento é aceitável; o risco é que um deploy que
esqueça de exportar as variáveis suba **funcionando** com credenciais públicas, em vez de falhar. Além disso,
`anthropic_api_key: str = ""` faz o sistema cair silenciosamente no modo simulado em produção — servindo texto de debug
como se fosse resposta.

**Correção sugerida.** Validar por ambiente no `Settings`:

```python
from pydantic import model_validator

@model_validator(mode="after")
def _reject_dev_defaults_in_prod(self):
    if self.environment == "production":
        weak = {"rag_dev_password", "minioadmin", "minioadmin_dev", "rag_neo4j_password", ""}
        for field in ("postgres_url", "minio_access_key", "minio_secret_key",
                      "neo4j_password", "anthropic_api_key"):
            if any(w and w in str(getattr(self, field)) for w in weak) or not getattr(self, field):
                raise ValueError(f"{field} usa credencial de desenvolvimento em produção")
    return self
```

#### 3.5 🟡 MÉDIO — Artefatos de runtime versionados no repositório

Estão sob controle de versão: `logs/pipeline.jsonl` (138 linhas), `logs/recall_history.jsonl` (11 linhas),
`benchmarks/results/*` e `runbook-construcao-rag.md.bak` (93 KB).

Logs de pipeline podem conter metadados de consultas reais (`query_hash`, `source_uri`, `tenant_id`) — não é lugar de
repositório. O `.bak` de 93 KB é lixo que o próprio Git torna desnecessário.

**Correção:** adicionar ao `.gitignore` e remover do índice:

```bash
printf 'logs/\nbenchmarks/results/\n*.bak\n' >> .gitignore
git rm -r --cached logs benchmarks/results runbook-construcao-rag.md.bak
```

Mantenha `logs/.gitkeep` e `benchmarks/results/.gitkeep` para preservar a estrutura.

#### 3.6 🟢 Positivo — não há segredos commitados

`.env` está corretamente ignorado no `.gitignore`, `.env.example` contém apenas placeholders, e uma varredura do histórico
não encontrou chaves de API reais. Bom.


---

### 4. Bugs e problemas de correção

#### 4.1 🔴 CRÍTICO — O Self-RAG descarta os chunks que acabou de validar

**Arquivo:** `src/generation/self_rag.py:134-234`

O fluxo de `adaptive_answer()` é:

```python
# 1. retrieval roteado, ACL-aware, com pool ampliado
candidates, intent, _ = routed_hybrid_search(query=query, top_k=max(top_k * 3, 15),
                                             tenant_id=tenant_id, principals=principals)   # :155
# 2. reranking cross-encoder
reranked = rerank(query, candidates, top_k=top_k)                                          # :165
# 3. avaliação de suficiência sobre os chunks rerankeados
decision, confidence = evaluate_context_sufficiency(query, reranked, ...)                  # :170
...
# 4. Suficiente: Gera resposta através do Claude (answer) usando os chunks validados
ans_res = answer(query=query, top_k=top_k, tenant_id=tenant_id, principals=principals)     # :229
```

O comentário na linha 228 diz *"usando os chunks validados"*, mas `reranked` **nunca é passado** para `answer()`.
E `answer()` → `_answer_impl()` executa `hybrid_search(...)` do zero na linha 497 de `answer.py`.

**Consequências, em ordem de gravidade:**

1. **A premissa do Self-RAG é anulada.** O sistema julga a suficiência de um conjunto de chunks (roteado + rerankeado) e
   então responde com um conjunto *diferente* (hybrid puro, sem routing e sem reranking). O `confidence_score` retornado
   ao chamador não descreve o contexto que gerou a resposta.
2. **Retrieval duplicado.** Dois pipelines completos por consulta. Considerando as latências registradas em
   `benchmarks/results/benchmark_report.md` (194 s para *Hybrid + Reranker*), o custo não é marginal.
3. **Uma rodada de RETRY é trabalho inteiramente jogado fora** — a query reescrita, o segundo retrieval e o segundo rerank
   afetam apenas `decision`/`confidence`; a geração final continua usando a query original.
4. A variável `intent` (linha 155) é atribuída e nunca usada (`ruff` F841).

**Correção sugerida.** Permitir que `answer()` receba chunks pré-computados e curto-circuite o retrieval:

```python
# src/generation/answer.py
def answer(query: str, top_k: int = MAX_CONTEXT_CHUNKS, *,
           tenant_id: Optional[str] = None,
           principals: Optional[List[str]] = None,
           prefetched_chunks: Optional[List[HybridResult]] = None,
           **kwargs) -> AnswerResult:
    ...

def _answer_impl(..., prefetched_chunks=None):
    chunks = prefetched_chunks if prefetched_chunks is not None else hybrid_search(
        query, top_k=top_k, tenant_id=tenant_id, principals=principals
    )
```

```python
# src/generation/self_rag.py:229
ans_res = answer(
    query=query,
    top_k=top_k,
    tenant_id=tenant_id,
    principals=principals,
    prefetched_chunks=reranked,     # ← usa de fato o que foi validado
)
```

Isso corrige o bug lógico **e** elimina metade da latência de uma vez. É a correção de maior retorno do relatório.

#### 4.2 🔴 CRÍTICO — O relatório de benchmark afirma conclusões que os dados refutam

**Arquivo:** `src/evaluation/benchmark.py:309-316`

```python
md_lines.extend([
    "",
    "## Conclusão e Diretrizes de Produção",
    "- **Perguntas Conceituais/Semânticas:** Dense BGE-M3 e Hybrid RRF apresentam melhor desempenho.",
    "- **Perguntas Tabulares/Fiscais/Hardware:** BM25 lexical e filtros prévios garantem precisão exata...",
    "- **Perguntas Relacionais:** A introdução do canal de Grafo ... eleva o Recall@5 substancialmente...",
    "- **Reranking:** O modelo cross-encoder eleva o MRR consistente em todas as classes de perguntas.",
])
```

São strings literais, emitidas **independentemente dos números medidos**. O artefato versionado
`benchmarks/results/benchmark_report.md` (execução de 2026-09-03) mostra:

| Configuração | Recall@5 | MRR | Latência média |
| :--- | :---: | :---: | ---: |
| BM25 Pure | 0.0000 | 0.0000 | 461 ms |
| Dense BGE-M3 Pure | 0.0000 | 0.0000 | 16.940 ms |
| Hybrid Uniform RRF | 0.0000 | 0.0000 | 1.315 ms |
| Hybrid RRF + BGE Reranker | 0.0000 | 0.0000 | 194.274 ms |
| Multichannel + Tuned RRF + Reranker | 0.0000 | 0.0000 | 253.141 ms |

Ou seja: o relatório afirma que "o reranking eleva o MRR consistentemente" em uma tabela onde **todo MRR é zero**, e que
"o grafo eleva o Recall@5 substancialmente" onde **todo recall é zero**. Um relatório de avaliação que não pode reprovar
nada não é uma ferramenta de avaliação — é um gerador de texto. Pior: mascara o fato de que o pipeline está com recall
nulo (§4.3), que é o problema real a ser investigado.

**Correção sugerida.** Derivar as conclusões dos dados e falhar alto quando eles forem degenerados:

```python
def _derive_conclusions(results: list[BenchmarkResult]) -> list[str]:
    if all(r.recall_at_k == 0.0 for r in results):
        return [
            "> ⚠️ **Resultado degenerado:** Recall@5 = 0 em todas as configurações. "
            "Isso indica falha de avaliação (golden dataset desalinhado com o índice) e não "
            "desempenho real dos pipelines. Nenhuma conclusão comparativa é válida.",
        ]
    best_recall  = max(results, key=lambda r: r.recall_at_k)
    best_mrr     = max(results, key=lambda r: r.mrr)
    fastest      = min(results, key=lambda r: r.avg_latency_ms)
    lines = [
        f"- **Melhor Recall@5:** {best_recall.config_name} ({best_recall.recall_at_k:.4f})",
        f"- **Melhor MRR:** {best_mrr.config_name} ({best_mrr.mrr:.4f})",
        f"- **Menor latência:** {fastest.config_name} ({fastest.avg_latency_ms:.0f} ms)",
    ]
    for cat in sorted({c for r in results for c in r.category_recalls}):
        win = max(results, key=lambda r: r.category_recalls.get(cat, 0.0))
        lines.append(f"- **{cat}:** {win.config_name} ({win.category_recalls.get(cat, 0.0):.2f})")
    return lines
```

Complemente com um `assert`/exit-code não-zero no `main()` quando todo o recall for zero, para que a execução em CI
reprove em vez de gerar um relatório otimista.

#### 4.3 🔴 CRÍTICO — O golden dataset está ancorado em UUIDs efêmeros

**Arquivos:** `data/golden_dataset.json` · `src/chunking/recursive.py:42`

O dataset de avaliação (48 perguntas) usa `expected_chunk_ids` contendo UUIDs. Mas o `chunk_id` é gerado assim:

```python
# src/chunking/recursive.py:42
chunk_id = str(uuid.uuid4())
```

`uuid4()` é aleatório: **cada reindexação produz IDs completamente novos**. Todo `expected_chunk_ids` do golden dataset
se torna inválido no instante em que o corpus é reprocessado. Esta é, com alta probabilidade, a causa raiz do
Recall@5 = 0.0000 em todas as configurações — o pipeline pode estar recuperando os chunks certos, mas comparando contra
IDs que não existem mais.

Note o contraste: `incremental_reindex.py` já faz a coisa certa para documentos, gerando `document_id` determinístico via
`uuid5`. A mesma disciplina não foi aplicada aos chunks.

**Correção sugerida — duas frentes.**

*(a) Tornar o `chunk_id` determinístico:*

```python
import hashlib, uuid

CHUNK_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")

def make_chunk_id(document_id: str, chunk_index: int, content: str) -> str:
    """ID estável: mesmo documento + mesma posição + mesmo texto ⇒ mesmo ID."""
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"{document_id}:{chunk_index}:{digest}"))
```

*(b) Reancorar o golden dataset em algo semanticamente estável.* Mesmo com IDs determinísticos, mudar o `chunk_size`
invalida o dataset. O gabarito deve apontar para `document_id` + um trecho literal esperado:

```json
{
  "question": "Qual o prazo de retenção fiscal?",
  "expected_documents": ["<document_id determinístico>"],
  "expected_spans": ["prazo de retenção de 5 (cinco) anos"],
  "category": "Fiscal"
}
```

O avaliador passa a marcar acerto quando um chunk recuperado pertence a `expected_documents` **e** contém o `span`
(normalizado). Assim o dataset sobrevive a mudanças de chunking, que é o ponto de um golden dataset.

#### 4.4 🟠 ALTO — Canal de grafo é write-only: Neo4j nunca é consultado

**Arquivo:** `src/retrieval/graph_search.py`

`KnowledgeGraphIndex` grava entidades e relações no Neo4j (`add_entity` / `add_relation`), mas `search_graph()`
(linha 256) percorre **apenas** o `nx.MultiDiGraph` mantido em memória e o `chunk_store` local do objeto. Em um processo
novo — que é exatamente o caso de um servidor de produção, um worker ou uma execução de benchmark — o grafo em memória
está vazio, `extract_entities_from_text(query)` não encontra correspondência e a função retorna `[]` na linha 266.

Consequência: o canal de grafo contribui com zero resultados em qualquer execução que não tenha feito a ingestão no mesmo
processo, e a persistência no Neo4j é puramente decorativa. Isso também explica por que o problema de ACL de §3.1 ainda
não se manifestou em produção — o canal simplesmente não retorna nada.

**Correção sugerida.** Implementar a travessia via Cypher, já com os filtros de segurança embutidos:

```python
CYPHER_NEIGHBORHOOD = """
MATCH (e:Entity)-[r*1..2]-(n:Entity)-[:MENTIONED_IN]->(c:Chunk)
WHERE e.name IN $entities
  AND ($tenant_id IS NULL OR c.tenant_id = $tenant_id)
  AND ($principals IS NULL OR any(p IN c.acl_principals WHERE p IN $principals))
RETURN c.chunk_id AS chunk_id, c.content AS content, c.source_uri AS source_uri,
       count(DISTINCT n) AS centrality, min(length(r)) AS hops
ORDER BY centrality DESC, hops ASC
LIMIT $top_k
"""
```

O grafo em memória continua útil como cache de processo, mas a fonte da verdade passa a ser o Neo4j.

#### 4.5 🟠 ALTO — `answer_stream()` lê o cache mas nunca escreve, e não registra custo nem trace

**Arquivo:** `src/generation/answer.py:558-606`

```python
def answer_stream(...) -> Iterator[str]:
    # 0. Cache check
    cached = cache.get_response(query=query, tenant_id=tenant_id or "", principals=principals)
    if cached is not None:
        ...
        return
    chunks = hybrid_search(query, top_k=top_k, tenant_id=tenant_id, principals=principals)
    ...
    yield from _stream_claude(query, chunks, model=model)     # nenhum set_response, nenhum record_generation_cost
```

Três lacunas no caminho de streaming — que é justamente o caminho que usuários finais tendem a usar:

- **Cache nunca é populado:** a taxa de acerto do cache no streaming será sempre 0% (a não ser por respostas gravadas pelo
  caminho não-streaming). O cache é write-never/read-always.
- **Custo não é contabilizado:** `_stream_claude()` não chama `record_generation_cost`, então tokens e USD do streaming
  ficam fora de toda a telemetria de custo.
- **Sem `pipeline_trace`:** nenhuma entrada em `logs/pipeline.jsonl` para requisições de streaming.

O resultado é um dashboard de custo e latência que subestima sistematicamente o uso real.

**Correção sugerida.** Acumular os tokens durante o yield e finalizar ao término do generator:

```python
    parts: list[str] = []
    with pipeline_trace(stage="answer_stream", query=query) as trace:
        for token in _stream_claude(query, chunks, model=model):
            parts.append(token)
            yield token
        full = "".join(parts)
        record_generation_cost(model=model, prompt=..., completion=full, trace=trace)
        cache.set_response(query=query, tenant_id=tenant_id or "", principals=principals,
                           value={"answer": full, "chunks_used": [_serialize_chunk(c) for c in chunks]})
```

Cubra com um teste que consuma o generator inteiro e verifique que `cache.get_response` passa a retornar valor.

#### 4.6 🟠 ALTO — Perda de contexto pai nos canais BM25 e reranker

**Dois defeitos que se somam no mesmo sintoma.**

*(a) `search_bm25()` seleciona `parent_chunk_id` e `parent_content` no SQL mas não os devolve.*

```sql
-- src/retrieval/bm25_index.py:143-144 — o SELECT traz os campos
    parent_chunk_id,
    parent_content,
```

```python
# src/retrieval/bm25_index.py:315-326 — mas o dict de saída os omite
return [
    {
        "chunk_id":       r["chunk_id"],
        "score":          round(float(r["score"]) / max_score, 4),
        "raw_score":      round(float(r["score"]), 6),
        "content":        r["content"],
        "source_uri":     r["source_uri"],
        "document_title": r["document_title"],
        "section_title":  r["section_title"],
        # parent_chunk_id e parent_content: buscados no banco, descartados aqui
    }
    for r in rows
]
```

*(b) `RerankResult` também não carrega esses campos:*

```python
# src/retrieval/reranker.py:84-94
@dataclass
class RerankResult:
    chunk_id: str
    content: str
    rerank_score: float
    original_rank: int
    new_rank: int
    source_uri: str = ""
    section_title: str = ""
    document_title: str = ""
    rrf_score: float = 0.0
    # sem parent_chunk_id / parent_content
```

**Impacto combinado.** A estratégia parent/child existe para dar ao LLM o contexto amplo do trecho recuperado. Um chunk
que chegue pelo canal BM25 perde essa expansão em `expand_parent_context()`; e **qualquer** chunk perde a expansão depois
de passar pelo reranker. Como o reranker está no caminho principal, na prática o parent/child só funciona no fluxo
`answer()` direto sem rerank. Há ainda inconsistência de tipos: `RerankResult` não é um `HybridResult`, então funções
downstream tipadas para `HybridResult` recebem outra coisa.

**Correção:** adicionar `parent_chunk_id: str = ""` e `parent_content: str = ""` ao dict de `search_bm25()` e ao
dataclass `RerankResult`, propagando-os em `rerank()`. Idealmente, unifique num único `RetrievedChunk` compartilhado por
todos os canais e estágios — hoje há pelo menos três formatos de chunk circulando (dict do BM25, `HybridResult`,
`RerankResult`).

#### 4.7 🟡 MÉDIO — Modo simulado é indistinguível de uma resposta real no texto

`_simulated_answer()` retorna texto de diagnóstico no campo `answer`. Existe a flag `simulated=True` no resultado, o que é
correto, mas nada impede que um consumidor renderize `result.answer` diretamente. Combinado com
`anthropic_api_key: str = ""` como default (§3.4), o cenário de falha é: deploy sem a variável de ambiente → API responde
200 OK → usuário recebe texto de debug como resposta oficial.

**Correção:** prefixar visivelmente a resposta simulada (`"[MODO SIMULADO — sem chave de API] ..."`) e emitir
`logger.error` uma vez na inicialização quando `anthropic_api_key` estiver vazio fora de ambiente de teste.

#### 4.8 🟡 MÉDIO — `embed_and_index()`: índice de chunk incorreto no fallback

```python
# src/embedding/embed_index.py:137,156
for i in range(0, len(chunks), batch_size):
    batch = chunks[i : i + batch_size]
    ...
        "chunk_index": chunk.metadata.get("chunk_index", i),   # ← "i" é o offset do batch
```

Quando `chunk_index` não está no metadata, o fallback usa `i`, que é o **offset inicial do batch**, não a posição do chunk.
Com `batch_size=12`, os 12 chunks do segundo batch recebem todos `chunk_index=12`. Isso corrompe qualquer ordenação por
posição e a reconstrução da sequência de um documento.

**Correção:** `for offset, chunk in enumerate(batch): ... chunk.metadata.get("chunk_index", i + offset)`.

#### 4.9 🟡 MÉDIO — `embedding_dimension()` retorna 1024 hardcoded

Funciona para BGE-M3, mas quebra silenciosamente ao trocar de modelo — e `src/embedding/migration.py` existe justamente
para suportar troca de modelo. Deve derivar de `model.get_sentence_embedding_dimension()` ou do vetor real, com cache.

#### 4.10 🟡 MÉDIO — Cache de embedding normaliza demais a chave

`_hash_key()` aplica `.strip().lower()` ao texto antes de gerar o hash. Consequência: `"Banco"` e `"banco"` compartilham
o mesmo vetor cacheado. Para BGE-M3, que é *case-sensitive*, isso introduz um erro sutil e difícil de diagnosticar —
especialmente prejudicial para siglas e nomes próprios (`"CNPJ"` vs `"cnpj"`, `"SP"` vs `"sp"`), que são exatamente os
casos onde o canal denso mais precisa de precisão. Faça o hash do texto exato.

#### 4.11 🟡 MÉDIO — `rerank()` calcula `original_rank` em O(n²)

```python
# src/retrieval/reranker.py:161-166
for new_rank, (chunk, score) in enumerate(scored, 1):
    original_rank = next(i + 1 for i, c in enumerate(chunks) if c.chunk_id == chunk.chunk_id)
```

Varredura linear dentro de um loop. Com pools de candidatos de 15–60 chunks o custo é irrelevante hoje, mas é uma
armadilha gratuita. Substitua por um dicionário construído antes do loop:

```python
rank_of = {c.chunk_id: i + 1 for i, c in enumerate(chunks)}
...
original_rank = rank_of[chunk.chunk_id]
```

#### 4.12 🟡 MÉDIO — Quebra de encapsulamento entre módulos

`src/ingestion/incremental_reindex.py` importa `_get_conn` de `bm25_index` — uma função com prefixo `_`, ou seja,
declarada como privada pelo próprio autor. Isso acopla o módulo de ingestão a um detalhe interno de conexão e impede que
`bm25_index` migre para um pool sem quebrar o consumidor. Exponha uma API pública (`get_connection()` ou, melhor, um
context manager `with bm25_connection() as conn:`).

#### 4.13 🟡 MÉDIO — `incremental_reindex` cobre apenas parte do pipeline de ingestão

O reindexador incremental: (i) usa somente `chunk_recursive`, ignorando as estratégias `semantic` e `parent_child`;
(ii) não indexa no grafo; (iii) não popula `acl_principals`; (iv) não remove do índice documentos apagados do disco.

O item (iii) é o mais sério: documentos reindexados incrementalmente entram **sem ACL**, e pela lógica fail-open de §3.3
tornam-se visíveis a todos. O item (iv) causa acúmulo de órfãos — chunks de documentos que não existem mais continuam
sendo retornados nas buscas.

#### 4.14 🟢 BAIXO — Thresholds mágicos sem calibração documentada

`COMPLEXITY_THRESHOLD = 0.6`, `MIN_WEIGHT_FLOOR = 0.10`, `CONFIDENCE_THRESHOLD = 0.0`, `RETRY_THRESHOLD = -2.5` e o
mapeamento ad-hoc `raw_score = (rrf_score * 100) - 2.0` em `evaluate_context_sufficiency()`. Nenhum deles tem
justificativa registrada nem procedimento de recalibração. `CONFIDENCE_THRESHOLD = 0.0` combinado com scores de
cross-encoder (logits, frequentemente negativos) merece atenção especial: pode estar reprovando contextos válidos.

Mova-os para `config.py` como campos de `Settings` (permitindo ajuste por ambiente) e registre em docstring o método usado
para chegar a cada valor — ou marque explicitamente como "chute inicial, pendente de calibração".

#### 4.15 🟢 BAIXO — Tabela de custos de LLM defasada

`_COST_TABLE` em `observability/instrumentation.py` está anotada como "preços Anthropic dez/2024". Estamos em set/2026 —
os valores de custo em USD reportados nos traces são, na melhor hipótese, aproximações antigas. Extraia para
`config.py`/YAML com data de vigência e emita um warning quando a tabela tiver mais de 6 meses.


---

### 5. Qualidade do código e conformidade com boas práticas Python

#### 5.1 Análise estática (`ruff check --select F,E,W`)

| Regra | Repo inteiro | Só `src/` + `rag.py` + `scripts/` | Descrição |
| :--- | ---: | ---: | :--- |
| `E501` | 588 | 352 | linha maior que 88 caracteres |
| `F401` | 107 | 52 | import não utilizado |
| `F541` | 11 | 7 | f-string sem placeholder |
| `E402` | 9 | 9 | import fora do topo do módulo |
| `F841` | 7 | 2 | variável atribuída e nunca usada |
| `E702` | 3 | 1 | múltiplos statements na mesma linha |
| `E741` | 2 | 2 | nome de variável ambíguo (`l`, `I`, `O`) |
| `W293` | 1 | 1 | linha em branco com espaços |
| **Total** | **728** | **426** | **excluindo `E501`: 140 / 74** |

O `E501` é subjetivo (o default do ruff é 88 colunas; muitos projetos adotam 100 ou 120) e pode ser desligado ou
recalibrado. Os demais **140 issues** são objetivos e majoritariamente auto-corrigíveis: `ruff check --fix` resolve 118
deles imediatamente.

Destaques que merecem atenção manual:

- **107 imports não utilizados** é sintoma de código movido entre módulos sem limpeza. Além do ruído, esconde dependências
  reais e infla o tempo de import (relevante aqui, onde importar `torch`/`FlagEmbedding` custa segundos).
- **9 `E402`** (imports no meio do arquivo) frequentemente indicam import circular sendo contornado. Vale investigar cada
  caso: se for circular, a solução é extrair a dependência comum, não esconder o import.
- **`B904`** (habilitando `--select B`): vários `raise` dentro de `except` sem `from e`, o que trunca a cadeia de exceções
  e dificulta o diagnóstico em produção.
- **`B905`**: `zip()` sem `strict=` — silenciosamente trunca quando as listas têm tamanhos diferentes. Em código de
  pipeline que casa embeddings com chunks, isso pode perder dados sem erro.

#### 5.2 🟡 90 `print()` contra 6 `logging.getLogger()`

Em `src/`: **90 chamadas a `print()`** e apenas **6 loggers instanciados**. `print()` aparece inclusive em caminhos
quentes como `embed_and_index()` (`"batch N: X chunks embedados"`) e no fluxo de streaming.

Por que isso importa:

- Impossível controlar verbosidade por ambiente (não há `DEBUG`/`INFO`/`WARNING`).
- Impossível redirecionar para arquivo, syslog ou coletor estruturado.
- Sem timestamp, sem nome de módulo, sem correlação com o `trace_id`.
- Uma biblioteca que escreve em `stdout` polui a saída de quem a consome — se `arquiteturaRAG` for importado por uma API
  ou por um worker, o `print` vaza no log da aplicação hospedeira.

**Correção:** substituição mecânica por módulo:

```python
logger = logging.getLogger(__name__)
# print(f"[BM25] Tabela e índices criados (ou já existiam).")
logger.info("Tabela e índices BM25 criados (ou já existiam)")
```

Depois, imponha a regra no lint: `ruff --select T20` (flake8-print) faz o CI reprovar novos `print()`.

#### 5.3 🟡 Efeito colateral no import de `instrumentation.py`

`_setup_logger()` é executado em nível de módulo (linha ~96 de `src/observability/instrumentation.py`): ao **importar** o
módulo, ele cria o diretório `logs/` e anexa um `StreamHandler`. Dois problemas:

1. **Import não deve ter efeito colateral no filesystem.** Um `import src.observability.instrumentation` num script de
   análise cria diretórios no CWD.
2. **Caminho relativo dependente do CWD.** Rodar a aplicação de outro diretório cria `logs/` no lugar errado — e é o
   mesmo motivo pelo qual os testes precisam de `sys.path.insert(0, ".")` (§7.2).

**Correção:** inicialização preguiçosa (`_get_logger()` chamado no primeiro uso) e caminho absoluto derivado da raiz do
pacote:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = Path(settings.log_dir) if settings.log_dir else PROJECT_ROOT / "logs"
```

#### 5.4 Tipagem e docstrings

| Métrica (só `src/`) | Valor |
| :--- | :--- |
| Funções totais | 240 |
| Com anotação de retorno | 207 (**86%**) |
| Com docstring | 175 (**73%**) |
| Classes | 36 — 23 com docstring (**64%**) |
| Módulos | 45 — 36 com docstring (**80%**) |
| `from __future__ import annotations` | uso consistente ✅ |

São números bons para um projeto deste porte. As lacunas se concentram em helpers privados. A qualidade das docstrings
existentes é alta (§2.2). Duas melhorias de baixo custo:

- Rodar `mypy --ignore-missing-imports src/` e tratar os erros incrementalmente. Com 86% de anotação, o retorno é alto.
- Adotar `Protocol`/dataclass único para o "chunk" que circula entre canais (§4.6) — hoje `dict`, `HybridResult` e
  `RerankResult` coexistem, o que anula boa parte do benefício da tipagem no caminho principal.

#### 5.5 🟢 Consistência de estilo

O código é **legível e uniforme**: nomes descritivos em português/inglês misturados mas coerentes dentro de cada módulo,
cabeçalhos de seção com separadores `# ---`, dataclasses para estruturas de retorno, `Optional[...]` explícito. Não há
"código espaguete". A adoção de `black` + `isort` (ou `ruff format`) fecharia o assunto de estilo definitivamente.

---

### 6. Tratamento de erros e robustez

#### 6.1 🟡 Onze exceções silenciadas em `src/`

Há **53 blocos `except Exception`** em `src/`, dos quais **11 seguem direto para `pass`**, sem log:

| Arquivo:linha | Contexto |
| :--- | :--- |
| `src/embedding/embed_index.py:233` | busca densa |
| `src/optimization/caching.py:243` | operação de cache |
| `src/parsing/pdf_parser.py:117` | parsing de PDF |
| `src/representation/multi_vector.py:170` | busca multi-vector |
| `src/retrieval/decomposition.py:105` | decomposição de query |
| `src/retrieval/graph_search.py:192, 217, 243` | travessia do grafo (3 ocorrências) |
| `src/retrieval/hyde.py:56` | geração HyDE |
| `src/observability/instrumentation.py:448` | escrita de trace |
| `src/evaluation/online_feedback.py:147` | registro de feedback |

Este é o achado 4.8 do review anterior, corrigido apenas parcialmente. O padrão é o mesmo em todos: falha silenciosa que
degrada a qualidade da resposta sem deixar rastro. Um pipeline de RAG que responde mal por causa de um parser quebrado é
muito mais difícil de diagnosticar do que um que falha alto.

Ponto positivo: **não há um único `except:` nu** no código — todos capturam `Exception` explicitamente.

**Correção mínima** (mantém a resiliência, elimina a cegueira):

```python
except Exception as e:
    logger.warning("Falha na travessia de 2 saltos para a entidade %s: %s", entity, e, exc_info=True)
    # segue com resultado parcial — comportamento intencional
```

Regra de bolso: `except: pass` é aceitável apenas quando o comentário adjacente explica **por que** a falha é
irrelevante. Nos 11 casos acima, não é.

#### 6.2 🟠 Cadeia de fallback do BM25 mascara erros reais de SQL

`search_bm25()` tenta `plainto_tsquery`; em caso de falha, refaz a busca com `to_tsquery` unindo os termos por `OR`; se
essa também falhar, executa `conn.rollback()` dentro de um `except Exception` sem log e retorna vazio.

O objetivo (tolerar queries com sintaxe problemática) é legítimo, mas do jeito atual **um erro de schema, uma conexão
morta ou um deadlock produzem exatamente o mesmo resultado que "nenhum documento encontrado"**. O canal lexical pode
estar 100% fora do ar e o sistema seguirá respondendo — pior, mas sem nenhum alarme.

**Correção:** distinguir os tipos de exceção do `psycopg2` e só aplicar fallback ao caso previsto:

```python
from psycopg2 import errors as pg_errors

try:
    ...  # plainto_tsquery
except pg_errors.SyntaxError:
    logger.info("Sintaxe de tsquery rejeitada; aplicando fallback OR-terms")
    ...  # fallback
except (pg_errors.OperationalError, pg_errors.InterfaceError) as e:
    logger.error("Canal BM25 indisponível: %s", e, exc_info=True)
    raise Bm25Unavailable(str(e)) from e     # falha alto — não finge que não há resultados
```

E exponha uma métrica `bm25_channel_errors_total` no trace, para que a degradação do canal seja visível.

#### 6.3 🟡 Falha do canal de grafo vira apenas um `warning`

```python
# src/retrieval/query_router.py:227-229
except Exception as e:
    logger.warning("Busca em grafo falhou (%s); prosseguindo com dense + bm25", e)
    graph_hits = []
```

Isso é razoável como política, mas hoje é o que mascara o `ImportError` de `networkx` (§8.1): o pacote está ausente,
todo import do canal de grafo falha, e o sistema opera permanentemente em modo degradado emitindo um warning que ninguém
lê. Registre a degradação no trace (`trace.metadata["graph_channel_degraded"] = True`) para que apareça na telemetria, e
faça a ausência de dependência falhar na inicialização, não a cada query.

#### 6.4 🟢 Pontos positivos de robustez

- Fallback Redis → cache em memória: correto e testado.
- `_get_conn()` verifica `_conn.closed` antes de reutilizar a conexão.
- Migração de schema idempotente.
- Fallback do reranker para candidatos brutos em caso de erro do cross-encoder (`self_rag.py:166-168`) — com log. Este é
  o padrão certo; replique-o nos 11 casos de §6.1.

---

### 7. Testes e integração contínua

#### 7.1 Inventário

| Métrica | Valor |
| :--- | :--- |
| Arquivos de teste | 33 |
| LOC de teste | 6.448 (35% do código total) |
| Funções `test_*` | 141 |
| Classes `Test*` | 21 |
| Coletados pelo `pytest --collect-only` | 145 testes, **19 erros de coleta** |
| Arquivos com mocks (`unittest.mock` / `monkeypatch`) | **4 de 33** |
| Arquivos com `skipif` / `pytest.skip` | **3 de 33** |
| Arquivos com `sys.path.insert(0, ".")` | **30 de 33** |
| `conftest.py` | ❌ ausente |
| `pytest.ini` / `pyproject.toml` / `setup.cfg` / `tox.ini` | ❌ todos ausentes |
| `.github/workflows/` | ❌ ausente — **não há CI** |

Os 19 erros de coleta observados neste ambiente decorrem de dependências não instaladas (ex.: `scenedetect`) e **não**
indicam bug no código — mas evidenciam que a suíte não roda em ambiente limpo sem preparação manual.

#### 7.2 🟠 ALTO — Sem CI e sem configuração de projeto

Não existe `.github/workflows/`. Todas as verificações desta revisão (lint, imports, coleta de testes) tiveram de ser
feitas manualmente — o que significa que nenhuma delas roda quando um commit é feito. Os 728 issues de lint e as
dependências faltantes teriam sido pegos automaticamente por um workflow de 20 linhas.

Além disso, a ausência de `conftest.py` e de `pyproject.toml` força **30 dos 33 arquivos** de teste a começarem com:

```python
sys.path.insert(0, ".")
```

Isso só funciona se `pytest` for invocado exatamente da raiz do repositório. Rodar `pytest tests/test_hybrid.py` de
dentro de `tests/` quebra tudo, e nenhuma IDE consegue descobrir os testes de forma confiável.

**Correção (3 arquivos, ~40 linhas, resolve os dois problemas):**

`pyproject.toml`:
```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths  = ["tests"]
markers = [
    "integration: exige serviços externos (qdrant, postgres, redis, neo4j)",
    "slow: carrega modelos pesados (torch, FlagEmbedding)",
]
addopts = "-m 'not integration and not slow' --strict-markers"

[tool.ruff]
line-length = 120
[tool.ruff.lint]
select = ["E", "F", "W", "B", "I", "T20", "UP"]
```

`tests/conftest.py`:
```python
import sys, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def _reachable(host: str, port: int) -> bool:
    import socket
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

@pytest.fixture(scope="session")
def qdrant_available():   return _reachable("localhost", 6333)

@pytest.fixture(scope="session")
def postgres_available(): return _reachable("localhost", 5432)
```

Depois disso, remova o `sys.path.insert` dos 30 arquivos (`sed -i '/sys.path.insert/d' tests/*.py`).

`.github/workflows/ci.yml`:
```yaml
name: CI
on: [push, pull_request]
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: ruff check .
      - run: python -c "import src.retrieval.graph_search"   # pega deps faltantes
      - run: pytest -m "not integration and not slow" --cov=src --cov-report=term-missing
```

#### 7.3 🟠 ALTO — Os testes dependem de infraestrutura viva

A grande maioria dos testes chama diretamente `search_bm25()`, `dense_search()`, `cache.*` — funções que abrem conexões
reais com Postgres, Qdrant e Redis. Com apenas **3 arquivos** usando `skipif`, o comportamento em ambiente sem
infraestrutura é falhar por `OperationalError`, e não pular de forma explícita.

O efeito prático é grave: **um teste que falha por falta de banco é indistinguível de um teste que falha por bug**. Isso
treina o time a ignorar a suíte vermelha, que é o pior estado possível para um conjunto de testes.

**Correção:** marque os testes de integração e faça-os pular quando o serviço estiver ausente:

```python
@pytest.mark.integration
def test_search_bm25_filtra_por_acl(postgres_available):
    if not postgres_available:
        pytest.skip("Postgres não disponível")
    ...
```

#### 7.4 🟠 Cobertura desequilibrada: falta teste onde mais dói

Apenas 4 de 33 arquivos usam mocks, o que significa que quase não há testes unitários puros — e as áreas de maior risco
identificadas neste review estão, justamente, sem cobertura:

| Comportamento crítico | Coberto? |
| :--- | :---: |
| `search_graph()` respeita `tenant_id`/`principals` | ❌ (o bug §3.1 passaria) |
| Toda função de busca exige `principals` | ❌ (§3.2/§3.3) |
| `adaptive_answer()` usa os chunks rerankeados na geração | ❌ (o bug §4.1 passaria) |
| `answer_stream()` popula o cache | ❌ (§4.5) |
| `search_bm25()` devolve `parent_chunk_id`/`parent_content` | ❌ (§4.6) |
| Conclusões do benchmark derivam dos dados | ❌ (§4.2) |
| Parsers (PDF/DOCX/XLSX) | ✅ **bem coberto**, com fixtures sintéticas via `reportlab` |
| Chunking | ✅ coberto |

Os testes de parsing são um destaque positivo — geram os arquivos de teste programaticamente em vez de versionar
binários, o que é a abordagem certa. O mesmo cuidado precisa chegar ao caminho de retrieval e segurança.

**Sugestão de teste de regressão de segurança** (barato e de altíssimo valor):

```python
import ast, pathlib, pytest

CAMINHOS = ["src/retrieval", "src/representation", "src/generation"]
ISENTAS = {"expand_parent_context", "rewrite_query_for_retry"}

def test_toda_funcao_de_busca_declara_tenant_e_principals():
    faltando = []
    for base in CAMINHOS:
        for p in pathlib.Path(base).rglob("*.py"):
            tree = ast.parse(p.read_text(encoding="utf-8"))
            for n in ast.walk(tree):
                if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if n.name.startswith("_") or n.name in ISENTAS:
                    continue
                if not any(k in n.name for k in ("search", "retrieve", "answer")):
                    continue
                args = {a.arg for a in n.args.args + n.args.kwonlyargs}
                if not {"tenant_id", "principals"} <= args:
                    faltando.append(f"{p}:{n.lineno} {n.name}")
    assert not faltando, "Funções de busca sem isolamento de tenant/ACL:\n" + "\n".join(faltando)
```

Esse único teste travaria a reintrodução de todos os problemas de §3.1 e §3.2.

---

### 8. Dependências e gestão de versões

#### 8.1 🔴 CRÍTICO — `networkx` e `pytest` ausentes de `requirements.txt`

Uma comparação por AST entre todos os imports de terceiros do projeto e o conteúdo de `requirements.txt` (24 dependências
efetivas) encontrou exatamente duas ausências:

| Pacote | Onde é usado | Consequência |
| :--- | :--- | :--- |
| `networkx` | `src/retrieval/graph_search.py` — **import de topo** | `ImportError` ao importar o módulo. Mascarado pelo `try/except` de `query_router.py:227` (vira warning), mas quebra `tests/test_graph_search.py` diretamente. |
| `pytest` | 33 arquivos em `tests/` | A suíte de testes é impossível de rodar após `pip install -r requirements.txt`. |

Um `pip install -r requirements.txt` num ambiente limpo produz um sistema com o canal de grafo permanentemente
desabilitado e sem forma de executar os testes. O review anterior já havia apontado 5 dependências faltantes
(`redis`, `anthropic`, `PyYAML`, `numpy`, `neo4j`) — todas corrigidas —, o que mostra que o processo funciona, apenas
precisa ser automatizado.

**Correção:**

```diff
# requirements.txt
+networkx>=3.2
```

E crie `requirements-dev.txt`:

```
-r requirements.txt
pytest>=8.0
pytest-cov>=5.0
ruff>=0.6
mypy>=1.11
```

Depois adicione ao CI o passo `python -c "import src.retrieval.graph_search"`, que pega esse tipo de omissão
automaticamente.

#### 8.2 🟡 Pinning inconsistente e ausência de lockfile

`requirements.txt` mistura `==`, `>=` e pacotes sem versão. Para um sistema que depende de `torch`, `FlagEmbedding` e
`qdrant-client` — ecossistemas com quebras de API frequentes — isso torna builds não reproduzíveis. O próprio código já
carrega uma cicatriz disso: o comentário em `embed_index.py` sobre `qdrant-client >= 1.7 usa query_points() em vez de
search()`.

**Sugestão:** adotar `pip-tools` (`requirements.in` → `requirements.txt` com hashes) ou `uv`/Poetry com lockfile
commitado. No mínimo, fixe as versões maiores (`torch>=2.3,<3`, `qdrant-client>=1.9,<2`).

#### 8.3 🟢 Organização do arquivo

`requirements.txt` está bem comentado, agrupando as dependências por camada do pipeline (parsing, embedding, retrieval,
geração). Isso ajuda quem for enxugar a instalação. Vale considerar *extras* opcionais
(`pip install .[video]`, `.[ocr]`) para evitar que todo mundo instale `scenedetect` e `torch` só para usar o canal BM25.

---

### 9. Performance e escalabilidade

#### 9.1 🟠 ALTO — Conexões e modelos como singletons globais sem pool nem thread-safety

```python
# src/retrieval/bm25_index.py:59-64
_conn = None

def _get_conn():
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(settings.postgres_url)
        _conn.autocommit = False
    return _conn
```

O mesmo padrão em `embed_index.py::_get_client()` (Qdrant) e nos singletons de modelo em `model.py` e `reranker.py`.

Problemas sob concorrência:

- **Uma única conexão Postgres compartilhada por todo o processo.** Conexões `psycopg2` não são thread-safe para uso
  simultâneo; dois workers concorrentes corrompem o protocolo.
- **`autocommit = False` sem `commit()` garantido** mantém transações abertas — o `rollback()` no fallback é o único
  ponto que as fecha. Transações ociosas seguram locks e consomem slots do servidor.
- **Sem `connect_timeout`, sem retry, sem health check.** Uma queda momentânea do banco derruba o processo até reiniciar.
- **Race na inicialização preguiçosa:** duas threads podem entrar no `if` simultaneamente e criar dois modelos BGE-M3 —
  cada um consumindo GB de RAM/VRAM.

**Correção:**

```python
from psycopg2 import pool

_pool: pool.ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()

def _get_pool() -> pool.ThreadedConnectionPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = pool.ThreadedConnectionPool(
                minconn=1, maxconn=settings.postgres_max_conn,
                dsn=settings.postgres_url, connect_timeout=5,
            )
    return _pool

@contextmanager
def bm25_connection():
    p = _get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)
```

Para os modelos, proteja a inicialização com `threading.Lock()` (double-checked locking).

#### 9.2 🟠 ALTO — `embed_and_index()` acumula todos os pontos em memória

```python
# src/embedding/embed_index.py:137-173
points = []
for i in range(0, len(chunks), batch_size):
    batch = chunks[i : i + batch_size]
    vectors = embed_texts(texts, batch_size=batch_size)
    ...
    points.append(PointStruct(...))          # acumula TUDO
client.upsert(collection_name=collection, points=points)   # um único upsert no final
```

O `batch_size` controla apenas o *embedding*; a indexação é monolítica. Cada `PointStruct` carrega o vetor de 1024
floats **e o texto completo do chunk no payload**. Para 100.000 chunks isso representa vários GB residentes antes do
primeiro byte chegar ao Qdrant — e, se o `upsert` falhar, todo o trabalho de embedding é perdido.

**Correção:** fazer upsert por batch, com retry:

```python
for i in range(0, len(chunks), batch_size):
    batch_points = [...]                                     # só este batch
    client.upsert(collection_name=collection, points=batch_points, wait=False)
    logger.info("indexados %d/%d chunks", min(i + batch_size, len(chunks)), len(chunks))
```

Isso torna o consumo de memória constante e a ingestão retomável.

#### 9.3 🟠 Latências medidas indicam problema sério de desempenho

De `benchmarks/results/benchmark_report.md`: *Dense BGE-M3 Pure* a **16,9 s**, *Hybrid + Reranker* a **194 s** e
*Multichannel + Tuned RRF + Reranker* a **253 s** de latência média por consulta. Ainda que a execução tenha sido em
hardware de desenvolvimento e provavelmente sem GPU, três ordens de grandeza acima do aceitável (< 2 s p95) merece
investigação dedicada. Suspeitos prováveis, em ordem:

1. Reranker em CPU processando pools de 15–60 candidatos (custo dominante — 194 s vs 1,3 s sem reranker).
2. Carregamento do modelo dentro do loop de medição, em vez de aquecido antes.
3. Retrieval duplicado do Self-RAG (§4.1).

**Sugestões:** aquecer os modelos antes de medir; limitar o pool do reranker (`top_k` de entrada ≤ 20); avaliar ONNX
Runtime ou quantização INT8 para o cross-encoder em CPU; reportar p50/p95/p99 em vez de apenas a média.

#### 9.4 🟡 Canais dense e BM25 executam em série

Em `hybrid_search()`/`routed_hybrid_search()`, `dense_search()` e `search_bm25()` são chamados sequencialmente, embora
sejam totalmente independentes (Qdrant e Postgres, sem dependência de dados entre si). Paralelizá-los reduz a latência do
retrieval ao máximo dos dois em vez da soma:

```python
with ThreadPoolExecutor(max_workers=3) as ex:
    f_dense = ex.submit(dense_search, query, top_k=pool, tenant_id=tenant_id, principals=principals)
    f_bm25  = ex.submit(search_bm25,  query, top_k=pool, tenant_id=tenant_id, principals=principals)
    f_graph = ex.submit(search_graph, query, top_k=pool, tenant_id=tenant_id, principals=principals)
    dense_hits, bm25_hits, graph_hits = f_dense.result(), f_bm25.result(), f_graph.result()
```

Ambos são I/O-bound, então threads bastam (sem custo de GIL relevante).

#### 9.5 🟡 Outros pontos de escalabilidade

- **Sem paginação nas buscas** — `top_k` é o único controle; não há cursor para navegar resultados.
- **Sem circuit breaker** para os serviços externos: se o Qdrant ficar lento, cada requisição espera o timeout completo.
- **Cache sem limite de memória configurável** no fallback LRU — em processos longos sem Redis, pode crescer indefinidamente
  dependendo da política adotada.
- **Ausência de batching no reranker** entre requisições concorrentes — cada query paga o overhead completo de inferência.

---

### 10. Documentação

#### 10.1 🟠 O README documenta apenas o "Passo 0"

`README.md` tem 58 linhas e cobre exclusivamente o setup do ambiente. Um projeto com 32 passos implementados, 45 módulos,
uma CLI com três subcomandos e cinco serviços de infraestrutura merece muito mais. Faltam:

- **O que o projeto faz** — hoje é preciso ler o código para descobrir que existe Graph RAG, Self-RAG e auto-tuning de RRF.
- **Diagrama do pipeline** (o de §1.2 deste relatório serve como ponto de partida).
- **Uso da CLI:** `python rag.py answer "pergunta"`, `python rag.py benchmark`, `python rag.py eval` — nem sequer
  mencionados no README, apesar de serem a interface principal.
- **Guia de ingestão:** como colocar documentos em `data/seed_corpus/` e indexá-los.
- **Modelo de segurança:** o que são `tenant_id` e `principals`, e como um chamador deve fornecê-los. Dado o achado §3.3,
  isso é documentação de segurança, não conveniência.
- **Variáveis de ambiente:** o `.env.example` está bom, mas o README não explica quais são obrigatórias.
- **Estado do projeto:** deixar explícito que é um protótipo avançado e não está pronto para produção.

O conteúdo de qualidade existe — `arquitetura-rag-revisada.md` (19 KB) e `runbook-construcao-rag.md` (78 KB) —, mas não
está referenciado a partir da porta de entrada. Basta um índice no README apontando para eles.

#### 10.2 🟢 Documentação inline é o ponto forte

Ver §2.2. 80% dos módulos e 73% das funções têm docstring, e a qualidade é acima da média porque explicam decisões, não
mecânica. Preserve esse padrão.

#### 10.3 🟡 `runbook-construcao-rag.md.bak` versionado

93 KB de backup manual dentro de um repositório Git. Remova (§3.5).

#### 10.4 🟢 Sugestão: registrar as decisões como ADRs

Boa parte do racional já escrito nas docstrings ("por que Postgres e não OpenSearch", "por que pré-filtro e não
pós-filtro", "por que `uuid5` para `document_id`") merece virar `docs/adr/NNN-titulo.md`. Isso as torna encontráveis e
versionáveis independentemente do código que as originou.

---

### 11. Status dos achados do review anterior

O arquivo `reviews/review_2026-09-04.md` documenta uma revisão prévia. Cada achado foi reverificado no código atual:

| # | Achado anterior | Status |
| :--- | :--- | :--- |
| 4.1 | Dependências faltando (`redis`, `anthropic`, `PyYAML`, `numpy`, `neo4j`) | ✅ **Corrigido** — porém surgiram/permaneceram `networkx` e `pytest` (§8.1) |
| 4.2 | `.env.example` desatualizado | ✅ **Corrigido** — 42 linhas, cobre Redis/Neo4j/cache |
| 4.3 | Senhas fracas hardcoded | ❌ **Não corrigido** (§3.4) |
| 4.4 | Cache sem isolamento por ACL | ✅ **Corrigido** — `_response_key()` inclui `principals` ordenados |
| 4.5 | `principals` não chegava à geração | ✅ **Corrigido** — propagado em `answer()`, `_answer_impl()`, `answer_stream()`, `adaptive_answer()`, `routed_hybrid_search()` |
| 4.6 | Streaming falso (resposta completa fatiada) | ✅ **Corrigido** — `_stream_claude()` usa `client.messages.stream` |
| 4.7 | Cache descartava `chunks_used` | ✅ **Corrigido** — `_serialize_chunk` / `_deserialize_chunk` |
| 4.8 | `except Exception: pass` | ⚠️ **Parcial** — ainda 11 ocorrências (§6.1) |
| 4.9 | Conexão Postgres singleton sem pool | ❌ **Não corrigido** (§9.1) |
| 4.11 | `Tuple` usado sem import em `benchmark.py` | ✅ **Corrigido** — import na linha 23 |
| 4.13 | Duplicação da lógica de RRF | ✅ **Corrigido** — `weighted_rrf()` delega a `reciprocal_rank_fusion()` |
| 5.1 | Testes fantasmas (arquivos sem função de teste) | ✅ **Corrigido** — `test_generation.py` e `test_evaluation.py` agora têm testes reais |
| 6 | `rag.py` só expunha `benchmark` | ✅ **Corrigido** — `answer` / `benchmark` / `eval` com `argparse` |

**9 corrigidos, 1 parcial, 3 pendentes.** É uma taxa de follow-through boa e o principal indicador de que este projeto
tende a evoluir bem. A recomendação estrutural é converter esse esforço manual em automação (§7.2): o que o CI verifica
não precisa ser lembrado.

---

### 12. Plano de ação priorizado

#### Sprint 1 — Bloqueadores (não promover a produção antes disto)

| # | Ação | Ref. | Esforço |
| :---: | :--- | :---: | :---: |
| 1 | Propagar `tenant_id`/`principals` para `search_graph()` e filtrar na travessia; enquanto isso, `use_graph=False` por padrão | §3.1 | M |
| 2 | Tornar o modelo de ACL fail-closed (`resolve_principals()` + sentinela `ANY_PRINCIPAL`) | §3.3 | M |
| 3 | Adicionar `principals` às 6 rotas alternativas (HyDE, decomposition, multi-vector, benchmark, offline_eval, migration) | §3.2 | M |
| 4 | Passar `reranked` para `answer()` via `prefetched_chunks` no Self-RAG | §4.1 | P |
| 5 | Adicionar `networkx` ao `requirements.txt` e criar `requirements-dev.txt` | §8.1 | P |
| 6 | Derivar as conclusões do benchmark dos dados; falhar quando o recall for degenerado | §4.2 | P |
| 7 | `chunk_id` determinístico + reancorar o golden dataset em `document_id` + spans | §4.3 | G |
| 8 | Criar `.github/workflows/ci.yml`, `pyproject.toml` e `tests/conftest.py` | §7.2 | P |
| 9 | Adicionar o teste de regressão de ACL por AST | §7.4 | P |

#### Sprint 2 — Alto impacto

| # | Ação | Ref. | Esforço |
| :---: | :--- | :---: | :---: |
| 10 | Pool de conexões Postgres + context manager público; locks nos singletons de modelo | §9.1, §4.12 | M |
| 11 | Upsert por batch em `embed_and_index()` + corrigir o fallback de `chunk_index` | §9.2, §4.8 | P |
| 12 | `answer_stream()`: gravar no cache, registrar custo e abrir trace | §4.5 | M |
| 13 | Propagar `parent_chunk_id`/`parent_content` no BM25 e no `RerankResult` | §4.6 | P |
| 14 | Implementar `search_graph()` via Cypher no Neo4j (com ACL embutida) | §4.4 | G |
| 15 | Marcar testes de integração e adicionar fixtures de disponibilidade de serviço | §7.3 | M |
| 16 | Investigar as latências de 194–253 s (aquecimento de modelo, tamanho do pool, ONNX) | §9.3 | G |
| 17 | Reescrever o README (visão geral, diagrama, CLI, ingestão, modelo de segurança) | §10.1 | M |

#### Sprint 3 — Melhoria contínua

| # | Ação | Ref. | Esforço |
| :---: | :--- | :---: | :---: |
| 18 | `ruff check --fix` + substituir os 90 `print()` por `logging`; ativar `T20` no lint | §5.1, §5.2 | M |
| 19 | Logar as 11 exceções silenciadas; diferenciar erros de sintaxe SQL de erros operacionais no BM25 | §6.1, §6.2 | M |
| 20 | Paralelizar os canais dense/BM25/graph com `ThreadPoolExecutor` | §9.4 | P |
| 21 | Validar credenciais de produção no `Settings`; prefixar o modo simulado | §3.4, §4.7 | P |
| 22 | Remover artefatos de runtime e o `.bak` do versionamento | §3.5, §10.3 | P |
| 23 | Mover thresholds mágicos para `Settings` e documentar a calibração | §4.14 | P |
| 24 | Unificar os três formatos de chunk num único `RetrievedChunk` | §4.6, §5.4 | G |
| 25 | Atualizar a tabela de custos de LLM com data de vigência | §4.15 | P |
| 26 | Corrigir `embedding_dimension()`, o hash do cache de embedding e o O(n²) do reranker | §4.9–§4.11 | P |
| 27 | Completar o `incremental_reindex` (ACL, grafo, remoção de órfãos, estratégias de chunking) | §4.13 | G |
| 28 | Introduzir lockfile e pinning consistente; extras opcionais de instalação | §8.2, §8.3 | M |
| 29 | Rodar `mypy` incrementalmente; registrar as decisões como ADRs | §5.4, §10.4 | M |

*Esforço: P = até meio dia · M = 1–3 dias · G = mais de 3 dias.*

---

### 13. Conclusão

`arquiteturaRAG` é um **protótipo avançado de alta qualidade**, com arquitetura bem pensada e amplitude funcional que
supera a de muitos produtos comerciais de RAG. O código é legível, as decisões de design estão documentadas com racional
explícito, e o histórico mostra um autor que revisa o próprio trabalho e efetivamente aplica as correções — 9 dos 13
achados do review anterior foram resolvidos.

O que separa o projeto de estar pronto para produção não é a arquitetura, e sim um conjunto pequeno e bem delimitado de
problemas: **o modelo de controle de acesso precisa deixar de ser fail-open e passar a cobrir todos os canais**, **o
Self-RAG precisa usar os chunks que valida**, **a avaliação precisa refletir os dados medidos** e **precisa existir um CI
que impeça a reintrodução de tudo isso**. Nada aqui exige reescrita; o Sprint 1 é majoritariamente propagação de
parâmetros, inversão de um default e três arquivos de configuração.

Uma observação estrutural que atravessa vários achados: **quase todos os problemas encontrados teriam sido capturados por
automação**. Dependências faltantes → um `import` no CI. Imports não usados → `ruff`. Bypass de ACL → o teste de AST de
§7.4. Relatório de benchmark falso → uma asserção sobre recall degenerado. O projeto já demonstrou que sabe corrigir o
que identifica; o próximo salto de maturidade é transferir a identificação do revisor humano para o pipeline.

**Recomendação:** executar o Sprint 1 antes de qualquer uso com dados reais de múltiplos clientes. Concluído esse
sprint, o sistema passa a ter uma base de segurança e avaliação confiável sobre a qual as otimizações do Sprint 2 valem
o investimento.

---

*Revisão estática realizada em 03/09/2026 sobre o commit `3cb602f`. Nenhum arquivo do repositório foi modificado. Os
testes não foram executados — as evidências vêm de leitura de código, análise de AST, `ruff` e coleta do `pytest`.*

*Nota operacional: para que revisões futuras alcancem repositórios privados sem intervenção manual, o acesso pode ser
concedido em [Github App](https://github.com/apps/abacusai/installations/select_target).*
