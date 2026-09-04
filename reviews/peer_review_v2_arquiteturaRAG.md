# Peer Review v2 — `ciecolopes01/arquiteturaRAG`

**Revisão de acompanhamento (follow-up) após a refatoração de resposta ao review v1**

| Campo | Valor |
| :--- | :--- |
| Repositório | `ciecolopes01/arquiteturaRAG` (branch `master`) |
| Commit revisado agora | `e8607ca` — *fix(security,retrieval,eval): address peer review findings* |
| Commit da revisão anterior | `3cb602f` |
| Autor do commit | Cristiano Lopes `<clopes.systems@alelo.com.br>` |
| Data do commit | 03/09/2026 23:53 (-03) |
| Diff analisado | 28 arquivos · +2.216 / −1.870 linhas |
| Volume atual | 10.070 LoC em `src/` · 6.605 LoC em `tests/` (33 arquivos, 143 funções de teste, 247 casos coletados) |
| Data desta revisão | 04/09/2026 |

---

### Sumário executivo

A refatoração é **substancial e majoritariamente bem direcionada**. Dos 5 achados críticos do review v1, **2 foram resolvidos de forma completa e correta**, 2 foram resolvidos **apenas parcialmente** (o mecanismo foi criado, mas não foi ligado ao caminho de execução real) e 1 foi resolvido no código mas **não no artefato entregue**. A infraestrutura de projeto que faltava (`pyproject.toml`, `requirements-dev.txt`, CI, `conftest.py`, teste de regressão de segurança por AST) foi introduzida — este é o maior ganho estrutural do commit.

Entretanto, a revisão identificou **uma regressão crítica nova, reproduzida em execução**, causada pela interação entre duas correções que foram feitas isoladamente: o caminho Self-RAG → `answer()` agora lança `AttributeError` em runtime. Além disso, **o pipeline de CI recém-criado falha no primeiro passo** (`ruff check .` retorna 180 erros) e no passo de testes (`test_acl.py` exige Qdrant/Postgres vivos, que o workflow não provisiona).

> [!IMPORTANT]
> **Veredito:** a direção está correta e o commit representa progresso real, mas **o sistema ainda não está pronto para promoção a produção**. O bloqueador mais urgente não é nenhum dos achados antigos — é a regressão nova no Self-RAG, que quebra em execução um fluxo que antes apenas gastava latência a mais.

#### Placar por severidade

| Severidade | v2 | Composição |
| :--- | :---: | :--- |
| 🔴 Crítico | 4 | 1 regressão nova · 2 persistem do v1 · 1 pendente apenas no artefato |
| 🟠 Alto | 7 | **4 novos** · 3 persistem do v1 |
| 🟡 Médio | 14 | **4 novos** · 10 persistem do v1 |
| 🟢 Baixo | 4 | **2 novos** · 2 persistem do v1 |

Contra o placar do v1 (6 críticos · 9 altos · 12 médios · 8 baixos), o resultado consolidado da tabela de status da §1 é: **12 achados resolvidos · 7 parciais · 14 não resolvidos · 10 novos**.

#### Nota por dimensão (0–10)

| Dimensão | v1 | v2 | Comentário |
| :--- | :---: | :---: | :--- |
| Arquitetura e design | 8,5 | 8,5 | Estrutura em camadas permanece limpa; nenhuma mudança estrutural |
| Qualidade do código | 6,5 | 6,5 | 180 violações de lint sob a configuração do próprio projeto |
| Tratamento de erros | 5,5 | 7,0 | Isolamento do `SyntaxError` no BM25 e logging no `graph_search` |
| Segurança | 4,0 | 6,0 | Grande avanço no canal de grafo; ainda *fail-open* por omissão |
| Testes | 5,0 | 5,5 | CI e `conftest.py` criados, porém CI está vermelho e fixtures não são usadas |
| Documentação | 6,0 | 8,5 | README reescrito e completo — melhor evolução do commit |
| Dependências | 6,0 | 7,5 | `networkx`/`pytest` incluídos, `requirements-dev.txt` e `pyproject.toml` criados |
| Performance | 5,0 | 6,0 | Upsert em lote e `rerank()` em O(n) corrigidos; singletons permanecem |
| **Média ponderada** | **≈ 5,8** | **≈ 6,9** | |

---

### 1. Status dos achados do review v1

Legenda: ✅ resolvido · 🟨 parcial · ❌ não resolvido · ⚠️ resolvido no código, pendente no artefato

| # (v1) | Achado | Sev. | Status | Evidência verificada nesta revisão |
| :--- | :--- | :---: | :---: | :--- |
| 3.1 | Bypass de ACL/tenant no canal de grafo | 🔴 | ✅ | `search_graph()` agora recebe `tenant_id`/`principals`; `chunk_metadata` populado em `index_chunks()`; filtro aplicado antes do corte de `top_k` |
| 3.2 | Rotas alternativas sem propagação de `principals` | 🔴 | ✅ | `hyde`, `decomposition`, `multi_vector`, `offline_eval` e `query_router` propagam `principals`; `acl_principals` indexado no Qdrant com `MatchAny` |
| 3.3 | Modelo de ACL *fail-open* | 🔴 | 🟨 | `ANY_PRINCIPAL` e `resolve_principals()` criados — porém **nenhum código de produção os chama** (ver §2.2) |
| 3.4 | Credenciais fracas hardcoded como default | 🟡 | ❌ | `src/config.py` ainda traz `rag_dev_password`, `minioadmin`, `rag_neo4j_password` como defaults |
| 3.5 | Artefatos de runtime versionados | 🟡 | 🟨 | `logs/*.jsonl` removidos e `.gitignore` ampliado; `benchmarks/results/benchmark_report.{md,json}` **continuam rastreados** |
| 4.1 | Self-RAG descarta chunks já validados | 🔴 | 🟨 | `prefetched_chunks` implementado corretamente, mas **quebra em runtime** (ver §2.1) |
| 4.2 | Conclusões de benchmark contradizem os dados | 🔴 | ⚠️ | `_derive_conclusions()` implementado com aviso de backend offline; o relatório commitado **ainda contém o texto falso antigo** |
| 4.3 | Golden dataset ancorado em UUIDs efêmeros | 🔴 | ❌ | `generate_chunk_id()` é UUIDv5, mas semeado por `Document.id`, que continua sendo `uuid4()` → **IDs continuam mudando a cada ingestão** (reproduzido, ver §2.3) |
| 4.4 | Canal de grafo é *write-only* (Neo4j nunca lido) | 🟠 | ❌ | `graph_search.py` só executa `MERGE`/`DETACH DELETE`; nenhum `MATCH` de leitura |
| 4.5 | `answer_stream()` não grava cache, custo nem trace | 🟠 | ✅ | Bloco `finally` acumula tokens, chama `record_generation_cost()`, grava cache e fecha o `pipeline_trace` |
| 4.6 | Perda de contexto pai em BM25 e reranker | 🟠 | ✅ | `SEARCH_SQL`/`SEARCH_SQL_SIMPLE` selecionam `parent_chunk_id`/`parent_content`; `RerankResult` ganhou os dois campos |
| 4.7 | Modo simulado indistinguível de resposta real | 🟡 | ❌ | Sem marcação no texto da resposta |
| 4.8 | `chunk_index` incorreto no fallback de `embed_and_index()` | 🟡 | ✅ | Corrigido para `i + offset` |
| 4.9 | `embedding_dimension()` retorna 1024 hardcoded | 🟡 | ❌ | Inalterado |
| 4.10 | Chave de cache de embedding normaliza demais | 🟡 | ❌ | `key_data.strip().lower()` mantido |
| 4.11 | `rerank()` calcula `original_rank` em O(n²) | 🟡 | ✅ | Substituído por dicionário `original_ranks` (O(n)) |
| 4.12 | Quebra de encapsulamento entre módulos | 🟡 | ❌ | Inalterado |
| 4.13 | `incremental_reindex` cobre parte do pipeline | 🟡 | ❌ | Inalterado |
| 4.14 | Thresholds mágicos sem calibração documentada | 🟢 | ❌ | Inalterado |
| 4.15 | Tabela de custos de LLM defasada | 🟢 | ❌ | Inalterado |
| 5.2 | 90 `print()` contra 6 `logging.getLogger()` | 🟡 | ❌ | Agora **91 `print()`** contra 7 loggers |
| 6.2 | Fallback do BM25 mascara erros reais de SQL | 🟠 | ✅ | `except pg_errors.SyntaxError` isolado; demais `psycopg2.Error` propagam |
| 6.3 | Falha do canal de grafo vira apenas um `warning` | 🟠 | ✅ | Agora marca `trace.metadata["graph_channel_degraded"] = True` |
| 7.2 | Sem CI e sem configuração de projeto | 🟠 | 🟨 | `pyproject.toml` + `.github/workflows/ci.yml` criados, mas o CI **falha** (ver §3.1) |
| 7.3 | Testes dependem de infraestrutura viva | 🟠 | 🟨 | `conftest.py` com fixtures `require_*` criado, mas **nenhum teste as utiliza** (ver §3.2) |
| 7.4 | Cobertura desequilibrada | 🟠 | 🟨 | `test_security_ast.py` adicionado (excelente ideia); demais lacunas permanecem |
| 8.1 | `networkx` e `pytest` ausentes de `requirements.txt` | 🔴 | ✅ | Ambos adicionados; `requirements-dev.txt` criado |
| 8.2 | Pinning inconsistente e sem lockfile | 🟡 | ❌ | Continua tudo em `>=`, sem lockfile |
| 9.1 | Singletons globais sem pool nem thread-safety | 🟠 | ❌ | `_conn`, `_client`, `_model`, `_reranker`, `_GRAPH_INDEX` inalterados |
| 9.2 | `embed_and_index()` acumula tudo em memória | 🟠 | ✅ | Upsert por batch com `total_upserted` |
| 9.4 | Canais dense e BM25 em série | 🟡 | ❌ | Inalterado |
| 10.1 | README documenta apenas o "Passo 0" | 🟠 | ✅ | README reescrito: diagrama Mermaid, capacidades, instalação, CLI, testes, estrutura |
| 10.3 | `runbook-construcao-rag.md.bak` versionado | 🟡 | ✅ | Removido (−1.587 linhas) e `*.bak` no `.gitignore` |

---

### 2. Achados críticos

#### 2.1 🔴 CRÍTICO · **REGRESSÃO NOVA** — o caminho Self-RAG → `answer()` lança `AttributeError` em runtime

Esta é a consequência não intencional de duas correções corretas feitas isoladamente:

1. O fix do achado 4.1 fez `adaptive_answer()` passar os chunks já rerankeados adiante:
   ```python
   # src/generation/self_rag.py:229-234
   ans_res = answer(..., prefetched_chunks=reranked)   # reranked: List[RerankResult]
   ```
2. O fix do achado 4.6 deu a `RerankResult` os campos `parent_chunk_id` e `parent_content`.

O problema é que `answer()` declara `prefetched_chunks: Optional[List[HybridResult]]` e repassa a lista, sem conversão, para `expand_parent_context()` (que roda por padrão, pois `expand_parents=True`). Essa função acessa atributos que **só existem em `HybridResult`**:

```python
# src/generation/answer.py:186-196
expanded.append(
    HybridResult(
        chunk_id=p_id,
        content=p_content,
        rrf_score=chunk.rrf_score,
        ...
        dense_rank=chunk.dense_rank,     # ❌ RerankResult não possui
        bm25_rank=chunk.bm25_rank,       # ❌
        dense_score=chunk.dense_score,   # ❌
        bm25_score=chunk.bm25_score,     # ❌
    )
)
```

`RerankResult` expõe apenas: `chunk_id`, `content`, `rerank_score`, `original_rank`, `new_rank`, `source_uri`, `section_title`, `document_title`, `rrf_score`, `parent_chunk_id`, `parent_content`.

**Reprodução executada nesta revisão** (`PYTHONPATH=. python`, no commit `e8607ca`):

```
1) expand_parent_context com RerankResult (caminho do Self-RAG):
   FALHOU -> AttributeError: 'RerankResult' object has no attribute 'dense_rank'
2) _serialize_chunk com RerankResult (gravação de cache):
   FALHOU -> AttributeError: 'RerankResult' object has no attribute 'dense_rank'
```

**Impacto:**

- **Crash não tratado:** basta que **um** chunk rerankeado tenha `parent_chunk_id` **e** `parent_content` preenchidos — exatamente o cenário que o fix do 4.6 acabou de habilitar — para `adaptive_answer()` estourar `AttributeError`. Não há `try/except` nesse caminho.
- **Falha silenciosa quando não há pai:** se nenhum chunk tiver contexto pai, o fluxo sobrevive, mas a gravação de cache (`_serialize_chunk`) estoura dentro de um `try/except ... logger.debug(...)`. Resultado: **toda resposta gerada via Self-RAG deixa de ser cacheada, sem nenhum sinal visível** — o log fica em nível `debug`.
- O `except` de fallback do reranker (`reranked = candidates[:top_k]`) torna o tipo da lista **heterogêneo** entre execuções (`RerankResult` no caminho feliz, `HybridResult` no fallback), o que faz o bug se manifestar de forma intermitente e difícil de diagnosticar.

**Correção sugerida** — converter explicitamente na fronteira, em vez de confiar em *duck typing*:

```python
# src/retrieval/reranker.py
def to_hybrid_result(r: "RerankResult") -> HybridResult:
    """Converte um RerankResult de volta ao contrato HybridResult."""
    return HybridResult(
        chunk_id=r.chunk_id,
        content=r.content,
        rrf_score=r.rrf_score,
        source_uri=r.source_uri,
        document_title=r.document_title,
        section_title=r.section_title,
        dense_rank=None,
        bm25_rank=None,
        graph_rank=None,
        dense_score=None,
        bm25_score=None,
        graph_score=r.rerank_score,   # ou campo próprio 'rerank_score'
        parent_chunk_id=r.parent_chunk_id,
        parent_content=r.parent_content,
    )

# src/generation/self_rag.py
from src.retrieval.reranker import to_hybrid_result

prefetched = [
    c if isinstance(c, HybridResult) else to_hybrid_result(c)
    for c in reranked
]
ans_res = answer(..., prefetched_chunks=prefetched)
```

E um teste de regressão que não dependa de infraestrutura:

```python
def test_answer_accepts_reranked_chunks_with_parent_context():
    rr = RerankResult(chunk_id="c1", content="filho", rerank_score=2.0,
                      original_rank=1, new_rank=1, parent_chunk_id="p1",
                      parent_content="pai")
    out = expand_parent_context([to_hybrid_result(rr)])
    assert out[0].content == "pai"
```

---

#### 2.2 🔴 CRÍTICO · **NÃO RESOLVIDO** — o *fail-open* de ACL continua intacto: `resolve_principals()` nunca é chamado

O commit criou o sentinela `ANY_PRINCIPAL` e a função `resolve_principals()` em `src/retrieval/acl_filter.py`. A ideia é exatamente a certa. O problema é que **o mecanismo não foi ligado a lugar nenhum**.

Busca em todo o repositório por usos fora do próprio módulo de definição:

```
./tests/test_security_ast.py:10, 62-84   ← únicas ocorrências
```

Ou seja: `search_dense()`, `search_bm25()`, `hybrid_search()`, `routed_hybrid_search()`, `answer()`, `adaptive_answer()` e `search_graph()` continuam com `principals: Optional[List[str]] = None` e continuam tratando `None` como "sem filtro de ACL". O comportamento em runtime é **byte a byte idêntico ao de antes do commit**: uma chamada que esqueça `principals` devolve todos os chunks do tenant.

Agrava o quadro o fato de que `resolve_principals()` tem `allow_unrestricted: bool = True` como padrão — isto é, mesmo se fosse chamado com a assinatura padrão, ele continuaria sendo *fail-open*.

O teste `test_acl_sentinel_and_resolution` passa, mas ele testa a função **em isolamento**. Ele não prova nada sobre o comportamento do sistema, e cria a impressão perigosa de que o problema foi endereçado.

**Correção sugerida** — invocar na fronteira de cada função pública de busca e inverter o padrão:

```python
# src/retrieval/acl_filter.py
def resolve_principals(principals, tenant_id=None, allow_unrestricted=False):  # ← padrão invertido
    ...

# src/retrieval/hybrid.py
def hybrid_search(query, top_k=5, tenant_id=None, principals=None, ...):
    principals = resolve_principals(principals, tenant_id=tenant_id)
    ...
```

E um teste que valide o **comportamento**, não a função:

```python
@pytest.mark.security
def test_busca_com_tenant_sem_principals_falha_fechada():
    with pytest.raises(ValueError, match="requer 'principals' explícito"):
        hybrid_search("consulta", tenant_id="tenant-A")
```

---

#### 2.3 🔴 CRÍTICO · **NÃO RESOLVIDO** — os `chunk_id` continuam mudando a cada ingestão

O commit introduziu `generate_chunk_id()` com UUIDv5 — o que está correto na intenção — mas a semente inclui `document.id`:

```python
# src/chunking/recursive.py:39-42
def generate_chunk_id(document_id: str, chunk_index: int, content: str) -> str:
    content_hash = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()[:16]
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"{document_id}:{chunk_index}:{content_hash}"))
```

E `Document.id` continua sendo aleatório:

```python
# src/parsing/canonical_model.py:34,58
def _new_id() -> str:
    return str(uuid.uuid4())
...
id: str = Field(default_factory=_new_id)
```

Um UUIDv5 semeado por um UUIDv4 é, na prática, um UUIDv4. **Reprodução executada nesta revisão** — mesmo arquivo, parseado e chunkado duas vezes:

```
document.id run1: 8b80d6d7-a0fa-4cd8-a9dc-1b8460f7a9cf
document.id run2: 9d340a1f-ee13-4db2-b540-06cde50b8817
chunk_id[0] run1: ba58f3f3-cfbb-5e32-81af-77f662dd8203
chunk_id[0] run2: e5b1af1e-1d0e-51cd-bca9-4ebd5106964a
=> IDs ESTÁVEIS ENTRE INGESTÕES? False
```

Consequências que permanecem exatamente como no v1:

- `data/golden_dataset.json` (48 perguntas, 79 `expected_chunk_ids`) continua com os UUIDs antigos, gerados em 2024 no commit `0bd20e8`. Não foi regenerado.
- **Recall@5 continuará 0.0000** em toda a suíte de benchmark após qualquer reindexação.
- Reindexações incrementais continuam criando pontos duplicados no Qdrant em vez de atualizar os existentes.
- O README afirma "Chunker Recursivo Determinístico UUIDv5" e "Chunks com identificadores determinísticos" — **a documentação agora declara uma propriedade que o código não entrega**.

**Correção sugerida** — ancorar a identidade do documento em algo estável do conteúdo/origem:

```python
# src/parsing/canonical_model.py
DOC_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "arquiteturaRAG.documents")

def document_id_for(source_uri: str, content_hash: str) -> str:
    return str(uuid.uuid5(DOC_NAMESPACE, f"{source_uri}:{content_hash}"))
```

Depois: regenerar `data/golden_dataset.json` com os IDs determinísticos (via `scripts/list_chunk_ids.py`) e **versionar o corpus semente** — hoje `data/seed_corpus/*` está no `.gitignore` e apenas o `.gitkeep` é rastreado, o que torna o golden dataset impossível de reproduzir a partir do repositório. Sete testes de `test_pdf_docx_parsers.py` falham com `FileNotFoundError` exatamente por isso.

---

#### 2.4 🔴 CRÍTICO · **PENDENTE NO ARTEFATO** — o relatório de benchmark commitado ainda afirma o que os dados refutam

O código foi corrigido: `_derive_conclusions()` gera conclusões a partir dos números observados e emite um bloco `> [!WARNING]` quando `max_recall == 0.0`. A implementação está boa.

Mas `benchmarks/results/benchmark_report.md` **continua rastreado no git** (o `.gitignore` só afeta arquivos ainda não versionados) e continua com o texto estático antigo, colado logo abaixo de uma tabela de zeros:

```
| **Multichannel + Tuned RRF + Reranker** | **0.0000** | 0.0000 | 253141.1 ms | ...

## Conclusão e Diretrizes de Produção
- **Perguntas Relacionais:** A introdução do canal de Grafo (Neo4j / Graph RAG) eleva
  o Recall@5 substancialmente ao explorar vizinhança multi-hop.
- **Reranking:** O modelo cross-encoder eleva o MRR consistente em todas as classes.
```

Qualquer pessoa que abra o repositório hoje lê uma afirmação de eficácia que os próprios números na linha acima desmentem.

**Confirmação de que o código novo funciona:** ao rodar a suíte de testes nesta revisão, o relatório foi regerado e produziu exatamente o comportamento pretendido —

```
## Conclusão e Diretrizes de Produção
> [!WARNING]
> **Atenção:** Todas as configurações obtiveram Recall@5 = 0.0. Isso geralmente indica que os
> serviços de backend (Qdrant, Postgres, Neo4j) estavam indisponíveis ou não indexados...
- **Melhor Configuração Global:** `BM25 Pure` com Recall@5 de **0.0000** e MRR de **0.0000**.
- **Menor Latência Média:** `Hybrid RRF + BGE Reranker` (0.7ms).
```

Ou seja, `_derive_conclusions()` está correta — falta apenas remover o artefato antigo do controle de versão.

**Efeito colateral notado:** essa regeração aconteceu como *side effect* de `pytest`. Como o arquivo continua rastreado, **rodar a suíte de testes suja a árvore de trabalho** (`git status` acusa `M benchmarks/results/benchmark_report.{md,json}`). Isso polui diffs e pode gerar commits acidentais de resultados de benchmark locais.

Correção: `git rm --cached benchmarks/results/benchmark_report.{md,json}` e regerar apenas após corrigir §2.3.

---

### 3. Achados de severidade alta

#### 3.1 🟠 ALTO · **NOVO** — o pipeline de CI criado neste commit está vermelho

O workflow `.github/workflows/ci.yml` é uma adição muito bem-vinda, mas **falha nos dois passos que importam**:

**Passo `Lint with Ruff` → `ruff check .`**

Executado nesta revisão contra a configuração do próprio projeto (`select = ["E","F","W","I"]`, `line-length = 120`):

```
Found 180 errors.
```

| Regra | Ocorrências | Descrição |
| :--- | :---: | :--- |
| `F401` | 105 | Import não utilizado |
| `I001` | 42 | Bloco de imports não ordenado |
| `F541` | 11 | f-string sem placeholder |
| `E402` | 9 | Import fora do topo do módulo |
| `F841` | 7 | Variável local atribuída e nunca usada |
| `E702` | 3 | Múltiplos statements na mesma linha |
| `E741` | 2 | Nome de variável ambíguo (`l`) |
| `W293` | 1 | Linha em branco com espaços |

Distribuição: 87 em `tests/`, 72 em `src/`, 21 em `scripts/`. Como o lint é o **primeiro** passo, o job aborta antes de rodar qualquer teste.

**Passo `Run offline unit tests`** — o workflow roda `pytest tests/test_acl.py`, mas:

- `test_acl.py` importa `embed_and_index` e chama `search_dense`, exigindo **FlagEmbedding + torch + Qdrant + Postgres vivos**;
- o workflow **não declara nenhum bloco `services:`** (verificado: 0 ocorrências), logo não há Qdrant nem Postgres no runner;
- `test_acl.py` não usa nenhuma fixture de skip.

Execução local desse conjunto exato: **8 testes falharam** com `ImportError: FlagEmbedding não instalado` — e falhariam de novo com `psycopg2.OperationalError` mesmo com o modelo instalado.

Observação adicional: `requirements-dev.txt` faz `-r requirements.txt`, que traz `torch>=2.0.0` e `FlagEmbedding>=1.2.10`. Cada execução de CI baixaria ~2–3 GB. Recomenda-se separar um `requirements-ci.txt` enxuto.

**Correção sugerida:**

```yaml
      - name: Lint with Ruff
        run: ruff check src/ tests/ scripts/ --output-format=github

      - name: Run offline unit tests
        run: pytest -m "not integration" --tb=short
```

Antes disso, rodar `ruff check . --fix` (155 dos 180 são autocorrigíveis) e revisar manualmente os `E402`/`E741` restantes.

---

#### 3.2 🟠 ALTO · **NOVO** — as fixtures de `conftest.py` foram criadas mas nenhum teste as utiliza

`tests/conftest.py` (72 linhas) define `qdrant_available`, `postgres_available`, `neo4j_available` e as fixtures de skip `require_qdrant`, `require_postgres`, `require_neo4j`. A implementação com `socket.create_connection` está correta e é elegante.

Verificação em todo o diretório `tests/`, excluindo o próprio `conftest.py`: **zero referências**. Do mesmo modo, o marcador `integration` declarado em `pyproject.toml` **nunca é aplicado** — apenas `security` (2 usos) e `parametrize` aparecem.

Resultado da suíte completa nesta revisão (sem Qdrant/Postgres/Neo4j, sem torch/FlagEmbedding/scenedetect):

```
43 failed, 186 passed, 12 skipped, 5 warnings, 6 errors
+ 2 módulos não coletáveis (test_audio_parser.py, test_video_parser.py — ModuleNotFoundError: scenedetect)
```

O achado 7.3 do v1 permanece funcionalmente aberto: a ferramenta existe, mas não está empunhada.

**Correção sugerida** — aplicar os marcadores e as fixtures nos módulos que dependem de infra:

```python
# tests/test_acl.py
pytestmark = pytest.mark.integration

def test_wrong_tenant_sees_nothing(require_qdrant, require_postgres):
    ...
```

---

#### 3.3 🟠 ALTO · **NOVO** — o filtro de ACL do canal de grafo é mais permissivo que os canais dense e BM25

O pré-filtro adicionado em `search_graph()` deixa passar chunks cujos metadados estejam vazios:

```python
# src/retrieval/graph_search.py:341-350
if tenant_id is not None and c_tenant and c_tenant != tenant_id:
    continue                                   # chunk com tenant_id == "" passa para QUALQUER tenant

if principals is not None and c_principals:
    if not any(p in c_principals for p in principals):
        continue                               # chunk com ACL vazia passa para QUALQUER usuário
```

Os outros dois canais fazem o oposto — negam por padrão:

```sql
-- src/retrieval/bm25_index.py (SEARCH_SQL)
AND (%(tenant_id)s = '' OR tenant_id = %(tenant_id)s)
AND (%(principals)s = '{}'::text[] OR acl_principals && %(principals)s)
```

O operador `&&` do Postgres não produz interseção com array vazio, e o `MatchAny` do Qdrant não casa com lista vazia. Logo, **um chunk sem ACL é invisível em dense/BM25 e visível no grafo**. A superfície de exposição do sistema passa a depender de qual canal o roteador escolheu — o pior tipo de inconsistência de segurança, porque é intermitente.

Além disso, o campo `matched_entities` devolvido por `search_graph()` é montado a partir de `matched_subgraph_entities`, que acumula entidades de **todo o grafo global** (`_GRAPH_INDEX` é um singleton compartilhado entre tenants), **antes** de qualquer filtragem. Nomes de entidades extraídos de documentos de outros tenants vazam nesse campo. Hoje `weighted_rrf()` descarta `matched_entities`, então não chega à resposta final — mas `search_graph()` é API pública e o campo aparece em qualquer inspeção ou debug.

**Correção sugerida:**

```python
# Nega por padrão, alinhado a dense/BM25
if tenant_id is not None and (c_tenant or "") != tenant_id:
    continue
if principals is not None and not set(principals) & set(c_principals or []):
    continue
```

E restringir `matched_entities` às entidades dos chunks efetivamente autorizados.

---

#### 3.4 🟠 ALTO · **PERSISTE** — canal de grafo continua *write-only*

`graph_search.py` grava no Neo4j (`MERGE (e:Entity ...)`, `MERGE (a)-[r]->(b)`) e apaga (`MATCH (n) DETACH DELETE n`), mas **não há uma única consulta de leitura**. Toda a busca roda sobre o `nx.MultiDiGraph` em memória. Em processo reiniciado ou em múltiplas réplicas, o grafo está vazio e o canal retorna `[]` silenciosamente — o `graph_channel_degraded` (bom acréscimo deste commit) só é marcado em exceção, não em grafo vazio.

O README anuncia "Graph RAG via Neo4j ou NetworkX (emulação in-memory)". Na prática, a persistência em Neo4j é decorativa.

---

#### 3.5 🟠 ALTO · **PERSISTE** — singletons globais sem pool nem thread-safety

Inalterados neste commit: `_conn` (`bm25_index.py:57`), `_client` (`embed_index.py:60`), `_model` (`model.py:28`), `_reranker` (`reranker.py:51`), `_WHISPER_MODEL` (`video_parser.py:35`), `_GRAPH_INDEX` (`graph_search.py:377`).

`_get_conn()` reutiliza uma única conexão Postgres com `autocommit = False`, protegida apenas por `if _conn is None or _conn.closed`. Sob concorrência, dois requests compartilham a mesma transação — e o `conn.rollback()` do fallback de BM25 pode desfazer trabalho de outro request. `_GRAPH_INDEX` compartilhado entre tenants agrava §3.3.

Recomendação: `psycopg2.pool.ThreadedConnectionPool` e proteção dos *lazy loaders* com `threading.Lock`.

---

#### 3.6 🟠 ALTO · **NOVO** — `answer_stream()` grava no cache respostas potencialmente truncadas

O bloco `finally` adicionado é conceitualmente correto, mas executa também quando o consumidor abandona o gerador (`GeneratorExit`, cliente desconectado). Nesse caso `full_answer` contém apenas o prefixo transmitido, e ele é gravado no cache como se fosse a resposta completa — envenenando o cache para todas as requisições subsequentes da mesma query/tenant.

Além disso, dois pontos menores no mesmo bloco:
- levantar exceção dentro de um `finally` durante `GeneratorExit` produz `RuntimeError: generator ignored GeneratorExit`;
- a contagem de tokens usa `len(texto.split())` (palavras), o que subestima o custo real em português em ~30–40%.

**Correção sugerida:** só gravar no cache quando a geração terminou naturalmente.

```python
    completed = False
    try:
        ...
        completed = True
    finally:
        full_answer = "".join(parts)
        if completed and full_answer:
            ...
```

---

#### 3.7 🟠 ALTO · **PERSISTE** — cobertura de testes ainda não cobre os pontos críticos

`test_security_ast.py` é uma adição de alto valor: valida via AST que 15 funções públicas declaram `tenant_id` e `principals`. Mas é um teste **estrutural**, não comportamental — ele teria passado mesmo com o bypass de grafo do v1, desde que a assinatura existisse. E, como visto em §2.2, ele passa hoje enquanto o *fail-open* segue ativo.

Continuam sem teste: `expand_parent_context()` com tipos mistos (o que teria pego §2.1), `weighted_rrf()` com pesos degenerados, `_derive_conclusions()`, `resolve_principals()` integrado ao pipeline, e o cache de `answer_stream()`.

---

### 4. Achados de severidade média

| # | Achado | Status | Detalhe |
| :--- | :--- | :---: | :--- |
| 4.1 | Respostas do Self-RAG nunca entram no cache | 🆕 | Consequência silenciosa de §2.1 — `_serialize_chunk` falha dentro de `try/except ... logger.debug` |
| 4.2 | Credenciais fracas como default em `config.py` | ❌ | `rag_dev_password`, `minioadmin`, `minioadmin_dev`, `rag_neo4j_password` |
| 4.3 | Relatórios de benchmark ainda rastreados no git | 🟨 | `.gitignore` correto, mas falta `git rm --cached` |
| 4.4 | `data/seed_corpus/*` ignorado torna o golden dataset irreproduzível | 🆕 | 7 falhas em `test_pdf_docx_parsers.py` por `FileNotFoundError` |
| 4.5 | 91 `print()` contra 7 `logging.getLogger()` em `src/` | ❌ | Piorou marginalmente (era 90 × 6) |
| 4.6 | `embedding_dimension()` retorna 1024 hardcoded | ❌ | Quebra silenciosamente ao trocar de modelo |
| 4.7 | Chave de cache aplica `.strip().lower()` | ❌ | Colide queries semanticamente distintas (siglas, códigos) |
| 4.8 | Modo simulado indistinguível de resposta real no texto | ❌ | Só o campo `simulated` diferencia |
| 4.9 | 8 blocos `except: pass` em `src/` | ❌ | Reduzidos no `graph_search.py`, persistem em parsers |
| 4.10 | `add_entity()` faz `**entity.metadata` sem sanitização | 🆕 | Chave `entity_type` ou `chunk_id` nos metadados sobrescreve o atributo do nó |
| 4.11 | `index_chunks()` assume que `c.acl` contém dicts | 🆕 | `e.get("principal")` estoura se a ACL vier como lista de strings — `multi_vector.py` já trata os dois formatos, `graph_search.py` não |
| 4.12 | Sem lockfile e pinning só com `>=` | ❌ | Builds não reproduzíveis |
| 4.13 | Canais dense e BM25 executam em série | ❌ | Latência somada em vez de paralela |
| 4.14 | `incremental_reindex` cobre apenas parte do pipeline | ❌ | Não cobre grafo nem multi-vector |

---

### 5. Achados de severidade baixa

- **🟢 `_derive_conclusions()` não distingue "backend offline" de "recall genuinamente zero".** O aviso é acionado por `max_recall == 0.0`, o que é uma boa heurística, mas a mensagem afirma a causa ("serviços indisponíveis") sem verificá-la. Sugestão: checar conectividade e ajustar o texto.
- **🟢 `resolve_principals()` tem uma linha tripla em branco** após a definição (`acl_filter.py`) — resíduo de edição.
- **🟢 Thresholds mágicos do Self-RAG sem calibração documentada** (`confidence_threshold`, `retry_threshold`) — persiste do v1.
- **🟢 Tabela de custos de LLM defasada** — persiste do v1.

---

### 6. Pontos fortes desta refatoração

Vale registrar o que foi feito bem, porque foi bastante:

1. **Propagação de `principals` no pipeline inteiro.** Nove arquivos alterados de forma consistente. O achado 3.2 do v1 foi resolvido com rigor, incluindo o `acl_principals` achatado no payload do Qdrant com `MatchAny` — a escolha correta de operador para arrays.
2. **Teste de regressão por AST.** Usar `ast.walk` para garantir que assinaturas de segurança não regridam é uma técnica sofisticada e barata de manter. É o tipo de teste que sobrevive a refatorações.
3. **`answer_stream()` alcançou paridade com `answer()`.** Cache, custo e trace agora são registrados no streaming. Era uma assimetria silenciosa e foi bem endereçada.
4. **README transformado.** De um documento de "Passo 0" para uma referência completa com diagrama Mermaid do pipeline, capacidades, instalação, CLI, testes e estrutura de diretórios. Salto de qualidade real.
5. **Isolamento correto do fallback de BM25.** Trocar `except psycopg2.Error` por `except pg_errors.SyntaxError` e deixar os demais erros propagarem é precisamente a correção certa — erros de conexão e de permissão não devem mais ser mascarados como problema de sintaxe de `tsquery`.
6. **Upsert em lote e `rerank()` em O(n).** Duas otimizações pequenas, corretas e sem efeito colateral.
7. **`graph_channel_degraded` no trace.** Degradação de canal agora é observável, não apenas logada.
8. **Higiene de repositório.** Remoção do `.bak` de 1.587 linhas, dos logs de runtime e `.gitignore` bem estruturado com `.gitkeep`.

---

### 7. Plano de ação priorizado

#### Sprint 1 — Bloqueadores (não promover a produção antes disto)

| # | Ação | Referência | Esforço |
| :--- | :--- | :--- | :---: |
| 1 | Converter `RerankResult` → `HybridResult` antes de `answer(prefetched_chunks=...)` + teste de regressão offline | §2.1 | 2h |
| 2 | Ancorar `Document.id` em `uuid5(source_uri, content_hash)`, regenerar `data/golden_dataset.json`, versionar `data/seed_corpus/` | §2.3 | 1d |
| 3 | Chamar `resolve_principals()` em todas as funções públicas de busca e inverter `allow_unrestricted` para `False` | §2.2 | 4h |
| 4 | Alinhar o filtro do grafo ao comportamento *deny-by-default* de dense/BM25 | §3.3 | 2h |
| 5 | `ruff check . --fix` + correção manual dos `E402`/`E741`; ajustar CI para `ruff check src/ tests/ scripts/` | §3.1 | 4h |
| 6 | Marcar testes de infra com `@pytest.mark.integration` + fixtures `require_*`; CI roda `pytest -m "not integration"` | §3.1, §3.2 | 4h |
| 7 | `git rm --cached benchmarks/results/benchmark_report.{md,json}` | §2.4 | 15min |

#### Sprint 2 — Alto impacto

| # | Ação | Referência | Esforço |
| :--- | :--- | :--- | :---: |
| 8 | Implementar leitura Cypher em `search_graph()` com fallback para NetworkX | §3.4 | 2d |
| 9 | `ThreadedConnectionPool` para Postgres + `threading.Lock` nos lazy loaders | §3.5 | 1d |
| 10 | Não cachear respostas truncadas em `answer_stream()`; usar contagem real de tokens | §3.6 | 3h |
| 11 | Mover credenciais default para `.env.example`; falhar no boot se ausentes em produção | §4.2 | 3h |
| 12 | Testes comportamentais de ACL que não exijam infra (fakes de Qdrant/Postgres) | §3.7 | 2d |
| 13 | Restringir `matched_entities` às entidades dos chunks autorizados | §3.3 | 2h |

#### Sprint 3 — Melhoria contínua

| # | Ação | Referência | Esforço |
| :--- | :--- | :--- | :---: |
| 14 | Substituir `print()` por `logging` estruturado em `src/` | §4.5 | 1d |
| 15 | `embedding_dimension()` derivado do modelo carregado | §4.6 | 1h |
| 16 | Lockfile (`pip-compile` ou `uv lock`) + `requirements-ci.txt` enxuto | §4.12, §3.1 | 4h |
| 17 | Paralelizar canais dense/BM25/grafo com `ThreadPoolExecutor` | §4.13 | 1d |
| 18 | Registrar decisões arquiteturais como ADRs em `docs/adr/` | — | contínuo |
| 19 | Documentar a calibração dos thresholds do Self-RAG | §5 | 4h |

---

### 8. Conclusão

O commit `e8607ca` é uma resposta séria ao review anterior. Dezenove dos trinta e três achados foram tocados, e doze foram **efetivamente resolvidos** — incluindo o bypass de ACL no grafo, que era o problema mais grave do v1. A qualidade das correções pontuais (isolamento do `SyntaxError` no BM25, `rerank()` em O(n), upsert em lote, paridade do `answer_stream`) é boa e demonstra entendimento real dos problemas apontados.

O padrão que mais chama atenção, e que vale a pena internalizar, é este: **quatro das correções pararam um passo antes da linha de chegada.**

- `resolve_principals()` foi escrita, testada em isolamento e **nunca chamada** (§2.2).
- `conftest.py` foi escrito com fixtures corretas e **nenhum teste as usa** (§3.2).
- `generate_chunk_id()` virou UUIDv5, mas a semente **continua aleatória** (§2.3).
- `_derive_conclusions()` foi implementada, mas o **artefato antigo continua no repositório** (§2.4).

Em cada caso o mecanismo existe e o teste passa; o comportamento em produção é o mesmo de antes. A lição prática é escrever testes que exerçam o **comportamento do sistema** (`hybrid_search()` sem `principals` deve falhar) em vez do **artefato criado** (`resolve_principals()` retorna o que se espera). O teste por AST é justamente a exceção que confirma a regra — ele funciona porque valida uma invariante estrutural que o time realmente quer preservar.

O achado mais urgente, porém, não é nenhum dos antigos: é a regressão de §2.1. Ela nasceu da interação entre duas correções corretas, feitas isoladamente, e quebra em runtime um caminho que antes apenas desperdiçava latência. É o argumento mais concreto possível a favor da ação nº 6 do Sprint 1 — uma suíte offline confiável rodando no CI teria pego isso antes do push.

**Recomendação:** executar o Sprint 1 integralmente e, depois, revalidar o Recall@5 do benchmark. Com o golden dataset ancorado em IDs estáveis, os números finalmente vão medir a arquitetura — que, essa sim, continua sólida.

---

*Revisão conduzida sobre o commit `e8607ca` em 04/09/2026. Todos os achados marcados como reproduzidos foram executados neste ambiente; achados de leitura de código estão identificados por arquivo e linha. Nenhum arquivo do repositório foi modificado durante esta revisão.*
