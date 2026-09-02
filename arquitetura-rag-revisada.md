# Arquitetura de Retrieval Engineering — Versão Revisada

Este documento incorpora as correções do diagnóstico à arquitetura original: fast path vs. full path, controle de acesso desde o Metadata Index, reindexação incremental, migração de modelo de embedding, routing suave (não binário), RRF explicitado, e avaliação tratada como workstream (não como lista de métricas).

---

## 1. O que mudou em relação ao desenho anterior

| Problema no desenho anterior | Correção nesta versão |
|---|---|
| Toda query paga o custo do pipeline completo | Roteador decide **fast path** (lookup direto) vs. **full path** (decomposição + multi-retrieval) |
| Router "duro" (escolhe 1 canal e elimina os outros) | Router **suave**: ajusta pesos/boost por canal, todos os canais relevantes continuam ativos |
| RRF ponderado sem fórmula definida | Fórmula explícita: `score = Σ weight_i / (k + rank_i)` |
| Sem ACL no pipeline | Metadata Index carrega `tenant_id` / `acl` desde a ingestão; todo retriever filtra por permissão antes de rankear |
| "Versioning" era só um bullet | Grafo de dependências explícito: documento → elementos → chunks → embeddings → entradas de índice |
| Sem plano para trocar modelo de embedding | Namespace de embedding versionado + dual-write + avaliação A/B antes do cutover |
| Avaliação = lista de métricas | Avaliação = workstream com geração assistida de dataset + revisão humana + loop de feedback de produção |
| V1 proposto era o pipeline quase inteiro | V1 realmente mínimo, sem reranker/router/decomposição |

---

## 2. Diagrama macro atualizado

```text
                    ┌────────────────────────────┐
                    │        DATA SOURCES         │
                    │ PDF │ DOCX │ XLSX │ HTML     │
                    │ IMG │ AUDIO │ VIDEO │ JSON   │
                    └──────────────┬─────────────┘
                                   ▼
                    ┌────────────────────────────┐
                    │       INGESTION LAYER       │
                    │ MIME Detection               │
                    │ Metadata Extraction          │
                    │ Tenant / ACL Tagging   ★NOVO │
                    │ Deduplication                │
                    │ Versioning + Content Hash ★  │
                    └──────────────┬─────────────┘
                                   ▼
             ┌─────────────────────────────────────────┐
             │          INTELLIGENT PARSING             │
             │ Text │ PDF Layout │ Table │ OCR │ Vision  │
             │ Audio Transcription │ Video Scene │ XLSX  │
             └────────────────────┬────────────────────┘
                                  ▼
             ┌─────────────────────────────────────────┐
             │        CANONICAL DOCUMENT MODEL          │
             │ Document → Sections → Paragraphs →       │
             │ Tables → Images → Figures → Pages →      │
             │ VideoScenes → AudioSegments               │
             │ + dependency_graph_id      ★NOVO          │
             └────────────────────┬────────────────────┘
                                  ▼
             ┌─────────────────────────────────────────┐
             │          CHUNKING ENGINE (plugin)        │
             │ Fixed │ Recursive │ Semantic │ Sentence   │
             │ Section │ ParentChild │ Table │ Multimodal│
             └────────────────────┬────────────────────┘
                                  ▼
             ┌─────────────────────────────────────────┐
             │       REPRESENTATION ENGINE              │
             │ Dense │ Sparse │ BM25 │ Summary │ Entities│
             │ HyDE │ Image Embed │ Audio Embed           │
             │ embedding_model_version tag   ★NOVO        │
             └────────────────────┬────────────────────┘
                                  ▼
      ┌─────────────────────────────────────────────────────────┐
      │                    INDEXING LAYER                        │
      │ Vector │ Full Text │ Graph │ Multimodal                  │
      │ METADATA INDEX (tenant_id, acl, doc_type, dates) ★NOVO   │
      └──────────────────────────┬──────────────────────────────┘
                                 ▼
                    ┌──────────────────────────┐
                    │   QUERY INTELLIGENCE      │
                    │ Classification            │
                    │ Rewrite / Expansion       │
                    │ Complexity Estimator ★NOVO│
                    └──────────────┬───────────┘
                                   ▼
                    ┌──────────────────────────┐
                    │     ROUTING DECISION      │
                    │  (soft, não elimina       │
                    │   canais)          ★NOVO  │
                    └──────┬─────────────┬──────┘
                            │             │
                     baixa complexidade   alta complexidade
                            ▼             ▼
                 ┌───────────────┐  ┌────────────────────┐
                 │   FAST PATH   │  │     FULL PATH       │
                 │ retrieval     │  │ Query Decomposition  │
                 │ direto,       │  │ → N sub-queries      │
                 │ sem decomp.,  │  │ → retrieval paralelo │
                 │ reranker leve │  │ → fusão dos sub-     │
                 │ ou nenhum     │  │   resultados          │
                 └───────┬───────┘  └──────────┬──────────┘
                          └───────────┬────────┘
                                      ▼
                    ┌──────────────────────────┐
                    │      MULTI RETRIEVAL      │
                    │ Dense │ BM25 │ Sparse │    │
                    │ Metadata │ Graph │ Parent  │
                    │  (filtrado por ACL) ★NOVO  │
                    └──────────────┬───────────┘
                                   ▼
                    ┌──────────────────────────┐
                    │      FUSION ENGINE        │
                    │ RRF = Σ w_i/(k+rank_i) ★  │
                    │ fórmula explícita          │
                    └──────────────┬───────────┘
                                   ▼
                    ┌──────────────────────────┐
                    │       RERANKING            │
                    │ Cross Encoder │ LLM │       │
                    │ Multimodal (só no full path)│
                    └──────────────┬───────────┘
                                   ▼
                    ┌──────────────────────────┐
                    │     CONTEXT ENGINE        │
                    │ Parent Expansion           │
                    │ Dedup de representações    │
                    │ redundantes (tabela) ★NOVO │
                    │ Token Budget                │
                    └──────────────┬───────────┘
                                   ▼
                    ┌──────────────────────────┐
                    │       GENERATION           │
                    │ LLM │ Citations │           │
                    │ Structured Answer           │
                    └──────────────┬───────────┘
                                   ▼
                    ┌──────────────────────────┐
                    │       EVALUATION           │
                    │ Offline: golden dataset     │
                    │ Online: feedback loop ★NOVO │
                    └──────────────┬───────────┘
                                   ▼
                    ┌──────────────────────────┐
                    │     OBSERVABILITY  ★NOVO   │
                    │ latência por estágio        │
                    │ custo por query              │
                    │ hit-rate ao longo do tempo   │
                    │ drift detection               │
                    └──────────────────────────┘
```

---

## 3. Fast path vs. full path

O erro do desenho anterior era tratar toda pergunta como se precisasse do pipeline inteiro (query analysis com LLM → decomposição → N canais → reranker → LLM). Isso encarece e atrasa até perguntas triviais.

```yaml
query_intelligence:
  complexity_estimator:
    # heurística leve, sem chamada de LLM cara:
    # nº de entidades, presença de comparativos ("compare", "diferença entre"),
    # presença de múltiplas perguntas na mesma query, comprimento
    signals: [entity_count, comparative_terms, multi_question, length]
    threshold_full_path: 0.6

routing:
  fast_path:
    condition: complexity_score < 0.6
    retrieval: [dense, bm25]
    fusion: rrf
    reranker: none        # ou cross-encoder leve, opcional
  full_path:
    condition: complexity_score >= 0.6
    query_decomposition: true
    retrieval: [dense, bm25, sparse, metadata, graph]
    fusion: weighted_rrf
    reranker: cross_encoder_or_llm
```

Isso resolve diretamente o problema de custo/latência: a maioria das perguntas factuais simples ("qual o CNPJ da empresa X?") não deveria disparar decomposição nem reranker pesado.

---

## 4. Routing suave, não binário

No desenho anterior, o router decidia "essa pergunta é Entity retrieval" e só buscava naquele canal — um erro de classificação derrubava a busca inteira. A correção é usar o router para **ajustar pesos**, mantendo todos os canais relevantes ativos:

```yaml
query_router:
  mode: soft   # nunca "hard" — não elimina canais, só reponde

  boosts_by_intent:
    fact_lookup:
      metadata: +0.20
      bm25: +0.15
      dense: 0.0
    table_lookup:
      table_retriever: +0.30
      dense: -0.10
    semantic_explanation:
      dense: +0.20
      bm25: -0.05
    multimodal:
      multimodal_index: +0.25

  # canal nunca cai a zero — apenas perde peso relativo
  min_weight_floor: 0.05
```

---

## 5. Fusão: fórmula explícita

```yaml
fusion:
  algorithm: weighted_rrf
  k: 60                      # constante padrão do RRF
  formula: "score(d) = Σ_i weight_i / (k + rank_i(d))"
  weights_source: query_router.boosts_by_intent   # não fixo, ajustado por intenção
  fallback_if_no_intent: rrf_uniforme               # weight_i = 1 para todos
```

Deixar a fórmula documentada no config evita que cada implementador invente sua própria variante.

---

## 6. Metadata Index com controle de acesso desde o início

```json
{
  "chunk_id": "chunk-8891",
  "document_id": "doc-123",
  "tenant_id": "org-456",
  "acl": {
    "read": ["group:finance", "user:jose@empresa.com"],
    "deny": []
  },
  "doc_type": "pdf",
  "classification": "confidencial",
  "created_at": "2026-03-01",
  "embedding_model_version": "bge-m3-v2"
}
```

Regra de retrieval: **todo canal (dense, BM25, graph, metadata) filtra por `tenant_id` + `acl` antes de rankear**, nunca depois. Filtrar depois do ranking vaza informação sobre a existência do documento (ex: contagem de resultados, posição no ranking) mesmo que o conteúdo não seja exibido.

```text
Query + user_context (tenant_id, groups)
         │
         ▼
   Pre-filter por ACL
         │
         ▼
   Retrieval (dense/bm25/graph/...)
         │
         ▼
   Fusão + Reranking
```

---

## 7. Reindexação incremental — grafo de dependências

```text
Document (v3)
   │
   ├── invalida → Elementos canônicos afetados
   │                  │
   │                  ├── invalida → Chunks derivados
   │                  │                  │
   │                  │                  ├── invalida → Embeddings
   │                  │                  │                  │
   │                  │                  │                  └── invalida → Entradas de índice
   │                  │                  │
   │                  │                  └── invalida → Nós/arestas de grafo relacionados
```

```yaml
reindex:
  trigger: content_hash_changed
  strategy: incremental
  steps:
    - recompute_canonical_elements: [changed_pages_only]
    - recompute_chunks: [chunks_derived_from_changed_elements]
    - recompute_embeddings: [affected_chunks]
    - update_index_entries: [affected_chunks]
    - invalidate_query_cache: [queries_touching_document_id]
```

Sem esse grafo, qualquer atualização de documento força reprocessar o corpus inteiro — inviável em escala.

---

## 8. Migração de modelo de embedding

```yaml
embedding_migration:
  strategy: dual_write

  steps:
    - write_new_embeddings_alongside_old: true
      namespace_old: "bge-m3-v1"
      namespace_new: "bge-m3-v2"
    - shadow_evaluate:
        # roda o golden dataset nos dois namespaces em paralelo
        compare: [recall_at_5, precision_at_5]
    - cutover:
        condition: "new_namespace.recall_at_5 >= old_namespace.recall_at_5"
        action: switch_default_namespace
    - decommission_old_namespace:
        after_days: 30
```

---

## 9. Avaliação como workstream (não como lista de métricas)

```text
Golden Dataset
   │
   ├── Geração assistida
   │     (LLM propõe pares pergunta→chunk esperado
   │      a partir do corpus real)
   │
   ├── Revisão humana
   │     (especialista de domínio aprova/corrige)
   │
   ├── Avaliação offline
   │     Recall@K, Precision@K, MRR, NDCG,
   │     Context Precision/Recall, Faithfulness
   │
   └── Avaliação online (produção)
         │
         ├── Feedback implícito (thumbs up/down, reformulação de query)
         ├── Alimenta de volta o golden dataset
         └── Dispara alerta se hit-rate cair (drift)
```

```yaml
evaluation:
  offline:
    dataset_generation: llm_assisted
    human_review: required
    metrics: [recall_at_5, precision_at_5, mrr, ndcg, faithfulness, context_precision]
  online:
    feedback_capture: [thumbs, query_reformulation_rate]
    feedback_to_dataset: true
    drift_alert:
      metric: recall_at_5_rolling_7d
      threshold_drop: 0.05
```

---

## 10. Observabilidade

```yaml
observability:
  per_stage_latency: [parsing, chunking, embedding, retrieval, fusion, reranking, generation]
  cost_per_query:
    breakdown: [llm_calls, embedding_calls, reranker_calls]
  retrieval_hit_rate:
    window: rolling_7d
  drift_detection:
    embedding_distribution_shift: true
    query_distribution_shift: true
```

Sem isso, não há como saber **qual estágio** está caro ou lento, nem detectar quando a qualidade de retrieval degrada silenciosamente com o tempo.

---

## 11. Roadmap revisado (V1 realmente mínimo)

O V1 anterior já incluía quase o pipeline inteiro. Esta versão separa o que é essencial para validar a ideia do que pode esperar.

```text
V1 — mínimo viável
│
├── PDF, DOCX, TXT, HTML
├── Canonical Document Model (sem tabelas estruturadas ainda)
├── 2 chunkers: Recursive + Parent/Child
├── Dense + BM25
├── RRF (uniforme, sem pesos por intenção ainda)
├── Metadata Index COM tenant_id/ACL desde o início   ★
├── Golden dataset mínimo (20–30 queries rotuladas manualmente)
└── Observabilidade básica (latência + custo por query)

V1.1
│
├── Reranker (cross-encoder)
├── Query rewrite simples
├── Context expansion (parent expansion)
└── Fast path / full path (complexity estimator)

V2
│
├── Tabelas estruturadas + XLSX
├── Multimodal (imagens)
├── Query Router (soft) + Query Decomposition
├── Reindexação incremental (grafo de dependências)
├── Migração de embedding (dual-write)
└── Avaliação online (feedback loop)

V3
│
├── Vídeo, Áudio
├── Graph RAG (opcional)
├── Adaptive Retrieval / Self-RAG
├── Auto-tuning de pesos de fusão
└── Plataforma de benchmark comparável entre configurações
```

---

## 12. Resumo das correções aplicadas

| Recomendação do diagnóstico | Onde entrou |
|---|---|
| Fast path para queries simples | Seção 3, roteamento antes do multi-retrieval |
| Router suave, não binário | Seção 4 |
| RRF ponderado com fórmula explícita | Seção 5 |
| ACL desde a ingestão, filtro antes do ranking | Seção 6 |
| Reindexação incremental via grafo de dependências | Seção 7 |
| Migração de modelo de embedding | Seção 8 |
| Avaliação como workstream com dataset assistido + feedback | Seção 9 |
| Observabilidade concreta | Seção 10 |
| V1 realmente mínimo | Seção 11 |

O que **não mudou** em relação à proposta original: Canonical Document Model, tabelas como objetos estruturados com múltiplas representações, chunking como plugin, Graph RAG como estratégia opcional, e a filosofia geral de tornar cada camada configurável via YAML. Essas decisões já estavam corretas.
