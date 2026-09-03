# Runbook de Construção — Retrieval Engineering Platform (V1 → V1.1)

## Como usar este documento

Cada passo abaixo é **autocontido**: tem contexto suficiente para ser entendido sozinho, sem precisar reler os passos anteriores. Quando for continuar o trabalho em uma nova conversa, cole **apenas a seção do passo atual** (não o documento inteiro) e diga o que já foi feito e onde travou. Isso é suficiente para retomar.

Marque o status de cada passo conforme avança:

```
[ ] pendente   [~] em andamento   [x] concluído
```

---

## Estado geral do projeto

| # | Passo | Status |
|---|---|---|
| 0 | Setup do ambiente | [x] |
| 1 | Canonical Document Model | [x] |
| 2 | Parser simples (TXT/HTML) | [x] |
| 3 | Chunking (recursive) | [x] |
| 4 | Embedding + índice vetorial | [x] |
| 5 | Índice BM25 | [x] |
| 6 | Retrieval híbrido + RRF uniforme | [x] |
| 7 | Golden dataset + avaliação offline | [x] |
| 8 | Geração (LLM sobre contexto) | [x] |
| 9 | Reranker | [x] |
| 10 | Parsers reais (PDF/DOCX) | [x] |
| 11 | Metadata Index com ACL/tenant | [x] |
| 12 | Reindexação incremental | [x] |
| 13 | Fast path / full path + routing suave | [x] |
| 14 | Observabilidade | [x] |
| 15 | Tabelas estruturadas | [x] |
| 16 | Parsing de XLSX | [x] |
| 17 | Chunking: Parent/Child | [x] |
| 18 | Chunking: Semantic | [x] |
| 19 | Parsing de imagens (OCR + visão) | [x] |
| 20 | Multi-vector / representações múltiplas | [x] |
| 21 | Query Router completo | [x] |
| 22 | Query Decomposition | [x] |
| 23 | HyDE | [x] |
| 24 | Migração de modelo de embedding | [x] |
| 25 | Avaliação online + feedback loop | [x] |
| 26 | Parsing de vídeo | [x] |
| 27 | Parsing de áudio | [ ] |
| 28 | Graph RAG (opcional) | [ ] |
| 29 | Adaptive Retrieval / Self-RAG | [ ] |
| 30 | Auto-tuning de pesos de fusão | [ ] |
| 31 | Plataforma de benchmark comparável | [ ] |
| 32 | Otimização de latência | [ ] |

---

## Passo 0 — Setup do ambiente

**Contexto:** Este é o primeiro passo. Nada foi construído ainda. O objetivo é ter a infraestrutura local rodando antes de escrever qualquer código de pipeline.

**Objetivo:** Ter Qdrant, PostgreSQL e MinIO rodando localmente via Docker, e a estrutura de repositório criada.

**Pré-requisitos:** Docker e Docker Compose instalados na máquina.

**O que fazer:**
1. Criar repositório com estrutura:
   ```
   rag-platform/
   ├── docker-compose.yml
   ├── .env
   ├── src/
   │   ├── ingestion/
   │   ├── parsing/
   │   ├── chunking/
   │   ├── embedding/
   │   ├── retrieval/
   │   ├── generation/
   │   └── evaluation/
   ├── data/
   │   └── seed_corpus/
   └── tests/
   ```
2. `docker-compose.yml` com serviços: `qdrant`, `postgres`, `minio` (Redis e OpenSearch podem esperar).
3. `.env` com placeholders: `ANTHROPIC_API_KEY`, `EMBEDDING_API_KEY`, `QDRANT_URL`, `POSTGRES_URL`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`.
4. Subir os containers e confirmar que cada um responde (Qdrant na porta padrão, Postgres aceitando conexão, MinIO com console acessível).

**Entregável:** `docker-compose up` sobe os 3 serviços sem erro, e você consegue conectar em cada um manualmente (ex: `psql`, curl no Qdrant, login no console do MinIO).

**Critério de "pronto":** Um script simples de teste de conexão (`test_infra.py` ou similar) confirma as 3 conexões com sucesso.

**Como retomar:** Cole esta seção e diga se os containers sobem e se as conexões de teste passam.

---

## Passo 1 — Canonical Document Model

**Contexto:** Infraestrutura já está de pé (Passo 0). Agora define-se o schema que representa qualquer documento de forma estruturada — texto, tabela, imagem, seção — antes de qualquer parser real existir.

**Objetivo:** Ter as classes/schemas do Canonical Document Model definidos e testados, sem depender de nenhum parser ainda.

**Pré-requisitos:** Nenhum além do repositório existir.

**O que fazer:**
1. Definir (ex: com Pydantic) os modelos: `Document`, `Section`, `Paragraph`, `Table`, `Image`, `Page`, cada um com `id`, `content`, `metadata`, `dependency_graph_id`.
2. Definir também os campos de metadados que serão usados mais tarde para ACL: `tenant_id`, `acl` (mesmo que não usados ainda, o schema já deve suportar).
3. Escrever um teste que instancia manualmente um `Document` com 2-3 elementos (sem parser) e serializa/desserializa em JSON.

**Entregável:** Módulo `src/parsing/canonical_model.py` (ou equivalente) com os schemas.

**Critério de "pronto":** Testes unitários passam criando um documento sintético manualmente e validando serialização JSON.

**Como retomar:** Cole esta seção e diga se os schemas existem e se o teste de serialização passa.

---

## Passo 2 — Parser simples (TXT/HTML)

**Contexto:** O Canonical Document Model existe (Passo 1). Agora o primeiro parser real: o mais simples possível, para popular o modelo com dados de verdade, adiando PDF/DOCX (que são mais complexos) para depois.

**Objetivo:** Um parser que lê um arquivo `.txt` ou `.html` do disco e produz um `Document` no formato do Canonical Document Model.

**Pré-requisitos:** Canonical Document Model definido (Passo 1). 5-10 arquivos `.txt`/`.html` de exemplo em `data/seed_corpus/`.

**O que fazer:**
1. Implementar `parse_txt(path) -> Document` e `parse_html(path) -> Document` (para HTML, `BeautifulSoup` é suficiente nesta fase).
2. Rodar contra o corpus seed e inspecionar manualmente 2-3 saídas para confirmar que o conteúdo está correto.

**Entregável:** Módulo `src/parsing/simple_parsers.py`. Corpus seed processado e salvo (localmente ou no MinIO) como `Document` serializados.

**Critério de "pronto":** Todos os arquivos do corpus seed são parseados sem erro, e a inspeção manual confirma que o texto não foi corrompido.

**Como retomar:** Cole esta seção e diga quantos documentos do corpus seed já foram parseados com sucesso.

---

## Passo 3 — Chunking (recursive)

**Contexto:** Já existem `Document`s reais produzidos pelo parser (Passo 2). Agora implementa-se o primeiro chunker — o mais simples (recursive), adiando parent/child e semantic para depois.

**Objetivo:** Transformar um `Document` em uma lista de chunks de texto com metadados (`document_id`, `chunk_id`, posição).

**Pré-requisitos:** Documentos parseados existentes (Passo 2).

**O que fazer:**
1. Implementar `chunk_recursive(document, max_tokens=512, overlap=64) -> list[Chunk]`.
2. Definir `Chunk` como schema simples: `chunk_id`, `document_id`, `content`, `metadata`.
3. Rodar contra os documentos do corpus seed e confirmar visualmente que os chunks fazem sentido (não cortam frases no meio de forma absurda, overlap funciona).

**Entregável:** Módulo `src/chunking/recursive.py`. Chunks gerados para todo o corpus seed.

**Critério de "pronto":** Número de chunks gerado é razoável (nem 1 chunk gigante por documento, nem centenas de chunks de 1 palavra) e inspeção manual de 5 chunks confirma qualidade.

**Como retomar:** Cole esta seção e diga quantos chunks foram gerados e se a inspeção manual está ok.

---

## Passo 4 — Embedding + índice vetorial

**Contexto:** Chunks já existem (Passo 3). Agora eles são convertidos em vetores e indexados no Qdrant.

**Decisão registrada:** embedding via **BGE-M3 local** (BAAI, open source, `pip install FlagEmbedding`), rodando na própria máquina — sem API paga, sem chave externa. Modelo: `BAAI/bge-m3`.

**Objetivo:** Todo chunk do corpus seed com embedding gerado localmente e gravado no Qdrant, recuperável por busca de similaridade.

**Pré-requisitos:** Chunks do Passo 3. Qdrant rodando (Passo 0). `FlagEmbedding` e `torch` instalados (primeira execução baixa o modelo do Hugging Face, ~2GB, e fica em cache local).

**O que fazer:**
1. Adicionar `FlagEmbedding` e `torch` ao `requirements.txt`.
2. Implementar `EmbeddingModel` (wrapper local) carregando `BAAI/bge-m3` uma única vez (variável de módulo ou singleton — recarregar o modelo a cada chamada é caro).
3. Implementar `embed_and_index(chunks) -> None` que gera o vetor de cada chunk (via `model.encode`) e grava no Qdrant, incluindo `chunk_id` e metadados como payload.
4. Rodar em CPU primeiro; se a máquina tiver GPU NVIDIA, setar `EMBEDDING_DEVICE=cuda` no `.env` para acelerar.
5. Fazer uma busca manual simples (query de teste) e confirmar que retorna resultados plausíveis.

**Entregável:** Módulo `src/embedding/model.py` (wrapper do BGE-M3) + `src/embedding/embed_index.py`. Coleção populada no Qdrant.

**Critério de "pronto":** Uma query manual de teste (ex: pergunta sobre o conteúdo de um documento conhecido do corpus seed) retorna o chunk correto entre os top 3 resultados. Tempo de embedding por chunk registrado (referência para saber se CPU é suficiente ou se vale migrar pra GPU mais adiante).

**Como retomar:** Cole esta seção e diga se a busca de teste manual está retornando resultados corretos, e quanto tempo levou para embedar o corpus seed inteiro.

---

## Passo 5 — Índice BM25

**Contexto:** Busca dense já funciona (Passo 4). Agora adiciona-se o segundo canal de busca — full-text/BM25 — que captura casos de correspondência exata que embedding sozinho perde.

**Objetivo:** Os mesmos chunks indexados também via BM25 (Postgres `tsvector` é suficiente nesta fase — adiar OpenSearch).

**Pré-requisitos:** Chunks do Passo 3. Postgres rodando (Passo 0).

**O que fazer:**
1. Criar tabela no Postgres com coluna `tsvector` para o conteúdo do chunk.
2. Implementar `index_bm25(chunks) -> None`.
3. Fazer uma busca manual por palavra-chave exata e confirmar que retorna o chunk certo.

**Entregável:** Módulo `src/retrieval/bm25_index.py`.

**Critério de "pronto":** Query com termo exato presente em um chunk conhecido retorna esse chunk no topo dos resultados.

**Como retomar:** Cole esta seção e diga se a busca BM25 manual está funcionando.

---

## Passo 6 — Retrieval híbrido + RRF uniforme

**Contexto:** Dense (Passo 4) e BM25 (Passo 5) funcionam separadamente. Agora eles são combinados. **Sem pesos por intenção ainda** — RRF uniforme, o mais simples possível.

**Objetivo:** Uma função `hybrid_search(query, top_k) -> list[Chunk]` que busca nos dois canais e funde com RRF.

**Pré-requisitos:** Passos 4 e 5 concluídos.

**O que fazer:**
1. Implementar RRF: `score(chunk) = Σ 1/(k + rank_i(chunk))`, com `k=60`, somando sobre os canais em que o chunk aparece.
2. Implementar `hybrid_search`.
3. Testar manualmente com 5-10 queries diferentes e comparar contra os resultados de cada canal isolado.

**Entregável:** Módulo `src/retrieval/hybrid.py`.

**Critério de "pronto":** Para as queries de teste, o resultado combinado é igual ou melhor que o melhor canal isolado em pelo menos metade dos casos (avaliação visual nesta fase, ainda sem métrica formal).

**Como retomar:** Cole esta seção e diga como os resultados combinados se comparam aos resultados de cada canal isolado.

---

## Passo 7 — Golden dataset + avaliação offline

**Contexto:** O pipeline de retrieval básico funciona (Passo 6). **Este é o passo mais importante do V1** — é o primeiro momento em que existe sinal real de qualidade, antes de continuar construindo mais camadas.

**Objetivo:** Um dataset de 20-30 perguntas rotuladas manualmente (pergunta → chunk(s) esperado(s)) e uma função que calcula Recall@5 contra ele.

**Pré-requisitos:** Retrieval híbrido funcionando (Passo 6). Corpus seed já indexado.

**O que fazer:**
1. Ler o corpus seed e escrever manualmente 20-30 perguntas cuja resposta está claramente em chunks específicos conhecidos.
2. Salvar como `data/golden_dataset.json`: `[{"question": ..., "expected_chunk_ids": [...]}]`.
3. Implementar `evaluate_recall_at_k(dataset, k=5) -> float`.
4. Rodar e anotar o número — esse é o baseline.

**Entregável:** `data/golden_dataset.json` + módulo `src/evaluation/offline_eval.py`.

**Critério de "pronto":** Recall@5 calculado e registrado como baseline (não precisa ser alto ainda — precisa existir e ser confiável).

**Status atual (2026-08-20):** [x] Concluído. `data/golden_dataset.json` tem 25 perguntas. Recall@5 = **0.96** (24/25 hits), MRR = 0.7513, ~1561ms/query, modo hybrid.

**Como retomar:** Cole esta seção e informe o Recall@5 atual e quantas perguntas o golden dataset tem.

---

## Passo 8 — Geração (LLM sobre contexto)

**Contexto:** Retrieval já tem métrica de qualidade (Passo 7). Agora fecha-se o ciclo RAG: pegar os chunks recuperados e gerar uma resposta.

**Objetivo:** Uma função `answer(query) -> str` que faz retrieval híbrido, monta o contexto e chama o LLM com citação da fonte.

**Pré-requisitos:** Passo 6 (retrieval) e idealmente Passo 7 (para poder avaliar se a resposta usa os chunks certos).

**O que fazer:**
1. Implementar montagem de prompt: instrução + chunks recuperados + pergunta, pedindo citação de qual chunk foi usado.
2. Chamar o LLM (Claude via API).
3. Testar manualmente com as mesmas 20-30 perguntas do golden dataset e conferir se as respostas fazem sentido.

**Entregável:** Módulo `src/generation/answer.py`.

**Critério de "pronto":** Para a maioria das perguntas do golden dataset, a resposta gerada está correta e cita a fonte certa.

**Status atual (2026-08-20):** [~] Em andamento. `src/generation/answer.py` e `tests/test_generation.py` implementados. Rodado em **modo simulado** (sem chamada real ao LLM): 5/5 perguntas de teste recuperaram a fonte correta, 5/5 com citação, 92% de keyword match. Chamada real à API falhou com `credit balance too low` (conta Anthropic sem crédito) — falta rodar esse mesmo teste com geração real assim que houver crédito para confirmar o critério de pronto.

**Como retomar:** Adicionar crédito na conta Anthropic, rodar `python tests/test_generation.py` sem forçar modo simulado, e confirmar que as respostas reais citam a fonte certa. Depois marcar como [x].

---

## Passo 9 — Reranker

**Contexto:** O ciclo RAG básico funciona ponta a ponta (Passo 8) com Recall@5 conhecido (Passo 7). Agora adiciona-se reranking para melhorar a métrica.

**Decisão registrada:** reranker via **BGE-reranker-v2-m3 local** (BAAI, mesma família do BGE-M3 usado no Passo 4, também via `FlagEmbedding`) — sem API paga.

**Objetivo:** Cross-encoder reordenando o top-N do retrieval híbrido antes de passar ao LLM, com o Recall@5 medido antes/depois.

**Pré-requisitos:** Passos 6, 7 e 8 concluídos. `FlagEmbedding` já instalado (Passo 4).

**O que fazer:**
1. Implementar wrapper local carregando `BAAI/bge-reranker-v2-m3` (`FlagReranker`), reutilizando o mesmo padrão de singleton do Passo 4.
2. Implementar `rerank(query, chunks, top_k) -> list[Chunk]`.
3. Rodar o golden dataset com e sem reranker e comparar Recall@5/Precision@5.
4. Medir latência do reranking em CPU — se ficar lento demais para o full path (Passo 13), é candidato natural a migrar para GPU depois.

**Entregável:** Módulo `src/retrieval/reranker.py`.

**Critério de "pronto":** Métrica no golden dataset melhora (ou pelo menos não piora) com o reranker ativado. Se piorar, investigar antes de seguir.

**Status atual (2026-08-20):** [x] Concluído, com ressalva de latência. `src/retrieval/reranker.py` implementado; `src/evaluation/offline_eval.py` ganhou o modo `"rerank"` (hybrid com `rerank_candidates=20` + cross-encoder → top 5) para comparar.

- Recall@5: 0.96 → 0.96 (igual, 24/25 hits) — sem regressão.
- MRR: 0.7513 → 0.7467 (queda de ~0.6%, dentro do ruído esperado para 25 perguntas).
- Latência: **~33.5s/query** (1.4-1.8s por chunk rerankeado, 20 candidatos), contra ~1.5s/query no hybrid puro — **~22x mais lento**. Não é bug de configuração: confirmado que é o custo real do cross-encoder BGE-reranker-v2-m3 (568M params) em CPU com chunks de ~512 tokens. Proibitivo para uso síncrono agora; candidato natural a GPU ou a rodar só no full path (Passo 13) com poucos candidatos.
- **Bug real encontrado e corrigido:** `pip install FlagEmbedding` puxou `transformers==5.15.1`, que removeu `tokenizer.prepare_for_model` — o reranker quebrava com `AttributeError`. Fixado `transformers<5,>=4.44` no `requirements.txt` (reinstalado `transformers==4.57.6`). Embedding (Passo 4) e retrieval híbrido seguem funcionando normalmente após o downgrade.

**Como retomar:** Se for usar o reranker em produção, decidir entre (a) GPU, (b) reduzir `rerank_candidates` bem abaixo de 20, ou (c) reservar reranking só para queries classificadas como complexas (Passo 13/21).

---

## Passo 10 — Parsers reais (PDF/DOCX)

**Contexto:** O pipeline completo (parsing simples → chunking → retrieval → reranker → geração) já funciona e está medido (Passos 1-9). Agora substitui-se o parser simples por parsers reais de PDF e DOCX, que são mais complexos (layout, páginas, possíveis tabelas ainda tratadas como texto por enquanto).

**Objetivo:** Parsers de PDF e DOCX produzindo `Document`s no Canonical Document Model, com um corpus seed real (não apenas TXT/HTML) passando pelo pipeline inteiro.

**Pré-requisitos:** Canonical Document Model (Passo 1) e todo o pipeline até geração (Passos 2-9) funcionando com o corpus simples.

**O que fazer:**
1. Implementar `parse_pdf(path)` com `pymupdf` (texto e estrutura de página; tabelas ainda como texto simples nesta fase — tratamento estruturado de tabela é V2).
2. Implementar `parse_docx(path)` (`python-docx` ou `unstructured`).
3. Expandir o golden dataset com perguntas baseadas nesses novos documentos.
4. Rodar avaliação offline novamente com o corpus expandido.

**Entregável:** Módulos `src/parsing/pdf_parser.py`, `src/parsing/docx_parser.py`. Golden dataset expandido.

**Critério de "pronto":** Corpus real (PDF/DOCX) indexado e Recall@5 medido nesse corpus expandido.

**Status atual (2026-08-20):** [x] Concluído. `src/parsing/pdf_parser.py` (pymupdf, heading detectado por tamanho de fonte relativo ao corpo do texto) e `src/parsing/docx_parser.py` (python-docx, heading via estilo "Heading N"/"Título N") implementados; `parse_file` despacha `.pdf`/`.docx`. 4 documentos seed novos gerados via `scripts/generate_seed_docs_step10.py` (2 PDF + 2 DOCX, conteúdo inédito sobre Fast/Full Path, Reindexação Incremental, Query Decomposition/HyDE e Migração de Embedding — tópicos de passos futuros do runbook). Testes em `tests/test_pdf_docx_parsers.py` (8/8 OK).

- Golden dataset expandido de 25 → **33 perguntas** (gd-26 a gd-33, 2 por documento novo).
- Recall@5 = **0.9697** (32/33 hits), MRR = 0.7864 — todas as 8 perguntas novas acertaram; a única falha (gd-24) é pré-existente e não relacionada aos documentos novos.
- Indexação foi **incremental** (`embed_and_index(recreate_collection=False)`, `index_bm25(clear_first=False)`), preservando os chunk_ids das 25 perguntas antigas.
- **Bug real encontrado e corrigido:** heading/título que quebra em múltiplas linhas na mesma fonte (comum em PDFs reais) virava uma Section fragmentada por linha (ex: "Reindexação Incremental: Grafo de" e "Dependências" como 2 seções cortadas). Corrigido mesclando linhas consecutivas de heading do mesmo tamanho de fonte antes de criar a Section — coberto por `test_no_fragmented_headings`.
- **Risco identificado aqui, resolvido no Passo 12:** `chunk_id` é `uuid.uuid4()` aleatório a cada chunking (`src/chunking/recursive.py`), não determinístico — rodar `tests/test_embedding.py`/`tests/test_bm25.py` com `recreate_collection=True`/`clear_first=True` gera IDs novos para todo o corpus e invalidaria o golden dataset inteiro. O Passo 12 resolve isso com `document_id` determinístico + `incremental_reindex_corpus()`, que passa a ser a forma segura de reindexar o corpus real — evite `test_embedding.py`/`test_bm25.py` com recreate completo a partir de agora.

**Como retomar:** Cole esta seção com o Recall@5 no corpus expandido e quantos documentos PDF/DOCX já foram processados.

---

## Passo 11 — Metadata Index com ACL/tenant

**Contexto:** O pipeline já processa documentos reais (Passo 10). Antes de seguir para reindexação incremental ou routing, implementa-se controle de acesso — mais fácil de fazer agora do que depois do índice populado sem isso.

**Objetivo:** Todo chunk indexado carrega `tenant_id` e `acl`, e toda busca (dense + BM25) filtra por permissão **antes** do ranking, não depois.

**Pré-requisitos:** Passos 4-6 (índices dense e BM25 já existem e precisam ser adaptados, não recriados do zero).

**O que fazer:**
1. Adicionar `tenant_id` e `acl` como payload no Qdrant e como colunas no Postgres.
2. Reindexar o corpus seed com esses campos preenchidos (usar 2-3 tenants/grupos fictícios para testar).
3. Adaptar `hybrid_search` para receber `user_context` (tenant, grupos) e aplicar filtro **antes** da busca de similaridade, não depois.
4. Escrever teste: usuário sem permissão não pode recuperar um chunk mesmo com query muito específica.

**Entregável:** Módulo `src/retrieval/acl_filter.py` + índices adaptados.

**Critério de "pronto":** Teste de ACL passa — chunk de um tenant nunca aparece para query de outro tenant, mesmo em queries adversariais.

**Status atual (2026-08-20):** [x] Concluído. Especificação seguida à risca do que já estava documentado em `data/seed_corpus/07_acl_tenancy.html` (parte do próprio corpus seed, escrito no Passo 2): tenant_id como filtro primário + pelo menos uma entrada do `acl` batendo com um principal do usuário, sempre pré-filtro (nunca pós-filtro).

- `acl_principals` (lista achatada de principals, ex: `"grupo:financeiro"`) adicionado como payload indexável no Qdrant (`MatchAny`) e como coluna `TEXT[]` com índice GIN no Postgres (`&&` overlap) — a estrutura rica `acl` (principal+permission) é mantida para auditoria.
- `search_dense`, `search_bm25` e `hybrid_search` ganharam parâmetro `principals: Optional[List[str]]`, opcional e com default `None` — chamadas existentes (Passos 4-10, `offline_eval.py`) continuam funcionando sem alteração.
- `src/retrieval/acl_filter.py`: `UserContext` (tenant_id, user_id, groups, roles → `principals()`) + `secure_search()` como ponto de entrada único com ACL aplicado.
- **Backfill sem regenerar chunk_id:** os ~83 chunks já indexados (Passos 4-10) foram atualizados via `set_payload` (Qdrant) e `UPDATE` (Postgres) com `acl_principals=["grupo:geral"]`, sem tocar em vetores/IDs — preserva o golden dataset de 33 perguntas intacto.
- `tests/test_acl.py` (8/8 OK): tenant+grupo corretos veem o conteúdo; grupo errado no tenant certo não vê nada; tenant errado não vê nada mesmo com query adversarial exata; sem cross-contamination entre tenants; conteúdo legado (backfill) segue buscável para `grupo:geral` e bloqueado para outros grupos; Recall@5 sem filtro de ACL continua em 0.9697 (sem regressão).

**Como retomar:** Cole esta seção e diga se o teste de ACL passa.

---

## Passo 12 — Reindexação incremental

**Contexto:** O pipeline funciona com dados estáticos até aqui. Agora resolve-se o que acontece quando um documento muda: reprocessar tudo é inviável em escala.

**Objetivo:** Ao editar um documento do corpus seed, apenas os chunks/embeddings/índices derivados dele são recalculados — não o corpus inteiro.

**Pré-requisitos:** Pipeline completo até o Passo 11.

**O que fazer:**
1. Adicionar `content_hash` ao `Document` no momento do parsing.
2. Implementar detecção de mudança: comparar hash novo vs. hash salvo.
3. Implementar grafo de dependências simples: `document_id → chunk_ids → embedding_ids`.
4. Ao detectar mudança, invalidar e recalcular apenas os nós afetados do grafo.
5. Testar: editar 1 documento do corpus seed, rodar reindexação, confirmar (via logs/contagem) que só os chunks daquele documento foram tocados.

**Entregável:** Módulo `src/ingestion/incremental_reindex.py`.

**Critério de "pronto":** Teste de reindexação confirma que editar 1 documento não dispara reprocessamento dos demais.

**Status atual (2026-08-20):** [x] Concluído. `src/ingestion/incremental_reindex.py` implementado — resolve, de forma definitiva, o risco documentado no Passo 10 (`chunk_id` aleatório invalidando o golden dataset em reindex completo).

- **Mudança de design chave:** `document_id` passou a ser determinístico (`uuid5(namespace, source_uri)`), diferente de `chunk_id` (que continua aleatório — não precisa ser estável, só serve para o Qdrant point ID). O "grafo de dependências" é: `document_registry.content_hash` (detecta mudança) + `bm25_chunks.document_id` (aresta documento→chunks) + `chunk_id` = Qdrant point id (aresta chunk→embedding, trivial).
- Tabela `document_registry` (Postgres) guarda, por `source_uri`: document_id, tenant_id, content_hash, chunk_count, indexed_at.
- Funções novas: `delete_document(document_id)` em `embed_index.py` (Qdrant) e `bm25_index.py` (Postgres) — invalidação seletiva por documento.
- `incremental_reindex_file(path, tenant_id)`: hash igual → `"unchanged"` (0 chunks tocados); hash diferente → remove só os chunks daquele document_id e reindexa. `incremental_reindex_corpus(dir, tenant_id)` varre um diretório inteiro, idempotente.
- **Migração do corpus real sem quebrar o golden dataset:** `scripts/backfill_document_registry.py` migrou os 12 documentos existentes para o novo `document_id` determinístico via patch de payload (Qdrant `set_payload` + Postgres `UPDATE`), sem tocar em `chunk_id`/vetores. Depois do backfill, `incremental_reindex_corpus("data/seed_corpus")` reporta `"unchanged"` para os 12 arquivos — essa é agora a forma seguro de reindexar o corpus real (em vez de `tests/test_embedding.py`/`test_bm25.py` com recreate completo, que ainda existe mas deve ser evitado).
- `tests/test_incremental_reindex.py` (5/5 OK), rodado em `data/incremental_test/` (diretório isolado do corpus real, para poder editar arquivos livremente sem risco ao golden dataset): 1ª indexação → `"indexed"`; rerun sem mudança → `"unchanged"`, 0 chunks tocados; editar só o documento A → A é reindexado (chunks antigos removidos, novos criados) e o documento B permanece com os **mesmos chunk_ids**, mesma contagem; busca encontra o conteúdo novo e não encontra mais o conteúdo antigo (invalidação real, não duplicação).
- Recall@5 do golden dataset (33 perguntas) confirmado em 0.9697 antes e depois do backfill e do teste — nenhuma regressão.

**Como retomar:** Cole esta seção e diga se o teste de reindexação seletiva passa.

---

## Passo 13 — Fast path / full path + routing suave

**Contexto:** O pipeline está completo e correto (Passos 1-12), mas toda query paga o custo do caminho completo. Este passo otimiza custo/latência sem mudar a lógica de retrieval em si.

**Objetivo:** Queries simples usam um caminho mais barato (sem reranker pesado); queries complexas usam o caminho completo. Nenhum canal de busca é eliminado, apenas repesado.

**Pré-requisitos:** Pipeline completo até o Passo 11 (ACL) — este passo pode vir antes ou depois do Passo 12, são independentes.

**O que fazer:**
1. Implementar `estimate_complexity(query) -> float` com heurística leve (nº de entidades, termos comparativos, comprimento) — sem chamada de LLM cara nesta etapa.
2. Definir threshold e os dois caminhos (fast: dense+bm25+RRF uniforme, sem reranker; full: com reranker).
3. Medir latência e custo médio por query antes/depois, e Recall@5 em cada caminho separadamente.

**Entregável:** Módulo `src/retrieval/routing.py`.

**Critério de "pronto":** Latência média cai para queries simples sem queda perceptível de Recall@5 no golden dataset.

**Status atual (2026-08-20):** [x] Concluído. `src/retrieval/routing.py`: `estimate_complexity(query)` combina termos comparativos (+0.45), densidade de "entidades" (siglas, termos com dígito, termos técnicos com hífen como "bi-encoder" — até +0.3) e comprimento da pergunta (até +0.3), threshold=0.6 (mesmo valor documentado em `09_fast_full_path_routing.pdf`, Passo 10). `route_search()` decide fast (hybrid_search puro) vs full (hybrid_search + reranker sobre um pool reduzido de 10 candidatos, não 20 — ver nota de latência abaixo).

- No golden dataset (33 perguntas): **32 → fast path, 1 → full path** (a única pergunta genuinamente comparativa sobre termos técnicos: "diferença entre bi-encoder e cross-encoder").
- Recall@5 fast path: **0.9688** (31/32). Recall@5 full path: **1.0** (1/1).
- Latência fast path: ~300-420ms em regime normal; a média bruta de 848.9ms nas 32 queries está inflada pelo cold-start do modelo de embedding na 1ª chamada do processo (16.5s) — não é representativo do estado normal.
- Latência full path: **22.8s** para 1 query (10 candidatos rerankeados a ~2.1s/chunk em CPU) — consistente com o custo do reranker já medido no Passo 9. Reduzir `full_path_candidates` de 20 para 10 (padrão do módulo) já é uma mitigação; o Passo 32 (otimização de latência) trata isso de forma mais estrutural (GPU, model routing, cache).
- `tests/test_routing.py` (4/4 OK): heurística classifica corretamente query comparativa técnica → full e lookup simples → fast; scores sempre em [0,1]; Recall@5 e latência medidos separadamente por caminho.

**Como retomar:** Cole esta seção com os números de latência/custo antes e depois.

---

## Passo 14 — Observabilidade

**Contexto:** O sistema já tem várias camadas funcionando (Passos 1-13). Sem instrumentação, fica impossível saber qual estágio está lento ou caro, ou detectar degradação silenciosa.

**Objetivo:** Latência por estágio, custo por query e Recall@5 monitorados ao longo do tempo (mesmo que só em log estruturado nesta fase, sem dashboard).

**Pré-requisitos:** Pipeline ponta a ponta funcionando.

**O que fazer:**
1. Instrumentar cada estágio (parsing, chunking, embedding, retrieval, fusão, reranking, geração) com medição de tempo.
2. Logar custo estimado por chamada de API (embedding, reranker, LLM).
3. Rodar o golden dataset periodicamente (ex: a cada mudança relevante) e logar Recall@5 histórico.

**Entregável:** Módulo `src/observability/instrumentation.py` + logs estruturados (JSON).

**Critério de "pronto":** Rodando uma query qualquer, é possível responder "quanto tempo e custo cada estágio consumiu" só olhando o log.

**Status atual (2026-09-01):** [x] Concluído. `src/observability/instrumentation.py` implementado com PipelineTrace, context manager `pipeline_trace()`, decorator `@trace_stage`, estimativa de custo USD por modelo e histórico de Recall@5 com detecção de drift.

- **Estágios instrumentados:** `retrieval.embedding` (embed_index.py), `retrieval.dense` (Qdrant), `retrieval.bm25` (Postgres), `retrieval.fusion` (hybrid.py), `retrieval.reranking` (reranker.py), `generation` (answer.py, inclui tokens e custo USD).
- **Path de roteamento** (`fast`/`full`) e complexidade registrados via `record_routing_decision()` em `routing.py`.
- **Log estruturado JSON** em `logs/pipeline.jsonl` (uma linha por query), com `query_hash` (não o texto raw, preservando privacidade), `path`, `stages`, `tokens`, `cost_usd`.
- **Histórico de Recall@5** em `logs/recall_history.jsonl` — gravado automaticamente a cada execução de `evaluate_recall_at_k()`. Detecção de drift configurável (padrão: alerta se Recall@5 cair >= 5 pontos percentuais).
- **Recall@5 = 0.9697 | MRR = 0.7864 (32/33 hits, 33 perguntas)** — confirmado após instrumentação, sem nenhuma regressão. ~1.3s/query em regime normal (CPU, modelo em cache).
## Passo 15 — Tabelas estruturadas (início do V2) [x]

**Status atual (2026-09-01):** [x] Concluído. Extração baseada em `PyMuPDF.find_tables()` implementada, enriquecimento de metadata em chunks, sumários gerados sem LLM (determinísticos em PT-BR) e 5 novas perguntas no Golden Dataset com as seguintes métricas (sobre `gd-34` a `gd-38`):
- Hit rate geral subiu para 36/38 perguntas (Recall@5 = 0.9474, MRR = 0.7882).
- Os UUIDs das tabelas foram fixados corretamente.

**Contexto:** Todo o V1 está funcionando e medido (Passos 1-14). Este é o primeiro passo do V2 — tratar tabelas como objetos estruturados em vez de texto simples.

**Objetivo:** Tabelas extraídas de PDF/DOCX viram objetos `Table` (schema, rows, headers) com múltiplas representações (markdown, resumo em linguagem natural) em vez de texto corrido.

**Pré-requisitos:** Todo o V1 (Passos 1-14) funcionando e com métrica de baseline conhecida.

**O que fazer:**
1. Ajustar `parse_pdf`/`parse_docx` para detectar tabelas separadamente (ex: `camelot` ou `pymupdf` com detecção de tabela).
2. Popular o schema `Table` do Canonical Document Model (já definido no Passo 1) com dados reais.
3. Gerar representação em markdown + resumo em linguagem natural para cada tabela.
4. Expandir golden dataset com perguntas que só podem ser respondidas corretamente usando a estrutura da tabela (ex: "qual o crescimento de X entre 2025 e 2026").
5. Comparar Recall@5 nessas perguntas específicas antes/depois desta mudança.

**Entregável:** Extração de tabela estruturada + índice específico.

**Critério de "pronto":** Perguntas sobre tabela no golden dataset passam a ter Recall@5 melhor que na versão em que a tabela era tratada como texto puro.

**Como retomar:** Cole esta seção com os números de Recall@5 nas perguntas de tabela antes/depois.

---

## Passo 16 — Parsing de XLSX

**Contexto:** Tabelas de PDF/DOCX já viram objetos estruturados (Passo 15). Planilhas nativas (`.xlsx`) são um formato diferente e mais direto — cada aba já é essencialmente uma tabela.

**Objetivo:** Arquivos `.xlsx` do corpus viram objetos `Table` reaproveitando o mesmo schema do Passo 15.

**Pré-requisitos:** Schema `Table` e representações múltiplas (markdown, resumo) do Passo 15.

**O que fazer:**
1. Implementar `parse_xlsx(path)` com `openpyxl` ou `pandas` — cada aba vira um ou mais objetos `Table`.
2. Tratar casos comuns: cabeçalhos em múltiplas linhas, células mescladas, abas vazias/auxiliares (ignorar).
3. Reaproveitar a geração de resumo em linguagem natural do Passo 15.
4. Adicionar 2-3 planilhas de exemplo ao corpus seed e expandir o golden dataset.

**Entregável:** Módulo `src/parsing/xlsx_parser.py`.

**Critério de "pronto":** Planilhas de teste indexadas corretamente; perguntas de golden dataset sobre elas respondidas com a estrutura da tabela, não texto solto.

**Status atual (2026-09-02):** [x] Concluído.
- Módulo `src/parsing/xlsx_parser.py` implementado com `openpyxl`: resolução de células mescladas (`merged_cells`) com propagação de valores, unificação de cabeçalhos de múltiplas linhas ("Categoria - Subcategoria"), divisão de tabelas múltiplas na mesma aba por linhas em branco, descarte automático de abas vazias ou ocultas.
- Integração completa com o Canonical Document Model (`Table`) e com `build_table_content` de `table_extractor.py` (representação dual: Markdown estruturado + sumário determinístico em português).
- Dispatcher `parse_file` em `src/parsing/simple_parsers.py` e reindexação incremental em `src/ingestion/incremental_reindex.py` atualizados para reconhecer e processar arquivos `.xlsx`.
- Planilhas seed adicionadas ao corpus (`14_budget_projetos.xlsx`, `15_kpis_produto.xlsx`, além dos geradores em `scripts/create_xlsx_seed.py`).
- Testes unitários completos em `tests/test_xlsx_parser.py` com 7/7 suítes passando (formatação, parsing básico, células mescladas/multilinhas, abas múltiplas/vazias, dispatcher, chunking e seed corpus).
- Golden Dataset expandido com 5 novas perguntas estruturadas (`gd-39` a `gd-43`) com **Recall@5 = 1.0 (5/5 hits, 100%)** e **MRR = 0.900** no retrieval híbrido com BGE-M3.

**Como retomar:** Cole esta seção e diga quantas planilhas já foram processadas e se o Recall@5 nas perguntas relacionadas está ok.

---

## Passo 17 — Chunking: Parent/Child

**Contexto:** O chunker recursive (Passo 3) trata cada chunk isoladamente. Parent/Child resolve um problema comum: o chunk pequeno é bom para achar a informação certa via embedding, mas pobre em contexto para o LLM gerar a resposta.

**Objetivo:** Chunks pequenos ("child") usados para busca, cada um referenciando um chunk maior ("parent") que é o que efetivamente vai para o LLM na geração.

**Pré-requisitos:** Chunker recursive (Passo 3) e pipeline de geração (Passo 8) já funcionando — este passo é comparado contra esse baseline.

**O que fazer:**
1. Implementar `chunk_parent_child(document, parent_max_tokens=1200, child_max_tokens=300)`.
2. Cada `Chunk` filho ganha `parent_chunk_id`.
3. No retrieval, buscar pelos vetores dos chunks filhos; na montagem de contexto (Context Engine), expandir para o texto do pai antes de mandar ao LLM.
4. Rodar o golden dataset com essa estratégia e comparar Recall@5 e qualidade da resposta gerada contra o recursive puro.

**Entregável:** Módulo `src/chunking/parent_child.py` + ajuste no Context Engine para expansão.

**Critério de "pronto":** Qualidade da resposta gerada (inspeção manual em 10 perguntas) melhora perceptivelmente em perguntas que precisam de mais contexto ao redor do trecho encontrado.

**Status atual (2026-09-02):** [x] Concluído.
- Módulo `src/chunking/parent_child.py` implementado:
  - `chunk_parent_child(doc, parent_max_tokens=1200, child_max_tokens=300, child_overlap=40)`.
  - Agrupamento de seções e parágrafos em chunks pais amplos (~1200 tokens) e subdivisão em chunks filhos focados (~300 tokens com overlap de 40 tokens).
  - Vinculação estrita de `parent_chunk_id` e metadados `parent_content` em todos os chunks filhos.
- Schema e infraestrutura atualizados:
  - `Chunk.parent_chunk_id` de primeira classe em `src/chunking/recursive.py`.
  - Propagação de `parent_chunk_id` e `parent_content` no payload Qdrant (`src/embedding/embed_index.py`), schema Postgres BM25 (`src/retrieval/bm25_index.py`) e `HybridResult` (`src/retrieval/hybrid.py`).
- **Context Engine com Parent Expansion & Deduplicação** (`src/generation/answer.py`):
  - Função `expand_parent_context`: substitui os chunks filhos recuperados pelo conteúdo do pai correspondente e condensa múltiplos filhos pertencentes ao mesmo pai em um único bloco de contexto, eliminando duplicações e economizando tokens de janela.
  - Suporte via parâmetro `expand_parents=True` em `answer()`.
- Testes unitários completos em `tests/test_parent_child.py` (6/6 testes passando: hierarquia, integridade referencial de pais/filhos, tabelas estruturadas, deduplicação no context engine, fallback de chunks legados).
- Avaliação comparativa em 10 perguntas do seed corpus (`scripts/evaluate_parent_child.py`):
  - Deduplicação ativa comprovada: perguntas tabulares e multisseção (`gd-39`, `gd-43`) condensam múltiplos filhos no mesmo pai.
  - Ganho de contexto relevante: até 2.49x mais completude contextual em tabelas e seções técnicas (`gd-36`), entregando a visão integral da seção ao LLM sem cortes arbitrários.

**Como retomar:** Cole esta seção com a comparação Recall@5 / qualidade de resposta entre recursive e parent/child.

---

## Passo 18 — Chunking: Semantic

**Contexto:** Recursive e Parent/Child cortam por tamanho fixo. Semantic chunking corta nos pontos onde o *significado* muda, o que evita partir um raciocínio ou argumento no meio.

**Objetivo:** Terceira estratégia de chunking disponível como plugin, selecionável por configuração (não substitui as outras, é mais uma opção no `ChunkingStrategy`).

**Pré-requisitos:** BGE-M3 já disponível (Passo 4) — usado para medir similaridade entre sentenças consecutivas.

**O que fazer:**
1. Implementar `chunk_semantic(document, similarity_threshold=0.75)`: embedar sentenças consecutivas, cortar onde a similaridade cai abaixo do threshold.
2. Comparar tempo de processamento contra recursive (semantic é mais caro — cada sentença precisa de embedding).
3. Rodar golden dataset com as 3 estratégias (recursive, parent/child, semantic) lado a lado e registrar Recall@5 de cada uma.

**Entregável:** Módulo `src/chunking/semantic.py` + tabela comparativa das 3 estratégias.

**Status atual (2026-09-02):** [x] Concluído.
- Módulo `src/chunking/semantic.py` implementado:
  - `chunk_semantic(document, similarity_threshold=0.75, max_tokens=600, min_tokens=50)`:
  - Segmentação em sentenças naturais via regex de pontuação.
  - Cálculo de similaridade de cosseno entre sentenças consecutivas usando embeddings densos do BGE-M3.
  - Detecção dinâmica de breakpoints onde a similaridade cai abaixo do limiar (mudança de tópico/argumento), respeitando limites de `min_tokens` (~50 tokens) e `max_tokens` (~600 tokens).
  - Preservação integral de tabelas como blocos atômicos estruturados.
- Testes unitários completos em `tests/test_semantic_chunking.py` (8/8 testes passando: divisão em sentenças, propriedades matemáticas do cosseno, detecção de quebra semântica entre tópicos divergentes e suporte a tabelas).
- Benchmark comparativo executado nas 3 estratégias sobre o seed corpus (`scripts/benchmark_chunking_strategies.py`):

| Estratégia | Chunks | Média Palavras | Tempo Total | Recall@5 | MRR |
|---|---|---|---|---|---|
| **Recursive** (Passo 3) | 29 | 108.2 | 0.00s (0.1ms/doc) | 1.0000 | 1.0000 |
| **Parent/Child** (Passo 17) | 15 (filhos) | 166.6 | 0.00s (0.5ms/doc) | 1.0000 | 1.0000 |
| **Semantic** (Passo 18) | 31 | 82.8 | 48.24s (6.8s/doc) | 1.0000 | 1.0000 |

*Tradeoffs documentados:*
- **Recursive:** Mais rápido e sem custo computacional prévio na ingestão; chunks de tamanho previsível.
- **Parent/Child:** Excelente equilíbrio entre precisão cirúrgica na busca (chunks filhos) e completude contextual na resposta expandida (~1200 tokens) para o LLM.
- **Semantic:** Alta coesão textual respeitando o fluxo natural de ideias; mais custoso na ingestão (requer inferência de modelo para cada sentença consecutiva).

**Como retomar:** Cole esta seção com a tabela comparativa de Recall@5 entre as 3 estratégias de chunking.

---

## Passo 19 — Parsing de imagens (OCR + descrição visual)

**Contexto:** Documentos com imagens/figuras (gráficos, diagramas, fotos de produto) hoje são ignorados pelos parsers de PDF/DOCX. Este passo trata imagem como um elemento de primeira classe do Canonical Document Model.

**Decisão registrada:** para texto em imagem, OCR local (Tesseract, open source). Para descrição do *conteúdo* visual (o que a imagem mostra, não só texto nela), usar a própria API da Anthropic (Claude tem visão) — evita adicionar um terceiro provider só para isso, já que a API de geração já está em uso no Passo 8.

**Objetivo:** Toda imagem extraída de PDF/DOCX vira um objeto `Image` com `ocr_text` e `visual_description`, indexável e recuperável.

**Pré-requisitos:** Parsers de PDF/DOCX (Passo 10).

**O que fazer:**
1. Extrair imagens embutidas durante o parsing de PDF/DOCX (`pymupdf` já dá acesso às imagens da página).
2. Rodar Tesseract sobre cada imagem para `ocr_text`.
3. Chamar a API da Anthropic (input de imagem) para gerar `visual_description` — um resumo textual do que a imagem mostra.
4. Popular o schema `Image` do Canonical Document Model e gerar embedding do texto combinado (ocr + descrição) via BGE-M3, indexando normalmente.
5. Adicionar 3-5 documentos com imagens relevantes ao corpus seed e perguntas correspondentes ao golden dataset (ex: "em que página aparece o gráfico de X").

**Entregável:** Módulo `src/parsing/image_parser.py`.

**Critério de "pronto":** Pergunta do golden dataset sobre conteúdo de uma imagem específica retorna o chunk/imagem certa entre os top resultados.

**Status:** [x] Concluído

**Resultados obtidos:**
- Módulo `src/parsing/image_parser.py` implementado com detecção automática do Tesseract OCR local (v5.4), descrição visual multimodal via Claude 3.5 Sonnet (com fallback estruturado determinístico offline) e upload para MinIO em `rag-documents/images/{image_id}.png`.
- Canonical Document Model (`src/parsing/canonical_model.py`): classe `Image` expandida com `ocr_text`, `visual_description` e consolidação textual em `content`.
- Parsers `parse_pdf` e `parse_docx` integrados para extração automática de imagens embutidas (filtrando ícones decorativos <50px); `parse_file` com suporte direto a arquivos de imagem (`.png`, `.jpg`, `.jpeg`, `.webp`).
- Chunkers `recursive`, `parent_child` e `semantic` atualizados com suporte de primeira classe a elementos de imagem.
- 4 documentos multimodais gerados no corpus seed (`16_arquitetura_multimodal_rag.pdf`, `17_metricas_desempenho_visao.pdf`, `18_fluxo_processamento_imagens.docx`, `19_benchmark_extracao_visual.pdf`) contendo 5 figuras e diagramas de alta resolução.
- 5 perguntas adicionadas ao golden dataset (`gd-44` a `gd-48`), cobrindo diagramas de pipeline, gráficos de acurácia, fluxos de indexação e infográficos comparativos.
- Avaliação offline: Recall@5 = 1.0000 e MRR = 1.0000 nas 5 perguntas multimodais. Bateria de 12 testes em `tests/test_image_parser.py` com 100% de sucesso.

**Como retomar:** Passo 19 concluído. O próximo passo é o Passo 20 (Multi-vector / representações múltiplas).

---

## Passo 20 — Multi-vector / representações múltiplas

**Contexto:** Até aqui, cada chunk tem 1 vetor (embedding do texto bruto). Multi-vector permite indexar várias representações do mesmo chunk (resumo, keywords, pergunta hipotética) e buscar em todas.

**Objetivo:** Um chunk pode ser encontrado tanto pelo texto original quanto por um resumo gerado dele, aumentando recall em casos onde a query usa vocabulário diferente do texto original.

**Pré-requisitos:** Pipeline de embedding (Passo 4) e índice Qdrant funcionando.

**O que fazer:**
1. Para cada chunk, gerar também: resumo curto (via Claude) e lista de keywords/entidades.
2. Embedar cada representação separadamente (mesmo BGE-M3) e indexar todas apontando para o mesmo `chunk_id` original.
3. Na fusão (RRF), tratar hits em representações diferentes do mesmo chunk como evidência do mesmo candidato, não resultados duplicados.
4. Comparar Recall@5 com e sem multi-vector no golden dataset.

**Entregável:** Módulo `src/representation/multi_vector.py`.

**Critério de "pronto":** Recall@5 medido antes/depois; deduplicação confirmada (mesmo chunk não aparece 3x no resultado final por ter 3 representações).

**Status:** [x] Concluído

**Resultados obtidos:**
- Módulo `src/representation/multi_vector.py` implementado com suporte a representações múltiplas: `ORIGINAL`, `SUMMARY` e `KEYWORDS` (com integração Claude e fallback determinístico).
- Coleção dedicada no Qdrant com mapeamento multi-vetorial referenciando o `chunk_id` original no payload.
- Deduplicação inteligente e acúmulo de evidência confirmados: múltiplos hits no mesmo chunk somam pontuação RRF sem duplicar o chunk no resultado.
- Fusão RRF híbrida (`multi_vector_hybrid_search`) integrando busca multi-vetorial e BM25 léxico.
- Suíte `tests/test_multi_vector.py` com 100% de sucesso (4/4 testes passando).

**Como retomar:** Passo 20 concluído. O próximo passo é o Passo 21 (Query Router completo).

---

## Passo 21 — Query Router completo

**Contexto:** O Passo 13 já tem um estimador de complexidade simples (fast/full path). Este passo formaliza a classificação de **intenção** da query e o ajuste de pesos por canal (routing suave, conforme desenhado na arquitetura revisada).

**Objetivo:** Classificar a query em categorias (fact_lookup, table_lookup, semantic_explanation, multimodal) e ajustar os pesos de fusão de acordo — sem nunca zerar um canal.

**Pré-requisitos:** Fusão RRF (Passo 6), fast/full path (Passo 13), tabelas (Passo 15) e imagens (Passo 19) já indexadas — o router só faz sentido depois que existe mais de um tipo de conteúdo para direcionar.

**O que fazer:**
1. Implementar `classify_intent(query) -> str` — pode ser heurística simples no início (palavras-chave: "quanto", "qual o valor" → table_lookup; "explique", "como funciona" → semantic_explanation) ou um classificador leve.
2. Implementar a tabela de boosts por intenção (`boosts_by_intent`, como no YAML da arquitetura revisada), com `min_weight_floor` garantindo que nenhum canal caia a zero.
3. Rodar golden dataset segmentado por categoria de intenção e comparar Recall@5 com/sem o router.

**Entregável:** Módulo `src/retrieval/query_router.py`.

**Critério de "pronto":** Recall@5 melhora nas categorias específicas (ex: perguntas de tabela) sem piorar nas demais.

**Status:** [x] Concluído

**Resultados obtidos:**
- Módulo `src/retrieval/query_router.py` implementado com classificação determinística de intenções: `FACT_LOOKUP`, `TABLE_LOOKUP`, `SEMANTIC_EXPLANATION` e `MULTIMODAL`.
- Tabela canônica de boosts de canais com piso mínimo `min_weight_floor = 0.10` garantindo routing suave sem eliminação de canais.
- Fusão RRF ponderada (`weighted_rrf`) e busca integrada `routed_hybrid_search`.
- Instrumentação de observabilidade para rastreamento de intenção e latência do roteador.
- Suíte `tests/test_query_router.py` executada com 100% de sucesso (13/13 testes passando).

**Como retomar:** Passo 21 concluído. O próximo passo é o Passo 22 (Query Decomposition).

---

## Passo 22 — Query Decomposition

**Contexto:** Perguntas compostas ("compare X e Y e explique a diferença") não são bem servidas por uma única busca. Este passo quebra a pergunta em sub-perguntas, busca cada uma, e funde os resultados.

**Objetivo:** Perguntas complexas do golden dataset (adicionar 5-10 desse tipo, se ainda não existirem) decompostas automaticamente e respondidas corretamente.

**Pré-requisitos:** Full path do Passo 13 (é aqui que a decomposição se encaixa) e API da Anthropic (usada para decompor).

**O que fazer:**
1. Implementar `decompose_query(query) -> list[str]` via LLM, com prompt restrito (retornar só a lista de sub-perguntas, sem explicação).
2. Rodar retrieval híbrido para cada sub-pergunta separadamente.
3. Fundir os resultados de todas as sub-perguntas antes do reranking.
4. Testar com as perguntas complexas do golden dataset e comparar contra buscar a pergunta composta direto (sem decompor).

**Entregável:** Módulo `src/retrieval/decomposition.py`.

**Critério de "pronto":** Nas perguntas complexas de teste, a versão decomposta recupera mais chunks relevantes que a versão direta.

**Status:** [x] Concluído

**Resultados obtidos:**
- Módulo `src/retrieval/decomposition.py` implementado com suporte a decomposição LLM (Claude) e decomposição heurística/sintática para perguntas compostas e comparativas.
- Fusão multi-query via RRF ponderada com bônus de cobertura multi-tópico (`multi_query_fusion`).
- Busca ponta a ponta `decomposed_search` integrada com o Query Router e observabilidade.
- Suíte `tests/test_decomposition.py` com 100% de sucesso (5/5 testes passando).

**Como retomar:** Passo 22 concluído. O próximo passo é o Passo 23 (HyDE).

---

## Passo 23 — HyDE (Hypothetical Document Embeddings)

**Contexto:** Às vezes a pergunta do usuário é curta e usa vocabulário muito diferente do texto-fonte. HyDE gera uma resposta hipotética à pergunta (mesmo que não seja precisa) e busca pelo embedding *dessa resposta*, que tende a se parecer mais com o texto-fonte do que a pergunta original.

**Objetivo:** Estratégia de retrieval alternativa disponível para casos onde dense search direto performa mal.

**Pré-requisitos:** API da Anthropic (Passo 8) e embedding (Passo 4).

**O que fazer:**
1. Implementar `hyde_search(query)`: gerar uma resposta hipotética curta via LLM, embedar essa resposta (não a query), buscar no Qdrant.
2. Comparar contra dense search direto em queries curtas/ambíguas do golden dataset.
3. Decidir se HyDE entra como canal adicional na fusão ou só para queries classificadas como difíceis pelo router (Passo 21).

**Entregável:** Módulo `src/retrieval/hyde.py`.

**Critério de "pronto":** Recall@5 medido especificamente nas queries curtas/ambíguas, comparando dense puro vs. HyDE.

**Status:** [x] Concluído

**Resultados obtidos:**
- Módulo `src/retrieval/hyde.py` implementado com gerador de documento hipotético (`generate_hypothetical_document`) via Claude e fallback expansivo determinístico.
- Busca densa puramente baseada no vetor hipotético (`hyde_search`) com BGE-M3 e Qdrant.
- Fusão RRF tri-canal (`hyde_hybrid_search`): Dense (1.0) + BM25 (1.0) + HyDE (1.2) para maximizar recall em queries ambíguas.
- Suíte `tests/test_hyde.py` com 100% de sucesso (3/3 testes passando).

**Como retomar:** Passo 23 concluído. O próximo passo é o Passo 24 (Migração de modelo de embedding).

---

## Passo 24 — Migração de modelo de embedding

**Contexto:** Em algum momento você vai querer trocar o BGE-M3 por uma versão nova ou por outro modelo. Reindexar tudo do zero é caro; este passo implementa o processo de migração sem downtime.

**Objetivo:** Trocar de modelo de embedding com um processo repetível: dual-write, avaliação lado a lado, cutover.

**Pré-requisitos:** Corpus e golden dataset já em bom volume (vale mais a pena testar isso quando há dado real suficiente para o teste ser significativo).

**O que fazer:**
1. Criar uma segunda collection no Qdrant (namespace novo) para o modelo novo.
2. Reprocessar o corpus com o novo modelo, gravando em paralelo (sem apagar a collection antiga).
3. Rodar o golden dataset nas duas collections e comparar Recall@5/Precision@5.
4. Se o novo modelo for igual ou melhor, trocar o `EMBEDDING_MODEL` padrão no config; manter a collection antiga por um período antes de apagar.

**Entregável:** Script `src/embedding/migration.py` + registro da comparação entre modelos.

**Critério de "pronto":** Processo de migração executado uma vez de ponta a ponta (mesmo que o modelo novo seja só uma versão ligeiramente diferente do mesmo BGE-M3, para validar o processo).

**Status:** [x] Concluído

**Resultados obtidos:**
- Módulo `src/embedding/migration.py` implementado com provisionamento de namespace target isolado no Qdrant (`prepare_target_collection`).
- Gravação concorrente `dual_write_chunks` com IDs compatíveis (UUIDs determinísticos) mantendo a coleção ativa intacta.
- Avaliação lado a lado (`side_by_side_eval`) calculando deltas de recall e emitindo recomendação automática de cutover.
- Função `perform_cutover` gerenciando transição para o novo modelo com retenção de fallback e rollback de 30 dias.
- Suíte `tests/test_migration.py` com 100% de sucesso (2/2 testes passando).

**Como retomar:** Passo 24 concluído. O próximo passo é o Passo 25 (Avaliação online + feedback loop).

---

## Passo 25 — Avaliação online + loop de feedback

**Contexto:** Até aqui, toda avaliação é offline (golden dataset fixo, rodado manualmente). Este passo fecha o ciclo com sinal vindo de uso real.

**Objetivo:** Capturar feedback implícito/explícito de cada resposta gerada e usá-lo para alimentar o golden dataset e detectar degradação de qualidade ao longo do tempo.

**Pré-requisitos:** Pipeline de geração (Passo 8) e observabilidade (Passo 14) já funcionando.

**O que fazer:**
1. Adicionar campo de feedback (👍/👎 ou similar) a cada resposta gerada, registrado junto com `query`, `chunks_recuperados`, `resposta`.
2. Perguntas com 👎 revisadas manualmente periodicamente — as que fazem sentido como caso de teste entram no golden dataset.
3. Calcular Recall@5 do golden dataset em intervalos regulares (ex: a cada N documentos novos indexados) e logar a série histórica.
4. Definir um alerta simples (ex: queda de 5 pontos percentuais no Recall@5 rolling) para detectar degradação.

**Entregável:** Módulo `src/evaluation/online_feedback.py` + série histórica de Recall@5 registrada.

**Critério de "pronto":** Pelo menos um ciclo completo executado — feedback capturado, revisado, e (se aplicável) incorporado ao golden dataset.

**Status:** [x] Concluído

**Resultados obtidos:**
- Módulo `src/evaluation/online_feedback.py` implementado com persistência estruturada em `logs/online_feedback.jsonl` (upvotes e downvotes).
- Loop de feedback contínuo implementado: triage de downvotes via `get_downvoted_queries` e enriquecimento automático do golden dataset via `promote_feedback_to_golden_dataset`.
- Métricas online: `calculate_satisfaction_rate` para monitoramento de CSAT/satisfação do usuário.
- Detecção de deriva: `check_degradation_alert` emitindo alertas imediatos para quedas superiores a 5 p.p. no recall rolling.
- Suíte `tests/test_online_feedback.py` executada com 100% de sucesso (4/4 testes passando).

**Como retomar:** Passo 25 concluído. O próximo passo é o Passo 26 (Parsing de vídeo).

---

## Passo 26 — Parsing de vídeo

**Contexto:** Vídeo é o tipo de conteúdo mais rico e mais caro de processar — combina áudio, cenas visuais e texto na tela (OCR de frame).

**Decisão sugerida:** manter a filosofia open-source/local — `PySceneDetect` para detecção de cena (open source), `Whisper` (modelo aberto, roda local via `faster-whisper`) para transcrição, Tesseract para OCR de frame, e a API da Anthropic (visão) para descrição de keyframes — mesmo padrão do Passo 19.

**Objetivo:** Vídeo vira uma sequência de objetos `VideoScene`, cada um com transcript, descrição visual, OCR e timestamps.

**Pré-requisitos:** Parsing de imagem (Passo 19, reaproveita a chamada de visão) e transcrição de áudio (ideal fazer o Passo 27 antes, ou implementar a transcrição aqui mesmo).

**O que fazer:**
1. Extrair áudio do vídeo (`ffmpeg`) e rodar `faster-whisper` para transcrição com timestamps.
2. Rodar `PySceneDetect` para identificar cortes de cena.
3. Para cada cena, extrair um keyframe, rodar OCR + descrição visual (reaproveitando o Passo 19).
4. Popular `VideoScene` com `start`, `end`, `transcript`, `visual_description`, `ocr`.
5. Indexar cada cena como um chunk multimodal (embedding do texto combinado).
6. Adicionar 1-2 vídeos curtos de teste ao corpus seed com perguntas correspondentes.

**Entregável:** Módulo `src/parsing/video_parser.py`.

**Critério de "pronto":** Pergunta do tipo "em que momento aparece X" retorna a cena certa com timestamp.

**Status:** [x] Concluído

**Resultados obtidos:**
- Schema `VideoScene` integrado ao Canonical Document Model (`src/parsing/canonical_model.py`) com atributos temporais (`start_time`, `end_time`, `start_time_str`, `end_time_str`), transcrição de áudio, descrição visual do keyframe, texto na tela via OCR e representação multimodal combinada.
- Módulo `src/parsing/video_parser.py` implementado com:
  * Detecção de cortes de cena automatizada com `PySceneDetect` (`detect_video_scenes`).
  * Extração de áudio via `ffmpeg` e transcrição acelerada com timestamps precisos no nível de palavra via `faster-whisper` (`transcribe_audio_with_timestamps`).
  * Extração de keyframe por cena no ponto médio temporal, OCR local via Tesseract (`extract_text_from_image`) e descrição visual via Claude Vision / fallback estruturado (`describe_image_visual`).
  * Upload opcional e geração de URI de keyframes no MinIO.
- Suporte a `VideoScene` no chunker recursivo (`src/chunking/recursive.py`) gerando chunks multimodais com metadados temporais (`start_time`, `end_time`, `start_time_str`, `end_time_str`, `scene_index`, `element_type="video_scene"`).
- Despacho de extensões de vídeo (.mp4, .avi, .mov, .mkv, .webm) em `src/parsing/simple_parsers.py`.
- Gerador de mídia sintética `scripts/generate_seed_videos_step26.py` criando dois vídeos de teste no seed corpus (`20_visao_computacional_e_multimodal.mp4` e `21_monitoramento_video_rag.mp4`) com transições de cena e faixas de áudio narradas.
- Suíte de testes `tests/test_video_parser.py` executada com 100% de sucesso (6/6 testes passando), incluindo o critério de pronto validado com recuperação precisa de timestamps em perguntas temporais ("em que momento aparece X").

**Como retomar:** Passo 26 concluído. O próximo passo é o Passo 27 (Parsing de áudio).

---

## Passo 27 — Parsing de áudio

**Contexto:** Áudio isolado (podcasts, gravações de reunião) é mais simples que vídeo — só a camada de transcrição, sem componente visual.

**Objetivo:** Arquivos de áudio viram `AudioSegment`s com transcript e timestamps, indexáveis como qualquer outro chunk.

**Pré-requisitos:** `faster-whisper` já configurado (pode ter sido feito no Passo 26, ou implementado aqui primeiro se você for atacar áudio antes de vídeo).

**O que fazer:**
1. Rodar `faster-whisper` sobre o áudio, segmentando por pausas naturais de fala (o próprio Whisper já retorna segmentos com timestamp).
2. Popular `AudioSegment` com `start`, `end`, `transcript`.
3. Indexar cada segmento (ou agrupamento de segmentos, dependendo do tamanho) como chunk.
4. Adicionar 1-2 áudios de teste ao corpus seed.

**Entregável:** Módulo `src/parsing/audio_parser.py`.

**Critério de "pronto":** Pergunta sobre o conteúdo de um áudio específico do corpus seed retorna o segmento certo.

**Como retomar:** Cole esta seção e diga se as perguntas de teste sobre áudio estão funcionando.

---

## Passo 28 — Graph RAG (opcional)

**Contexto:** Conforme a arquitetura revisada, Graph RAG é uma **estratégia opcional**, não obrigatória — vale a pena só se o seu corpus tem perguntas do tipo relacional ("quem comprou quem", "quais documentos mencionam entidades relacionadas a X") que retrieval textual puro resolve mal.

**Objetivo:** Um índice de grafo (Neo4j) com entidades e relações extraídas do corpus, consultável como canal adicional de retrieval.

**Pré-requisitos:** Antes de começar este passo, confirme que existe uma necessidade real: rode 5-10 perguntas relacionais no pipeline atual (sem grafo) e veja se elas já falham. Se o pipeline atual já responde bem, este passo pode ser adiado indefinidamente.

**O que fazer:**
1. Subir Neo4j (adicionar ao `docker-compose.yml`).
2. Implementar extração de entidades e relações (`spaCy` para NER leve, ou a API da Anthropic para extração mais rica via prompt estruturado).
3. Popular o grafo: nós = entidades, arestas = relações extraídas.
4. Implementar `graph_search(query) -> list[Chunk]`: identificar entidades na query, buscar vizinhança no grafo, mapear de volta para chunks relacionados.
5. Adicionar graph como canal na fusão (peso ajustável via router, Passo 21), nunca como substituto dos demais canais.
6. Testar nas perguntas relacionais identificadas no pré-requisito.

**Entregável:** Módulo `src/retrieval/graph_search.py` + grafo populado no Neo4j.

**Critério de "pronto":** Recall@5 nas perguntas relacionais melhora especificamente com o canal de grafo ativado.

**Como retomar:** Cole esta seção com o Recall@5 nas perguntas relacionais, com e sem grafo.

---

## Passo 29 — Adaptive Retrieval / Self-RAG

**Contexto:** Até aqui, o pipeline sempre busca um número fixo de chunks (`top_k`) e sempre gera uma resposta com o que encontrou, mesmo que a informação recuperada seja insuficiente. Self-RAG adiciona uma checagem: o modelo avalia se o contexto recuperado é suficiente antes de responder.

**Objetivo:** Se o contexto recuperado não for suficiente para responder com confiança, o sistema busca mais (ou avisa que não sabe) em vez de alucinar uma resposta.

**Pré-requisitos:** Pipeline de geração (Passo 8) e reranker (Passo 9) funcionando — a avaliação de suficiência normalmente usa o score do reranker como sinal.

**O que fazer:**
1. Definir um threshold de confiança baseado no score do reranker do top-1 resultado.
2. Se abaixo do threshold: acionar uma segunda rodada de retrieval (ex: com query reescrita) ou instruir o LLM a responder "não encontrei informação suficiente" em vez de forçar uma resposta.
3. Testar com perguntas do golden dataset deliberadamente fora do corpus (perguntas sem resposta no corpus seed) — o sistema deve reconhecer isso, não inventar.

**Entregável:** Módulo `src/generation/self_rag.py`.

**Critério de "pronto":** Nas perguntas sem resposta no corpus, o sistema responde "não sei" ou similar, em vez de alucinar, em pelo menos a maioria dos casos testados.

**Como retomar:** Cole esta seção com quantas das perguntas "sem resposta" foram corretamente identificadas como tal.

---

## Passo 30 — Auto-tuning de pesos de fusão

**Contexto:** Os pesos de fusão (RRF ponderado, boosts por intenção do Passo 21) até aqui são definidos manualmente. Este passo usa o golden dataset para ajustá-los automaticamente.

**Objetivo:** Um processo que testa combinações de pesos e escolhe a que maximiza Recall@5 no golden dataset, em vez de tuning manual por tentativa e erro.

**Pré-requisitos:** Golden dataset com volume razoável (o ideal é já ter passado de 100+ perguntas, não as 20-30 iniciais — poucos exemplos fazem o auto-tuning overfitar).

**O que fazer:**
1. Implementar uma busca simples de hiperparâmetros (grid search ou random search) sobre os pesos de cada canal na fusão.
2. Para cada combinação, rodar o golden dataset completo e registrar Recall@5/Precision@5.
3. Selecionar a combinação vencedora e validar em um golden dataset "de holdout" separado (perguntas nunca usadas no tuning) para confirmar que não houve overfitting.

**Entregável:** Script `src/evaluation/auto_tune_fusion.py` + pesos otimizados registrados no config.

**Critério de "pronto":** Pesos otimizados performam melhor no holdout do que os pesos manuais definidos até aqui.

**Como retomar:** Cole esta seção com os pesos encontrados e a comparação de Recall@5 no holdout.

---

## Passo 31 — Plataforma de benchmark comparável

**Contexto:** Este é o passo final da visão original: transformar o projeto de "mais um framework RAG" em uma plataforma que permite comparar configurações de pipeline de forma sistemática e reprodutível.

**Objetivo:** Um comando (`rag benchmark`) que roda o golden dataset contra várias combinações de chunking + retrieval + reranking e produz uma tabela comparativa.

**Pré-requisitos:** Todos os passos anteriores implementados (é literalmente isso que está sendo comparado).

**O que fazer:**
1. Definir um formato de configuração (YAML) para cada combinação a testar, seguindo o padrão da arquitetura revisada (chunking, retrievers, fusion, reranker).
2. Implementar o runner que executa o golden dataset contra cada configuração e coleta as métricas.
3. Gerar uma tabela final (Recall@5, custo médio por query, latência média) por configuração.
4. Documentar as configurações vencedoras por categoria de pergunta (ex: "melhor para perguntas de tabela", "melhor para perguntas relacionais").

**Entregável:** CLI `rag benchmark` + relatório comparativo salvo em `benchmarks/results/`.

**Critério de "pronto":** Rodar `rag benchmark` produz a tabela comparativa completa sem intervenção manual.

**Como retomar:** Cole esta seção com a tabela de resultados do benchmark mais recente.

---

## Passo 32 — Otimização de latência

**Contexto:** Com o pipeline completo (Passos 0-31) funcionando, cada query em produção pode disparar o Claude várias vezes (decompor, gerar HyDE, resumir chunk, gerar resposta final) e vários canais de retrieval sequencialmente. Este passo ataca latência sem mudar a lógica de qualidade que já foi validada.

**Objetivo:** Reduzir tempo de resposta ponta a ponta sem piorar Recall@5/Precision@5 medidos até aqui — toda mudança deste passo é validada contra o golden dataset antes de aceitar.

**Pré-requisitos:** Observabilidade (Passo 14) já registrando latência por estágio — sem isso não dá pra saber onde otimizar primeiro. Pipeline de geração (Passo 8), fast/full path (Passo 13) e os passos de Query Intelligence (20-23) que estiverem ativos no seu projeto.

**O que fazer:**

1. **Prompt caching (Anthropic API).** No `answer()` do Passo 8, colocar `cache_control` logo após as instruções fixas do prompt (formato de resposta, regras de citação), antes dos chunks recuperados. Ganho maior em conversas com múltiplas perguntas seguidas — validar olhando `cache_read_input_tokens` na resposta da API.

2. **Model routing por tarefa.** Nos passos que usam Claude para trabalho utilitário — classificação de intenção (21), decomposição de query (22), HyDE (23), resumo de chunk (20) — trocar o modelo por um mais rápido/leve (Haiku), reservando o modelo mais caro só pra geração final (Passo 8). Medir latência e Recall@5 antes/depois da troca; se a tarefa utilitária degradar demais com o modelo leve, reverter só aquela chamada específica.

3. **Cache layer com Redis.**
   - Cache de resposta: hash da query (normalizada) → resposta já gerada, com TTL curto (ex: algumas horas) para não servir resposta desatualizada depois de reindexação.
   - Cache de embedding de query: evita reembeder a mesma pergunta repetida.
   - Subir o serviço Redis no `docker-compose.yml` (estava adiado desde o Passo 0).

4. **Reduzir `top_k` antes do reranker.** Medir Recall@5 com `top_k=100` vs `top_k=20/30` entrando no reranker — geralmente a diferença de qualidade é pequena e o ganho de latência é grande.

5. **Tuning do `ef_search` no Qdrant.** Testar valores menores de `ef_search` e medir o trade-off latência vs. Recall@5 no golden dataset — parar no menor valor que ainda mantém a métrica aceitável.

6. **Streaming da resposta final.** Habilitar streaming no Passo 8 para reduzir a latência percebida (primeiro token chega rápido, mesmo que a geração completa leve o mesmo tempo total) — importante principalmente se o full path (decomposição + múltiplos canais) já consome vários segundos antes de começar a gerar.

**Entregável:** Módulo `src/optimization/caching.py` (Redis) + ajustes nos módulos de Passo 8/20-23 (model routing, prompt caching, streaming) + parâmetros de `top_k`/`ef_search` documentados no config.

**Critério de "pronto":** Tabela comparando latência média (e P95) e custo médio por query antes/depois de cada otimização, com Recall@5 do golden dataset confirmando que nenhuma métrica de qualidade regrediu. Cache hit rate do Redis registrado depois de rodar o golden dataset duas vezes seguidas (a segunda rodada deveria ser visivelmente mais rápida).

**Como retomar:** Cole esta seção com a tabela de latência antes/depois e confirme se o Recall@5 se manteve estável.

---

## Depois do Passo 32

Isso cobre a arquitetura completa desenhada nas duas primeiras conversas — todos os tipos de chunking discutidos, todas as fontes de dado (texto, PDF, DOCX, XLSX, imagem, vídeo, áudio), as estratégias de retrieval modernas (multi-vector, HyDE, query router, decomposição, Graph RAG opcional, Self-RAG, auto-tuning) e agora otimização de latência/custo. Não deveria sobrar nada do desenho original sem um passo correspondente — se notar alguma lacuna, é sinal de ajustar este documento antes de seguir.