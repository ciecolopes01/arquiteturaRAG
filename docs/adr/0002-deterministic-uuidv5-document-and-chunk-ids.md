# ADR 0002: Identificadores Determinísticos UUIDv5 para Documentos e Chunks

## Status
Aceito

## Data
2026-09-04

## Contexto
Originalmente, `Document.id` era instanciado via `uuid.uuid4()`. Embora `generate_chunk_id()` utilizasse `uuid.uuid5()`, sua semente continha `document.id`.
Como consequência, um UUIDv5 semeado por um UUIDv4 aleatório resultava em identificadores de chunks instáveis que mudavam a cada nova ingestão do mesmo documento.

Isso causava:
1. Ruptura total do `data/golden_dataset.json` (que ancora ground-truth em IDs fixos), fazendo o Recall@5 cair para 0.0000 após qualquer reindexação.
2. Criação de pontos vetoriais duplicados no Qdrant durante ingestões subsequentes do mesmo arquivo.
3. Impossibilidade de validação de cache de chunks entre pipelines.

## Decisão
1. Definir namespace padrão para documentos: `DOC_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "arquiteturaRAG.documents")`.
2. Calcular `Document.id` de maneira determinística através da função `document_id_for(source_uri: str, content_hash: str) -> str`, combinando o URI de origem e o hash SHA-256 do conteúdo.
3. Chunks herdam o `Document.id` determinístico e combinam com seu `chunk_index` e hash do trecho (`content_hash`).
4. Versionar o corpus semente em `data/seed_corpus/` no controle de versão Git para permitir reprodução fiel do golden dataset em qualquer ambiente.

## Consequências
- **Positivas:** Re-ingestões do mesmo arquivo produzem exatamente os mesmos IDs de documentos e chunks (idempotência).
- **Positivas:** Reindexações incrementais realizam *upsert* fiel no Qdrant e no Postgres sem duplicação de dados.
- **Positivas:** Métricas de Recall@k e MRR no benchmark offline tornam-se 100% confiáveis e reprodutíveis.
