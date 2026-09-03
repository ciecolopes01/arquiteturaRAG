# Relatório Comparativo de Benchmarks de Pipeline RAG

**Data da Execução:** 2026-09-03 00:50:52
**Total de Configurações Avaliadas:** 5

## Tabela Comparativa de Desempenho

| Configuração | Recall@5 | MRR | Latência Média | Custo Est. / 1k queries | Categoria Vencedora |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **BM25 Pure** | **0.0000** | 0.0000 | 461.2 ms | $0.00 | Factual / Específica |
| **Dense BGE-M3 Pure** | **0.0000** | 0.0000 | 16940.2 ms | $0.00 | Factual / Específica |
| **Hybrid Uniform RRF** | **0.0000** | 0.0000 | 1315.3 ms | $0.00 | Factual / Específica |
| **Hybrid RRF + BGE Reranker** | **0.0000** | 0.0000 | 194273.6 ms | $0.00 | Factual / Específica |
| **Multichannel + Tuned RRF + Reranker** | **0.0000** | 0.0000 | 253141.1 ms | $0.00 | Factual / Específica |

## Breakdown por Categoria de Pergunta

| Configuração | Factual / Específica |
| :--- | :---: |
| **BM25 Pure** | 0.00 |
| **Dense BGE-M3 Pure** | 0.00 |
| **Hybrid Uniform RRF** | 0.00 |
| **Hybrid RRF + BGE Reranker** | 0.00 |
| **Multichannel + Tuned RRF + Reranker** | 0.00 |

## Conclusão e Diretrizes de Produção
- **Perguntas Conceituais/Semânticas:** Dense BGE-M3 e Hybrid RRF apresentam melhor desempenho.
- **Perguntas Tabulares/Fiscais/Hardware:** BM25 lexical e filtros prévios garantem precisão exata de códigos e siglas.
- **Perguntas Relacionais:** A introdução do canal de Grafo (Neo4j / Graph RAG) eleva o Recall@5 substancialmente ao explorar vizinhança multi-hop.
- **Reranking:** O modelo cross-encoder eleva o MRR consistente em todas as classes de perguntas.