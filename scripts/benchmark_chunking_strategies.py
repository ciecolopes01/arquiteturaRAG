"""
Benchmark comparativo entre as 3 estratégias de chunking — Passo 18.

Estratégias avaliadas:
1. Recursive Character Chunking (Passo 3)
2. Parent/Child Chunking (Passo 17)
3. Semantic Chunking com BGE-M3 (Passo 18)

Métricas medidas:
- Tempo de processamento / latência de chunking (ms por documento).
- Quantidade total de chunks gerados.
- Tamanho médio dos chunks (palavras).
- Recall@5 e MRR sobre o conjunto de perguntas representativas do Golden Dataset.
- Tradeoff de custo computacional vs preservação semântica.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, ".")

from src.chunking.parent_child import chunk_parent_child
from src.chunking.recursive import chunk_recursive
from src.chunking.semantic import chunk_semantic
from src.embedding.model import embed_query, embed_texts
from src.generation.answer import expand_parent_context
from src.parsing.simple_parsers import parse_file
from src.retrieval.hybrid import HybridResult

SEED_FILES = [
    "data/seed_corpus/09_fast_full_path_routing.pdf",
    "data/seed_corpus/10_reindexacao_incremental.pdf",
    "data/seed_corpus/11_query_decomposition_hyde.docx",
    "data/seed_corpus/12_migracao_embedding.docx",
    "data/seed_corpus/13_tabelas_financeiras.pdf",
    "data/seed_corpus/14_budget_projetos.xlsx",
    "data/seed_corpus/15_kpis_produto.xlsx",
]

BENCHMARK_QUERIES = [
    {
        "id": "gd-26",
        "query": "Qual é o valor de threshold usado para decidir entre fast path e full path?",
        "expected_doc": "09_fast_full_path_routing.pdf",
        "keywords": ["threshold", "0.6", "fast path", "full path"],
    },
    {
        "id": "gd-28",
        "query": "Qual algoritmo de hash é usado para detectar se um documento mudou antes de reindexar?",
        "expected_doc": "10_reindexacao_incremental.pdf",
        "keywords": ["sha-256", "content_hash", "reindex"],
    },
    {
        "id": "gd-30",
        "query": "O que é HyDE e o que ele gera antes de buscar?",
        "expected_doc": "11_query_decomposition_hyde.docx",
        "keywords": ["hyde", "hipotética", "resposta", "embedding"],
    },
    {
        "id": "gd-31",
        "query": "Qual a diferença entre o que Query Decomposition e HyDE resolvem?",
        "expected_doc": "11_query_decomposition_hyde.docx",
        "keywords": ["decomposition", "compostas", "hyde", "ambíguas"],
    },
    {
        "id": "gd-32",
        "query": "Na migração de modelo de embedding, o que é feito antes de apagar o modelo antigo?",
        "expected_doc": "12_migracao_embedding.docx",
        "keywords": ["rollback", "período", "coleção antiga", "migração"],
    },
    {
        "id": "gd-34",
        "query": "Qual estratégia de retrieval tem o maior MRR e qual é seu valor segundo a tabela de performance?",
        "expected_doc": "13_tabelas_financeiras.pdf",
        "keywords": ["0.8182", "mrr", "reranker", "tabela"],
    },
    {
        "id": "gd-36",
        "query": "Qual modelo de embedding tem o menor numero de parametros segundo o comparativo?",
        "expected_doc": "13_tabelas_financeiras.pdf",
        "keywords": ["all-minilm", "22m", "384", "parâmetros"],
    },
    {
        "id": "gd-39",
        "query": "Qual o orçamento e o valor gasto no projeto DataLake Migration?",
        "expected_doc": "14_budget_projetos.xlsx",
        "keywords": ["datalake migration", "1.200.000", "orçamento"],
    },
    {
        "id": "gd-41",
        "query": "Qual foi a eficiência percentual e o saldo em reais no trimestre Q2_2026?",
        "expected_doc": "14_budget_projetos.xlsx",
        "keywords": ["q2_2026", "73.8", "550.000", "eficiência"],
    },
    {
        "id": "gd-43",
        "query": "Qual foi o MRR e o ARR em reais registrado no mês de Jan_2026?",
        "expected_doc": "15_kpis_produto.xlsx",
        "keywords": ["jan_2026", "420.000", "5.040.000", "mrr"],
    },
]


def _match_score(text: str, keywords: List[str]) -> float:
    text_lower = text.lower()
    return sum(1.0 for kw in keywords if kw.lower() in text_lower)


def run_benchmark():
    print("=" * 85)
    print("      BENCHMARK DE ESTRATÉGIAS DE CHUNKING: RECURSIVE vs PARENT/CHILD vs SEMANTIC")
    print("=" * 85)

    # 1. Carrega e parseia os documentos canônicos
    print("\n[1/3] Carregando documentos do seed corpus...")
    docs = []
    for p in SEED_FILES:
        if os.path.exists(p):
            docs.append(parse_file(p))
    print(f"      {len(docs)} documentos carregados.")

    # 2. Executa cada estratégia de chunking medindo latência e volume
    print("\n[2/3] Executando chunking nas 3 estratégias...")
    strategies = ["recursive", "parent_child", "semantic"]
    chunks_by_strategy: Dict[str, list] = {}
    time_by_strategy: Dict[str, float] = {}

    for strat in strategies:
        t0 = time.time()
        strat_chunks = []
        for doc in docs:
            if strat == "recursive":
                strat_chunks.extend(chunk_recursive(doc, max_tokens=512, overlap=50))
            elif strat == "parent_child":
                res = chunk_parent_child(doc, parent_max_tokens=1200, child_max_tokens=300, child_overlap=40)
                strat_chunks.extend(res.children)
            elif strat == "semantic":
                strat_chunks.extend(chunk_semantic(doc, similarity_threshold=0.75, max_tokens=600, min_tokens=50))
        elapsed_s = time.time() - t0
        chunks_by_strategy[strat] = strat_chunks
        time_by_strategy[strat] = elapsed_s

        words = [len(c.content.split()) for c in strat_chunks]
        avg_words = sum(words) / max(len(words), 1)
        print(f"      -> {strat:<13}: {len(strat_chunks):>3} chunks | {elapsed_s:>6.2f}s total ({elapsed_s/len(docs)*1000:>5.1f}ms/doc) | média {avg_words:.1f} palavras/chunk")

    # 3. Avalia Recall@5 e MRR no benchmark de queries
    print("\n[3/3] Avaliando Retrieval (Recall@5 e MRR) nas queries do Golden Dataset...")
    eval_metrics: Dict[str, dict] = {}

    for strat in strategies:
        c_list = chunks_by_strategy[strat]
        hits_at_5 = 0
        reciprocal_ranks = []

        for q_item in BENCHMARK_QUERIES:
            kws = q_item["keywords"]
            exp_doc = q_item["expected_doc"]

            # Ranqueia chunks por correspondência
            ranked = sorted(c_list, key=lambda c: _match_score(c.content, kws), reverse=True)
            top_5 = ranked[:5]

            # Se for Parent/Child, expande para o pai
            if strat == "parent_child":
                hybrid_candidates = [
                    HybridResult(
                        chunk_id=c.chunk_id,
                        content=c.content,
                        rrf_score=0.03,
                        source_uri=c.metadata.get("source_uri", ""),
                        parent_chunk_id=c.parent_chunk_id,
                        parent_content=c.metadata.get("parent_content"),
                    )
                    for c in top_5
                ]
                expanded = expand_parent_context(hybrid_candidates)
                is_hit = False
                rank_hit = 0
                for rank, exp_c in enumerate(expanded, 1):
                    if exp_doc.lower() in (exp_c.source_uri or "").lower() and any(kw.lower() in exp_c.content.lower() for kw in kws):
                        is_hit = True
                        rank_hit = rank
                        break
            else:
                is_hit = False
                rank_hit = 0
                for rank, c in enumerate(top_5, 1):
                    src = c.metadata.get("source_uri", "")
                    if exp_doc.lower() in src.lower() and any(kw.lower() in c.content.lower() for kw in kws):
                        is_hit = True
                        rank_hit = rank
                        break

            if is_hit:
                hits_at_5 += 1
                reciprocal_ranks.append(1.0 / rank_hit)
            else:
                reciprocal_ranks.append(0.0)

        recall_5 = hits_at_5 / len(BENCHMARK_QUERIES)
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        eval_metrics[strat] = {
            "recall_5": recall_5,
            "mrr": mrr,
            "hits": hits_at_5,
            "total": len(BENCHMARK_QUERIES),
        }

    # 4. Tabela Consolidada de Resultados
    print("\n" + "=" * 85)
    print("                    TABELA COMPARATIVA DAS 3 ESTRATÉGIAS")
    print("=" * 85)
    print(f"{'Estratégia':<15} | {'Chunks':<7} | {'Média Palavras':<15} | {'Tempo Total':<12} | {'Recall@5':<9} | {'MRR':<6}")
    print("-" * 85)
    for strat in strategies:
        n_chunks = len(chunks_by_strategy[strat])
        avg_w = sum(len(c.content.split()) for c in chunks_by_strategy[strat]) / max(n_chunks, 1)
        t_tot = f"{time_by_strategy[strat]:.2f}s"
        rec = f"{eval_metrics[strat]['recall_5']:.4f}"
        mrr_val = f"{eval_metrics[strat]['mrr']:.4f}"
        print(f"{strat:<15} | {n_chunks:<7} | {avg_w:<15.1f} | {t_tot:<12} | {rec:<9} | {mrr_val:<6}")
    print("-" * 85)

    print("\n[Análise Comparativa]")
    print("1. Recursive    : Mais rápido (sem custo de embedding prévio na quebra), bom baseline.")
    print("2. Parent/Child : Busca de alta precisão via filhos + contexto expandido (~1200 tokens) para o LLM.")
    print("3. Semantic     : Respeita fronteiras de significado natural das sentenças, maior custo de indexação.")
    print("=" * 85)


if __name__ == "__main__":
    run_benchmark()
