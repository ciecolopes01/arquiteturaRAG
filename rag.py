"""
Interface de Linha de Comando (CLI) unificada da Plataforma RAG.

Subcomandos:
  rag answer "sua pergunta"   Executa resposta de pergunta via pipeline RAG
  rag benchmark               Executa a plataforma de benchmark comparável (Passo 31)
  rag eval                    Executa avaliação offline sobre o golden dataset
"""

import argparse
import sys
from pathlib import Path

# Garante raiz no PYTHONPATH
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _cmd_answer(args):
    from src.generation.answer import answer, answer_stream
    from src.generation.self_rag import adaptive_answer

    principals = [p.strip() for p in args.principals.split(",")] if args.principals else None

    if args.adaptive:
        print(f"[CLI] Modo Adaptive Self-RAG para: '{args.query}'\n")
        res = adaptive_answer(
            query=args.query,
            top_k=args.top_k,
            tenant_id=args.tenant,
            principals=principals,
        )
        print(f"Decisão: {res.decision.value} (confiança: {res.confidence_score:.2f})")
        print("-" * 60)
        print(res.answer)
        return

    if args.stream:
        print(f"[CLI] Streaming para: '{args.query}'\n")
        for token in answer_stream(
            query=args.query,
            top_k=args.top_k,
            tenant_id=args.tenant,
            principals=principals,
        ):
            print(token, end="", flush=True)
        print()
    else:
        res = answer(
            query=args.query,
            top_k=args.top_k,
            tenant_id=args.tenant,
            principals=principals,
        )
        res.pretty_print()


def _cmd_benchmark(args):
    from src.evaluation.benchmark import run_benchmark_suite

    print("=" * 70)
    print("  PLATAFORMA DE BENCHMARK COMPARÁVEL (Passo 31) — rag benchmark")
    print("=" * 70)
    suite_res = run_benchmark_suite(
        configs_path=Path(args.configs),
        sample_size=args.sample,
        top_k=args.top_k,
    )
    print("\n" + "=" * 70)
    print("TABELA COMPARATIVA FINAL:")
    print("=" * 70)
    print(f"{'Configuração':<32} {'Recall@5':>10} {'MRR':>8} {'Latência':>12}")
    print("-" * 70)
    for r in sorted(suite_res, key=lambda x: (x.recall_at_5, x.mrr), reverse=True):
        print(f"{r.config_name:<32} {r.recall_at_5:>10.4f} {r.mrr:>8.4f} {r.avg_latency_ms:>10.1f}ms")


def _cmd_eval(args):
    from src.evaluation.offline_eval import evaluate_recall_at_k, load_golden_dataset

    dataset = load_golden_dataset()
    print(f"Golden dataset carregado: {len(dataset)} perguntas")
    print(f"Avaliando modo: {args.mode.upper()} (k={args.k})...\n")

    r = evaluate_recall_at_k(
        dataset,
        k=args.k,
        mode=args.mode,
        tenant_id=args.tenant,
        verbose=True,
    )
    print("\n" + "=" * 60)
    print(f"Recall@{args.k}: {r['recall_at_k']:.4f} | MRR: {r['mrr']:.4f} | Hits: {r['hits']}/{r['n_questions']}")
    print(f"Latência média: {r['ms_per_query']:.1f}ms/query")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        prog="rag",
        description="CLI unificada da Plataforma RAG (arquiteturaRAG)",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Comando a executar")

    # Subcomando: answer
    parser_ans = subparsers.add_parser("answer", help="Responde uma pergunta usando o pipeline RAG")
    parser_ans.add_argument("query", type=str, help="Pergunta do usuário")
    parser_ans.add_argument("--tenant", type=str, default=None, help="Identificador do tenant")
    parser_ans.add_argument("--principals", type=str, default=None, help="Lista de principals separados por vírgula (ex: user:1,grupo:eng)")
    parser_ans.add_argument("--top-k", type=int, default=5, help="Número de chunks de contexto (default: 5)")
    parser_ans.add_argument("--stream", action="store_true", help="Habilita streaming token-a-token")
    parser_ans.add_argument("--adaptive", action="store_true", help="Usa Self-RAG adaptativo com recusa inteligente")
    parser_ans.set_defaults(func=_cmd_answer)

    # Subcomando: benchmark
    parser_bench = subparsers.add_parser("benchmark", help="Executa a suíte comparativa de pipelines RAG")
    parser_bench.add_argument("--configs", type=str, default="benchmarks/configs", help="Diretório com arquivos YAML de configuração")
    parser_bench.add_argument("--sample", type=int, default=None, help="Amostra do golden dataset")
    parser_bench.add_argument("--top-k", type=int, default=5, help="Top K chunks avaliados (default: 5)")
    parser_bench.set_defaults(func=_cmd_benchmark)

    # Subcomando: eval
    parser_eval = subparsers.add_parser("eval", help="Avaliação offline do golden dataset")
    parser_eval.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid", help="Modo de busca a avaliar")
    parser_eval.add_argument("--k", type=int, default=5, help="K para Recall@K (default: 5)")
    parser_eval.add_argument("--tenant", type=str, default="tenant-test", help="Tenant ID para avaliação")
    parser_eval.set_defaults(func=_cmd_eval)

    # Parsing
    args = parser.parse_args()
    if not args.subcommand:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
