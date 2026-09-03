"""
Interface de Linha de Comando (CLI) unificada da Plataforma RAG.

Comandos:
  rag benchmark       Executa a plataforma de benchmark comparável (Passo 31)
"""

import sys
from pathlib import Path

# Garante raiz no PYTHONPATH
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        from src.evaluation.benchmark import main as benchmark_main
        sys.argv = [sys.argv[0]] + sys.argv[2:] + ["benchmark"]
        benchmark_main()
    else:
        print("Uso: python rag.py benchmark [--configs CAMINHO] [--sample N]")
        sys.exit(1)


if __name__ == "__main__":
    main()
