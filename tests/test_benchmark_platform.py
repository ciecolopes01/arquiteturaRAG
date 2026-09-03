"""
Testes unitários e de integração para a Plataforma de Benchmark Comparável — Passo 31.

Valida:
1. Carregamento e validação de configurações de pipeline em YAML (benchmarks/configs/).
2. Categorização de perguntas por intenção/domínio do golden dataset.
3. Execução do runner de benchmark e cálculo das métricas comparativas.
4. Geração automática de relatórios em Markdown e JSON (benchmarks/results/).
5. Critério de "pronto": Execução do comando CLI `rag benchmark` produzindo
   a tabela comparativa sem intervenção manual.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, ".")

from src.evaluation.benchmark import (
    BenchmarkResult,
    PipelineConfig,
    benchmark_config,
    categorize_question,
    generate_reports,
    run_benchmark_suite,
)

CONFIGS_DIR = Path("benchmarks/configs")
RESULTS_DIR = Path("benchmarks/results")


def test_pipeline_config_yaml_parsing():
    """Valida leitura e parse das configurações YAML oficiais."""
    yaml_files = list(CONFIGS_DIR.glob("*.yaml"))
    assert len(yaml_files) >= 4, f"Esperado pelo menos 4 configs em {CONFIGS_DIR}, encontrado {len(yaml_files)}"

    for yf in yaml_files:
        cfg = PipelineConfig.from_yaml(yf)
        assert len(cfg.name) > 0
        assert len(cfg.retrievers) > 0
        assert cfg.fusion_strategy in ["single", "rrf", "weighted_rrf"]


def test_categorize_question_domains():
    """Valida classificação em categorias de domínio do RAG."""
    assert categorize_question("Qual a relação entre as entidades A e B?", "relacional") == "Relacional / Grafo"
    assert categorize_question("Em que página aparece a figura 1 do diagrama?", "multimodal") == "Multimodal / Imagem"
    assert categorize_question("Qual o MRR e ARR registrado na planilha financeira?", "xlsx") == "Tabelas / XLSX"
    assert categorize_question("Explique como funciona a fusão de rankings no RRF", "conceito") == "Semântica / Conceitual"
    assert categorize_question("Qual a porta padrão do serviço Qdrant?", "infra") == "Factual / Específica"


def test_benchmark_config_and_report_generation(tmp_path: Path):
    """Valida a execução de benchmark sobre amostra e geração de relatórios."""
    sample_dataset = [
        {
            "id": "q1",
            "question": "O que é RRF?",
            "expected_chunk_ids": ["chk_01"],
            "notes": "conceitual",
        },
        {
            "id": "q2",
            "question": "Qual a porta do Qdrant?",
            "expected_chunk_ids": ["chk_02"],
            "notes": "factual",
        },
    ]

    cfg = PipelineConfig(
        name="Test Pipeline",
        description="Pipeline sintético de teste",
        retrievers=["dense", "bm25"],
        fusion_strategy="rrf",
    )

    res = benchmark_config(cfg, sample_dataset, top_k=5)
    assert isinstance(res, BenchmarkResult)
    assert res.config_name == "Test Pipeline"
    assert 0.0 <= res.recall_at_5 <= 1.0
    assert 0.0 <= res.mrr <= 1.0
    assert res.total_queries == 2
    assert res.avg_latency_ms >= 0.0

    # Gera relatórios no diretório temporário
    md_file, json_file = generate_reports([res], tmp_path)
    assert md_file.exists()
    assert json_file.exists()
    assert md_file.stat().st_size > 50

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)
        assert len(data) == 1
        assert data[0]["config_name"] == "Test Pipeline"


def test_cli_rag_benchmark_command():
    """
    Critério de "pronto" do Passo 31:
    O comando de benchmark executa e produz a tabela comparativa sem intervenção manual.
    """
    cmd = [sys.executable, "rag.py", "benchmark", "--sample", "2"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert res.returncode == 0, f"Comando falhou com código {res.returncode}:\nSTDERR: {res.stderr}"
    assert "PLATAFORMA DE BENCHMARK" in res.stdout
    assert "TABELA COMPARATIVA FINAL" in res.stdout
    assert "Recall@5" in res.stdout
    assert "MRR" in res.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
