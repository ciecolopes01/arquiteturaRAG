"""
Testes unitários para o chunking Parent/Child e Context Engine (Passo 17).

Valida:
1. Divisão hierárquica em chunks pais e filhos.
2. Integridade referencial: todo filho possui parent_chunk_id válido apontando para um pai existente.
3. Propagação de metadados e parent_content para os filhos.
4. Tratamento de elementos especiais (seções, parágrafos grandes, tabelas).
5. Expansão de contexto e deduplicação no Context Engine (múltiplos filhos do mesmo pai
   condensados em um único bloco expandido).
6. Fallback gracioso para chunks legados/sem pai.
"""

import sys
import unittest

sys.path.insert(0, ".")

from src.chunking.parent_child import (
    chunk_parent_child,
)
from src.generation.answer import _build_context_block, expand_parent_context
from src.parsing.canonical_model import Document, Paragraph, Section, Table
from src.retrieval.hybrid import HybridResult


class TestParentChildChunking(unittest.TestCase):
    """Testes da lógica de chunking hierárquico."""

    def setUp(self):
        self.doc = Document(
            id="doc-test-pc-01",
            title="Arquitetura de Sistemas RAG",
            source_uri="docs/arquitetura_rag.txt",
            tenant_id="tenant-acme",
        )
        # Seção 1: Introdução
        self.doc.elements.append(Section(
            id="sec-01",
            title="1. Fundamentos de RAG",
            content="RAG (Retrieval-Augmented Generation) combina modelos de linguagem com recuperação de informação."
        ))
        for i in range(1, 6):
            self.doc.elements.append(Paragraph(
                id=f"p-{i}",
                content=(
                    f"Parágrafo {i}: O processo de busca vetorial calcula a similaridade de cosseno entre a query "
                    f"e os chunks indexados no banco vetorial. Quanto maior o cosseno, maior a relevância semântica. "
                    f"Isso permite recuperar contextos altamente pertinentes para o modelo gerador formular a resposta final."
                )
            ))

        # Seção 2: Tabelas e Métricas
        self.doc.elements.append(Section(
            id="sec-02",
            title="2. Métricas de Avaliação",
            content="Métricas clássicas incluem Recall@K, MRR e NDCG."
        ))
        for i in range(6, 11):
            self.doc.elements.append(Paragraph(
                id=f"p-{i}",
                content=(
                    f"Parágrafo {i}: O Recall@5 mede se ao menos um chunk relevante está presente no top 5 resultados retornados. "
                    f"Já o MRR (Mean Reciprocal Rank) avalia a posição média do primeiro resultado relevante. "
                    f"Essas métricas são computadas offline através de um golden dataset cuidadosamente curado."
                )
            ))

    def test_parent_and_child_generation(self):
        """Verifica que pais e filhos são gerados e respeitam a hierarquia."""
        result = chunk_parent_child(
            self.doc,
            parent_max_tokens=600,
            child_max_tokens=150,
            child_overlap=30,
        )

        self.assertGreater(result.total_parents, 0)
        self.assertGreater(result.total_children, result.total_parents)

        # Valida propriedades dos pais
        parent_ids = {p.chunk_id for p in result.parents}
        self.assertEqual(len(parent_ids), result.total_parents, "IDs de pais devem ser únicos")

        for parent in result.parents:
            self.assertIsNone(parent.parent_chunk_id, "Pai não deve ter parent_chunk_id")
            self.assertTrue(parent.metadata.get("is_parent"))
            self.assertFalse(parent.metadata.get("is_child"))
            self.assertEqual(parent.tenant_id, "tenant-acme")
            self.assertEqual(parent.document_id, "doc-test-pc-01")

        # Valida integridade referencial dos filhos
        for child in result.children:
            self.assertIsNotNone(child.parent_chunk_id, "Filho deve possuir parent_chunk_id")
            self.assertIn(child.parent_chunk_id, parent_ids, "parent_chunk_id deve apontar para pai existente")
            self.assertTrue(child.metadata.get("is_child"))
            self.assertFalse(child.metadata.get("is_parent"))
            self.assertIn("parent_content", child.metadata)
            self.assertGreater(len(child.metadata["parent_content"]), len(child.content))
            self.assertIn(child.content.split()[0], child.metadata["parent_content"])

    def test_lookup_helpers(self):
        """Testa métodos de busca do ParentChildResult."""
        result = chunk_parent_child(self.doc, parent_max_tokens=500, child_max_tokens=150)
        first_parent = result.parents[0]

        # Busca pelo pai existente
        found_parent = result.get_parent(first_parent.chunk_id)
        self.assertEqual(found_parent, first_parent)

        # Busca por pai inexistente
        self.assertIsNone(result.get_parent("id-inexistente"))

        # Busca filhos do pai
        children_of_first = result.get_children_for_parent(first_parent.chunk_id)
        self.assertGreater(len(children_of_first), 0)
        for c in children_of_first:
            self.assertEqual(c.parent_chunk_id, first_parent.chunk_id)

    def test_table_element_handling(self):
        """Testa documento contendo tabela estruturada."""
        doc_with_table = Document(
            id="doc-tbl-pc",
            title="Relatório de Custos",
            source_uri="finance/custos.xlsx",
            tenant_id="tenant-fin",
        )
        table_content = (
            "| Projeto | Orçamento | Gasto Real |\n"
            "|---|---|---|\n"
            "| RAG Core | R$ 150.000 | R$ 110.000 |\n"
            "| Cloud Ops | R$ 80.000 | R$ 75.000 |\n\n"
            "## Sumário\nTabela com 2 projetos orçados totalizando R$ 230.000."
        )
        doc_with_table.elements.append(Table(
            id="tbl-01",
            content=table_content,
            headers=["Projeto", "Orçamento", "Gasto Real"],
            rows=[["RAG Core", "R$ 150.000", "R$ 110.000"], ["Cloud Ops", "R$ 80.000", "R$ 75.000"]],
        ))

        result = chunk_parent_child(doc_with_table, parent_max_tokens=800, child_max_tokens=200)
        self.assertEqual(result.total_parents, 1)
        parent = result.parents[0]
        self.assertTrue(parent.metadata.get("has_table_summary"))
        self.assertEqual(parent.metadata.get("table_row_count"), 2)

        # Filhos da tabela
        self.assertGreaterEqual(result.total_children, 1)
        for child in result.children:
            self.assertEqual(child.parent_chunk_id, parent.chunk_id)
            self.assertIn("RAG Core", child.metadata["parent_content"])


class TestContextEngineExpansion(unittest.TestCase):
    """Testes da expansão de contexto e deduplicação no Context Engine."""

    def setUp(self):
        self.parent1_id = "parent-uuid-001"
        self.parent1_content = (
            "CONTEÚDO PAI COMPLETO 1: Seção completa cobrindo arquitetura do banco vetorial Qdrant, "
            "configurações de HNSW, distância coseno e escalabilidade horizontal em múltiplos nós."
        )

        self.parent2_id = "parent-uuid-002"
        self.parent2_content = (
            "CONTEÚDO PAI COMPLETO 2: Seção completa cobrindo o modelo de embedding BAAI/bge-m3, "
            "suas representações densas de 1024 dimensões e suporte multilíngue nativo para português."
        )

    def test_deduplication_of_sibling_children(self):
        """Dois filhos do mesmo pai devem ser consolidados em um único bloco do pai."""
        child1_a = HybridResult(
            chunk_id="child-1a",
            content="Trecho pequeno A do pai 1 (Qdrant HNSW)",
            rrf_score=0.033,
            source_uri="docs/vetorial.txt",
            section_title="Bancos Vetoriais",
            parent_chunk_id=self.parent1_id,
            parent_content=self.parent1_content,
        )
        child1_b = HybridResult(
            chunk_id="child-1b",
            content="Trecho pequeno B do pai 1 (escalabilidade de nós)",
            rrf_score=0.025,
            source_uri="docs/vetorial.txt",
            section_title="Bancos Vetoriais",
            parent_chunk_id=self.parent1_id,
            parent_content=self.parent1_content,
        )
        child2 = HybridResult(
            chunk_id="child-2",
            content="Trecho pequeno C do pai 2 (BGE-M3 1024 dim)",
            rrf_score=0.020,
            source_uri="docs/embeddings.txt",
            section_title="Modelos de Embedding",
            parent_chunk_id=self.parent2_id,
            parent_content=self.parent2_content,
        )

        retrieved = [child1_a, child1_b, child2]
        expanded = expand_parent_context(retrieved)

        # Eram 3 filhos (sendo 2 do mesmo pai), viram 2 pais únicos!
        self.assertEqual(len(expanded), 2)
        self.assertEqual(expanded[0].chunk_id, self.parent1_id)
        self.assertEqual(expanded[0].content, self.parent1_content)
        self.assertEqual(expanded[1].chunk_id, self.parent2_id)
        self.assertEqual(expanded[1].content, self.parent2_content)

    def test_fallback_for_unparented_chunks(self):
        """Chunks legados sem parent_content devem ser preservados sem alteração."""
        legacy_chunk = HybridResult(
            chunk_id="legacy-chunk-01",
            content="Chunk padrão antigo sem pai.",
            rrf_score=0.015,
            source_uri="docs/legado.txt",
            parent_chunk_id=None,
            parent_content=None,
        )

        expanded = expand_parent_context([legacy_chunk])
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0].chunk_id, "legacy-chunk-01")
        self.assertEqual(expanded[0].content, "Chunk padrão antigo sem pai.")

    def test_context_block_formatting(self):
        """Verifica se a formatação final do bloco de contexto exibe tag de expansão."""
        expanded_chunk = HybridResult(
            chunk_id="parent-01",
            content="Texto expandido do documento pai.",
            rrf_score=0.040,
            source_uri="docs/guia.pdf",
            section_title="Introdução",
            document_title="Guia RAG",
            parent_chunk_id="parent-01",
            parent_content="Texto expandido do documento pai.",
        )

        block = _build_context_block([expanded_chunk])
        self.assertIn("[Fonte 1 (Contexto Expandido)]", block)
        self.assertIn("Seção: Introdução", block)
        self.assertIn("Doc: Guia RAG", block)
        self.assertIn("Texto expandido do documento pai.", block)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestParentChildChunking)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestContextEngineExpansion))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
