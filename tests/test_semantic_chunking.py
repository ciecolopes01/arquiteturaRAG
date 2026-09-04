"""
Testes unitários para o chunker semântico (Passo 18).

Valida:
1. Divisão precisa em sentenças.
2. Cálculo da similaridade de cosseno.
3. Detecção de breakpoints semânticos entre tópicos divergentes.
4. Preservação estrutural de tabelas no modelo canônico.
5. Metadados e respeito aos limites min_tokens e max_tokens.
"""

import sys
import unittest

sys.path.insert(0, ".")

from src.chunking.semantic import (
    _cosine_similarity,
    _split_into_sentences,
    chunk_semantic,
)
from src.parsing.canonical_model import Document, Paragraph, Section, Table


class TestSentenceSplitter(unittest.TestCase):
    """Testes da divisão de texto em sentenças."""

    def test_basic_sentences(self):
        text = "Primeira sentença sobre RAG. Segunda sentença com mais detalhes! Terceira pergunta aqui?"
        sentences = _split_into_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0], "Primeira sentença sobre RAG.")
        self.assertEqual(sentences[1], "Segunda sentença com mais detalhes!")
        self.assertEqual(sentences[2], "Terceira pergunta aqui?")

    def test_single_sentence(self):
        text = "Apenas uma sentença sem quebras adicionais."
        sentences = _split_into_sentences(text)
        self.assertEqual(len(sentences), 1)

    def test_empty_string(self):
        sentences = _split_into_sentences("   ")
        self.assertEqual(len(sentences), 0)


class TestCosineSimilarity(unittest.TestCase):
    """Testes da função de similaridade de cosseno."""

    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(_cosine_similarity(v, v), 1.0, places=4)

    def test_orthogonal_vectors(self):
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        self.assertAlmostEqual(_cosine_similarity(v1, v2), 0.0, places=4)

    def test_zero_vector(self):
        v1 = [0.0, 0.0]
        v2 = [1.0, 2.0]
        self.assertEqual(_cosine_similarity(v1, v2), 0.0)


class TestSemanticChunking(unittest.TestCase):
    """Testes ponta a ponta do chunking semântico."""

    def test_semantic_split_topics(self):
        """Verifica se tópicos semanticamente distintos causam corte."""
        doc = Document(
            id="doc-semantic-test",
            title="Comparativo de Tópicos",
            source_uri="docs/test.txt",
            tenant_id="tenant-test",
        )
        # Seção com dois tópicos radicalmente diferentes
        doc.elements.append(Section(id="s1", title="Tópicos Mistos"))
        doc.elements.append(Paragraph(
            id="p1",
            content=(
                "Redes neurais artificiais aprendem representações através de gradiente descendente estocástico. "
                "O modelo otimiza pesos sinápticos minimizando uma função de perda calculada no conjunto de treino. "
                "Arquiteturas transformers utilizam camadas de auto-atenção multi-cabeça para capturar dependências textuais."
            )
        ))
        doc.elements.append(Paragraph(
            id="p2",
            content=(
                "Para fazer um bolo de cenoura tradicional, bata as cenouras cruas com óleo e ovos no liquidificador. "
                "Em seguida, acrescente açúcar e farinha de trigo, misturando até obter uma massa homogênea. "
                "Leve ao forno pré-aquecido a 180 graus por aproximadamente quarenta minutos."
            )
        ))

        chunks = chunk_semantic(doc, similarity_threshold=0.75, min_tokens=15, max_tokens=300)
        self.assertGreaterEqual(len(chunks), 2, "Tópicos de IA e Culinária devem ser separados semanticamente")

        # Verifica metadados dos chunks
        for c in chunks:
            self.assertEqual(c.metadata["strategy"], "semantic")
            self.assertEqual(c.tenant_id, "tenant-test")
            self.assertIn("similarity_threshold", c.metadata)

    def test_table_preservation(self):
        """Verifica que tabelas são preservadas integralmente como blocos atômicos."""
        doc = Document(id="doc-tbl", title="Tabela Teste", source_uri="docs/tabela.txt")
        tbl_content = (
            "| Modelo | Parâmetros | Latência |\n"
            "|---|---|---|\n"
            "| BGE-M3 | 560M | 15ms |\n"
            "| MiniLM | 22M | 2ms |\n\n"
            "## Sumário\nTabela comparativa de modelos de embedding."
        )
        doc.elements.append(Table(
            id="t1",
            content=tbl_content,
            headers=["Modelo", "Parâmetros", "Latência"],
            rows=[["BGE-M3", "560M", "15ms"], ["MiniLM", "22M", "2ms"]],
        ))

        chunks = chunk_semantic(doc)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].metadata["has_table_summary"])
        self.assertEqual(chunks[0].metadata["table_row_count"], 2)
        self.assertIn("BGE-M3", chunks[0].content)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSentenceSplitter)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCosineSimilarity))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSemanticChunking))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
