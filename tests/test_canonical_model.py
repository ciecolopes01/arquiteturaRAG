"""
Testes do Canonical Document Model — Passo 1.

Cria documentos sintéticos manualmente (sem nenhum parser real) e valida:
1. Instanciação de cada tipo de elemento.
2. IDs únicos gerados automaticamente.
3. Serialização para JSON e desserialização de volta (round-trip).
4. Helpers: get_full_text(), elements_by_type().
5. Campos de ACL presentes no schema.

Rodar com: python tests/test_canonical_model.py
"""

import json
import sys

sys.path.insert(0, ".")

from src.parsing.canonical_model import (
    ACLEntry,
    Document,
    ElementType,
    Image,
    Page,
    Paragraph,
    Section,
    Table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_sample_document() -> Document:
    """Monta um documento sintético com vários tipos de elemento."""
    sec = Section(
        title="Introdução",
        level=1,
        content="",
        metadata={"source_heading": "1. Introdução"},
    )

    p1 = Paragraph(
        content="Retrieval-Augmented Generation (RAG) combina busca vetorial com LLMs.",
        metadata={"char_offset": 0},
    )

    p2 = Paragraph(
        content="O objetivo desta plataforma é indexar documentos corporativos e responder perguntas com citação de fonte.",
        metadata={"char_offset": 70},
    )

    tbl = Table(
        content="| Componente | Tecnologia |\n|---|---|\n| Vector DB | Qdrant |\n| Metadata DB | PostgreSQL |",
        headers=["Componente", "Tecnologia"],
        rows=[["Vector DB", "Qdrant"], ["Metadata DB", "PostgreSQL"]],
        caption="Tabela 1 — Stack técnico",
    )

    img = Image(
        content="Diagrama de arquitetura RAG",
        storage_uri="minio://rag-documents/assets/arch-diagram.png",
        width=1200,
        height=800,
    )

    page = Page(
        page_number=1,
        content="Página 1",
        element_ids=[sec.id, p1.id, p2.id, tbl.id, img.id],
    )

    doc = Document(
        source_uri="data/seed_corpus/intro-rag.txt",
        title="Introdução à plataforma RAG",
        elements=[sec, p1, p2, tbl, img, page],
        tenant_id="tenant-acme",
        acl=[
            ACLEntry(principal="grupo:engenharia", permission="read"),
            ACLEntry(principal="user:alice@acme.com", permission="admin"),
        ],
        content_hash="abc123",  # placeholder; SHA-256 real vem do parser
        language="pt",
        metadata={"source": "manual-test", "version": "1.0"},
    )
    return doc


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

def test_instantiation() -> bool:
    """Todos os tipos de elemento instanciam sem erro."""
    print("  [*] test_instantiation...")
    try:
        doc = make_sample_document()
        assert len(doc.elements) == 6, f"Esperado 6 elementos, got {len(doc.elements)}"
        print(f"      Document id={doc.id[:8]}… com {len(doc.elements)} elementos")
        print("  [OK]")
        return True
    except Exception as e:
        print(f"  [FALHOU] {e}")
        return False


def test_unique_ids() -> bool:
    """Cada elemento tem um ID único (não há colisão)."""
    print("  [*] test_unique_ids...")
    try:
        doc = make_sample_document()
        ids = [el.id for el in doc.elements] + [doc.id]
        assert len(ids) == len(set(ids)), "IDs duplicados detectados!"
        print(f"      {len(ids)} IDs todos únicos")
        print("  [OK]")
        return True
    except Exception as e:
        print(f"  [FALHOU] {e}")
        return False


def test_json_roundtrip() -> bool:
    """Serialização → JSON → desserialização produz objeto idêntico."""
    print("  [*] test_json_roundtrip...")
    try:
        doc = make_sample_document()
        json_str = doc.to_json()

        # valida que é JSON válido
        parsed = json.loads(json_str)
        assert parsed["id"] == doc.id
        assert parsed["tenant_id"] == "tenant-acme"

        # desserializa de volta
        doc2 = Document.from_json(json_str)
        assert doc2.id == doc.id
        assert doc2.title == doc.title
        assert len(doc2.elements) == len(doc.elements)
        assert doc2.tenant_id == doc.tenant_id
        assert len(doc2.acl) == 2

        print(f"      JSON round-trip OK — {len(json_str)} bytes")
        print("  [OK]")
        return True
    except Exception as e:
        print(f"  [FALHOU] {e}")
        return False


def test_get_full_text() -> bool:
    """get_full_text() concatena os conteúdos sem levantar exceção."""
    print("  [*] test_get_full_text...")
    try:
        doc = make_sample_document()
        text = doc.get_full_text()
        assert "Introdução" in text
        assert "RAG" in text
        assert len(text) > 50
        print(f"      Texto completo: {len(text)} chars")
        print("  [OK]")
        return True
    except Exception as e:
        print(f"  [FALHOU] {e}")
        return False


def test_elements_by_type() -> bool:
    """elements_by_type() filtra corretamente por tipo."""
    print("  [*] test_elements_by_type...")
    try:
        doc = make_sample_document()
        paragraphs = doc.elements_by_type(ElementType.PARAGRAPH)
        tables = doc.elements_by_type(ElementType.TABLE)
        images = doc.elements_by_type(ElementType.IMAGE)
        sections = doc.elements_by_type(ElementType.SECTION)
        pages = doc.elements_by_type(ElementType.PAGE)

        assert len(paragraphs) == 2, f"Esperado 2 parágrafos, got {len(paragraphs)}"
        assert len(tables) == 1, f"Esperado 1 tabela, got {len(tables)}"
        assert len(images) == 1, f"Esperado 1 imagem, got {len(images)}"
        assert len(sections) == 1, f"Esperado 1 seção, got {len(sections)}"
        assert len(pages) == 1, f"Esperado 1 página, got {len(pages)}"

        print(f"      {len(paragraphs)} parágrafos, {len(sections)} seções, {len(tables)} tabelas, {len(images)} imagens, {len(pages)} páginas")
        print("  [OK]")
        return True
    except Exception as e:
        print(f"  [FALHOU] {e}")
        return False


def test_acl_fields() -> bool:
    """ACL e tenant_id estão presentes e tipados corretamente."""
    print("  [*] test_acl_fields...")
    try:
        doc = make_sample_document()
        assert doc.tenant_id == "tenant-acme"
        assert len(doc.acl) == 2
        assert doc.acl[0].principal == "grupo:engenharia"
        assert doc.acl[0].permission == "read"
        assert doc.acl[1].principal == "user:alice@acme.com"
        assert doc.acl[1].permission == "admin"

        # Document sem ACL deve funcionar — campos são opcionais nesta fase
        doc_no_acl = Document(source_uri="test://empty")
        assert doc_no_acl.tenant_id is None
        assert doc_no_acl.acl == []

        print("      tenant_id e acl validados; Document sem ACL também funciona")
        print("  [OK]")
        return True
    except Exception as e:
        print(f"  [FALHOU] {e}")
        return False


def test_element_type_enum() -> bool:
    """ElementType enum cobre todos os tipos esperados."""
    print("  [*] test_element_type_enum...")
    try:
        expected = {"paragraph", "section", "table", "image", "page"}
        actual = {e.value for e in ElementType}
        assert actual == expected, f"Tipos inesperados: {actual ^ expected}"
        print(f"      Tipos: {sorted(actual)}")
        print("  [OK]")
        return True
    except Exception as e:
        print(f"  [FALHOU] {e}")
        return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("Testando Canonical Document Model (Passo 1)...\n")

    tests = [
        test_instantiation,
        test_unique_ids,
        test_json_roundtrip,
        test_get_full_text,
        test_elements_by_type,
        test_acl_fields,
        test_element_type_enum,
    ]

    results = {}
    for test_fn in tests:
        name = test_fn.__name__
        results[name] = test_fn()
        print()

    print("Resumo:")
    for name, ok in results.items():
        status = "OK" if ok else "FALHOU"
        print(f"  {name}: {status}")

    if all(results.values()):
        print("\nPasso 1 concluído — Canonical Document Model pronto.")
        sys.exit(0)
    else:
        failed = [n for n, ok in results.items() if not ok]
        print(f"\nPasso 1 incompleto — {len(failed)} teste(s) falharam: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
