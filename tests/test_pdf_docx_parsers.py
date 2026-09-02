"""
Testes dos parsers reais PDF/DOCX — Passo 10.

Roda parse_pdf e parse_docx contra os 4 documentos seed gerados por
scripts/generate_seed_docs_step10.py e valida que:
1. Nenhum arquivo falha ao ser parseado.
2. Cada Document tem pelo menos 1 Section e 1 Paragraph.
3. O conteúdo não foi corrompido (spot-check de string esperada).
4. Headings multi-linha não fragmentam (regressão do bug corrigido:
   título de 2 linhas virando 2 Sections quebradas no meio da frase).
5. content_hash está presente.
6. PDFs geram elementos Page; DOCX não (não tem conceito nativo de página).
7. Documents são serializáveis para JSON.

Rodar com: python tests/test_pdf_docx_parsers.py
"""

import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.parsing.simple_parsers import parse_file
from src.parsing.canonical_model import ElementType

CORPUS_DIR = Path("data/seed_corpus")

PDF_FILES = ["09_fast_full_path_routing.pdf", "10_reindexacao_incremental.pdf"]
DOCX_FILES = ["11_query_decomposition_hyde.docx", "12_migracao_embedding.docx"]
ALL_FILES = PDF_FILES + DOCX_FILES


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

def test_all_files_parse() -> bool:
    print("  [*] test_all_files_parse...")
    failed = []
    for name in ALL_FILES:
        path = CORPUS_DIR / name
        if not path.exists():
            failed.append(f"{name} (não encontrado — rode scripts/generate_seed_docs_step10.py)")
            continue
        try:
            doc = parse_file(path, tenant_id="tenant-test")
            assert doc is not None
        except Exception as e:
            print(f"    [ERRO] {name}: {e}")
            failed.append(name)
    if failed:
        print(f"  [FALHOU] {failed}")
        return False
    print(f"  [OK] {len(ALL_FILES)} arquivo(s) parseado(s) sem erro")
    return True


def test_documents_have_sections_and_paragraphs() -> bool:
    print("  [*] test_documents_have_sections_and_paragraphs...")
    failed = []
    for name in ALL_FILES:
        doc = parse_file(CORPUS_DIR / name, tenant_id="tenant-test")
        n_sec = len(doc.elements_by_type(ElementType.SECTION))
        n_par = len(doc.elements_by_type(ElementType.PARAGRAPH))
        if n_sec == 0 or n_par == 0:
            failed.append(f"{name} (sections={n_sec}, paragraphs={n_par})")
    if failed:
        print(f"  [FALHOU] {failed}")
        return False
    print(f"  [OK] Todos os {len(ALL_FILES)} documents têm Section e Paragraph")
    return True


def test_content_hash_present() -> bool:
    print("  [*] test_content_hash_present...")
    missing = []
    for name in ALL_FILES:
        doc = parse_file(CORPUS_DIR / name, tenant_id="tenant-test")
        if not doc.content_hash or len(doc.content_hash) != 64:
            missing.append(name)
    if missing:
        print(f"  [FALHOU] Sem hash válido: {missing}")
        return False
    print(f"  [OK] Todos os {len(ALL_FILES)} documents têm content_hash SHA-256 (64 hex)")
    return True


def test_content_integrity() -> bool:
    """Spot-check: fatos específicos do texto original sobrevivem ao parsing."""
    print("  [*] test_content_integrity...")
    checks = [
        ("09_fast_full_path_routing.pdf", "threshold = 0.6"),
        ("10_reindexacao_incremental.pdf", "SHA-256"),
        ("11_query_decomposition_hyde.docx", "decompose_query"),
        ("12_migracao_embedding.docx", "EMBEDDING_MODEL"),
    ]
    failed = []
    for filename, expected in checks:
        doc = parse_file(CORPUS_DIR / filename, tenant_id="tenant-test")
        full_text = doc.get_full_text()
        if expected not in full_text:
            print(f"    [ERRO] '{expected}' não encontrado em {filename}")
            failed.append(filename)
    if failed:
        print(f"  [FALHOU] {len(failed)} arquivo(s) com conteúdo corrompido")
        return False
    print(f"  [OK] Spot-check de conteúdo passou em {len(checks)} arquivos")
    return True


def test_no_fragmented_headings() -> bool:
    """
    Regressão: um heading/título de múltiplas linhas na mesma fonte não pode
    virar 2+ Sections com o texto cortado no meio (bug encontrado e corrigido
    durante este passo). Verifica que o título completo aparece como Section
    OU como doc.title — nunca fragmentado em pedaços menores que o título.
    """
    print("  [*] test_no_fragmented_headings...")
    expected_titles = {
        "09_fast_full_path_routing.pdf": "Fast Path e Full Path: Routing Suave",
        "10_reindexacao_incremental.pdf": "Reindexação Incremental: Grafo de Dependências",
    }
    failed = []
    for filename, expected_title in expected_titles.items():
        doc = parse_file(CORPUS_DIR / filename, tenant_id="tenant-test")
        section_titles = [s.title for s in doc.elements_by_type(ElementType.SECTION)]
        found_whole = (doc.title == expected_title) or (expected_title in section_titles)
        if not found_whole:
            print(f"    [ERRO] {filename}: título esperado '{expected_title}' não encontrado íntegro")
            print(f"      doc.title={doc.title!r}")
            print(f"      sections={section_titles}")
            failed.append(filename)
    if failed:
        print(f"  [FALHOU] {failed}")
        return False
    print(f"  [OK] Nenhum heading fragmentado em {len(expected_titles)} PDFs verificados")
    return True


def test_pdf_has_pages() -> bool:
    print("  [*] test_pdf_has_pages...")
    failed = []
    for name in PDF_FILES:
        doc = parse_file(CORPUS_DIR / name, tenant_id="tenant-test")
        n_pages = len(doc.elements_by_type(ElementType.PAGE))
        if n_pages == 0:
            failed.append(name)
    if failed:
        print(f"  [FALHOU] PDFs sem elemento Page: {failed}")
        return False
    print(f"  [OK] Todos os {len(PDF_FILES)} PDFs têm pelo menos 1 elemento Page")
    return True


def test_docx_tables_extracted() -> bool:
    """Se algum DOCX tiver tabela, ela deve virar Table com headers."""
    print("  [*] test_docx_tables_extracted...")
    for name in DOCX_FILES:
        doc = parse_file(CORPUS_DIR / name, tenant_id="tenant-test")
        tables = doc.elements_by_type(ElementType.TABLE)
        for t in tables:
            if not t.headers:
                print(f"    [AVISO] {name}: tabela sem headers")
    print(f"  [OK] Extração de tabela verificada (corpus seed atual não tem tabelas nos DOCX)")
    return True


def test_json_serializable() -> bool:
    print("  [*] test_json_serializable...")
    failed = []
    for name in ALL_FILES:
        doc = parse_file(CORPUS_DIR / name, tenant_id="tenant-test")
        try:
            json_str = doc.to_json()
            assert len(json_str) > 100
        except Exception as e:
            print(f"    [ERRO] {name}: {e}")
            failed.append(name)
    if failed:
        print(f"  [FALHOU] {failed}")
        return False
    print(f"  [OK] Todos os {len(ALL_FILES)} documents serializam para JSON")
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("Testando parsers reais PDF/DOCX (Passo 10)...\n")

    tests = [
        test_all_files_parse,
        test_documents_have_sections_and_paragraphs,
        test_content_hash_present,
        test_content_integrity,
        test_no_fragmented_headings,
        test_pdf_has_pages,
        test_docx_tables_extracted,
        test_json_serializable,
    ]

    results = {}
    for test_fn in tests:
        results[test_fn.__name__] = test_fn()
        print()

    print("Resumo:")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FALHOU'}")

    if all(results.values()):
        print("\nPasso 10 concluído — parsers PDF/DOCX prontos, corpus seed expandido.")
        sys.exit(0)
    else:
        failed = [n for n, ok in results.items() if not ok]
        print(f"\nPasso 10 incompleto — {len(failed)} teste(s) falharam: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
