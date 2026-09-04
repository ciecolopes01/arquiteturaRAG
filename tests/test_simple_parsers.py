"""
Testes dos parsers simples — Passo 2.

Roda parse_txt e parse_html contra todos os arquivos do corpus seed e
valida que:
1. Nenhum arquivo falha ao ser parseado.
2. Cada Document tem pelo menos 1 elemento.
3. O conteúdo não foi corrompido (spot-check de string esperada).
4. content_hash está presente.
5. Inspeciona manualmente 2-3 Documents no terminal.

Rodar com: python tests/test_simple_parsers.py
"""

import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.parsing.canonical_model import ElementType
from src.parsing.simple_parsers import parse_file

CORPUS_DIR = Path("data/seed_corpus")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def inspect_document(doc, max_elements: int = 4) -> None:
    """Imprime um resumo legível do Document."""
    print(f"    title    : {doc.title}")
    print(f"    source   : {doc.source_uri}")
    print(f"    elements : {len(doc.elements)}")
    print(f"    hash     : {doc.content_hash[:12]}…")
    counts = {}
    for el in doc.elements:
        counts[el.element_type.value] = counts.get(el.element_type.value, 0) + 1
    print(f"    tipos    : {dict(sorted(counts.items()))}")
    print(f"    primeiros {min(max_elements, len(doc.elements))} elementos:")
    for el in doc.elements[:max_elements]:
        snippet = getattr(el, "content", "") or getattr(el, "title", "")
        snippet = snippet[:80].replace("\n", " ")
        print(f"      [{el.element_type.value}] {snippet}")


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

def test_all_files_parse() -> bool:
    """Todos os arquivos do corpus seed parseam sem exceção."""
    print("  [*] test_all_files_parse...")
    files = list(CORPUS_DIR.glob("*.txt")) + list(CORPUS_DIR.glob("*.html"))

    if not files:
        print("  [AVISO] Nenhum arquivo encontrado em", CORPUS_DIR)
        return False

    failed = []
    for f in sorted(files):
        try:
            doc = parse_file(f, tenant_id="tenant-test")
            assert doc is not None
        except Exception as e:
            print(f"    [ERRO] {f.name}: {e}")
            failed.append(f.name)

    if failed:
        print(f"  [FALHOU] {len(failed)} arquivo(s) com erro: {failed}")
        return False

    print(f"  [OK] {len(files)} arquivo(s) parseado(s) sem erro")
    return True


def test_documents_have_elements() -> bool:
    """Cada Document tem pelo menos 1 elemento."""
    print("  [*] test_documents_have_elements...")
    files = list(CORPUS_DIR.glob("*.txt")) + list(CORPUS_DIR.glob("*.html"))
    empties = []
    for f in sorted(files):
        doc = parse_file(f)
        if len(doc.elements) == 0:
            empties.append(f.name)

    if empties:
        print(f"  [FALHOU] Documents sem elementos: {empties}")
        return False

    print(f"  [OK] Todos os {len(files)} documents têm pelo menos 1 elemento")
    return True


def test_content_hash_present() -> bool:
    """Todos os Documents têm content_hash SHA-256."""
    print("  [*] test_content_hash_present...")
    files = list(CORPUS_DIR.glob("*.txt")) + list(CORPUS_DIR.glob("*.html"))
    missing = []
    for f in sorted(files):
        doc = parse_file(f)
        if not doc.content_hash or len(doc.content_hash) != 64:
            missing.append(f.name)

    if missing:
        print(f"  [FALHOU] Sem hash válido: {missing}")
        return False

    print(f"  [OK] Todos os {len(files)} documents têm content_hash SHA-256 (64 hex)")
    return True


def test_txt_content_integrity() -> bool:
    """Spot-check: strings esperadas estão presentes nos Documents TXT."""
    print("  [*] test_txt_content_integrity...")
    checks = [
        ("01_introducao_rag.txt", "Recall@5"),
        ("02_embeddings_busca_vetorial.txt", "Qdrant"),
        ("03_chunking.txt", "overlap"),
        ("04_retrieval_hibrido_rrf.txt", "k = 60"),
        ("05_avaliacao_offline.txt", "golden dataset"),
        ("08_observabilidade.txt", "pipeline rag"),
    ]
    # "latência" pode ter encoding diferente — busca case-insensitive
    failed = []
    for filename, expected in checks:
        path = CORPUS_DIR / filename
        if not path.exists():
            print(f"    [SKIP] {filename} não encontrado")
            continue
        doc = parse_file(path)
        full_text = doc.get_full_text().lower()
        # normaliza acentos simples para busca
        expected_lower = expected.lower().replace("â", "a").replace("ê", "e").replace("ê", "e")
        if expected_lower not in full_text and expected.lower() not in full_text:
            print(f"    [ERRO] '{expected}' não encontrado em {filename}")
            failed.append(filename)

    if failed:
        print(f"  [FALHOU] {len(failed)} arquivo(s) com conteúdo corrompido")
        return False

    print(f"  [OK] Spot-check de conteúdo passou em {len(checks)} arquivos")
    return True


def test_html_has_sections() -> bool:
    """Documentos HTML têm pelo menos uma Section (heading extraído)."""
    print("  [*] test_html_has_sections...")
    html_files = list(CORPUS_DIR.glob("*.html"))
    if not html_files:
        print("  [SKIP] Nenhum HTML no corpus")
        return True
    failed = []
    for f in html_files:
        doc = parse_file(f)
        sections = doc.elements_by_type(ElementType.SECTION)
        if not sections:
            print(f"    [ERRO] {f.name} não tem nenhuma Section")
            failed.append(f.name)
    if failed:
        print(f"  [FALHOU] {len(failed)} HTML(s) sem Section")
        return False
    print(f"  [OK] Todos os {len(html_files)} HTMLs têm pelo menos 1 Section")
    return True


def test_json_serializable() -> bool:
    """Todos os Documents são serializáveis para JSON (sem tipos não-serializáveis)."""
    print("  [*] test_json_serializable...")
    files = list(CORPUS_DIR.glob("*.txt")) + list(CORPUS_DIR.glob("*.html"))
    failed = []
    for f in sorted(files):
        doc = parse_file(f)
        try:
            json_str = doc.to_json()
            assert len(json_str) > 100
        except Exception as e:
            print(f"    [ERRO] {f.name}: {e}")
            failed.append(f.name)
    if failed:
        print(f"  [FALHOU] {len(failed)} arquivo(s) não serializáveis")
        return False
    print(f"  [OK] Todos os {len(files)} documents serializam para JSON")
    return True


# ---------------------------------------------------------------------------
# Inspeção manual
# ---------------------------------------------------------------------------

def inspect_sample_documents() -> None:
    """Inspeciona 3 Documents para verificação visual."""
    samples = [
        CORPUS_DIR / "01_introducao_rag.txt",
        CORPUS_DIR / "04_retrieval_hibrido_rrf.txt",
        CORPUS_DIR / "06_reranking.html",
    ]
    print("\nInspeção manual de 3 Documents:")
    print("=" * 60)
    for path in samples:
        if not path.exists():
            print(f"  {path.name}: arquivo não encontrado, pulando")
            continue
        doc = parse_file(path, tenant_id="tenant-test")
        print(f"\n  [{path.name}]")
        inspect_document(doc)
    print("=" * 60)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("Testando parsers simples TXT/HTML (Passo 2)...\n")

    tests = [
        test_all_files_parse,
        test_documents_have_elements,
        test_content_hash_present,
        test_txt_content_integrity,
        test_html_has_sections,
        test_json_serializable,
    ]

    results = {}
    for test_fn in tests:
        name = test_fn.__name__
        results[name] = test_fn()
        print()

    inspect_sample_documents()

    print("\nResumo:")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FALHOU'}")

    total_files = len(list(CORPUS_DIR.glob("*.txt"))) + len(list(CORPUS_DIR.glob("*.html")))
    print(f"\n  Total de arquivos no corpus seed: {total_files}")

    if all(results.values()):
        print("\nPasso 2 concluído — parsers TXT/HTML prontos, corpus seed processado.")
        sys.exit(0)
    else:
        failed = [n for n, ok in results.items() if not ok]
        print(f"\nPasso 2 incompleto — {len(failed)} teste(s) falharam: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
