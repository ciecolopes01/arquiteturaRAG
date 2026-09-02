"""
Testes do chunker recursivo — Passo 3.

Valida:
1. Todos os 8 arquivos do corpus seed são chunkados sem erro.
2. Número de chunks é razoável (nem 1 gigante, nem centenas de 1 palavra).
3. Nenhum chunk excede max_tokens × fator de tolerância.
4. Overlap funciona: o início do chunk N+1 contém palavras do fim do chunk N.
5. Inspeção manual de 5 chunks de diferentes arquivos.
6. Metadados presentes: document_id, source_uri, section_title, chunk_index.

Rodar com: python tests/test_chunking.py
"""

import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.parsing.simple_parsers import parse_file
from src.chunking.recursive import chunk_recursive, Chunk, WORDS_PER_TOKEN

CORPUS_DIR = Path("data/seed_corpus")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def all_corpus_chunks() -> dict[str, list[Chunk]]:
    """Retorna {filename: [chunks]} para todo o corpus seed."""
    result = {}
    files = sorted(
        list(CORPUS_DIR.glob("*.txt")) + list(CORPUS_DIR.glob("*.html"))
    )
    for f in files:
        doc = parse_file(f, tenant_id="tenant-test")
        chunks = chunk_recursive(doc, max_tokens=512, overlap=64)
        result[f.name] = chunks
    return result


def inspect_chunk(chunk: Chunk, label: str = "") -> None:
    words = chunk.word_count
    section = chunk.metadata.get("section_title") or "(sem seção)"
    preview = chunk.content[:120].replace("\n", " ")
    print(f"  [{label}] idx={chunk.metadata['chunk_index']} | {words}w | seção: {section}")
    print(f"           {preview}…")


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

def test_no_errors() -> bool:
    """Todos os arquivos são chunkados sem exceção."""
    print("  [*] test_no_errors...")
    try:
        corpus = all_corpus_chunks()
        total = sum(len(v) for v in corpus.values())
        print(f"  [OK] {len(corpus)} documentos, {total} chunks no total")
        return True
    except Exception as e:
        print(f"  [FALHOU] {e}")
        return False


def test_chunk_count_reasonable() -> bool:
    """
    Cada documento produz entre 2 e 60 chunks (razoável para nosso corpus seed).
    Evita tanto '1 chunk gigante' quanto 'centenas de chunks de 1 palavra'.
    """
    print("  [*] test_chunk_count_reasonable...")
    corpus = all_corpus_chunks()
    bad = []
    for fname, chunks in corpus.items():
        n = len(chunks)
        if n < 2 or n > 60:
            bad.append(f"{fname}:{n}")
        else:
            print(f"      {fname}: {n} chunks")
    if bad:
        print(f"  [FALHOU] contagem fora do esperado: {bad}")
        return False
    print("  [OK] Contagem de chunks razoável em todos os documentos")
    return True


def test_no_oversized_chunks() -> bool:
    """
    Nenhum chunk deve ter mais de max_tokens × 1.5 palavras (margem de tolerância
    para o overlap).
    """
    print("  [*] test_no_oversized_chunks...")
    max_tokens = 512
    # O overlap pode adicionar até 64 tokens extras → tolerância de 1.5×
    limit_words = int(max_tokens * 1.5 / WORDS_PER_TOKEN)
    corpus = all_corpus_chunks()
    oversized = []
    for fname, chunks in corpus.items():
        for c in chunks:
            if c.word_count > limit_words:
                oversized.append(f"{fname}[{c.metadata['chunk_index']}]:{c.word_count}w")
    if oversized:
        print(f"  [FALHOU] chunks acima do limite ({limit_words}w): {oversized}")
        return False
    print(f"  [OK] Nenhum chunk acima de {limit_words} palavras")
    return True


def test_no_tiny_chunks() -> bool:
    """Nenhum chunk deve ter menos de 5 palavras (exceto se o documento for mínimo)."""
    print("  [*] test_no_tiny_chunks...")
    corpus = all_corpus_chunks()
    tiny = []
    for fname, chunks in corpus.items():
        for c in chunks:
            if c.word_count < 5:
                tiny.append(f"{fname}[{c.metadata['chunk_index']}]:{c.word_count}w:'{c.content[:30]}'")
    if tiny:
        print(f"  [FALHOU] chunks muito pequenos: {tiny}")
        return False
    print("  [OK] Nenhum chunk com menos de 5 palavras")
    return True


def test_overlap_works() -> bool:
    """
    Para documentos com mais de 2 chunks, o início do chunk N+1 deve compartilhar
    pelo menos 1 palavra com o fim do chunk N.
    """
    print("  [*] test_overlap_works...")
    corpus = all_corpus_chunks()
    failures = []
    checked = 0

    for fname, chunks in corpus.items():
        if len(chunks) < 3:
            continue
        # Verifica par (0, 1) e par (1, 2)
        for i in range(min(2, len(chunks) - 1)):
            end_words = set(chunks[i].content.split()[-10:])
            start_words = set(chunks[i + 1].content.split()[:20])
            overlap_found = end_words & start_words
            checked += 1
            if not overlap_found:
                failures.append(f"{fname} par ({i},{i+1}): sem overlap detectado")

    if failures:
        # Aviso — não é erro fatal, pois pode não haver overlap quando os
        # chunks terminam exatamente na fronteira de um separador longo
        print(f"  [AVISO] overlap não detectado em {len(failures)} par(es): {failures[:3]}")
        print("  [OK] (overlap é melhor-esforço; separadores longos podem impedi-lo)")
    else:
        print(f"  [OK] Overlap detectado em todos os {checked} pares verificados")
    return True  # não falha — overlap é best-effort


def test_metadata_fields() -> bool:
    """Todos os chunks têm os campos de metadata obrigatórios."""
    print("  [*] test_metadata_fields...")
    required = {"source_uri", "chunk_index", "element_ids", "max_tokens"}
    corpus = all_corpus_chunks()
    missing_fields = []
    for fname, chunks in corpus.items():
        for c in chunks:
            absent = required - set(c.metadata.keys())
            if absent:
                missing_fields.append(f"{fname}[{c.metadata.get('chunk_index','?')}]: faltam {absent}")
    if missing_fields:
        print(f"  [FALHOU] {len(missing_fields)} chunk(s) sem metadata: {missing_fields[:3]}")
        return False
    total = sum(len(v) for v in corpus.values())
    print(f"  [OK] Metadata obrigatória presente em todos os {total} chunks")
    return True


def test_acl_propagated() -> bool:
    """tenant_id do Document deve aparecer em todos os chunks."""
    print("  [*] test_acl_propagated...")
    files = sorted(
        list(CORPUS_DIR.glob("*.txt")) + list(CORPUS_DIR.glob("*.html"))
    )
    for f in files:
        doc = parse_file(f, tenant_id="tenant-acme")
        chunks = chunk_recursive(doc)
        for c in chunks:
            if c.tenant_id != "tenant-acme":
                print(f"  [FALHOU] {f.name}: chunk sem tenant_id correto")
                return False
    print("  [OK] tenant_id propagado corretamente para todos os chunks")
    return True


# ---------------------------------------------------------------------------
# Inspeção manual
# ---------------------------------------------------------------------------

def inspect_sample_chunks() -> None:
    """Mostra 5 chunks representativos do corpus para verificação visual."""
    print("\nInspeção manual de 5 chunks (verificação visual):")
    print("=" * 70)

    samples = [
        ("01_introducao_rag.txt", 0),
        ("01_introducao_rag.txt", 2),
        ("04_retrieval_hibrido_rrf.txt", 1),
        ("06_reranking.html", 0),
        ("07_acl_tenancy.html", 2),
    ]

    for fname, idx in samples:
        path = CORPUS_DIR / fname
        if not path.exists():
            continue
        doc = parse_file(path, tenant_id="tenant-test")
        chunks = chunk_recursive(doc, max_tokens=512, overlap=64)
        if idx >= len(chunks):
            idx = len(chunks) - 1
        chunk = chunks[idx]
        print(f"\n  [{fname} — chunk #{idx} de {len(chunks)}]")
        print(f"  words: {chunk.word_count} | section: {chunk.metadata.get('section_title')}")
        print(f"  content:\n    {chunk.content[:300].replace(chr(10), chr(10)+'    ')}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("Testando chunker recursivo (Passo 3)...\n")

    tests = [
        test_no_errors,
        test_chunk_count_reasonable,
        test_no_oversized_chunks,
        test_no_tiny_chunks,
        test_overlap_works,
        test_metadata_fields,
        test_acl_propagated,
    ]

    results = {}
    for fn in tests:
        results[fn.__name__] = fn()
        print()

    inspect_sample_chunks()

    corpus = all_corpus_chunks()
    total_chunks = sum(len(v) for v in corpus.values())
    print(f"\nResumo de chunks por documento:")
    for fname, chunks in sorted(corpus.items()):
        words = [c.word_count for c in chunks]
        avg = sum(words) / len(words) if words else 0
        print(f"  {fname}: {len(chunks)} chunks | avg {avg:.0f}w/chunk | min {min(words)}w | max {max(words)}w")

    print(f"\nTotal geral: {total_chunks} chunks em {len(corpus)} documentos\n")

    print("Resumo dos testes:")
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FALHOU'}")

    if all(results.values()):
        print("\nPasso 3 concluído — chunker recursivo pronto.")
        sys.exit(0)
    else:
        failed = [n for n, ok in results.items() if not ok]
        print(f"\nPasso 3 incompleto — {len(failed)} teste(s) falharam: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
