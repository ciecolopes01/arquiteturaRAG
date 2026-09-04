"""
Geração de resposta com LLM (Claude) sobre contexto recuperado — Passo 8.

Fecha o ciclo RAG completo:
    query → hybrid_search → montagem de contexto → Claude → resposta + citação

Função principal:
    answer(query, top_k, tenant_id, stream) -> AnswerResult

AnswerResult contém:
    - answer      : str — resposta gerada pelo LLM
    - chunks_used : list[HybridResult] — chunks que foram passados como contexto
    - model       : str — modelo Claude usado
    - input_tokens / output_tokens : int
    - latency_ms  : float
    - source_uris : list[str] — URIs das fontes citadas

Prompt design (deliberado):
  1. Instrução fixa no system (candidata a cache_control em V1.1).
  2. Contexto = chunks numerados com seção e URI, no human turn.
  3. LLM instrucido a citar pelo número do chunk: [Fonte 1], [Fonte 2], etc.
  4. Se nenhum chunk contém a resposta, LLM deve dizer "não encontrei nos
     documentos fornecidos" — sem alucinar.

Modo degradado (sem ANTHROPIC_API_KEY):
    Se a chave não estiver configurada, answer() retorna uma resposta simulada
    mostrando os chunks recuperados — útil para testar o pipeline sem API.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger("rag.answer")

# Observabilidade (Passo 14)
try:
    from src.observability.instrumentation import (
        pipeline_trace,
        get_current_trace,
        record_generation_cost,
    )
    _OBS_AVAILABLE = True
except ImportError:
    _OBS_AVAILABLE = False
    def record_generation_cost(model, input_tokens, output_tokens, latency_ms):
        pass
    def get_current_trace():
        return None
    # fallback: pipeline_trace nao-op
    import contextlib
    @contextlib.contextmanager
    def pipeline_trace(query, log=True):
        yield None

from src.config import settings
from src.retrieval.hybrid import HybridResult, hybrid_search


def _serialize_chunk(c: HybridResult) -> Dict[str, Any]:
    return {
        "chunk_id": c.chunk_id,
        "content": c.content,
        "rrf_score": c.rrf_score,
        "source_uri": c.source_uri,
        "document_title": c.document_title,
        "section_title": c.section_title,
        "dense_rank": c.dense_rank,
        "bm25_rank": c.bm25_rank,
        "graph_rank": c.graph_rank,
        "dense_score": c.dense_score,
        "bm25_score": c.bm25_score,
        "graph_score": c.graph_score,
        "parent_chunk_id": c.parent_chunk_id,
        "parent_content": c.parent_content,
    }


def _deserialize_chunk(d: Dict[str, Any]) -> HybridResult:
    return HybridResult(
        chunk_id=d.get("chunk_id", ""),
        content=d.get("content", ""),
        rrf_score=float(d.get("rrf_score", 0.0)),
        source_uri=d.get("source_uri"),
        document_title=d.get("document_title"),
        section_title=d.get("section_title"),
        dense_rank=d.get("dense_rank"),
        bm25_rank=d.get("bm25_rank"),
        graph_rank=d.get("graph_rank"),
        dense_score=d.get("dense_score"),
        bm25_score=d.get("bm25_score"),
        graph_score=d.get("graph_score"),
        parent_chunk_id=d.get("parent_chunk_id"),
        parent_content=d.get("parent_content"),
    )


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULT_MODEL  = "claude-3-5-haiku-20241022"   # rápido e barato para V1
FALLBACK_MODEL = "claude-3-5-sonnet-20241022"  # mais capaz se necessário
MAX_CONTEXT_CHUNKS = 5   # top-K passados ao LLM após retrieval
MAX_OUTPUT_TOKENS  = 400 # teto reduzido para economia de tokens de saída e menor latência

SYSTEM_PROMPT = """Você é um assistente especialista em recuperação de informação e sistemas RAG.
Responda de forma estritamente concisa, direta e em português (pt-BR), baseando-se APENAS no contexto fornecido.

Regras:
1. Seja objetivo: responda em no máximo 2 a 4 frases assertivas. Evite preâmbulos ("Com base no contexto..."), saudações ou conclusões prolixas.
2. Cite a fonte usando [Fonte N] imediatamente após cada informação retirada dela.
3. Se a resposta não estiver no contexto, diga explicitamente: "Não encontrei essa informação nos documentos fornecidos."
4. Não invente informações além do que está no contexto.
5. Se houver informações contraditórias entre fontes, mencione a contradição de forma direta."""


# ---------------------------------------------------------------------------
# Schema de resultado
# ---------------------------------------------------------------------------

@dataclass
class AnswerResult:
    """Resultado completo de uma chamada answer()."""
    answer: str
    chunks_used: List[HybridResult]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    source_uris: List[str] = field(default_factory=list)
    simulated: bool = False   # True quando rodou sem API key (modo degradado)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def pretty_print(self) -> None:
        """Imprime o resultado formatado no terminal."""
        print("\n" + "=" * 70)
        print(f"RESPOSTA ({self.model}{'  [simulada]' if self.simulated else ''}):")
        print("=" * 70)
        print(self.answer)
        print()
        print(f"Chunks utilizados: {len(self.chunks_used)}")
        for i, c in enumerate(self.chunks_used, 1):
            score_info = f"rrf={c.rrf_score:.4f}"
            print(f"  [Fonte {i}] {c.source_uri or '(sem URI)'} | {score_info}")
            print(f"            '{c.content[:80].replace(chr(10), ' ')}'")
        if not self.simulated:
            print(f"\nTokens: {self.input_tokens} entrada + "
                  f"{self.output_tokens} saída = {self.total_tokens} total")
        print(f"Latência: {self.latency_ms:.0f}ms")
        print("=" * 70)


# ---------------------------------------------------------------------------
# Montagem e expansão do contexto (Context Engine — Passos 8 e 17)
# ---------------------------------------------------------------------------

def expand_parent_context(chunks: List[HybridResult]) -> List[HybridResult]:
    """
    Expande chunks filhos para seus respectivos chunks pais com deduplicação — Passo 17.

    Lógica:
    - Se múltiplos chunks filhos no top-K pertencem ao mesmo chunk pai (mesmo `parent_chunk_id`),
      eles são condensados em uma única entrada do pai, evitando duplicações no contexto do LLM.
    - O texto enviado ao LLM passa a ser `parent_content` (contexto rico e amplo de ~1200 tokens).
    - Se o chunk não possui `parent_chunk_id` ou `parent_content` (ex: chunks do baseline recursive
      ou tabelas sem hierarquia), mantém o chunk original.
    """
    expanded: List[HybridResult] = []
    seen_parents: set = set()

    for chunk in chunks:
        p_id = chunk.parent_chunk_id
        p_content = chunk.parent_content

        if p_id and p_content:
            if p_id in seen_parents:
                continue
            seen_parents.add(p_id)
            expanded.append(
                HybridResult(
                    chunk_id=p_id,
                    content=p_content,
                    rrf_score=chunk.rrf_score,
                    source_uri=chunk.source_uri,
                    document_title=chunk.document_title,
                    section_title=chunk.section_title,
                    dense_rank=chunk.dense_rank,
                    bm25_rank=chunk.bm25_rank,
                    dense_score=chunk.dense_score,
                    bm25_score=chunk.bm25_score,
                    parent_chunk_id=p_id,
                    parent_content=p_content,
                )
            )
        else:
            expanded.append(chunk)

    return expanded


def _build_context_block(chunks: List[HybridResult]) -> str:
    """
    Monta o bloco de contexto com os chunks numerados.

    Formato:
        [Fonte 1 (Contexto Expandido)] Seção: <título> | Documento: <uri>
        <content>

        [Fonte 2] ...
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        is_parent = bool(chunk.parent_chunk_id and chunk.parent_content)
        type_tag = " (Contexto Expandido)" if is_parent else ""
        header_parts = [f"[Fonte {i}{type_tag}]"]
        if chunk.section_title:
            header_parts.append(f"Seção: {chunk.section_title}")
        if chunk.document_title:
            header_parts.append(f"Doc: {chunk.document_title}")
        elif chunk.source_uri:
            header_parts.append(f"URI: {chunk.source_uri}")
        header = " | ".join(header_parts)
        parts.append(f"{header}\n{chunk.content.strip()}")

    return "\n\n".join(parts)


def _build_human_prompt(query: str, context: str) -> str:
    return f"""Contexto recuperado:

{context}

---
Pergunta: {query}"""


# ---------------------------------------------------------------------------
# Geração simulada (sem API key)
# ---------------------------------------------------------------------------

def _simulated_answer(query: str, chunks: List[HybridResult]) -> AnswerResult:
    """
    Resposta simulada para ambiente sem ANTHROPIC_API_KEY.
    Exibe os chunks que seriam passados ao LLM.
    """
    context = _build_context_block(chunks)
    simulated = (
        f"[MODO SIMULADO — sem ANTHROPIC_API_KEY]\n\n"
        f"Query: {query}\n\n"
        f"Chunks que seriam passados ao LLM ({len(chunks)}):\n\n"
        f"{context}\n\n"
        f"Configure ANTHROPIC_API_KEY no .env para obter resposta real."
    )
    return AnswerResult(
        answer=simulated,
        chunks_used=chunks,
        model="simulado",
        simulated=True,
        source_uris=[c.source_uri for c in chunks if c.source_uri],
    )


# ---------------------------------------------------------------------------
# Geração real (Claude API)
# ---------------------------------------------------------------------------

def _call_claude(
    query: str,
    chunks: List[HybridResult],
    model: str,
    stream: bool,
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> AnswerResult:
    """Chama a API da Anthropic e retorna o AnswerResult."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "SDK Anthropic não instalado. Execute:\n"
            "  pip install anthropic"
        )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    context = _build_context_block(chunks)
    human_prompt = _build_human_prompt(query, context)

    system_blocks = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    t0 = time.time()

    if stream:
        # Streaming: acumula o texto e retorna no final
        full_text = ""
        usage = {"input_tokens": 0, "output_tokens": 0}
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system_blocks,
            messages=[{"role": "user", "content": human_prompt}],
        ) as stream_ctx:
            for text in stream_ctx.text_stream:
                full_text += text
                print(text, end="", flush=True)
            final = stream_ctx.get_final_message()
            usage["input_tokens"]  = final.usage.input_tokens
            usage["output_tokens"] = final.usage.output_tokens
        print()  # nova linha após streaming
        answer_text = full_text
    else:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_blocks,
            messages=[{"role": "user", "content": human_prompt}],
        )
        answer_text = response.content[0].text
        usage = {
            "input_tokens":  response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    latency_ms = (time.time() - t0) * 1000

    # Registra tokens e custo no trace ativo (Passo 14)
    record_generation_cost(
        model=model,
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        latency_ms=latency_ms,
    )

    return AnswerResult(
        answer=answer_text,
        chunks_used=chunks,
        model=model,
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        latency_ms=latency_ms,
        source_uris=list({c.source_uri for c in chunks if c.source_uri}),
    )


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def _stream_claude(
    query: str,
    chunks: List[HybridResult],
    model: str,
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> Iterator[str]:
    """Transmite tokens de resposta diretamente da API Anthropic."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "SDK Anthropic não instalado. Execute:\n"
            "  pip install anthropic"
        )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    context = _build_context_block(chunks)
    human_prompt = _build_human_prompt(query, context)

    system_blocks = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system_blocks,
        messages=[{"role": "user", "content": human_prompt}],
    ) as stream_ctx:
        for text in stream_ctx.text_stream:
            yield text


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def answer(
    query: str,
    top_k: int = MAX_CONTEXT_CHUNKS,
    tenant_id: Optional[str] = None,
    principals: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL,
    stream: bool = False,
    dense_candidates: int = 20,
    bm25_candidates: int = 20,
    expand_parents: bool = True,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    prefetched_chunks: Optional[List[HybridResult]] = None,
) -> AnswerResult:
    """
    Responde uma query fazendo retrieval híbrido e geração com Claude.

    Se não houver um `pipeline_trace` ativo, cria um automaticamente
    para que cada chamada a answer() gere entrada no log de observabilidade
    (logs/pipeline.jsonl) com latência por estágio e custo estimado.

    Parâmetros:
    - query             : pergunta do usuário.
    - top_k             : número de chunks passados ao LLM (contexto).
    - tenant_id         : filtro de tenant — repassado ao retrieval.
    - principals        : filtro de ACL — repassado ao retrieval e chave de cache.
    - model             : modelo Claude a usar.
    - stream            : True habilita streaming (imprime tokens conforme chegam).
    - dense_candidates  : candidatos do canal dense antes do RRF.
    - bm25_candidates   : candidatos do canal BM25 antes do RRF.
    - expand_parents    : True expande chunks filhos para o chunk pai correspondente (Passo 17).
    - max_output_tokens : teto máximo de tokens de saída para o LLM.
    - prefetched_chunks : se fornecido, usa esses chunks diretamente e ignora hybrid_search.

    Retorna AnswerResult com resposta, chunks, tokens e latência.
    """
    # Se já há trace ativo (ex: chamado de dentro de pipeline_trace),
    # não abre um novo. Se não há, cria um automático para garantir
    # que cada query seja logada (critério do Passo 14).
    _active_trace = get_current_trace()
    _auto_trace_ctx = None

    if _active_trace is None and _OBS_AVAILABLE:
        _auto_trace_ctx = pipeline_trace(query, log=True)
        _active_trace = _auto_trace_ctx.__enter__()

    try:
        result = _answer_impl(
            query=query,
            top_k=top_k,
            tenant_id=tenant_id,
            principals=principals,
            model=model,
            stream=stream,
            dense_candidates=dense_candidates,
            bm25_candidates=bm25_candidates,
            expand_parents=expand_parents,
            max_output_tokens=max_output_tokens,
            prefetched_chunks=prefetched_chunks,
        )
    finally:
        if _auto_trace_ctx is not None:
            _auto_trace_ctx.__exit__(None, None, None)

    return result


def _answer_impl(
    query: str,
    top_k: int = MAX_CONTEXT_CHUNKS,
    tenant_id: Optional[str] = None,
    principals: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL,
    stream: bool = False,
    dense_candidates: int = 20,
    bm25_candidates: int = 20,
    expand_parents: bool = True,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    prefetched_chunks: Optional[List[HybridResult]] = None,
) -> AnswerResult:
    # --- 0. Cache de Resposta (Passo 32, isolado por tenant e ACL) ---
    try:
        from src.optimization.caching import cache
        cached = cache.get_response(query=query, tenant_id=tenant_id or "", principals=principals)
        if cached is not None:
            print(f"[answer] Cache HIT para query: '{query[:40]}...'")
            chunks_raw = cached.get("chunks_used", [])
            restored_chunks = [_deserialize_chunk(c) for c in chunks_raw] if chunks_raw else []
            return AnswerResult(
                answer=cached.get("answer", ""),
                chunks_used=restored_chunks,
                model=cached.get("model", model) + " (cached)",
                simulated=cached.get("simulated", False),
                latency_ms=1.0,
                source_uris=cached.get("source_uris", [c.source_uri for c in restored_chunks if c.source_uri]),
            )
    except Exception as e:
        logger.debug("Falha na consulta ao cache: %s", e)

    # --- 1. Retrieval híbrido ou chunks pré-computados (Self-RAG) ---
    t_retrieval = time.time()
    if prefetched_chunks is not None:
        chunks = prefetched_chunks
        retrieval_ms = 0.0
        print(f"[answer] Usando {len(chunks)} chunks pré-computados (Self-RAG)")
    else:
        chunks = hybrid_search(
            query,
            top_k=top_k,
            tenant_id=tenant_id,
            principals=principals,
            dense_candidates=dense_candidates,
            bm25_candidates=bm25_candidates,
        )
        retrieval_ms = (time.time() - t_retrieval) * 1000
        print(f"[answer] Retrieval: {len(chunks)} chunks em {retrieval_ms:.0f}ms")

    if not chunks:
        return AnswerResult(
            answer="Não encontrei documentos relevantes para essa pergunta.",
            chunks_used=[],
            model=model,
            latency_ms=retrieval_ms,
        )

    # --- 1.5 Context Engine: Expansão de chunks filhos para pais e deduplicação (Passo 17) ---
    if expand_parents and chunks:
        before_count = len(chunks)
        chunks = expand_parent_context(chunks)
        if any(c.parent_chunk_id for c in chunks):
            print(f"[answer] Context Engine: {before_count} chunks expandidos/deduplicados para {len(chunks)} contextos pais")

    # --- 2. Geração ---
    if not settings.anthropic_api_key:
        print("[answer] ANTHROPIC_API_KEY não configurada — modo simulado")
        result = _simulated_answer(query, chunks)
        result.latency_ms = retrieval_ms
    else:
        result = _call_claude(
            query, chunks, model=model, stream=stream, max_tokens=max_output_tokens
        )
        print(
            f"[answer] Geração: {result.latency_ms:.0f}ms | "
            f"{result.input_tokens}→{result.output_tokens} tokens"
        )

    # Salva no cache de respostas (Passo 32, isolado por tenant e ACL)
    try:
        from src.optimization.caching import cache
        cache.set_response(
            query=query,
            tenant_id=tenant_id or "",
            principals=principals,
            response_payload={
                "answer": result.answer,
                "model": result.model,
                "simulated": result.simulated,
                "source_uris": result.source_uris,
                "chunks_used": [_serialize_chunk(c) for c in result.chunks_used],
            },
        )
    except Exception as e:
        logger.debug("Falha na gravação do cache: %s", e)

    return result


def answer_stream(
    query: str,
    top_k: int = MAX_CONTEXT_CHUNKS,
    tenant_id: Optional[str] = None,
    principals: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL,
    expand_parents: bool = True,
    prefetched_chunks: Optional[List[HybridResult]] = None,
) -> Iterator[str]:
    """
    Generator para streaming token-a-token da resposta final.
    Executa retrieval com pré-filtro de tenant e ACL (principals),
    grava no cache ao término da transmissão e registra telemetria.
    """
    # 0. Cache check
    try:
        from src.optimization.caching import cache
        cached = cache.get_response(query=query, tenant_id=tenant_id or "", principals=principals)
        if cached is not None:
            words = cached.get("answer", "").split(" ")
            for i, w in enumerate(words):
                yield w + (" " if i < len(words) - 1 else "")
            return
    except Exception as e:
        logger.debug("Falha na consulta ao cache em answer_stream: %s", e)

    # Inicia trace se observabilidade estiver disponível
    _active_trace = get_current_trace()
    _auto_trace_ctx = None
    if _active_trace is None and _OBS_AVAILABLE:
        _auto_trace_ctx = pipeline_trace(query, log=True)
        _active_trace = _auto_trace_ctx.__enter__()

    # 1. Retrieval ou prefetched chunks
    if prefetched_chunks is not None:
        chunks = prefetched_chunks
    else:
        chunks = hybrid_search(
            query,
            top_k=top_k,
            tenant_id=tenant_id,
            principals=principals,
        )

    if not chunks:
        yield "Não encontrei documentos relevantes para essa pergunta."
        if _auto_trace_ctx is not None:
            _auto_trace_ctx.__exit__(None, None, None)
        return

    if expand_parents and chunks:
        chunks = expand_parent_context(chunks)

    # 2. Streaming e acumulação para cache/telemetria
    parts: List[str] = []
    t0 = time.time()
    try:
        if not settings.anthropic_api_key:
            sim = _simulated_answer(query, chunks)
            words = sim.answer.split(" ")
            for i, w in enumerate(words):
                token = w + (" " if i < len(words) - 1 else "")
                parts.append(token)
                yield token
        else:
            for token in _stream_claude(query, chunks, model=model):
                parts.append(token)
                yield token
    finally:
        full_answer = "".join(parts)
        if full_answer:
            latency_ms = (time.time() - t0) * 1000
            in_tokens = sum(len(c.content.split()) for c in chunks)
            out_tokens = max(1, len(full_answer.split()))
            record_generation_cost(
                model=model,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                latency_ms=latency_ms,
            )
            try:
                from src.optimization.caching import cache
                cache.set_response(
                    query=query,
                    tenant_id=tenant_id or "",
                    principals=principals,
                    response_payload={
                        "answer": full_answer,
                        "model": model,
                        "simulated": not bool(settings.anthropic_api_key),
                        "source_uris": list({c.source_uri for c in chunks if c.source_uri}),
                        "chunks_used": [_serialize_chunk(c) for c in chunks],
                    },
                )
            except Exception as e:
                logger.debug("Falha na gravação do cache em answer_stream: %s", e)

        if _auto_trace_ctx is not None:
            _auto_trace_ctx.__exit__(None, None, None)



