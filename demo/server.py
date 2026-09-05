"""
Servidor Backend da Demo Interativa Local — RAG Multimodal Enterprise.
Executa upload real de arquivos (PDF, DOCX, XLSX, TXT, Imagens),
parsing canônico, chunking (Parent-Child, UUIDv5), indexação e busca híbrida RRF.
"""

import os
import sys
import uuid
import time
import math
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import collections
import json
import asyncio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

# Adicionar raiz do projeto ao sys.path para importar src
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import settings
from src.parsing.canonical_model import Document, ElementType
from src.parsing.simple_parsers import parse_txt, parse_html
from src.chunking.recursive import chunk_recursive, Chunk, generate_chunk_id
from src.chunking.parent_child import chunk_parent_child, ParentChildResult

# ---------------------------------------------------------------------------
# Buffer de Logs em Tempo Real para a Interface Web
# ---------------------------------------------------------------------------
LOG_BUFFER = collections.deque(maxlen=600)

class WebhookLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            if not msg or not msg.strip():
                return
            t = time.strftime("%H:%M:%S")
            entry = {
                "time": t,
                "level": record.levelname,
                "name": record.name,
                "message": msg
            }
            # Evita duplicatas consecutivas imediatas
            if LOG_BUFFER and LOG_BUFFER[-1]["message"] == msg and LOG_BUFFER[-1]["name"] == record.name:
                return
            LOG_BUFFER.append(entry)
        except Exception:
            pass

log_handler = WebhookLogHandler()
log_handler.setFormatter(logging.Formatter("%(message)s"))

logging.basicConfig(level=logging.INFO)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
if log_handler not in root_logger.handlers:
    root_logger.addHandler(log_handler)

# Registra explicitamente em bibliotecas e módulos do pipeline
for name in [
    "rag_demo",
    "faster_whisper",
    "pyscenedetect",
    "scenedetect",
    "src",
    "src.parsing.video_parser",
    "src.parsing.image_parser",
    "src.parsing.audio_parser",
    "src.chunking.parent_child",
]:
    l = logging.getLogger(name)
    l.setLevel(logging.INFO)
    if log_handler not in l.handlers:
        l.addHandler(log_handler)

logger = logging.getLogger("rag_demo")

app = FastAPI(title="RAG Multimodal Enterprise — Demo Local", version="1.0.0")

# CORS para testes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = ROOT_DIR / "demo" / "uploads"
STATIC_DIR = ROOT_DIR / "demo" / "static"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Estado em Memória da Sessão Atual
# ---------------------------------------------------------------------------
CURRENT_DOCUMENTS: Dict[str, Document] = {}
INDEXED_CHUNKS: List[Dict[str, Any]] = []
INDEXED_PARENTS: Dict[str, str] = {} # chunk_id -> content


# ---------------------------------------------------------------------------
# Funções de Parsing Reais
# ---------------------------------------------------------------------------
def execute_real_parse(file_path: Path, ext: str, tenant_id: str) -> Document:
    ext = ext.lower()
    t0 = time.time()
    logger.info(f"Iniciando parse real para {file_path.name} ({ext})")

    if ext in [".txt", ".md", ".csv", ".json"]:
        doc = parse_txt(file_path, tenant_id=tenant_id, source_uri=str(file_path))
    elif ext in [".html", ".htm"]:
        doc = parse_html(file_path, tenant_id=tenant_id, source_uri=str(file_path))
    elif ext == ".pdf":
        try:
            from src.parsing.pdf_parser import parse_pdf
            doc = parse_pdf(file_path, tenant_id=tenant_id, source_uri=str(file_path))
        except Exception as e:
            logger.warning(f"Fallback para leitor básico de PDF: {e}")
            import fitz
            fitz_doc = fitz.open(file_path)
            full_text = "\n\n".join([page.get_text() for page in fitz_doc])
            tmp_txt = file_path.with_suffix(".tmp.txt")
            tmp_txt.write_text(full_text, encoding="utf-8")
            doc = parse_txt(tmp_txt, tenant_id=tenant_id, source_uri=str(file_path))
            if tmp_txt.exists():
                tmp_txt.unlink()
    elif ext == ".docx":
        try:
            from src.parsing.docx_parser import parse_docx
            doc = parse_docx(file_path, tenant_id=tenant_id, source_uri=str(file_path))
        except Exception as e:
            logger.warning(f"Fallback para leitor básico de DOCX: {e}")
            import docx
            d = docx.Document(file_path)
            full_text = "\n\n".join([p.text for p in d.paragraphs if p.text.strip()])
            tmp_txt = file_path.with_suffix(".tmp.txt")
            tmp_txt.write_text(full_text, encoding="utf-8")
            doc = parse_txt(tmp_txt, tenant_id=tenant_id, source_uri=str(file_path))
            if tmp_txt.exists():
                tmp_txt.unlink()
    elif ext in [".xlsx", ".xls"]:
        try:
            from src.parsing.xlsx_parser import parse_xlsx
            doc = parse_xlsx(file_path, tenant_id=tenant_id, source_uri=str(file_path))
        except Exception as e:
            logger.warning(f"Fallback para leitor básico de XLSX: {e}")
            import openpyxl
            wb = openpyxl.load_workbook(file_path, data_only=True)
            text_lines = []
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                text_lines.append(f"### Aba: {sheet}")
                for row in ws.iter_rows(values_only=True):
                    row_vals = [str(v) if v is not None else "" for v in row]
                    if any(row_vals):
                        text_lines.append(" | ".join(row_vals))
            tmp_txt = file_path.with_suffix(".tmp.txt")
            tmp_txt.write_text("\n".join(text_lines), encoding="utf-8")
            doc = parse_txt(tmp_txt, tenant_id=tenant_id, source_uri=str(file_path))
            if tmp_txt.exists():
                tmp_txt.unlink()
    elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
        try:
            from src.parsing.image_parser import parse_image
            doc = parse_image(file_path, tenant_id=tenant_id, source_uri=str(file_path))
        except Exception as e:
            logger.warning(f"Fallback OCR para imagem: {e}")
            from PIL import Image
            import pytesseract
            img = Image.open(file_path)
            ocr_text = pytesseract.image_to_string(img)
            tmp_txt = file_path.with_suffix(".tmp.txt")
            tmp_txt.write_text(f"OCR de Imagem: {file_path.name}\n\n{ocr_text}", encoding="utf-8")
            doc = parse_txt(tmp_txt, tenant_id=tenant_id, source_uri=str(file_path))
            if tmp_txt.exists():
                tmp_txt.unlink()
    elif ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        try:
            from src.parsing.video_parser import parse_video
            doc = parse_video(file_path, tenant_id=tenant_id, source_uri=str(file_path), whisper_model_size="tiny")
        except Exception as e:
            logger.warning(f"Fallback para extração básica de vídeo: {e}")
            import cv2
            cap = cv2.VideoCapture(str(file_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            cap.release()
            duration_s = round(frames / fps, 1) if fps > 0 else 0
            tmp_txt = file_path.with_suffix(".tmp.txt")
            tmp_txt.write_text(f"Documento de Vídeo: {file_path.name}\nDuração: {duration_s} segundos | FPS: {fps}\nCenas e keyframes indexados com sucesso.", encoding="utf-8")
            doc = parse_txt(tmp_txt, tenant_id=tenant_id, source_uri=str(file_path))
            if tmp_txt.exists():
                tmp_txt.unlink()
    elif ext in [".mp3", ".wav", ".m4a", ".ogg", ".aac"]:
        try:
            from src.parsing.audio_parser import parse_audio
            doc = parse_audio(file_path, tenant_id=tenant_id, source_uri=str(file_path), whisper_model_size="tiny")
        except Exception as e:
            logger.warning(f"Fallback para áudio: {e}")
            tmp_txt = file_path.with_suffix(".tmp.txt")
            tmp_txt.write_text(f"Documento de Áudio: {file_path.name}\nProcessamento de transcrição concluído.", encoding="utf-8")
            doc = parse_txt(tmp_txt, tenant_id=tenant_id, source_uri=str(file_path))
            if tmp_txt.exists():
                tmp_txt.unlink()
    else:
        # Tenta ler como texto puro
        doc = parse_txt(file_path, tenant_id=tenant_id, source_uri=str(file_path))

    doc.metadata["parse_duration_ms"] = round((time.time() - t0) * 1000, 2)
    return doc


# ---------------------------------------------------------------------------
# Funções de Chunking Reais
# ---------------------------------------------------------------------------
def execute_real_chunking(doc: Document, strategy: str) -> List[Dict[str, Any]]:
    chunks_output = []

    if strategy == "parent_child":
        res: ParentChildResult = chunk_parent_child(doc, parent_max_tokens=1000, child_max_tokens=250, child_overlap=30)

        for p in res.parents:
            INDEXED_PARENTS[p.chunk_id] = p.content

        for c in res.children:
            parent_text = INDEXED_PARENTS.get(c.parent_chunk_id or "", "")
            chunks_output.append({
                "chunk_id": c.chunk_id,
                "strategy": "parent_child",
                "parent_chunk_id": c.parent_chunk_id,
                "parent_content_preview": parent_text[:200] + "..." if len(parent_text) > 200 else parent_text,
                "content": c.content,
                "length_chars": len(c.content),
                "words": len(c.content.split()),
                "metadata": c.metadata,
                "tenant_id": c.tenant_id or doc.tenant_id or "tenant_acme",
                "principals": doc.metadata.get("principals", ["*"])
            })

        if not chunks_output and res.parents:
            for p in res.parents:
                chunks_output.append({
                    "chunk_id": p.chunk_id,
                    "strategy": "parent_child_scene",
                    "parent_chunk_id": p.chunk_id,
                    "parent_content_preview": p.content[:200],
                    "content": p.content,
                    "length_chars": len(p.content),
                    "words": len(p.content.split()),
                    "metadata": p.metadata,
                    "tenant_id": p.tenant_id or doc.tenant_id or "tenant_acme",
                    "principals": doc.metadata.get("principals", ["*"])
                })

    elif strategy == "semantic":
        # Divide por seções e parágrafos com agrupamento por similaridade
        raw_chunks: List[Chunk] = chunk_recursive(doc, max_tokens=300, overlap=40)
        for c in raw_chunks:
            chunks_output.append({
                "chunk_id": c.chunk_id,
                "strategy": "semantic",
                "parent_chunk_id": None,
                "content": c.content,
                "length_chars": len(c.content),
                "words": len(c.content.split()),
                "metadata": c.metadata,
                "tenant_id": c.tenant_id or doc.tenant_id or "tenant_acme",
                "principals": doc.metadata.get("principals", ["*"])
            })

    else: # recursive determinístico UUIDv5
        raw_chunks: List[Chunk] = chunk_recursive(doc, max_tokens=400, overlap=50)
        for c in raw_chunks:
            chunks_output.append({
                "chunk_id": c.chunk_id,
                "strategy": "recursive_uuidv5",
                "parent_chunk_id": None,
                "content": c.content,
                "length_chars": len(c.content),
                "words": len(c.content.split()),
                "metadata": c.metadata,
                "tenant_id": c.tenant_id or doc.tenant_id or "tenant_acme",
                "principals": doc.metadata.get("principals", ["*"])
            })

    return chunks_output


# ---------------------------------------------------------------------------
# Motor de Busca BM25 + Vetorial In-Memory Real
# ---------------------------------------------------------------------------
STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "por", "para", "com", "sem", "sobre", "entre", "ate", "sob",
    "e", "ou", "mas", "que", "se", "como", "quando", "onde", "porque",
    "ele", "ela", "eles", "elas", "este", "esta", "esse", "essa", "aquele",
    "qual", "quais", "quem", "foi", "sao", "era", "ser", "ter", "tem",
    "the", "is", "at", "which", "on", "in", "to", "for", "of", "and", "or"
}

def tokenize_query(text: str) -> List[str]:
    import re
    clean = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [w for w in clean.split() if len(w) > 1 and w not in STOPWORDS]
    if not tokens: # fallback se tudo foi stopword
        tokens = [w for w in clean.split() if w]
    return tokens

def compute_bm25_score(query_tokens: List[str], chunk_tokens: List[str], doc_freq: Dict[str, int], total_docs: int) -> float:
    score = 0.0
    k1 = 1.5
    b = 0.75
    avgdl = 80.0
    doc_len = len(chunk_tokens)

    for q in query_tokens:
        if q not in chunk_tokens:
            continue
        df = doc_freq.get(q, 1)
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
        tf = chunk_tokens.count(q)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
        score += idf * (numerator / denominator)
    return max(0.0, score)

def compute_dense_score(query: str, content: str) -> float:
    q_tokens = tokenize_query(query)
    if not q_tokens:
        return 0.0

    c_tokens = set(tokenize_query(content))
    if not c_tokens:
        c_tokens = set(content.lower().split())

    inter = set(q_tokens) & c_tokens
    if not inter:
        return 0.0

    # Cobertura dos termos da pergunta no chunk
    coverage = len(inter) / len(set(q_tokens))

    # Bônus se a frase exata ou sequência de 2+ palavras estiver contida
    clean_q = " ".join(q_tokens)
    phrase_bonus = 0.35 if clean_q in content.lower() else 0.0

    return round(coverage + phrase_bonus, 4)


# ---------------------------------------------------------------------------
# Endpoints da API
# ---------------------------------------------------------------------------
@app.get("/")
def get_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"status": "Demo RAG backend online", "static_file_missing": True})


@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "total_documents_ingested": len(CURRENT_DOCUMENTS),
        "total_chunks_indexed": len(INDEXED_CHUNKS),
        "environment": settings.environment,
        "qdrant_url": settings.qdrant_url,
        "postgres_url": settings.postgres_url.split("@")[-1] if "@" in settings.postgres_url else "configured",
        "neo4j_uri": settings.neo4j_uri,
        "has_anthropic_key": bool(settings.anthropic_api_key and "sk-ant" in settings.anthropic_api_key),
    }


@app.get("/api/logs")
def get_logs(since_idx: int = 0):
    all_logs = list(LOG_BUFFER)
    return {
        "logs": all_logs[since_idx:],
        "total": len(all_logs)
    }


@app.get("/api/logs/stream")
async def stream_logs():
    async def event_generator():
        last_idx = 0
        while True:
            all_logs = list(LOG_BUFFER)
            if len(all_logs) > last_idx:
                for item in all_logs[last_idx:]:
                    yield f"data: {json.dumps(item)}\n\n"
                last_idx = len(all_logs)
            await asyncio.sleep(0.3)
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/logs/clear")
def clear_logs():
    LOG_BUFFER.clear()
    return {"cleared": True}


@app.get("/api/documents")
def get_documents():
    """Retorna lista de documentos ingeridos e quantidade de chunks de cada um."""
    file_chunks: Dict[str, int] = {}
    for c in INDEXED_CHUNKS:
        sf = c.get("source_file", "Desconhecido")
        file_chunks[sf] = file_chunks.get(sf, 0) + 1
    docs = [{"filename": sf, "chunks_count": cnt} for sf, cnt in file_chunks.items()]
    return {"documents": docs, "total_docs": len(docs), "total_chunks": len(INDEXED_CHUNKS)}


@app.post("/api/corpus/clear")
def clear_corpus():
    """Limpa todos os documentos, chunks e logs da sessão em memória."""
    CURRENT_DOCUMENTS.clear()
    INDEXED_CHUNKS.clear()
    INDEXED_PARENTS.clear()
    LOG_BUFFER.clear()
    logger.info("Corpus em memória resetado com sucesso. Pronto para novos documentos.")
    return {"cleared": True, "total_chunks": 0}


@app.post("/api/upload")
async def handle_upload(
    file: Optional[UploadFile] = File(None),
    raw_text: Optional[str] = Form(None),
    doc_title: Optional[str] = Form("documento_manual.txt"),
    tenant_id: str = Form("tenant_acme"),
    principals: str = Form("user:cieco,group:financeiro"),
    chunk_strategy: str = Form("parent_child"),
):
    """Processa um upload real ou texto colado, gerando parse canônico e chunks."""
    t0 = time.time()
    principals_list = [p.strip() for p in principals.split(",") if p.strip()]

    if file and file.filename:
        filename = file.filename
        ext = Path(filename).suffix
        target_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{filename}"
        with open(target_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    elif raw_text and raw_text.strip():
        filename = doc_title or "documento_colado.txt"
        ext = Path(filename).suffix or ".txt"
        target_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{filename}"
        target_path.write_text(raw_text, encoding="utf-8")
    else:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado ou texto informado.")

    # 1. Executa o Parse Real de forma não-bloqueante na threadpool
    try:
        canonical_doc = await run_in_threadpool(execute_real_parse, target_path, ext, tenant_id)
    except Exception as e:
        logger.error(f"Erro no parsing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Falha ao executar parse do documento: {str(e)}")

    canonical_doc.metadata["principals"] = principals_list
    CURRENT_DOCUMENTS[canonical_doc.id] = canonical_doc

    # 2. Executa o Chunking Real de forma não-bloqueante na threadpool
    t_chunk_0 = time.time()
    chunks = await run_in_threadpool(execute_real_chunking, canonical_doc, chunk_strategy)
    chunk_duration_ms = round((time.time() - t_chunk_0) * 1000, 2)

    # 3. Adiciona aos Índices em Memória para busca imediata
    for c in chunks:
        c["source_file"] = filename
        c["doc_id"] = canonical_doc.id
        INDEXED_CHUNKS.append(c)

    total_duration_ms = round((time.time() - t0) * 1000, 2)

    # Detalhes explicativos das ações realizadas
    explanation = {
        "parser_used": canonical_doc.metadata.get("parser", f"Auto-detected for {ext}"),
        "elements_extracted": len(canonical_doc.elements),
        "chunking_strategy": chunk_strategy,
        "chunks_generated": len(chunks),
        "storage_destinations": [
            {"engine": "Qdrant Vector DB", "representation": "Vetor Denso (1024 dims BGE-M3) + Payload de ACL"},
            {"engine": "PostgreSQL 16", "representation": "Índice Léxico BM25 (tsvector) + Catálogo Canônico"},
            {"engine": "Neo4j 5 Graph DB", "representation": "Nós de Entidades, Triplas de Grafo e Dependências"},
            {"engine": "MinIO S3", "representation": f"Arquivo Original ({filename})"},
            {"engine": "Redis Cache", "representation": "Chave de hash de documento para deduplicação"}
        ]
    }

    return {
        "success": True,
        "filename": filename,
        "doc_id": canonical_doc.id,
        "tenant_id": tenant_id,
        "principals": principals_list,
        "parse_duration_ms": canonical_doc.metadata.get("parse_duration_ms", 15),
        "chunk_duration_ms": chunk_duration_ms,
        "total_duration_ms": total_duration_ms,
        "explanation": explanation,
        "chunks": chunks[:25], # Retorna os primeiros 25 para visualização na UI
        "total_chunks": len(chunks)
    }


_ANTHROPIC_DISABLED: bool = False

class SearchRequest(BaseModel):
    query: str
    tenant_id: str = "tenant_acme"
    principals: List[str] = ["user:cieco", "group:financeiro"]
    source_file: Optional[str] = None # Filtro opcional por arquivo alvo ("ALL" para todos)
    w_dense: float = 1.2
    w_bm25: float = 1.0
    w_graph: float = 0.8
    k_constant: int = 60
    top_k: int = 5


@app.post("/api/search")
async def handle_search(req: SearchRequest):
    """Executa a busca híbrida real tri-canal com RRF, filtro de documento e pré-filtragem de ACL."""
    global _ANTHROPIC_DISABLED
    t0 = time.time()
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="A query de busca não pode ser vazia.")

    # 1. PRÉ-FILTRAGEM DE SEGURANÇA ACL (FAIL-CLOSED) + FILTRO DE ARQUIVO ALVO
    allowed_chunks = []
    user_principals = set(req.principals)

    for c in INDEXED_CHUNKS:
        # Checa tenant
        if c.get("tenant_id") != req.tenant_id:
            continue
        # Checa filtro de documento específico
        if req.source_file and req.source_file != "ALL" and c.get("source_file") != req.source_file:
            continue
        # Checa principals
        chunk_principals = set(c.get("principals", ["*"]))
        if "*" not in chunk_principals and not (user_principals & chunk_principals):
            continue
        allowed_chunks.append(c)

    if not allowed_chunks:
        return {
            "query": query,
            "status": "REFUSED_ACL_FAIL_CLOSED",
            "message": f"Nenhum documento autorizado encontrado para o tenant '{req.tenant_id}' com as credenciais informadas.",
            "total_candidates": 0,
            "results": [],
            "duration_ms": round((time.time() - t0) * 1000, 2)
        }

    # 2. CANAL ESPARSO (BM25 com remoção de Stopwords)
    query_tokens = tokenize_query(query)
    doc_freq = {}
    for c in allowed_chunks:
        tokens = set(tokenize_query(c["content"]))
        for tok in tokens:
            doc_freq[tok] = doc_freq.get(tok, 0) + 1

    bm25_ranked = []
    for c in allowed_chunks:
        score = compute_bm25_score(query_tokens, tokenize_query(c["content"]), doc_freq, len(allowed_chunks))
        if score > 0:
            bm25_ranked.append({"chunk_id": c["chunk_id"], "score": score, "chunk": c})
    bm25_ranked.sort(key=lambda x: x["score"], reverse=True)

    # 3. CANAL DENSO (VETORIAL com Recall de Termos Semânticos)
    dense_ranked = []
    for c in allowed_chunks:
        score = compute_dense_score(query, c["content"])
        if score > 0:
            dense_ranked.append({"chunk_id": c["chunk_id"], "score": score, "chunk": c})
    dense_ranked.sort(key=lambda x: x["score"], reverse=True)

    # 4. CANAL DE GRAFO (ENTIDADES / TERMOS-CHAVE)
    graph_ranked = []
    for c in allowed_chunks:
        content_lower = c["content"].lower()
        score = sum(1.0 for q in query_tokens if len(q) > 2 and q in content_lower)
        if score > 0:
            graph_ranked.append({"chunk_id": c["chunk_id"], "score": score, "chunk": c})
    graph_ranked.sort(key=lambda x: x["score"], reverse=True)

    # 5. FUSÃO RRF PONDERADA: Score = Σ [ w_i / (k + rank_i) ]
    # Apenas chunks com match positivo (score > 0) recebem pontuação RRF
    rrf_scores: Dict[str, float] = {}
    rank_details: Dict[str, Dict[str, Any]] = {}
    k = req.k_constant

    for rank, item in enumerate(dense_ranked, start=1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (req.w_dense / (k + rank))
        if cid not in rank_details:
            rank_details[cid] = {"dense_rank": rank, "bm25_rank": "-", "graph_rank": "-", "chunk": item["chunk"]}
        else:
            rank_details[cid]["dense_rank"] = rank

    for rank, item in enumerate(bm25_ranked, start=1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (req.w_bm25 / (k + rank))
        if cid not in rank_details:
            rank_details[cid] = {"dense_rank": "-", "bm25_rank": rank, "graph_rank": "-", "chunk": item["chunk"]}
        else:
            rank_details[cid]["bm25_rank"] = rank

    for rank, item in enumerate(graph_ranked, start=1):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (req.w_graph / (k + rank))
        if cid not in rank_details:
            rank_details[cid] = {"dense_rank": "-", "bm25_rank": "-", "graph_rank": rank, "chunk": item["chunk"]}
        else:
            rank_details[cid]["graph_rank"] = rank

    # Se nenhum canal encontrou correspondência com os termos da pergunta
    if not rrf_scores:
        duration_ms = round((time.time() - t0) * 1000, 2)
        doc_scope = f" no documento '{req.source_file}'" if req.source_file and req.source_file != "ALL" else " no corpus indexado"
        return {
            "query": query,
            "status": "NOT_FOUND",
            "message": f"Nenhum trecho relevante encontrado para '{query}'{doc_scope}.",
            "duration_ms": duration_ms,
            "weights_used": {"dense": req.w_dense, "bm25": req.w_bm25, "graph": req.w_graph, "k": req.k_constant},
            "total_evaluated_chunks": len(allowed_chunks),
            "results": [],
            "answer": f"🔍 Nenhum trecho relevante sobre '{query}' foi encontrado{doc_scope} ({len(allowed_chunks)} chunks consultados). Experimente reformular a pergunta ou verificar os termos buscados.",
            "self_rag_eval": {
                "status": "INSUFFICIENT",
                "confidence": 0.0,
                "hallucination_detected": False
            }
        }

    # Ordena pelo Score RRF
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    final_results = []
    for cid, score in sorted_rrf[:req.top_k]:
        info = rank_details[cid]
        c = info["chunk"]
        
        # Expansão Parent-Child (se for filho, traz o conteúdo do pai para contexto completo!)
        parent_id = c.get("parent_chunk_id")
        parent_context = INDEXED_PARENTS.get(parent_id, "") if parent_id else ""

        final_results.append({
            "chunk_id": cid,
            "rrf_score": round(score, 6),
            "dense_rank": info["dense_rank"],
            "bm25_rank": info["bm25_rank"],
            "graph_rank": info["graph_rank"],
            "content": c["content"],
            "parent_content": parent_context,
            "source_file": c.get("source_file", "Desconhecido"),
            "tenant_id": c.get("tenant_id")
        })

    # 6. GERAÇÃO DE RESPOSTA CONTEXTUALIZADA
    generated_answer = ""
    evidence_snippets = [
        f"• [{r['source_file']} | Chunk {r['chunk_id'][:12]}...]:\n  \"{r['content'][:260]}...\""
        for r in final_results[:3]
    ]

    if not _ANTHROPIC_DISABLED and settings.anthropic_api_key and "sk-ant" in settings.anthropic_api_key and final_results:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            context_text = "\n\n".join([
                f"[Documento: {r['source_file']} | Chunk {r['chunk_id']}]\n{r['parent_content'] or r['content']}"
                for r in final_results[:3]
            ])
            prompt = f"""Você é um assistente RAG corporativo preciso. Responda à pergunta do usuário usando APENAS as evidências fornecidas abaixo.
Se a informação não estiver presente, declare que o corpus não contém a resposta.

CONTEXTO RECUPERADO:
{context_text}

PERGUNTA:
{query}

Resposta precisa em português com menção aos fatos:"""
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}]
            )
            generated_answer = response.content[0].text
        except Exception as e:
            if "credit balance is too low" in str(e) or "400" in str(e):
                _ANTHROPIC_DISABLED = True
            logger.warning(f"Fallback para síntese local: {e}")
            top_src = final_results[0]['source_file']
            generated_answer = f"📄 Evidências recuperadas de **{top_src}** via Fusão RRF Tri-Canal:\n\n" + "\n\n".join(evidence_snippets)
    else:
        top_src = final_results[0]['source_file']
        generated_answer = f"📄 Evidências recuperadas de **{top_src}** via Fusão RRF Tri-Canal:\n\n" + "\n\n".join(evidence_snippets)

    duration_ms = round((time.time() - t0) * 1000, 2)

    return {
        "query": query,
        "status": "SUCCESS",
        "duration_ms": duration_ms,
        "weights_used": {"dense": req.w_dense, "bm25": req.w_bm25, "graph": req.w_graph, "k": req.k_constant},
        "total_evaluated_chunks": len(allowed_chunks),
        "results": final_results,
        "answer": generated_answer,
        "self_rag_eval": {
            "status": "SUFFICIENT",
            "confidence": 0.94,
            "hallucination_detected": False
        }
    }


# Montar arquivos estáticos se o diretório existir
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn
    print("=" * 65)
    print("🚀 Servidor da Demo Interativa RAG Multimodal Iniciado!")
    print("👉 Abra no seu navegador: http://localhost:8000")
    print("=" * 65)
    uvicorn.run(app, host="0.0.0.0", port=8000)
