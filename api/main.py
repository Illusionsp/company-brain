# api/main.py
# Company Brain — FastAPI app.
#
# Endpoints:
#   POST /ingest/text       add text to knowledge base
#   POST /ingest/file       upload PDF / TXT / MD
#   POST /chat              ask a question
#   GET  /documents         list indexed documents
#   DELETE /documents/{n}   remove a document
#   POST /feedback          thumbs up/down
#   DELETE /sessions/{id}   clear conversation
#   GET  /health            status + stats
#   GET  /                  web dashboard

import logging, os
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
from ingestion.chunker import chunk_text, extract_text_from_pdf
from ingestion.embedder import embed_texts
from retrieval.vector_store import (
    init_store, store_chunks, hybrid_search,
    list_documents, delete_document,
    get_stats, save_feedback, get_top_questions,
)
from retrieval.reranker import rerank
from retrieval.rag_pipeline import generate_answer, clear_session, session_turns

logger = logging.getLogger(__name__)
app    = FastAPI(title="Company Brain", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question:   str
    session_id: Optional[str] = "default"
    top_k:      int = 4
    doc_filter: Optional[str] = None

class ChatResponse(BaseModel):
    answer:      str
    sources:     List[str]
    used_chunks: List[dict]   # exact chunks passed to AI with all scores
    session_id:  str
    turns:       int

class IngestRequest(BaseModel):
    content:  str
    doc_name: str

class FeedbackRequest(BaseModel):
    question:   str
    answer:     str
    thumbs:     str
    session_id: Optional[str] = ""


# ── Startup ─────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    init_store()
    logger.info(f"Company Brain | AI: {settings.ai_provider()} | Embed: all-MiniLM-L6-v2 (local)")


# ── Endpoints ───────────────────────────────────────────────

@app.get("/health")
async def health():
    s = get_stats()
    return {
        "status": "ok",
        "ai_provider": settings.ai_provider(),
        "embedding_provider": "all-MiniLM-L6-v2 (local)",
        "retrieval": "hybrid_bm25_semantic",
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "documents": s["total_documents"],
        "chunks": s["total_chunks"],
        "feedback": s["total_feedback"],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


@app.post("/ingest/text")
async def ingest_text(req: IngestRequest):
    if not req.content.strip():
        raise HTTPException(400, "Content cannot be empty")
    chunks = chunk_text(req.content, req.doc_name, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(422, "No content could be extracted")
    embs = await embed_texts([c.content for c in chunks])
    store_chunks(chunks, embs)
    return {"status": "success", "doc_name": req.doc_name, "chunks": len(chunks),
            "message": f"'{req.doc_name}' indexed. Ready to answer questions about it."}


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    fname = file.filename or "document"
    ext   = Path(fname).suffix.lower()
    if ext not in (".pdf", ".txt", ".md"):
        raise HTTPException(400, f"Unsupported: {ext}. Use .pdf .txt .md")
    content = await file.read()
    if not content:
        raise HTTPException(400, "File is empty")
    text = extract_text_from_pdf(content) if ext == ".pdf" else content.decode("utf-8", "ignore")
    if not text.strip():
        raise HTTPException(422, "No text found in file")
    doc_name = Path(fname).stem
    chunks   = chunk_text(text, doc_name, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    embs     = await embed_texts([c.content for c in chunks])
    store_chunks(chunks, embs)
    return {"status": "success", "doc_name": doc_name,
            "file_size": f"{len(content)/1024:.1f} KB", "chunks": len(chunks),
            "message": f"'{doc_name}' indexed. Ready to query."}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Full pipeline:
    1. Hybrid search (BM25 + semantic) → 20 candidates
    2. Cross-encoder reranking → top 4
    3. AI answers from top 4 only
    4. Returns answer + sources + exact chunks used
    """
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    s = get_stats()
    if s["total_chunks"] == 0:
        raise HTTPException(400, "No documents indexed. Upload a document first via /ingest/file")

    # Embed question
    q_embs = await embed_texts([req.question])
    q_vec  = q_embs[0]

    # Hybrid search
    candidates = hybrid_search(
        query=req.question, query_embedding=q_vec,
        top_k=settings.INITIAL_TOP_K, doc_filter=req.doc_filter)

    if not candidates:
        return ChatResponse(answer="I couldn't find relevant information in the documents.",
                            sources=[], used_chunks=[], session_id=req.session_id or "default", turns=0)

    # Cross-encoder rerank
    reranked = rerank(req.question, candidates, top_k=req.top_k)

    # Generate answer
    answer  = await generate_answer(req.question, reranked, req.session_id or "default")
    sources = list(dict.fromkeys(r.doc_name for r in reranked))

    used_chunks = [
        {"chunk_id": c.chunk_id, "doc_name": c.doc_name, "content": c.content,
         "semantic_score": c.semantic_score, "bm25_score": c.bm25_score,
         "hybrid_score": c.hybrid_score, "rerank_score": c.rerank_score}
        for c in reranked
    ]

    return ChatResponse(answer=answer, sources=sources, used_chunks=used_chunks,
                        session_id=req.session_id or "default",
                        turns=session_turns(req.session_id or "default"))


@app.get("/documents")
async def documents():
    return {"documents": list_documents(), "stats": get_stats()}


@app.delete("/documents/{doc_name}")
async def remove_doc(doc_name: str):
    n = delete_document(doc_name)
    if n == 0:
        raise HTTPException(404, f"'{doc_name}' not found")
    return {"deleted": doc_name, "chunks_removed": n}


@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    if req.thumbs not in ("up", "down"):
        raise HTTPException(400, "thumbs must be 'up' or 'down'")
    save_feedback(req.question, req.answer, req.thumbs, req.session_id or "")
    return {"status": "saved"}


@app.delete("/sessions/{session_id}")
async def clear(session_id: str):
    clear_session(session_id)
    return {"cleared": session_id}


# ── Root ─────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "Company Brain API is running. Use Streamlit for the UI."}
