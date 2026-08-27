"""
Core FastAPI application for the Company Brain RAG system.
Handles document ingestion, vector retrieval, and chat endpoints.
"""

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


# ── Dashboard ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    s        = get_stats()
    docs     = list_documents()
    top_q    = get_top_questions(8)
    provider = settings.ai_provider()
    embed    = "all-MiniLM-L6-v2 (local, free)"

    ai_ok    = provider != "none"
    ai_color = "#0F6E56" if ai_ok else "#854F0B"
    ai_bg    = "#E1F5EE" if ai_ok else "#FAEEDA"
    ai_lbl   = f"✅ {provider.capitalize()}" if ai_ok else "⚠️ No AI key — set GROQ_API_KEY in .env"

    doc_rows = "".join([
        f"<tr><td>{d['doc_name']}</td><td style='text-align:center'>{d['chunks']}</td>"
        f"<td style='color:#64748b;font-size:12px'>{(d['added_at'] or '')[:16]}</td>"
        f"<td><button class='del' onclick=\"delDoc('{d['doc_name']}')\">🗑 Remove</button></td></tr>"
        for d in docs
    ]) or "<tr><td colspan='4' style='text-align:center;color:#94a3b8;padding:20px'>No documents yet — upload one below</td></tr>"

    q_rows = "".join([
        f"<tr><td>{q['question'][:70]}</td><td style='text-align:center'>{q['count']}</td>"
        f"<td style='text-align:center;color:#16a34a'>👍 {q['thumbs_up']}</td>"
        f"<td style='text-align:center;color:#dc2626'>👎 {q['thumbs_down']}</td></tr>"
        for q in top_q
    ]) or "<tr><td colspan='4' style='text-align:center;color:#94a3b8;padding:20px'>No feedback yet</td></tr>"

    no_key_banner = "" if ai_ok else """
    <div style="background:#FAEEDA;border:1px solid #F0C4B4;border-radius:10px;padding:14px 18px;margin-bottom:20px;font-size:13px;color:#854F0B">
      ⚠️ <strong>No AI key configured.</strong> Add <code>GROQ_API_KEY=your_key</code> to your .env file.
      Get a free key (no credit card) at <a href="https://console.groq.com" target="_blank" style="color:#854F0B">console.groq.com</a>
    </div>"""

    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{settings.APP_TITLE}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,sans-serif;background:#f8fafc;color:#1e293b}}
.wrap{{max-width:1100px;margin:0 auto;padding:32px 20px}}
h1{{font-size:26px;font-weight:700}}
h2{{font-size:17px;font-weight:600;margin-bottom:14px;color:#1e293b}}
.sub{{color:#64748b;margin-top:4px;font-size:14px}}
.badges{{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}}
.badge{{padding:4px 12px;border-radius:20px;font-size:12px;font-weight:500}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:14px;margin:24px 0}}
.stat{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:18px;text-align:center}}
.stat-n{{font-size:30px;font-weight:700;color:#2563eb}}.stat-l{{font-size:12px;color:#64748b;margin-top:3px}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:22px;margin-bottom:20px}}
.row{{display:flex;gap:10px;margin-bottom:12px}}
input,textarea{{flex:1;padding:11px 14px;border:1px solid #e2e8f0;border-radius:9px;font-size:14px;outline:none;font-family:inherit}}
input:focus,textarea:focus{{border-color:#2563eb;box-shadow:0 0 0 3px #dbeafe}}
.btn{{background:#2563eb;color:#fff;border:none;padding:11px 22px;border-radius:9px;cursor:pointer;font-size:14px;font-weight:500;white-space:nowrap}}
.btn:hover{{background:#1d4ed8}}.btn-sm{{padding:6px 14px;font-size:12px;border-radius:7px}}
.del{{background:#fee2e2;color:#dc2626;border:none;padding:5px 12px;border-radius:6px;cursor:pointer;font-size:12px}}
.answer-box{{display:none;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:18px;margin-top:14px;line-height:1.75;font-size:14px}}
.sources{{margin-top:10px;font-size:12px;color:#64748b;border-top:1px solid #d1fae5;padding-top:8px}}
.fb-row{{display:flex;gap:8px;align-items:center;margin-top:10px}}
.fb-btn{{background:none;border:1px solid #e2e8f0;border-radius:7px;padding:4px 12px;cursor:pointer;font-size:14px}}
.fb-btn:hover{{background:#f1f5f9}}
.chunks-area{{margin-top:14px}}
.chunk{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;margin-bottom:8px;font-size:12px}}
.chunk-top{{display:flex;justify-content:space-between;margin-bottom:6px;font-weight:500;font-size:12px}}
.chunk-scores{{color:#94a3b8;font-size:11px;margin-bottom:5px}}
.chunk-text{{color:#64748b;line-height:1.5;max-height:70px;overflow:hidden}}
.pipeline-trace{{background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:10px 14px;margin-top:10px;font-size:12px;color:#1d4ed8}}
.loading{{color:#64748b;font-style:italic;padding:10px 0;font-size:14px}}
.upload-zone{{border:2px dashed #e2e8f0;border-radius:10px;padding:22px;text-align:center;cursor:pointer;transition:border-color .2s}}
.upload-zone:hover{{border-color:#2563eb}}
.msg{{margin-top:10px;font-size:13px;min-height:18px}}
table{{width:100%;border-collapse:collapse}}
th{{background:#f8fafc;color:#64748b;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;padding:10px 14px;text-align:left;border-bottom:1px solid #e2e8f0}}
td{{padding:10px 14px;border-bottom:1px solid #f1f5f9;font-size:13px;vertical-align:middle}}
tr:last-child td{{border-bottom:none}}
code{{background:#f1f5f9;padding:1px 6px;border-radius:4px;font-size:12px}}
</style></head><body><div class="wrap">

<div style="margin-bottom:28px">
  <h1>🧠 {settings.APP_TITLE}</h1>
  <p class="sub">Upload your SOPs, policies, manuals and FAQs — your team asks questions, AI answers instantly.</p>
  <div class="badges">
    <span class="badge" style="color:{ai_color};background:{ai_bg}">{ai_lbl}</span>
    <span class="badge" style="color:#185FA5;background:#eff6ff">Embeddings: {embed}</span>
    <span class="badge" style="color:#854F0B;background:#FAEEDA">Reranker: cross-encoder</span>
    <span class="badge" style="color:#0F5C47;background:#E1F5EE">Hybrid BM25 + Semantic</span>
  </div>
</div>

{no_key_banner}

<div class="stats">
  <div class="stat"><div class="stat-n">{s['total_documents']}</div><div class="stat-l">Documents</div></div>
  <div class="stat"><div class="stat-n">{s['total_chunks']}</div><div class="stat-l">Knowledge Chunks</div></div>
  <div class="stat"><div class="stat-n">{s['total_feedback']}</div><div class="stat-l">Feedback Given</div></div>
  <div class="stat"><div class="stat-n">{'✅' if ai_ok else '❌'}</div><div class="stat-l">AI Connected</div></div>
</div>

<!-- Ask -->
<div class="card">
  <h2>💬 Ask a Question</h2>
  <div class="row">
    <input type="text" id="q" placeholder="What is our refund policy? How do I reset my password?" onkeypress="if(event.key==='Enter')ask()">
    <button class="btn" onclick="ask()">Ask</button>
  </div>
  <div id="answer-area"></div>
</div>

<!-- Upload -->
<div class="card">
  <h2>📄 Upload Document</h2>
  <div class="upload-zone" onclick="document.getElementById('fi').click()">
    <div style="font-size:28px;margin-bottom:6px">📁</div>
    <span style="color:#2563eb;font-weight:500">Click to upload PDF, TXT, or MD</span>
    <p style="color:#94a3b8;font-size:13px;margin-top:4px">Drag and drop supported</p>
    <input type="file" id="fi" accept=".pdf,.txt,.md" style="display:none" onchange="upload(this)">
  </div>
  <div class="msg" id="umsg"></div>
  <div style="margin-top:16px">
    <p style="font-size:13px;color:#64748b;margin-bottom:8px">Or paste text directly:</p>
    <textarea id="tc" placeholder="Paste your document content here..." style="height:90px;resize:vertical;width:100%"></textarea>
    <div class="row" style="margin-top:8px">
      <input type="text" id="dn" placeholder="Document name (e.g. refund-policy)">
      <button class="btn btn-sm" onclick="ingestText()">Add Text</button>
    </div>
  </div>
</div>

<!-- Documents -->
<div class="card">
  <h2>📚 Knowledge Base</h2>
  <table><thead><tr><th>Document</th><th style="text-align:center">Chunks</th><th>Added</th><th></th></tr></thead>
  <tbody>{doc_rows}</tbody></table>
</div>

<!-- Top questions -->
<div class="card">
  <h2>🔥 Most Asked Questions</h2>
  <table><thead><tr><th>Question</th><th style="text-align:center">Times Asked</th><th style="text-align:center">👍</th><th style="text-align:center">👎</th></tr></thead>
  <tbody>{q_rows}</tbody></table>
</div>

</div>

<script>
let lastQ='', lastA='';

async function ask() {{
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  lastQ = q;
  const area = document.getElementById('answer-area');
  area.innerHTML = '<div class="loading">🔍 Searching documents with hybrid BM25 + semantic search, then reranking with cross-encoder...</div>';

  try {{
    const r    = await fetch('/chat', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{question:q,session_id:'web'}})}});
    const data = await r.json();

    if (!r.ok) {{
      area.innerHTML = `<div class="answer-box" style="display:block;background:#fef2f2;border-color:#fecaca">❌ ${{data.detail}}</div>`;
      return;
    }}

    lastA = data.answer;
    const sources = data.sources.length ? `<div class="sources">📄 <strong>Sources:</strong> ${{data.sources.join(', ')}}</div>` : '';

    const chunkHtml = data.used_chunks.map((c,i) => `
      <div class="chunk">
        <div class="chunk-top">
          <span>📄 ${{c.doc_name}} — Chunk ${{i+1}}</span>
          <span style="color:#2563eb">Rerank: ${{c.rerank_score.toFixed(3)}}</span>
        </div>
        <div class="chunk-scores">Semantic: ${{c.semantic_score.toFixed(3)}} | BM25: ${{c.bm25_score.toFixed(3)}} | Hybrid: ${{c.hybrid_score.toFixed(3)}}</div>
        <div class="chunk-text">${{c.content.slice(0,200)}}...</div>
      </div>`).join('');

    area.innerHTML = `
      <div class="answer-box" style="display:block">
        ${{data.answer.replace(/\\n/g,'<br>')}}
        ${{sources}}
        <div class="fb-row">
          <span style="font-size:12px;color:#64748b">Was this helpful?</span>
          <button class="fb-btn" onclick="fb('up')">👍</button>
          <button class="fb-btn" onclick="fb('down')">👎</button>
          <span id="fbm" style="font-size:12px;color:#16a34a"></span>
        </div>
      </div>
      <div class="pipeline-trace">
        🔬 <strong>Pipeline:</strong> Retrieved ${{data.used_chunks.length > 0 ? 20 : 0}} candidates → Cross-encoder reranked to ${{data.used_chunks.length}} | AI: {provider} | Method: hybrid_bm25_semantic
      </div>
      <div class="chunks-area">
        <h2 style="margin-top:16px;margin-bottom:10px">📑 Chunks Passed to AI</h2>
        ${{chunkHtml}}
      </div>`;
  }} catch(e) {{
    area.innerHTML = '<div class="answer-box" style="display:block;background:#fef2f2;border-color:#fecaca">❌ Request failed. Is the server running?</div>';
  }}
}}

async function fb(t) {{
  await fetch('/feedback', {{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{question:lastQ,answer:lastA,thumbs:t}})}});
  document.getElementById('fbm').textContent = t==='up' ? '✅ Thanks!' : '✅ Noted, we\'ll improve.';
}}

async function upload(input) {{
  const f = input.files[0]; if (!f) return;
  const msg = document.getElementById('umsg');
  msg.style.color='#64748b'; msg.textContent=`⏳ Uploading ${{f.name}}...`;
  const fd = new FormData(); fd.append('file', f);
  const r  = await fetch('/ingest/file', {{method:'POST',body:fd}});
  const d  = await r.json();
  msg.style.color = r.ok ? '#16a34a' : '#dc2626';
  msg.textContent = r.ok ? `✅ ${{d.message}} (${{d.chunks}} chunks created)` : `❌ ${{d.detail}}`;
  if (r.ok) setTimeout(() => location.reload(), 1500);
}}

async function ingestText() {{
  const c = document.getElementById('tc').value.trim();
  const n = document.getElementById('dn').value.trim();
  if (!c || !n) {{ alert('Please add both text and a document name'); return; }}
  const msg = document.getElementById('umsg');
  msg.style.color='#64748b'; msg.textContent='⏳ Processing text...';
  const r = await fetch('/ingest/text', {{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{content:c,doc_name:n}})}});
  const d = await r.json();
  msg.style.color = r.ok ? '#16a34a' : '#dc2626';
  msg.textContent = r.ok ? `✅ ${{d.message}}` : `❌ ${{d.detail}}`;
  if (r.ok) {{ document.getElementById('tc').value=''; document.getElementById('dn').value=''; setTimeout(()=>location.reload(),1500); }}
}}

async function delDoc(name) {{
  if (!confirm(`Remove '${{name}}' from the knowledge base? This cannot be undone.`)) return;
  const r = await fetch('/documents/'+encodeURIComponent(name), {{method:'DELETE'}});
  if (r.ok) location.reload();
  else alert('Failed to delete document');
}}
</script></body></html>"""
