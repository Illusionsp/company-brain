# retrieval/vector_store.py
# Hybrid BM25 + semantic search with SQLite.
# Also stores feedback and session history for analytics.

import json, math, sqlite3, logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from ingestion.chunker import Chunk

logger  = logging.getLogger(__name__)
DB_PATH = "data/company_brain.db"


@dataclass
class SearchResult:
    chunk_id:       str
    doc_name:       str
    content:        str
    semantic_score: float
    bm25_score:     float
    hybrid_score:   float
    rerank_score:   float = 0.0


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    ma  = math.sqrt(sum(x * x for x in a))
    mb  = math.sqrt(sum(x * x for x in b))
    return dot / (ma * mb) if ma and mb else 0.0


def _bm25(query_terms, doc, avg_dl, n_docs, df, k1=1.5, b=0.75):
    words    = doc.lower().split()
    dl       = len(words)
    tf_map   = {}
    for w in words:
        tf_map[w] = tf_map.get(w, 0) + 1
    score = 0.0
    for term in query_terms:
        tf  = tf_map.get(term.lower(), 0)
        idf = math.log((n_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1)
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
    return score


def init_store():
    Path("data").mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id   TEXT PRIMARY KEY,
            doc_name   TEXT NOT NULL,
            content    TEXT NOT NULL,
            embedding  TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            question   TEXT,
            answer     TEXT,
            thumbs     TEXT,
            session_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_doc ON chunks(doc_name);
    """)
    conn.commit(); conn.close()
    logger.info("Vector store initialised")


def store_chunks(chunks: List[Chunk], embeddings: List[List[float]]):
    conn = sqlite3.connect(DB_PATH)
    for ch, emb in zip(chunks, embeddings):
        conn.execute("INSERT OR REPLACE INTO chunks (chunk_id,doc_name,content,embedding) VALUES (?,?,?,?)",
            (ch.chunk_id, ch.doc_name, ch.content, json.dumps(emb)))
    conn.commit(); conn.close()
    logger.info(f"Stored {len(chunks)} chunks")


def hybrid_search(
    query: str,
    query_embedding: List[float],
    top_k: int = 20,
    doc_filter: Optional[str] = None,
    semantic_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[SearchResult]:
    conn = sqlite3.connect(DB_PATH)
    q    = "SELECT chunk_id,doc_name,content,embedding FROM chunks"
    rows = conn.execute(q + " WHERE doc_name=?" if doc_filter else q,
                        (doc_filter,) if doc_filter else ()).fetchall()
    conn.close()
    if not rows: return []

    terms  = query.lower().split()
    docs   = [r[2] for r in rows]
    avg_dl = sum(len(d.split()) for d in docs) / len(docs)
    df     = {t: sum(1 for d in docs if t in d.lower()) for t in terms}

    sems, bms25s, results = [], [], []
    for cid, dn, content, emb_j in rows:
        sem  = _cosine(query_embedding, json.loads(emb_j))
        bm   = _bm25(terms, content, avg_dl, len(docs), df)
        sems.append(sem); bms25s.append(bm)
        results.append(SearchResult(chunk_id=cid, doc_name=dn, content=content,
                                    semantic_score=sem, bm25_score=bm, hybrid_score=0.0))

    sm = max(sems)  or 1.0
    bm = max(bms25s) or 1.0
    for r, s, b in zip(results, sems, bms25s):
        r.semantic_score = round(s, 4)
        r.bm25_score     = round(b, 4)
        r.hybrid_score   = round(semantic_weight * s/sm + bm25_weight * b/bm, 4)

    results.sort(key=lambda x: x.hybrid_score, reverse=True)
    return results[:top_k]


def list_documents() -> List[dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""SELECT doc_name, COUNT(*) as chunks, MAX(created_at)
        FROM chunks GROUP BY doc_name ORDER BY MAX(created_at) DESC""").fetchall()
    conn.close()
    return [{"doc_name": r[0], "chunks": r[1], "added_at": r[2]} for r in rows]


def delete_document(doc_name: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.execute("DELETE FROM chunks WHERE doc_name=?", (doc_name,))
    conn.commit(); n = cur.rowcount; conn.close()
    return n


def get_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    ch   = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    docs = conn.execute("SELECT COUNT(DISTINCT doc_name) FROM chunks").fetchone()[0]
    fb   = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    conn.close()
    return {"total_chunks": ch, "total_documents": docs, "total_feedback": fb}


def save_feedback(question: str, answer: str, thumbs: str, session_id: str = ""):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO feedback (question,answer,thumbs,session_id) VALUES (?,?,?,?)",
                 (question, answer[:200], thumbs, session_id))
    conn.commit(); conn.close()


def get_top_questions(limit: int = 10) -> List[dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""SELECT question, COUNT(*) as cnt,
        SUM(CASE WHEN thumbs='up' THEN 1 ELSE 0 END),
        SUM(CASE WHEN thumbs='down' THEN 1 ELSE 0 END)
        FROM feedback GROUP BY question ORDER BY cnt DESC LIMIT ?""", (limit,)).fetchall()
    conn.close()
    return [{"question": r[0], "count": r[1], "thumbs_up": r[2], "thumbs_down": r[3]} for r in rows]
