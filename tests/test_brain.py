# tests/test_brain.py
# Full test suite — no network, no API calls.
# Run: pytest tests/ -v

import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def tmp_db(monkeypatch, tmp_path):
    import retrieval.vector_store as vs
    monkeypatch.setattr(vs, "DB_PATH", str(tmp_path / "test.db"))
    vs.init_store()
    yield


def make_chunks(doc="test_doc", n=5):
    from ingestion.chunker import chunk_text
    return chunk_text(
        "\n\n".join([f"Section {i}: This is paragraph content for testing." for i in range(n)]),
        doc,
    )


# ── Chunker ─────────────────────────────────────────────────

class TestChunker:
    def test_basic_chunking(self):
        from ingestion.chunker import chunk_text
        chunks = chunk_text("Para one.\n\nPara two.\n\nPara three.", "doc")
        assert len(chunks) >= 1

    def test_empty_returns_empty(self):
        from ingestion.chunker import chunk_text
        assert chunk_text("", "doc") == []

    def test_whitespace_returns_empty(self):
        from ingestion.chunker import chunk_text
        assert chunk_text("   \n\n   ", "doc") == []

    def test_doc_name_correct(self):
        from ingestion.chunker import chunk_text
        chunks = chunk_text("A.\n\nB.\n\nC.", "my_policy")
        assert all(c.doc_name == "my_policy" for c in chunks)

    def test_unique_chunk_ids(self):
        from ingestion.chunker import chunk_text
        long = "\n\n".join([f"Para {i}" for i in range(20)])
        ids  = [c.chunk_id for c in chunk_text(long, "doc")]
        assert len(ids) == len(set(ids))

    def test_sequential_indices(self):
        from ingestion.chunker import chunk_text
        chunks = chunk_text("\n\n".join([f"P{i}" for i in range(10)]), "doc")
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_splits_long_text(self):
        from ingestion.chunker import chunk_text
        text   = "\n\n".join(["Word " * 30 for _ in range(10)])
        chunks = chunk_text(text, "doc", chunk_size=200)
        assert len(chunks) >= 2

    def test_content_not_empty(self):
        from ingestion.chunker import chunk_text
        chunks = chunk_text("A.\n\nB.\n\nC.", "doc")
        assert all(c.content.strip() for c in chunks)

    def test_total_chunks_consistent(self):
        from ingestion.chunker import chunk_text
        chunks = chunk_text("\n\n".join([f"P{i}" for i in range(15)]), "doc")
        total  = chunks[0].total_chunks
        assert all(c.total_chunks == total for c in chunks)


# ── Vector store ─────────────────────────────────────────────

class TestVectorStore:
    def test_store_and_stats(self):
        from retrieval.vector_store import store_chunks, get_stats
        chunks = make_chunks()
        store_chunks(chunks, [[0.1] * 10] * len(chunks))
        s = get_stats()
        assert s["total_chunks"] == len(chunks)
        assert s["total_documents"] == 1

    def test_list_documents(self):
        from retrieval.vector_store import store_chunks, list_documents
        c1 = make_chunks("alpha")
        c2 = make_chunks("beta")
        store_chunks(c1, [[0.1] * 10] * len(c1))
        store_chunks(c2, [[0.9] * 10] * len(c2))
        names = [d["doc_name"] for d in list_documents()]
        assert "alpha" in names and "beta" in names

    def test_delete_document(self):
        from retrieval.vector_store import store_chunks, delete_document, list_documents
        chunks = make_chunks("to_delete")
        store_chunks(chunks, [[0.5] * 10] * len(chunks))
        n = delete_document("to_delete")
        assert n == len(chunks)
        assert "to_delete" not in [d["doc_name"] for d in list_documents()]

    def test_feedback_saved(self):
        from retrieval.vector_store import save_feedback, get_top_questions
        save_feedback("What is the refund policy?", "30 days", "up")
        save_feedback("What is the refund policy?", "30 days", "up")
        save_feedback("How do I cancel?", "Email us", "down")
        top = get_top_questions()
        q   = next((q for q in top if q["question"] == "What is the refund policy?"), None)
        assert q is not None
        assert q["count"] == 2
        assert q["thumbs_up"] == 2

    def test_stats_empty(self):
        from retrieval.vector_store import get_stats
        s = get_stats()
        assert s["total_chunks"] == 0
        assert s["total_documents"] == 0

    def test_doc_filter(self):
        from retrieval.vector_store import store_chunks, hybrid_search
        c1 = make_chunks("policy_doc")
        c2 = make_chunks("hr_doc")
        store_chunks(c1, [[1.0] * 10] * len(c1))
        store_chunks(c2, [[0.1] * 10] * len(c2))
        results = hybrid_search("test", [1.0] * 10, top_k=10, doc_filter="policy_doc")
        assert all(r.doc_name == "policy_doc" for r in results)


# ── Hybrid search ─────────────────────────────────────────────

class TestHybridSearch:
    def test_returns_results(self):
        from retrieval.vector_store import store_chunks, hybrid_search
        chunks = make_chunks()
        store_chunks(chunks, [[1.0, 0.0] * 5] * len(chunks))
        results = hybrid_search("test query", [1.0, 0.0] * 5, top_k=3)
        assert len(results) > 0

    def test_empty_store(self):
        from retrieval.vector_store import hybrid_search
        assert hybrid_search("q", [0.5] * 10, top_k=3) == []

    def test_sorted_by_hybrid_score(self):
        from retrieval.vector_store import store_chunks, hybrid_search
        chunks = make_chunks()
        store_chunks(chunks, [[float(i) / 10] * 10 for i in range(len(chunks))])
        results = hybrid_search("test", [1.0] * 10, top_k=5)
        scores  = [r.hybrid_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_hybrid_score_range(self):
        from retrieval.vector_store import store_chunks, hybrid_search
        chunks = make_chunks()
        store_chunks(chunks, [[0.5] * 10] * len(chunks))
        for r in hybrid_search("test", [0.5] * 10):
            assert 0.0 <= r.hybrid_score <= 1.0


# ── Cosine similarity ─────────────────────────────────────────

class TestCosine:
    def test_identical(self):
        from retrieval.vector_store import _cosine
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine(v, v) - 1.0) < 1e-6

    def test_orthogonal(self):
        from retrieval.vector_store import _cosine
        assert abs(_cosine([1, 0, 0], [0, 1, 0])) < 1e-6

    def test_opposite(self):
        from retrieval.vector_store import _cosine
        assert abs(_cosine([1, 0], [-1, 0]) + 1.0) < 1e-6

    def test_zero_vector(self):
        from retrieval.vector_store import _cosine
        assert _cosine([1, 2, 3], [0, 0, 0]) == 0.0


# ── Reranker ──────────────────────────────────────────────────

class TestReranker:
    def _candidates(self, n=5):
        from retrieval.vector_store import SearchResult
        return [
            SearchResult(chunk_id=str(i), doc_name="doc",
                         content=f"Sentence {i} about the refund policy.",
                         semantic_score=float(i)/10, bm25_score=float(i)/10,
                         hybrid_score=float(i)/10)
            for i in range(n)
        ]

    def test_empty_returns_empty(self):
        from retrieval.reranker import rerank
        assert rerank("q", [], top_k=4) == []

    def test_respects_top_k(self, monkeypatch):
        import builtins, retrieval.reranker as rr
        real = builtins.__import__
        def mock(name, *a, **kw):
            if name == "sentence_transformers": raise ImportError
            return real(name, *a, **kw)
        monkeypatch.setattr(builtins, "__import__", mock)
        monkeypatch.setattr(rr, "_model", None)
        result = rr.rerank("test", self._candidates(10), top_k=4)
        assert len(result) == 4

    def test_fallback_uses_hybrid_score(self, monkeypatch):
        import builtins, retrieval.reranker as rr
        real = builtins.__import__
        def mock(name, *a, **kw):
            if name == "sentence_transformers": raise ImportError
            return real(name, *a, **kw)
        monkeypatch.setattr(builtins, "__import__", mock)
        monkeypatch.setattr(rr, "_model", None)
        cands  = self._candidates(5)
        result = rr.rerank("test", cands, top_k=3)
        scores = [r.hybrid_score for r in result]
        assert scores == sorted(scores, reverse=True)
