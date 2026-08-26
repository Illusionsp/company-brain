# retrieval/reranker.py
# Cross-encoder reranker.
# Takes (question, chunk) pairs → scores true relevance together.
# Much more accurate than cosine similarity alone.

import logging
from typing import List
from retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)
_model = None


def rerank(query: str, candidates: List[SearchResult], top_k: int = 4) -> List[SearchResult]:
    """Rerank candidates using cross-encoder. Falls back to hybrid score if model unavailable."""
    if not candidates:
        return []
    return _rerank_local(query, candidates, top_k)


def _rerank_local(query: str, candidates: List[SearchResult], top_k: int) -> List[SearchResult]:
    global _model
    try:
        from sentence_transformers import CrossEncoder
        if _model is None:
            logger.info("Loading cross-encoder (~80MB first time)...")
            _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Cross-encoder loaded ✅")

        pairs  = [(query, c.content) for c in candidates]
        scores = _model.predict(pairs)

        for c, s in zip(candidates, scores):
            c.rerank_score = round(float(s), 4)

        candidates.sort(key=lambda x: x.rerank_score, reverse=True)
        logger.info(f"Reranked {len(candidates)} → {top_k} | top: {candidates[0].rerank_score:.4f}")
        return candidates[:top_k]

    except ImportError:
        logger.warning("sentence-transformers not installed — using hybrid scores")
        candidates.sort(key=lambda x: x.hybrid_score, reverse=True)
        return candidates[:top_k]
    except Exception as e:
        logger.warning(f"Rerank failed ({e}) — using hybrid scores")
        candidates.sort(key=lambda x: x.hybrid_score, reverse=True)
        return candidates[:top_k]
