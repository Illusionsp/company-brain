"""
Embedding module using sentence-transformers.
Generates local embeddings for document chunks and user queries.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)
_local_model = None


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts. Returns list of float vectors."""
    if not texts:
        return []
    return _local(texts)


def _local(texts: List[str]) -> List[List[float]]:
    """
    Local sentence-transformers implementation.
    Model: all-MiniLM-L6-v2 (Dimensions: 384)
    """
    global _local_model
    try:
        from sentence_transformers import SentenceTransformer
        if _local_model is None:
            logger.info("Loading all-MiniLM-L6-v2 (~90MB, first time only)...")
            _local_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded ✅")
        return _local_model.encode(texts, show_progress_bar=False).tolist()
    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        return [[0.0] * 384 for _ in texts]
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return [[0.0] * 384 for _ in texts]
