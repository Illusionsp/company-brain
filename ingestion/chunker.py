"""
Document parsing and chunking module.
Handles text extraction and overlapping text segmentation.
"""
import uuid, logging
from typing import List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id:     str
    doc_name:     str
    content:      str
    chunk_index:  int
    total_chunks: int
    metadata:     dict = field(default_factory=dict)


def chunk_text(text: str, doc_name: str, chunk_size=500, overlap=50) -> List[Chunk]:
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    raw, current = [], ""

    for para in paragraphs:
        if current and len(current) + len(para) + 2 > chunk_size:
            raw.append(current.strip())
            current = current[-overlap:] + "\n\n" + para if overlap else para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        raw.append(current.strip())

    total  = len(raw)
    chunks = [
        Chunk(chunk_id=str(uuid.uuid4()), doc_name=doc_name, content=t,
              chunk_index=i, total_chunks=total, metadata={"doc_name": doc_name})
        for i, t in enumerate(raw)
    ]
    avg = sum(len(c.content) for c in chunks) // max(total, 1)
    logger.info(f"'{doc_name}' → {total} chunks (avg {avg} chars)")
    return chunks


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        import fitz
        doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n\n".join(p.get_text() for p in doc)
        doc.close()
        return text
    except ImportError:
        logger.error("Run: pip install pymupdf")
        return ""
    except Exception as e:
        logger.error(f"PDF error: {e}")
        return ""
