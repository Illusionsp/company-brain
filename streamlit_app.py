import streamlit as st
import asyncio
import os
from pathlib import Path
from config import settings
from ingestion.chunker import chunk_text, extract_text_from_pdf
from ingestion.embedder import embed_texts
from retrieval.vector_store import (
    init_store, store_chunks, hybrid_search,
    list_documents, delete_document, get_stats,
    save_feedback, get_top_questions
)
from retrieval.reranker import rerank
from retrieval.rag_pipeline import generate_answer, clear_session

# --- Initialization ---
st.set_page_config(page_title="Company Brain", page_icon="🧠", layout="wide")

@st.cache_resource
def setup_environment():
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    init_store()

setup_environment()

# Ensure we have a session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = "streamlit_session"

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: Knowledge Base Management ---
with st.sidebar:
    st.title("🧠 Company Brain")
    st.write("Upload your SOPs, policies, manuals and FAQs — your team asks questions, AI answers instantly.")
    
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload PDF, TXT, or MD", type=["pdf", "txt", "md"])
    if uploaded_file:
        with st.spinner("Processing document..."):
            fname = uploaded_file.name
            ext = Path(fname).suffix.lower()
            content = uploaded_file.read()
            
            if content:
                text = extract_text_from_pdf(content) if ext == ".pdf" else content.decode("utf-8", "ignore")
                doc_name = Path(fname).stem
                chunks = chunk_text(text, doc_name, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
                
                if chunks:
                    embs = asyncio.run(embed_texts([c.content for c in chunks]))
                    store_chunks(chunks, embs)
                    st.success(f"'{doc_name}' indexed successfully! ({len(chunks)} chunks)")
                else:
                    st.error("No extractable text found.")

    st.header("📚 Knowledge Base")
    docs = list_documents()
    if docs:
        for d in docs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{d['doc_name']}** ({d['chunks']} chunks)")
            with col2:
                if st.button("🗑️", key=f"del_{d['doc_name']}", help="Delete Document"):
                    delete_document(d['doc_name'])
                    st.rerun()
    else:
        st.info("No documents uploaded yet.")
        
    st.header("📊 Stats")
    stats = get_stats()
    st.metric("Total Documents", stats["total_documents"])
    st.metric("Knowledge Chunks", stats["total_chunks"])
    
    if st.button("Clear Conversation"):
        clear_session(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()

# --- Main Chat Interface ---
st.title("💬 Ask a Question")

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption(f"**Sources:** {', '.join(msg['sources'])}")

# Chat input
if prompt := st.chat_input("What is our refund policy?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    stats = get_stats()
    if stats["total_chunks"] == 0:
        st.error("No documents indexed. Upload a document first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching documents & reranking..."):
                # Embed query
                q_embs = asyncio.run(embed_texts([prompt]))
                q_vec = q_embs[0]
                
                # Hybrid Search
                candidates = hybrid_search(prompt, q_vec, top_k=settings.INITIAL_TOP_K)
                
                if not candidates:
                    answer = "I couldn't find relevant information in the documents."
                    sources = []
                    used_chunks = []
                else:
                    # Rerank
                    reranked = rerank(prompt, candidates, top_k=4)
                    sources = list(dict.fromkeys(r.doc_name for r in reranked))
                    used_chunks = reranked
                    
                    # Generate Answer
                    answer = asyncio.run(generate_answer(prompt, reranked, st.session_state.session_id))
            
            st.markdown(answer)
            if sources:
                st.caption(f"**Sources:** {', '.join(sources)}")
                
            with st.expander("Pipeline Trace (Exact Chunks Used)"):
                for i, c in enumerate(used_chunks):
                    st.write(f"**Chunk {i+1} from `{c.doc_name}`**")
                    st.write(f"*Semantic: {c.semantic_score:.3f} | BM25: {c.bm25_score:.3f} | Hybrid: {c.hybrid_score:.3f} | Rerank: {c.rerank_score:.3f}*")
                    st.info(c.content)

        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "sources": sources
        })
