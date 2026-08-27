# Company Brain

**An enterprise-grade internal AI knowledge base.**

Company Brain is a production-ready RAG (Retrieval-Augmented Generation) application designed to automate internal team support. By uploading SOPs, policies, and manuals, teams can ask questions via a web dashboard or Telegram and receive instant, hallucination-free answers with exact source citations.

## 🚀 Key Features

- **Advanced RAG Pipeline**: Implements Hybrid Search (BM25 keyword + Semantic embeddings) for high-recall retrieval.
- **Cross-Encoder Reranking**: Re-ranks the top 20 candidates down to the top 4 using a cross-encoder model to ensure highest contextual accuracy.
- **Full-Stack Interface**: Features a native Python Streamlit UI for web interactions and a Telegram Bot for mobile accessibility.
- **Strict Grounding**: The LLM is strictly prompted to only answer based on retrieved context, gracefully returning "I don't know" to prevent hallucinations.
- **Production-Ready Backend**: API powered by FastAPI, containerized with Docker, running behind a Gunicorn ASGI process manager.

## 🧠 System Architecture

```text
User Query
   │
   ▼
Query Embedding (sentence-transformers)
   │
   ▼
Hybrid Retrieval (BM25 + Semantic Search) ───► Top 20 Candidates
   │
   ▼
Cross-Encoder Reranker (ms-marco-MiniLM)  ───► Top 4 Chunks
   │
   ▼
LLM Generation (Groq/Gemini/Anthropic)    ───► Grounded Answer + Citations
```

## 🛠️ Technology Stack

- **Backend Framework**: FastAPI
- **Frontend UI**: Streamlit
- **ML / NLP**: `sentence-transformers`, `cross-encoder`
- **Infrastructure**: Docker, Gunicorn
- **Testing**: `pytest` (26 passing tests covering core logic)

## 📁 Project Structure

```text
company-brain/
├── api/                 # FastAPI application and endpoints
├── ingestion/           # Document parsing and chunking logic
├── retrieval/           # Hybrid search, reranking, and RAG pipeline
├── bot/                 # Telegram bot integration
├── tests/               # Comprehensive unit test suite
├── streamlit_app.py     # Streamlit web dashboard
├── docker-compose.yml   # Multi-container orchestration
└── Dockerfile           # Production container definition
```

## ⚙️ Local Development Setup

To run this project locally:

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Add your preferred LLM API Key (e.g., GROQ_API_KEY)
   ```
3. **Run the Streamlit Dashboard**
   ```bash
   streamlit run streamlit_app.py
   ```
   *Dashboard will be available at http://localhost:8501*

4. **Run the API (Optional)**
   ```bash
   uvicorn api.main:app --reload
   ```

## 🐳 Docker Deployment

To spin up the entire stack (API + Telegram Bot) in an isolated container:

```bash
docker-compose up -d
```

## 🧪 Testing

The project includes a comprehensive test suite that runs entirely offline (no network calls required) in under 2 seconds.

```bash
pytest tests/ -v
```
