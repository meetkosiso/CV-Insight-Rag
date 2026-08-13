# CV Insight Rag

Intelligent semantic search, question-answering, and insight generation over resumes and CV documents using modern Retrieval-Augmented Generation (RAG).

Built with **FastAPI** + **LangChain** + **Ollama** + **Chroma** + Hybrid Search & Reranking

## ✨ Features

- Semantic + Keyword Hybrid Search (0.7 vector / 0.3 BM25)
- FlashRank Reranking for improved relevance
- Stable, content-aware chunk IDs that survive re-indexing
- Incremental indexing with upsert (no full rebuilds)
- Fully local & offline-first (Ollama + HuggingFace embeddings)
- FastAPI REST API with clean dependency injection
- Production-oriented patterns (process-level singletons, proper error handling)

## 🛠️ Tech Stack

| Component           | Technology                             | Purpose                                |
| ------------------- | -------------------------------------- | -------------------------------------- |
| Backend             | FastAPI                                | High-performance API                   |
| LLM                 | Ollama (local)                         | Answer generation                      |
| Embeddings          | HuggingFace sentence-transformers      | Dense vector generation                |
| Vector Store        | Chroma                                 | Persistent local vector database       |
| Retrieval           | LangChain (Ensemble + Compression)     | Hybrid search + reranking              |
| Reranker            | FlashRank                              | Fast & lightweight relevance reranking |
| Document Processing | PyPDF + RecursiveCharacterTextSplitter | PDF loading & intelligent chunking     |

## 🚀 Quick Start

### Prerequisites

- Python 3.9–3.12
- Ollama is installed and running locally with a model pulled  
  (example: `ollama pull llama3.1:8b` or `qwen2.5:7b`)
- Some PDF resumes/CVs are placed in the `docs/` folder

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/advance-cv-insight.git
cd advance-cv-insight
```

### Install dependencies

```
pip install -r requirements.txt
```

### Paths

```
DOCS_PATH=./docs
CHROMA_PATH=./chroma_db
```

### LLM & Embeddings

```
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.1:8b
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

### (Optional) Manually trigger indexing when documents change

```
python -m app.rag.indexer
```

### Start the FastAPI server

```
uvicorn app.main:app --reload --port 8000
```

### Run all tests

```
pytest
```

### Run with coverage

```
pytest --cov=app --cov-report=term-missing
```

### Run specific tests

```
pytest tests/test_indexer.py
pytest tests/api/
```
