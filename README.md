---
title: "Production Agentic RAG API"
emoji: "🚀"
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🚀 Production Agentic RAG API

A modular Retrieval-Augmented Generation (RAG) system built with Python, FastAPI, FAISS, BM25, SentenceTransformers, Cross-Encoder reranking, and FLAN-T5.

The project combines hybrid document retrieval with an LLM tool-calling agent and an Agentic RAG workflow designed to improve answer reliability through context evaluation, query rewriting, controlled retrieval retries, and final answer validation.

## ✨ Key Features

### 1. Advanced RAG Pipeline

- PDF ingestion and document chunking
- SentenceTransformer embeddings
- Dense retrieval using FAISS
- Sparse retrieval using BM25
- Hybrid search with Reciprocal Rank Fusion (RRF)
- Cross-Encoder reranking
- FLAN-T5 answer generation
- Source-aware document retrieval

### 2. Agentic RAG

- **Reflection:** Analyzes questions and helps determine whether rewriting is needed.
- **Context-Aware Query Rewriting:** Improves questions using conversation context when necessary.
- **Retrieval Evaluation:** Evaluates whether the retrieved context sufficiently supports an answer.
- **Search Query Rewriting:** Improves search queries when initial retrieval is insufficient.
- **Adaptive Retrieval:** Adjusts retrieval depth during retries within configured limits.
- **Retry Control:** Limits repeated retrieval attempts.
- **Retrieval Guard:** Detects repeated retrieval results and prevents ineffective retries.
- **Final Answer Guard:** Returns an insufficient-information response when the available evidence cannot support a reliable answer.
- **Conversation Memory:** Stores interactions to support conversational workflows.
- **Source-Aware Responses:** Provides retrieved document information alongside the answer.

### 3. LLM Tool-Calling Agent

The project includes a modular tool framework with:

- Calculator tool
- PDF tool
- RAG tool
- Tool registry
- Planner and executor components
- Agent output and response handling

### 4. API and Production Infrastructure

- FastAPI REST API
- Swagger and OpenAPI documentation
- Health-check endpoint
- Dependency injection
- Centralized configuration
- Structured logging and exception handling
- Persistent FAISS and BM25 index support
- Docker-based deployment configuration

## 🏗️ Architecture

```text
                         User Question
                               |
                               v
                      Agentic RAG Workflow
                               |
                    Reflection / Query Rewrite
                               |
                               v
                       Hybrid Retrieval
                         /           \
                        v             v
                      FAISS          BM25
                         \           /
                          v         v
                    Reciprocal Rank Fusion
                               |
                               v
                      Cross-Encoder Reranking
                               |
                               v
                       Retrieved Context
                               |
                               v
                       Context Evaluation
                         /           \
                        v             v
                  Sufficient      Insufficient
                      |                |
                      v                v
                  Generation     Search Query Rewrite
                      |                |
                      v                v
                Answer Guard      Controlled Retry
                      |                |
                      +-------<--------+
                      |
                      v
                Final Answer + Sources
```

The workflow uses bounded retries and retrieval-result checks to avoid repeatedly executing ineffective searches. Final answer validation helps prevent unsupported responses when the available documents do not contain sufficient evidence.

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| Language | Python |
| API | FastAPI, Uvicorn |
| Dense Retrieval | FAISS |
| Sparse Retrieval | BM25, rank-bm25 |
| Embeddings | SentenceTransformers |
| Reranking | Cross-Encoder |
| Generation | Hugging Face Transformers, FLAN-T5 |
| LLM Integration | Groq API |
| Persistence | FAISS and BM25 indexes |
| Deployment | Docker, Docker Compose |
| API Documentation | Swagger, OpenAPI |

## 📁 Project Structure

```text
app/
├── agent/             # Tool-calling agent
│   └── tools/         # Calculator, PDF, and RAG tools
├── agentic_rag/       # Reflection, evaluation, rewriting, retries
├── api/               # API routes
├── config/            # Application settings
├── container/         # Dependency injection
├── embeddings/        # Embedding components
├── generation/        # Answer generation
├── index/             # Index management
├── langgraph_agent/   # LangGraph-related components
├── llm/               # LLM client and models
├── memory/            # Conversation memory
├── persistence/       # Index persistence
├── reranker/          # Cross-Encoder reranking
├── retrieval/         # BM25 and hybrid retrieval
├── services/          # RAG services
├── vectorstore/       # FAISS vector store
├── ingestion.py       # Document ingestion
└── main.py            # Application entry point
```

## ⚙️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/abdallahyasse/advanced-rag.git
cd advanced-rag
```

### 2. Create a Virtual Environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root and configure the variables supported by your application.

Example:

```env
PDF_PATH=data/Abdullah.yasser.pdf
INDEX_DIRECTORY=indexes

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
LLM_MODEL=google/flan-t5-base

CHUNK_SIZE=500
CHUNK_OVERLAP=50

PLANNER_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=your_groq_api_key
```

Ensure the PDF path and environment variable names match your actual configuration. Never commit real API keys or secrets.

### 5. Run the API

```bash
python -m uvicorn app.main:app --reload
```

Local endpoints:

- API: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs
- OpenAPI schema: http://127.0.0.1:8000/openapi.json

### 6. Health Check

```http
GET /health
```

The response depends on the health-check implementation in the application.

## 🐳 Docker

Build the application:

```bash
docker compose build
```

Start the services:

```bash
docker compose up
```

Open Swagger UI at http://localhost:8000/docs.

Ensure required environment variables, documents, and persistent storage are configured for your deployment environment.

## 🧪 Testing

Run the available tests:

```bash
pytest
```

If a dedicated test directory exists:

```bash
pytest tests/
```

Run tests in your current environment before considering a release complete.

## 🔍 Reliability and Evaluation

The Agentic RAG workflow is designed to handle both answerable and unsupported questions.

Test scenarios include:

- Questions answerable from indexed documents
- Questions that benefit from query rewriting
- Questions requiring another retrieval attempt
- Questions for which the documents lack sufficient evidence
- Repeated retrieval results that should stop further retries

For unsupported questions, the final answer guard is intended to return an insufficient-information response rather than accept an unsupported generated answer.

## 🚀 Roadmap

- [x] Hybrid retrieval using FAISS and BM25
- [x] Cross-Encoder reranking
- [x] Production-oriented FastAPI architecture
- [x] Tool-calling agent framework
- [x] Agentic RAG workflow with evaluation and bounded retries
- [x] Retrieval guard and final answer validation
- [ ] GraphRAG and knowledge graph integration
- [ ] Expanded automated evaluation and regression testing
- [ ] Semantic caching
- [ ] Further monitoring and observability
- [ ] Production cloud deployment
- [ ] Advanced agent workflows and MCP integration

## 👨‍💻 Author

**Abdalla Yasser** — AI Engineer

- GitHub: https://github.com/abdallahyasse
- LinkedIn: https://www.linkedin.com/in/abdullah-yasser-6a7748183/
