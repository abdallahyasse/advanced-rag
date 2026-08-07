---
title: 🚀 Production Agentic RAG API
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🚀 Production RAG API

A production-ready Agentic Retrieval-Augmented Generation (RAG) system built with FastAPI, FAISS, BM25, SentenceTransformers, CrossEncoder, FLAN-T5, Reflection Agents, Conversation Memory, Query Rewriting, Retrieval Evaluation and Retry Pipeline.

The project follows a modular AI architecture inspired by modern Agentic RAG systems with clean dependency injection, persistent indexes, memory, evaluation loops, Docker deployment and scalable components..

---

# ✨ Features

### Core RAG

- 📄 PDF Ingestion
- ✂️ Automatic Chunking
- 🧠 SentenceTransformer Embeddings
- 🔍 FAISS Dense Retrieval
- 🔎 BM25 Sparse Retrieval
- ⚡ Hybrid Search
- 🎯 CrossEncoder Reranking
- 🤖 FLAN-T5 Generation

### Agentic RAG

- 🧠 Reflection Agent
- ✍️ Query Rewriter
- 🔍 Search Query Rewriter
- 📊 Retrieval Evaluation
- 🔁 Automatic Retrieval Retry
- 💬 Conversation Memory
- 📚 Context Builder

### Agent Framework

- 🛠 Tool Registry
- 📄 PDF Tool
- 🔢 Calculator Tool
- 🤖 RAG Tool
- 🧩 Planner
- ⚙️ Executor
- 📤 Output Parser

### LangGraph

- 🌐 LangGraph Ready
- 🧠 Graph State
- 🔀 Nodes & Edges

### API

- 🌐 FastAPI
- 📚 Swagger
- ❤️ Health Endpoint

### Production

- 📝 Structured Logging
- ⚠️ Exception Handling
- 💾 Persistent Indexes
- 🐳 Docker
- 🧪 Unit Tests
- 🏗 Clean Architecture

---

# 🏗 Architecture

```text
                    User
                      │
                      ▼
             Agentic RAG Workflow
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
   Reflection Agent        Conversation Memory
          │                       │
          └───────────┬───────────┘
                      ▼
              Query Rewriter
                      ▼
             Hybrid Retrieval
              ┌──────────────┐
              │              │
              ▼              ▼
          FAISS         BM25 Search
              └──────┬──────┘
                     ▼
          CrossEncoder Reranker
                     ▼
             FLAN-T5 Generator
                     ▼
          Retrieval Evaluation
                     │
      enough? ───────┴──────── no
          │                    │
         yes                   ▼
          │            Search Query Rewriter
          │                    │
          └────────────Retry────┘
                     ▼
                Final Answer

---

```text
app/
│
├── agent/
│
├── agentic_rag/
│
├── api/
│
├── config/
│
├── container/
│
├── embeddings/
│
├── generation/
│
├── index/
│
├── langgraph_agent/
│
├── llm/
│
├── logging/
│
├── memory/
│
├── persistence/
│
├── reranker/
│
├── retrieval/
│
├── services/
│
├── vectorstore/
│
└── main.py
```

---

# ⚙ Installation

## Clone Repository

```bash
git clone https://github.com/abdallahyasse/advanced-rag.git
cd advanced-rag
```

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## Install Requirements

```bash
pip install -r requirements.txt
```

---

# ⚙ Environment Variables

Create a `.env` file:

```env
PDF_PATH=data/Abdullah.yasser.pdf

INDEX_DIRECTORY=indexes

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

LLM_MODEL=google/flan-t5-base

CHUNK_SIZE=500

CHUNK_OVERLAP=50
```env
PLANNER_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=your_key_here
```

---

# ▶ Running the API

```bash
python -m uvicorn app.main:app --reload
```

API:

```
http://127.0.0.1:8000
```

---

# 📚 API Documentation

Swagger

```
http://127.0.0.1:8000/docs
```

OpenAPI

```
http://127.0.0.1:8000/openapi.json
```

---

# ❤️ Health Check

```http
GET /health
```

Response

```json
{
  "status": "healthy",
  "service": "Production RAG API",
  "version": "1.0.0"
}
```

---

# 🚀 Root Endpoint

```http
GET /
```

```json
{
  "message": "Production RAG API",
  "docs": "/docs",
  "health": "/health"
}
```

---

# 🔍 Query Endpoint

```http
POST /query
```

Request

```json
{
  "question": "What is Machine Learning?",
  "top_k": 3
}
```

Response

```json
{
  "answer": "Machine Learning is a branch of Artificial Intelligence."
}
```

---

# 🐳 Docker

Build

```bash
docker compose build
```

Run

```bash
docker compose up
```

Swagger

```
http://localhost:8000/docs
```

---

# 🧪 Run Tests

```bash
pytest
```

or

```bash
pytest tests/
```

---

# 🛠 Tech Stack

- Python
- FastAPI
- FAISS
- BM25
- SentenceTransformers
- CrossEncoder
- Transformers
- FLAN-T5
- Groq API
- LangGraph
- Docker
- Pytest

---

# 🏆 Production Features

## Retrieval

- ✅ Dense Retrieval
- ✅ Sparse Retrieval
- ✅ Hybrid Search
- ✅ CrossEncoder Reranking

## Agentic RAG

- ✅ Reflection
- ✅ Query Rewriting
- ✅ Retrieval Evaluation
- ✅ Automatic Retry
- ✅ Conversation Memory
- ✅ Context Builder

## AI Agent

- ✅ Tool Framework
- ✅ Planner
- ✅ Executor
- ✅ LangGraph Ready

## Infrastructure

- ✅ Dependency Injection
- ✅ Docker
- ✅ Persistent Indexes
- ✅ Logging
- ✅ Exception Handling
- ✅ REST API
- ✅ Swagger

---
# 🚧 Roadmap

- ✅ Production RAG
- ✅ Agentic RAG
- 🔄 Full Tool Calling Agent
- 🔄 GraphRAG
- 🔄 Semantic Cache
- 🔄 LangGraph Agent
- 🔄 Multi-Agent System
- 🔄 Browser Agent
- 🔄 Text-to-SQL
- 🔄 MCP Server / Client
- 🔄 LLMOps Platform
- 🔄 Kubernetes
- 🔄 AWS Deployment
---

# 👨‍💻 Author

**Abdalla Yasser**

AI Engineer

**GitHub**

https://github.com/abdallahyasse

**LinkedIn**

https://www.linkedin.com/in/abdullah-yasser-6a7748183