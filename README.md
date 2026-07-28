---
title: Production RAG API
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🚀 Production RAG API

A **production-ready Retrieval-Augmented Generation (RAG)** system built with **FastAPI**, **FAISS**, **BM25**, **SentenceTransformers**, **CrossEncoder Reranker**, and **FLAN-T5**.

The project follows a clean, scalable architecture with dependency injection, persistent indexes, Docker deployment, logging, exception handling, and unit testing.

---

# ✨ Features

- 📄 PDF Ingestion
- ✂️ Automatic Text Chunking
- 🧠 SentenceTransformer Embeddings
- 🔍 FAISS Dense Vector Search
- 🔎 BM25 Sparse Retrieval
- ⚡ Hybrid Search
- 🎯 CrossEncoder Reranker
- 🤖 FLAN-T5 Text Generation
- 🌐 FastAPI REST API
- 📚 Interactive Swagger Documentation
- ❤️ Health Check Endpoint
- 📝 Structured Logging
- ⚠️ Global Exception Handling
- 💾 Persistent FAISS & BM25 Indexes
- 🐳 Docker & Docker Compose
- 🧪 Unit Testing
- 🏗 Clean Production Architecture

---

# 🏗 Architecture

```

User
│
▼
FastAPI API
│
▼
Dependency Container
│
▼
RAG Service
│
▼
Hybrid Retrieval
│
├──────────────┐
▼ ▼
FAISS BM25
│ │
└──────┬───────┘
▼
CrossEncoder Reranker
▼
FLAN-T5 Generator
▼
Final Answer
📂 Project Structure
Advanced-RAG/
│
├── app/
│   ├── api/
│   ├── config/
│   ├── container/
│   ├── embeddings/
│   ├── exceptions/
│   ├── generation/
│   ├── index/
│   ├── logging/
│   ├── middleware/
│   ├── persistence/
│   ├── reranker/
│   ├── retrieval/
│   ├── services/
│   ├── vectorstore/
│   └── main.py
│
├── data/
├── indexes/
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
⚙ Installation
Clone Repository
git clone https://github.com/abdallahyasse/advanced-rag.git

cd advanced-rag
Create Virtual Environment
python -m venv venv

Windows

venv\Scripts\activate

Linux / macOS

source venv/bin/activate
Install Requirements
pip install -r requirements.txt
⚙ Environment Variables

Create a .env

PDF_PATH=data/Abdullah.yasser.pdf

INDEX_DIRECTORY=indexes

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

LLM_MODEL=google/flan-t5-base

CHUNK_SIZE=500

CHUNK_OVERLAP=50
▶ Running the API
python -m uvicorn app.main:app --reload

The API will start at

http://127.0.0.1:8000
📚 API Documentation

Swagger UI

http://127.0.0.1:8000/docs

OpenAPI JSON

http://127.0.0.1:8000/openapi.json
❤️ Health Check
GET /health

Response

{
    "status":"healthy",
    "service":"Production RAG API",
    "version":"1.0.0"
}
🚀 Root Endpoint
GET /

Response

{
    "message":"Production RAG API",
    "docs":"/docs",
    "health":"/health"
}
🔍 Query Endpoint
POST /query

Request

{
    "question":"What is Machine Learning?",
    "top_k":3
}

Example Response

{
    "answer":"Machine Learning is a branch of Artificial Intelligence."
}
🐳 Docker
Build
docker compose build
Run
docker compose up

Swagger

http://localhost:8000/docs
🧪 Run Tests
pytest

or

pytest tests/
🛠 Tech Stack
Python
FastAPI
FAISS
Rank-BM25
Sentence Transformers
CrossEncoder
Transformers
FLAN-T5
Docker
Docker Compose
Pytest
🏆 Production Features
✅ Hybrid Retrieval
✅ Dense + Sparse Search
✅ CrossEncoder Reranking
✅ Dependency Injection
✅ Persistent Vector Index
✅ Configuration Management
✅ Structured Logging
✅ Exception Handling
✅ Docker Deployment
✅ REST API
✅ Swagger Documentation
✅ Unit Tests
👨‍💻 Author

Abdalla Yasser

AI Engineer

GitHub

https://github.com/abdallahyasse

LinkedIn

https://www.linkedin.com/in/abdullah-yasser-6a7748183

هذا الشكل احترافي جدًا ومناسب لـ **GitHub** و**Hugging Face Spaces**، ويعكس أن المشروع Production-ready وليس مجرد مشروع تعليمي.