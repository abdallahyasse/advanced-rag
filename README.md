---
title: Production RAG API
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🚀 Production RAG API

A production-ready Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **FAISS**, **BM25**, **CrossEncoder Reranker**, and **FLAN-T5**.

The project supports PDF ingestion, hybrid retrieval, semantic search, reranking, Docker deployment, and a production-ready REST API.

---

# ✨ Features

- 📄 PDF Ingestion
- ✂️ Automatic Text Chunking
- 🧠 SentenceTransformer Embeddings
- 🔍 FAISS Dense Vector Search
- 🔎 BM25 Sparse Keyword Search
- ⚡ Hybrid Retrieval
- 🎯 CrossEncoder Reranker
- 🤖 FLAN-T5 Text Generation
- 🌐 FastAPI REST API
- 📚 Swagger Documentation
- 📝 Production Logging
- ⚠️ Global Exception Handling
- ❤️ Health Check Endpoint
- 💾 Persistent FAISS & BM25 Indexes
- 🐳 Docker & Docker Compose Support

---

# 🏗 System Architecture

```text
                User
                  │
                  ▼
             FastAPI API
                  │
                  ▼
             RAG Service
                  │
                  ▼
       Hybrid Retrieval Service
          │               │
          ▼               ▼
      FAISS Search    BM25 Search
          │               │
          └───────┬───────┘
                  ▼
         CrossEncoder Reranker
                  ▼
            Top Documents
                  ▼
            FLAN-T5 Generator
                  ▼
                Answer
```

---

# 📂 Project Structure

```text
Advanced-RAG/
│
├── app/
│   ├── api/
│   ├── config/
│   ├── container/
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
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
└── README.md
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/abdallahyasse/advanced-rag.git

cd advanced-rag
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file

```env
PDF_PATH=data/Abdullah.yasser.pdf

INDEX_DIRECTORY=indexes

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

LLM_MODEL=google/flan-t5-base

CHUNK_SIZE=500

CHUNK_OVERLAP=50
```

---

# Running the API

```bash
python -m uvicorn app.main:app --reload
```

Swagger UI

```
http://127.0.0.1:8000/docs
```

Health Endpoint

```
http://127.0.0.1:8000/health
```

---

# Docker

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

# API Example

## POST `/query`

Request

```json
{
    "question":"What is Machine Learning?",
    "top_k":3
}
```

Response

```json
{
    "answer":"Machine Learning is a branch of Artificial Intelligence."
}
```

---

# Tech Stack

- Python
- FastAPI
- FAISS
- Rank-BM25
- Sentence Transformers
- CrossEncoder
- Transformers
- FLAN-T5
- Docker
- Docker Compose

---

# Production Features

- Hybrid Retrieval
- Semantic Search
- Reranking
- Dependency Injection
- Configuration Management
- Logging
- Exception Handling
- Health Monitoring
- Persistent Index Storage
- Docker Deployment

---

# Future Improvements

- Multi-file Upload API
- Streaming Responses
- Conversation Memory
- ChromaDB / Milvus Support
- Redis Cache
- Authentication
- Prometheus Metrics
- CI/CD Pipeline
- Cloud Deployment (Azure / AWS / GCP)

---

# Author

**Abdalla Yasser**

AI Engineer

GitHub:
https://github.com/abdallahyasse

LinkedIn:
https://www.linkedin.com/in/abdullah-yasser-6a7748183