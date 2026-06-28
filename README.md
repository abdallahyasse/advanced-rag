# Advanced RAG with Hybrid Search, Reranker & FastAPI

## Overview

This project is an advanced Retrieval-Augmented Generation (RAG) system that combines dense retrieval, keyword search, reranking, and text generation to provide accurate answers from PDF documents.

The system supports multiple PDF files, preserves document metadata (source and page number), and exposes a REST API using FastAPI. The project is fully containerized with Docker.

---

## Features

* Hybrid Search (BM25 + FAISS)
* Dense Embeddings using Sentence Transformers
* Cross-Encoder Reranker
* Multi-PDF Support
* Source & Page Metadata
* FLAN-T5 Text Generation
* FastAPI REST API
* Docker Support

---

## Project Structure

```text
Advanced-RAG/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── data/
├── src/
│   ├── embeddings.py
│   ├── bm25_search.py
│   ├── vector_store.py
│   ├── hybrid_search.py
│   ├── reranker.py
│   ├── ingestion.py
│   ├── generator.py
│   └── rag_pipeline.py
│
├── test_pdf.py
├── test_hybrid.py
└── test_rag.py
```

---

## Architecture

```
PDF Documents
      │
      ▼
Document Loader
      │
      ▼
Chunking
      │
      ▼
SentenceTransformer Embeddings
      │
      ▼
FAISS Vector Database
      │
      ├───────────────┐
      ▼               ▼
Dense Search      BM25 Search
      │               │
      └──────┬────────┘
             ▼
      Hybrid Retrieval
             ▼
 Cross Encoder Reranker
             ▼
      FLAN-T5 Generator
             ▼
         Final Answer
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/abdallahyasse/advanced-rag.git
cd advanced-rag
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the API

```bash
uvicorn app:app --reload
```

API will be available at:

```
http://127.0.0.1:8000
```

Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## Docker

Build the image:

```bash
docker build -t advanced-rag .
```

Run the container:

```bash
docker run -p 8000:8000 advanced-rag
```

---

## Technologies

* Python
* FastAPI
* FAISS
* Sentence Transformers
* Rank-BM25
* Transformers
* FLAN-T5
* Docker

---

## Future Improvements

* Persistent Vector Database (ChromaDB / Milvus)
* Streaming Responses
* Authentication
* CI/CD Pipeline
* Cloud Deployment
* LLM APIs (OpenAI, Gemini, Claude)

---

## Author

**Abdalla Yasser**

Artificial Intelligence Engineer

GitHub: https://github.com/abdallahyasse

LinkedIn: https://www.linkedin.com/in/abdullah-yasser-6a7748183
