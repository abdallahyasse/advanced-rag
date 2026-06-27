from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.rag_pipeline import AdvancedRAG

app = FastAPI(title="Advanced RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = AdvancedRAG()


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3


@app.get("/")
def health():
    return {
        "status": "ok",
        "ready": rag.ready,
        "documents": rag.doc_count,
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted.")
    contents = await file.read()
    try:
        chunk_count = rag.ingest(contents, filename=file.filename)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {
        "message": f"{file.filename} ingested.",
        "chunks": chunk_count,
        "total_documents": rag.doc_count,
    }


@app.post("/ask")
def ask(req: QuestionRequest):
    if not rag.ready:
        raise HTTPException(400, "No PDF uploaded yet.")
    result = rag.ask(req.question, top_k=req.top_k)
    return result