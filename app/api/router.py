from fastapi import APIRouter

from app.api.schemas import (
    QueryRequest,
    QueryResponse,
)

from app.container.dependencies import container
from app.exceptions.custom_exceptions import RAGException

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
)
def query(request: QueryRequest):

    answer = container.rag_service.ask(
        question=request.question,
        top_k=request.top_k,
    )

    return QueryResponse(
        answer=answer,
    )


@router.get("/")
def root():
    return {
        "message": "Production RAG API",
        "docs": "/docs",
        "health": "/health",
    }