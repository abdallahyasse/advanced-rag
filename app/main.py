from fastapi import FastAPI

from app.api.router import router

from app.exceptions.handlers import (
    rag_exception_handler,
    generic_exception_handler,
)

from app.exceptions.custom_exceptions import (
    RAGException,
)

from app.middleware.logging_middleware import LoggingMiddleware


app = FastAPI(
    title="Production RAG API",
    version="1.0.0",
)

# Middleware
app.add_middleware(
    LoggingMiddleware,
)

# Routers
app.include_router(router)

# Exception Handlers
app.add_exception_handler(
    RAGException,
    rag_exception_handler,
)

app.add_exception_handler(
    Exception,
    generic_exception_handler,
)