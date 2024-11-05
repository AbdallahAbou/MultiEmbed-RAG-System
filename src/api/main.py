"""
FastAPI application for RAG system.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os


class QueryRequest(BaseModel):
    """Query request schema."""
    query: str
    top_k: int = 5
    level: str = "sentence"
    rerank: bool = True


class QueryResponse(BaseModel):
    """Query response schema."""
    answer: str
    sources: List[Dict[str, Any]]
    tokens_used: int


class IngestRequest(BaseModel):
    """Document ingestion request schema."""
    documents: List[Dict[str, Any]]
    text_key: str = "text"
    id_key: str = "id"


class IngestResponse(BaseModel):
    """Document ingestion response schema."""
    status: str
    documents_processed: int
    embeddings_created: Dict[str, int]


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="MultiEmbed RAG API",
        description="Multi-level embedding RAG system",
        version="0.1.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    return app
