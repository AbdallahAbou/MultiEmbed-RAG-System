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
