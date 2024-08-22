"""
Reranking models for improving retrieval quality.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np


class Reranker(ABC):
    """Abstract base class for rerankers."""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[int, float, str, Dict[str, Any]]],
        top_k: int = 10
    ) -> List[Tuple[int, float, str, Dict[str, Any]]]:
        """Rerank candidates based on query relevance."""
        pass


class CrossEncoderReranker(Reranker):
    """Cross-encoder based reranker for accurate relevance scoring."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
    
    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_name, device=self.device)
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[int, float, str, Dict[str, Any]]],
        top_k: int = 10
    ) -> List[Tuple[int, float, str, Dict[str, Any]]]:
        """Rerank candidates using cross-encoder scores."""
        if self._model is None:
            self._load_model()
        
        if not candidates:
            return []
        
        pairs = [(query, candidate[2]) for candidate in candidates]
        scores = self._model.predict(pairs)
        
        reranked = [
            (candidate[0], float(score), candidate[2], candidate[3])
            for candidate, score in zip(candidates, scores)
        ]
        
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
