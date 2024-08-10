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
