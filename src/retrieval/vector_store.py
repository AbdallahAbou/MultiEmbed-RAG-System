"""
Vector store implementations for similarity search.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add embeddings to the store."""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[int, float, str, Dict[str, Any]]]:
        """Search for similar embeddings."""
        pass
