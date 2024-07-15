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
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the vector store from disk."""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for efficient similarity search."""
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        self.dimension = dimension
        self.index_type = index_type
        self._index = None
        self._texts: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
        self._init_index()
    
    def _init_index(self) -> None:
        """Initialize the FAISS index."""
        import faiss
        
        if self.index_type == "flat":
            self._index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dimension)
            self._index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
