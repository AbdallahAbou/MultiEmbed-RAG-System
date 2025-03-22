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
    
    def add(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Add embeddings to the FAISS index."""
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Normalize for inner product similarity
        faiss_module = __import__('faiss')
        faiss_module.normalize_L2(embeddings)
        
        self._index.add(embeddings)
        self._texts.extend(texts)
        
        if metadata:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([{} for _ in texts])
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple[int, float, str, Dict[str, Any]]]:
        """Search for similar embeddings."""
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        faiss_module = __import__('faiss')
        faiss_module.normalize_L2(query_embedding)
        
        scores, indices = self._index.search(query_embedding, top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self._texts):
                results.append((
                    int(idx),
                    float(score),
                    self._texts[idx],
                    self._metadata[idx]
                ))
        
        return results
    
    def save(self, path: str) -> None:
        """Save the index and metadata to disk."""
        import faiss
        import json
        
        faiss.write_index(self._index, f"{path}.index")
        
        with open(f"{path}.meta.json", 'w') as f:
            json.dump({
                'texts': self._texts,
                'metadata': self._metadata,
                'dimension': self.dimension,
                'index_type': self.index_type
            }, f)
    
    def load(self, path: str) -> None:
        """Load the index and metadata from disk."""
        import faiss
        import json
        
        self._index = faiss.read_index(f"{path}.index")
        
        with open(f"{path}.meta.json", 'r') as f:
            data = json.load(f)
            self._texts = data['texts']
            self._metadata = data['metadata']
            self.dimension = data['dimension']
            self.index_type = data['index_type']
    
    @property
    def size(self) -> int:
        """Return number of vectors in the index."""
        return self._index.ntotal
