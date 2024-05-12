"""
Embedding model wrappers for different backends.
"""
from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._model = None
    
    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass
    
    @abstractmethod
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode text(s) into embeddings."""
        pass

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        raise NotImplementedError


class SentenceTransformerModel(EmbeddingModel):
    """Sentence-Transformers based embedding model."""
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "cuda"
    ):
        super().__init__(model_name, device)
        self._dimension = None
    
    def load(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._dimension = self._model.get_sentence_embedding_dimension()
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if self._model is None:
            self.load()
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings
    
    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self.load()
        return self._dimension
