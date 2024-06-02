"""
Multi-level embedding generator for hierarchical document representation.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from .models import EmbeddingModel, SentenceTransformerModel


@dataclass
class EmbeddingLevel:
    """Represents a single level of embeddings."""
    name: str
    embeddings: np.ndarray
    texts: List[str]
    metadata: List[Dict[str, Any]]


class MultiLevelEmbedder:
    """
    Generate embeddings at multiple granularity levels:
    - Document level: entire document
    - Paragraph level: logical sections
    - Sentence level: individual sentences
    """
    
    def __init__(
        self,
        model: Optional[EmbeddingModel] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.model = model or SentenceTransformerModel()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._levels: Dict[str, EmbeddingLevel] = {}
