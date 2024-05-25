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
