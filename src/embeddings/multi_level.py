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
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Handle multiple sentence delimiters including Arabic
        pattern = r'(?<=[.!?؟،])\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks
