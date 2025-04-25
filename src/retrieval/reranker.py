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


class HybridReranker(Reranker):
    """
    Combines multiple signals for reranking:
    - Semantic similarity score
    - Cross-encoder score
    - BM25 lexical score
    """
    
    def __init__(
        self,
        cross_encoder_weight: float = 0.6,
        semantic_weight: float = 0.3,
        bm25_weight: float = 0.1
    ):
        self.cross_encoder_weight = cross_encoder_weight
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self._cross_encoder = CrossEncoderReranker()
    
    def _compute_bm25_scores(
        self,
        query: str,
        documents: List[str]
    ) -> np.ndarray:
        """Compute BM25 scores for documents."""
        from rank_bm25 import BM25Okapi
        
        tokenized_docs = [doc.lower().split() for doc in documents]
        tokenized_query = query.lower().split()
        
        bm25 = BM25Okapi(tokenized_docs)
        return np.array(bm25.get_scores(tokenized_query))
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[int, float, str, Dict[str, Any]]],
        top_k: int = 10
    ) -> List[Tuple[int, float, str, Dict[str, Any]]]:
        """Rerank using hybrid scoring."""
        if not candidates:
            return []
        
        documents = [c[2] for c in candidates]
        semantic_scores = np.array([c[1] for c in candidates])
        
        # Normalize semantic scores
        if semantic_scores.max() > semantic_scores.min():
            semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min())
        
        # BM25 scores
        bm25_scores = self._compute_bm25_scores(query, documents)
        if bm25_scores.max() > bm25_scores.min():
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        
        # Cross-encoder rerank
        ce_reranked = self._cross_encoder.rerank(query, candidates, len(candidates))
        ce_scores = np.array([c[1] for c in ce_reranked])
        if ce_scores.max() > ce_scores.min():
            ce_scores = (ce_scores - ce_scores.min()) / (ce_scores.max() - ce_scores.min())
        
        # Combine scores
        final_scores = (
            self.semantic_weight * semantic_scores +
            self.bm25_weight * bm25_scores +
            self.cross_encoder_weight * ce_scores
        )
        
        reranked = [
            (candidates[i][0], float(final_scores[i]), candidates[i][2], candidates[i][3])
            for i in range(len(candidates))
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked[:top_k]
