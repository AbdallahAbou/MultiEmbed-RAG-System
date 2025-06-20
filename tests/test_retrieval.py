"""Tests for retrieval components."""
import pytest
import numpy as np


class TestFAISSVectorStore:
    """Tests for FAISS vector store."""
    
    def test_add_and_search(self):
        from src.retrieval import FAISSVectorStore
        
        store = FAISSVectorStore(dimension=3)
        
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        texts = ["doc1", "doc2", "doc3"]
        store.add(embeddings, texts)
        
        assert store.size == 3
    
    def test_search_returns_results(self):
        from src.retrieval import FAISSVectorStore
        
        store = FAISSVectorStore(dimension=3)
        
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        
        store.add(embeddings, ["doc1", "doc2"])
        
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=1)
        
        assert len(results) == 1
        assert results[0][2] == "doc1"
