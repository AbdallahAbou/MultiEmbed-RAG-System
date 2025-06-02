"""Tests for embedding models."""
import pytest
import numpy as np


class TestEmbeddingLevel:
    """Tests for EmbeddingLevel dataclass."""
    
    def test_embedding_level_creation(self):
        from src.embeddings import EmbeddingLevel
        
        level = EmbeddingLevel(
            name="test",
            embeddings=np.array([[1.0, 2.0]]),
            texts=["test text"],
            metadata=[{"key": "value"}]
        )
        
        assert level.name == "test"
        assert len(level.texts) == 1


class TestMultiLevelEmbedder:
    """Tests for MultiLevelEmbedder class."""
    
    def test_split_into_sentences(self):
        from src.embeddings import MultiLevelEmbedder
        
        embedder = MultiLevelEmbedder()
        sentences = embedder._split_into_sentences("Hello world. How are you?")
        
        assert len(sentences) >= 2
    
    def test_split_into_paragraphs(self):
        from src.embeddings import MultiLevelEmbedder
        
        embedder = MultiLevelEmbedder()
        text = "First paragraph.\n\nSecond paragraph."
        paragraphs = embedder._split_into_paragraphs(text)
        
        assert len(paragraphs) == 2
