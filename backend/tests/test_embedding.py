import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the backend/src directory to the Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.embedding.embedding_generator import EmbeddingGenerator

class TestEmbeddingGenerator:
    @patch('cohere.Client')
    def test_embed_single_text(self, mock_cohere_client):
        """Test embedding generation for a single text"""
        # Mock the Cohere client response
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3] + [0.0] * 1021]  # 1024-dimensional vector
        mock_cohere_client.return_value.embed.return_value = mock_response
        
        generator = EmbeddingGenerator()
        embedding = generator.embed_single("Test text for embedding")
        
        assert len(embedding) == 1024
        assert embedding[0] == 0.1
        assert embedding[1] == 0.2
        assert embedding[2] == 0.3
    
    @patch('cohere.Client')
    def test_embed_multiple_texts(self, mock_cohere_client):
        """Test embedding generation for multiple texts"""
        # Mock the Cohere client response
        mock_response = MagicMock()
        mock_response.embeddings = [
            [0.1, 0.2, 0.3] + [0.0] * 1021,  # 1024-dimensional vector
            [0.4, 0.5, 0.6] + [0.0] * 1021   # 1024-dimensional vector
        ]
        mock_cohere_client.return_value.embed.return_value = mock_response
        
        generator = EmbeddingGenerator()
        embeddings = generator.embed(["Test text 1", "Test text 2"])
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1024
        assert len(embeddings[1]) == 1024
    
    def test_validate_embedding_quality_valid(self):
        """Test validation of a valid embedding"""
        generator = EmbeddingGenerator()
        valid_embedding = [0.1] * 1024  # 1024-dimensional vector
        
        result = generator.validate_embedding_quality(valid_embedding, "Test text")
        assert result is True
    
    def test_validate_embedding_quality_invalid_dimension(self):
        """Test validation of an embedding with wrong dimension"""
        generator = EmbeddingGenerator()
        invalid_embedding = [0.1] * 100  # Wrong dimension
        
        result = generator.validate_embedding_quality(invalid_embedding, "Test text")
        assert result is False
    
    def test_validate_embedding_quality_non_finite_values(self):
        """Test validation of an embedding with non-finite values"""
        generator = EmbeddingGenerator()
        invalid_embedding = [float('inf')] + [0.1] * 1023  # Contains infinity
        
        result = generator.validate_embedding_quality(invalid_embedding, "Test text")
        assert result is False