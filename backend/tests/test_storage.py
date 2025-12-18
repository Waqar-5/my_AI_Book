import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the backend/src directory to the Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.storage.qdrant_client import QdrantStorage

class TestQdrantStorage:
    @patch('qdrant_client.QdrantClient')
    def test_create_collection(self, mock_qdrant_client):
        """Test creating a collection in Qdrant"""
        # Mock the client methods
        mock_instance = Mock()
        mock_qdrant_client.return_value = mock_instance
        mock_instance.get_collections.return_value = Mock()
        mock_instance.get_collections.return_value.collections = []
        
        storage = QdrantStorage()
        result = storage.create_collection()
        
        assert result is True
        mock_instance.create_collection.assert_called_once()
    
    @patch('qdrant_client.QdrantClient')
    def test_save_chunk_to_qdrant(self, mock_qdrant_client):
        """Test saving a chunk to Qdrant"""
        # Mock the client methods
        mock_instance = Mock()
        mock_qdrant_client.return_value = mock_instance
        
        storage = QdrantStorage()
        embedding = [0.1] * 1024  # 1024-dimensional vector
        metadata = {'source_url': 'test.com', 'page_title': 'Test', 'chunk_index': 0}
        
        result = storage.save_chunk_to_qdrant('test-id', embedding, metadata)
        
        assert result is True
        mock_instance.upsert.assert_called_once()
    
    @patch('qdrant_client.QdrantClient')
    def test_search(self, mock_qdrant_client):
        """Test searching in Qdrant"""
        # Mock the client methods and search results
        mock_instance = Mock()
        mock_qdrant_client.return_value = mock_instance
        
        mock_search_result = [
            Mock(id='result1', score=0.9, payload={'source_url': 'test.com', 'page_title': 'Test'})
        ]
        mock_instance.search.return_value = mock_search_result
        
        storage = QdrantStorage()
        query_embedding = [0.1] * 1024
        results = storage.search(query_embedding, top_k=1)
        
        assert len(results) == 1
        assert results[0]['id'] == 'result1'
        assert results[0]['score'] == 0.9
        assert results[0]['payload']['source_url'] == 'test.com'