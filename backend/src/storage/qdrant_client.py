import logging
import sys
import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Add the backend directory to Python path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from config.settings import settings
from src.utils.helpers import setup_logger, exponential_backoff

logger = setup_logger(__name__)

class QdrantStorage:
    """
    Class for storing and retrieving embeddings in Qdrant vector database.
    """
    
    def __init__(self):
        # Initialize Qdrant client with settings
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            prefer_grpc=False  # Using REST for simplicity, can switch to gRPC for better performance
        )
        self.collection_name = settings.collection_name

    def create_collection(self) -> bool:
        """
        Create a collection in Qdrant with appropriate settings for embeddings.
        
        Returns:
            Boolean indicating if the collection was created or already exists
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)
            
            if collection_exists:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return True
            
            # Create new collection with 1024-dimensional vectors (for Cohere embeddings)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            
            logger.info(f"Collection '{self.collection_name}' created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection '{self.collection_name}': {str(e)}")
            return False

    def save_chunk_to_qdrant(self, chunk_id: str, embedding: List[float], metadata: Dict[str, Any]) -> bool:
        """
        Save a single text chunk with its embedding to Qdrant.
        
        Args:
            chunk_id: Unique identifier for the chunk
            embedding: The embedding vector (1024-dimensional for Cohere)
            metadata: Additional metadata to store with the embedding
            
        Returns:
            Boolean indicating success
        """
        try:
            # Prepare the point to be inserted
            point = PointStruct(
                id=chunk_id,
                vector=embedding,
                payload=metadata
            )
            
            # Upsert the point into the collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.debug(f"Successfully saved chunk {chunk_id} to Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Error saving chunk {chunk_id} to Qdrant: {str(e)}")
            return False

    def save_chunks_batch(self, chunk_data: List[Dict[str, Any]]) -> bool:
        """
        Save multiple text chunks with their embeddings to Qdrant in a batch operation.
        
        Args:
            chunk_data: List of dictionaries, each containing 'id', 'embedding', and 'metadata'
            
        Returns:
            Boolean indicating success
        """
        try:
            # Prepare points for batch insertion
            points = []
            for data in chunk_data:
                point = PointStruct(
                    id=data['id'],
                    vector=data['embedding'],
                    payload=data['metadata']
                )
                points.append(point)
            
            # Upsert the points into the collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Successfully saved {len(chunk_data)} chunks to Qdrant in batch operation")
            return True
            
        except Exception as e:
            logger.error(f"Error saving chunks batch to Qdrant: {str(e)}")
            return False

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings in the Qdrant collection.
        
        Args:
            query_embedding: The embedding vector to search for similar ones
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing similar chunks with their metadata
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'id': result.id,
                    'score': result.score,
                    'payload': result.payload
                })
            
            logger.debug(f"Search returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching Qdrant: {str(e)}")
            return []

    @exponential_backoff()
    def validate_connection(self) -> bool:
        """
        Validate the connection to the Qdrant database.
        
        Returns:
            Boolean indicating if the connection is valid
        """
        try:
            # Try to get the list of collections to verify connection
            collections = self.client.get_collections()
            logger.info("Qdrant connection validated successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to validate Qdrant connection: {str(e)}")
            return False

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check if a chunk with the given ID already exists in the collection.
        
        Args:
            chunk_id: The ID to check for duplicates
            
        Returns:
            Boolean indicating if the chunk already exists
        """
        try:
            # Try to retrieve the point by ID
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[chunk_id],
                with_payload=True,
                with_vectors=False
            )
            
            exists = len(points) > 0
            if exists:
                logger.debug(f"Chunk {chunk_id} already exists in collection")
            return exists
        except Exception as e:
            logger.error(f"Error checking for duplicate chunk {chunk_id}: {str(e)}")
            return False  # If there's an error, assume it doesn't exist to avoid blocking

# Standalone functions for backward compatibility with the original design
@exponential_backoff()
def create_collection() -> bool:
    """
    Standalone function to create the RAG_Embedding collection in Qdrant.
    """
    storage = QdrantStorage()
    return storage.create_collection()

def save_chunk_to_qdrant(chunk_id: str, embedding: List[float], metadata: Dict[str, Any]) -> bool:
    """
    Standalone function to save a chunk with its embedding to Qdrant.
    """
    storage = QdrantStorage()
    return storage.save_chunk_to_qdrant(chunk_id, embedding, metadata)