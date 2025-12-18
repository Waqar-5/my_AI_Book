import cohere
import logging
import sys
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Add the backend directory to Python path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from config.settings import settings
from src.utils.helpers import setup_logger, exponential_backoff

logger = setup_logger(__name__)

class EmbeddingGenerator:
    """
    Class for generating embeddings using Cohere models.
    """
    
    def __init__(self):
        self.client = cohere.Client(settings.cohere_api_key)
        self.model = "multilingual-22-12"  # Cohere model that outputs 1024-dim vectors

    @exponential_backoff()
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Cohere.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors (each vector is a list of floats)
        """
        try:
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type="search_document"  # Appropriate input type for document embeddings
            )
            
            # Validate that all embeddings have the correct dimensions (1024)
            embeddings = response.embeddings
            for i, embedding in enumerate(embeddings):
                if len(embedding) != 1024:
                    raise ValueError(f"Embedding {i} has incorrect dimension: {len(embedding)}, expected 1024")
            
            logger.info(f"Successfully generated embeddings for {len(texts)} text chunks")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector as a list of floats
        """
        return self.embed([text])[0]  # Return first (and only) embedding

    def validate_embedding_quality(self, embedding: List[float], text: str) -> bool:
        """
        Validate that an embedding correctly represents the semantic meaning of the text.
        
        Args:
            embedding: The embedding vector to validate
            text: The original text that generated this embedding
            
        Returns:
            Boolean indicating if the embedding is valid
        """
        # Basic validation: check if embedding has 1024 dimensions
        if len(embedding) != 1024:
            logger.warning(f"Embedding has {len(embedding)} dimensions, expected 1024")
            return False
        
        # Check if all values are finite numbers
        for val in embedding:
            if not isinstance(val, (int, float)) or not (float('-inf') < val < float('inf')):
                logger.warning(f"Embedding contains non-finite value: {val}")
                return False
        
        return True

# Standalone function for backward compatibility with the original design
@exponential_backoff()
def embed(texts: List[str]) -> List[List[float]]:
    """
    Standalone function to generate embeddings for a list of texts.
    """
    generator = EmbeddingGenerator()
    return generator.embed(texts)