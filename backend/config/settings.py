import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional

# Load environment variables from .env file
load_dotenv()

class Settings(BaseModel):
    """
    Application settings loaded from environment variables.
    """
    cohere_api_key: str = Field(..., description="Cohere API key for embedding generation")
    qdrant_api_key: str = Field(..., description="Qdrant API key for vector database access")
    qdrant_url: str = Field(..., description="Qdrant cluster URL")
    source_url: str = Field(default="https://my-ai-book-gamma.vercel.app/", description="Source URL for the Docusaurus book")
    collection_name: str = Field(default="RAG_Embedding", description="Name of the Qdrant collection")
    chunk_size: int = Field(default=1000, description="Size of text chunks for embedding")
    chunk_overlap: int = Field(default=100, description="Overlap between text chunks")
    
    class Config:
        # This allows using environment variables as fallbacks
        env_file = ".env"
        case_sensitive = True

    @classmethod
    def load_from_env(cls):
        """
        Load settings from environment variables.
        """
        return cls(
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_url=os.getenv("QDRANT_URL"),
            source_url=os.getenv("SOURCE_URL", "https://my-ai-book-gamma.vercel.app/"),
            collection_name=os.getenv("COLLECTION_NAME", "RAG_Embedding"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        )

# Global settings instance
settings = Settings.load_from_env()