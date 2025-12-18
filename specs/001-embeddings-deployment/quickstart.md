# Quickstart Guide: Website Embeddings Deployment & Vector DB Storage

## Overview
This guide provides instructions to quickly set up and run the embedding pipeline that extracts content from Docusaurus-based book websites, generates embeddings using Cohere models, and stores them in Qdrant vector database.

## Prerequisites
- Python 3.11+
- UV package manager
- Cohere API key
- Qdrant Cloud account and API key

## Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Initialize the Backend
```bash
# Navigate to the backend directory
cd backend

# Create a virtual environment using UV
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install project dependencies
uv pip install -r requirements.txt
# Or if using pyproject.toml
uv pip install -e .
```

### 4. Configure Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your credentials
nano .env
```

Add your credentials:
```
COHERE_API_KEY=your_cohere_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_cluster_url
SOURCE_URL=https://my-ai-book-gamma.vercel.app/
COLLECTION_NAME=RAG_Embedding
```

## Usage

### 1. Run the Full Pipeline
Execute the complete pipeline: fetch URLs, extract text, chunk content, generate embeddings, and store in Qdrant.

```bash
python main.py
```

The script will:
1. Discover all URLs from the Docusaurus site
2. Extract clean text from each page
3. Chunk the content into appropriate sizes
4. Generate embeddings using Cohere
5. Create the "RAG_Embedding" collection in Qdrant
6. Store all embeddings with metadata

### 2. Run with Custom Parameters
```bash
# With custom chunk size and overlap
python main.py --chunk-size 1000 --chunk-overlap 100

# For a different URL
python main.py --url https://different-book-site.com
```

## Verification

### 1. Check Collection in Qdrant
After running the pipeline, verify that:
- The "RAG_Embedding" collection exists in your Qdrant instance
- The collection contains the expected number of vectors
- Each vector has the correct metadata (source_url, page_title, chunk_index)

### 2. Test Query
You can also run a simple test query to verify the embeddings are working:

```python
from src.storage.qdrant_client import QdrantStorage

# Initialize the storage client
storage = QdrantStorage()
# Query for similar content
results = storage.search("your search query", top_k=5)
print(results)
```

## Troubleshooting

### Common Issues
1. **Rate Limiting**: If you get rate limit errors from Cohere, the pipeline includes backoff mechanisms but you might need to adjust the rate limits in your Cohere account.

2. **Invalid URLs**: If certain URLs return 404s, check that the source website hasn't changed its structure.

3. **Memory Issues**: For very large sites, you may need to adjust the chunk size parameters to process smaller pieces of content at a time.

## Next Steps
- Integrate with your RAG chatbot application
- Set up scheduled runs for content updates
- Monitor embedding quality and adjust chunking strategy as needed