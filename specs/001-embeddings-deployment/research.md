# Research Summary: Website Embeddings Deployment & Vector DB Storage

## Overview
This document summarizes the research and decisions made for implementing the embedding pipeline to extract content from Docusaurus-based book websites, generate embeddings using Cohere models, and store them in Qdrant vector database.

## Key Decisions

### 1. Content Extraction Approach
**Decision**: Use BeautifulSoup4 with Requests for content extraction from Docusaurus websites
**Rationale**: BeautifulSoup4 provides reliable HTML parsing capabilities and can effectively extract text content while filtering out navigation and UI elements specific to Docusaurus sites. It's a well-established Python library for web scraping and content extraction.

**Alternatives considered**:
- Selenium: More resource-intensive, better for JavaScript-heavy sites
- Newspaper3k: Designed for news articles, may not work well with documentation sites
- Custom regex: Less reliable and harder to maintain

### 2. Chunking Strategy
**Decision**: Implement recursive character text splitting with overlap
**Rationale**: This approach maintains semantic coherence while ensuring chunks fit within Cohere's token limits. It allows for some overlap to preserve context across chunk boundaries.

**Parameters**:
- Chunk size: 1000 characters
- Chunk overlap: 100 characters

**Alternatives considered**:
- Sentence-based splitting: May result in chunks of uneven sizes
- Fixed-length splitting: May break semantic context in the middle of concepts

### 3. Qdrant Collection Design
**Decision**: Create a collection with 1024-dimensional vectors (matching Cohere embedding dimensions)
**Rationale**: Cohere's multilingual-22-12 model outputs 1024-dimensional vectors, which is suitable for most embedding tasks while being efficient to store and query.

### 4. Metadata Storage
**Decision**: Store source URL, page title, and chunk index as payload in Qdrant
**Rationale**: These fields are essential for retrieving the original context of embeddings during RAG operations, as specified in the functional requirements.

**Fields**:
- source_url: URL of the original page
- page_title: Title of the page
- chunk_index: Index of the chunk within the page
- text_content: The actual text chunk (for verification purposes)

### 5. URL Discovery for Docusaurus Sites
**Decision**: Use sitemap.xml and robots.txt for URL discovery
**Rationale**: Docusaurus sites typically generate sitemap.xml files with all public pages, making it the most reliable way to discover all content pages.

**Alternative approaches**:
- Manual URL specification: Less scalable
- Web crawling: More complex and potentially missing content

### 6. Error Handling & Retries
**Decision**: Implement exponential backoff for Cohere API calls and Qdrant operations
**Rationale**: Both services may have rate limits or temporary outages. Using retry mechanisms with exponential backoff ensures robust operation.

### 7. Duplicate Prevention
**Decision**: Use URL and chunk index as a composite identifier for deduplication
**Rationale**: This ensures that re-running the pipeline won't create duplicate entries, satisfying FR-008 requirement.

## Technology Stack Best Practices

### Python Development
- Use Pydantic for data validation
- Use python-dotenv for environment management
- Follow PEP 8 guidelines
- Use type hints throughout

### Cohere Integration
- Use Cohere's Python SDK
- Implement proper error handling for API responses
- Validate response formats before processing

### Qdrant Integration
- Use Qdrant's Python client
- Implement proper connection management
- Use batch operations for efficient data insertion