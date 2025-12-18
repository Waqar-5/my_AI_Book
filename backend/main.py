#!/usr/bin/env python3
"""
Main script for the Website Embeddings Deployment & Vector DB Storage pipeline.

This script orchestrates the full pipeline:
1. Discover all URLs from a Docusaurus site
2. Extract clean text from each page
3. Chunk the content into appropriate sizes
4. Generate embeddings using Cohere
5. Create the "RAG_Embedding" collection in Qdrant
6. Store all embeddings with metadata
"""

import argparse
import logging
import sys
import os
import time
from typing import List, Dict, Any
import uuid

# Add the backend directory to Python path for proper imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.extraction.content_extractor import ContentExtractor, get_all_urls
from src.embedding.embedding_generator import EmbeddingGenerator
from src.storage.qdrant_client import QdrantStorage
from src.utils.helpers import chunk_text, setup_logger
from config.settings import settings

# Set up logging for the main module
logger = setup_logger(__name__)


def process_single_page(
    url: str,
    extractor: ContentExtractor,
    embedder: EmbeddingGenerator,
    storage: QdrantStorage,
    chunk_size: int,
    chunk_overlap: int,
    dry_run: bool,
    reprocess: bool
) -> int:
    """
    Process a single page: extract content, chunk it, generate embeddings, and store.

    Args:
        url: The URL of the page to process
        extractor: ContentExtractor instance
        embedder: EmbeddingGenerator instance
        storage: QdrantStorage instance
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        dry_run: Whether to skip storing to Qdrant
        reprocess: Whether to reprocess existing content

    Returns:
        Number of chunks processed from this page
    """
    logger.info(f"Processing: {url}")

    # Extract content from the URL
    content_data = extractor.extract_text_from_url(url)
    if not content_data:
        logger.warning(f"Could not extract content from {url}, skipping...")
        return 0

    # Chunk the content
    raw_content = content_data['raw_content']
    chunks = chunk_text(raw_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not chunks:
        logger.warning(f"No content to process for {url}")
        return 0

    # Generate embeddings for each chunk
    embeddings = embedder.embed(chunks)
    chunks_stored = 0

    # Store each chunk with its embedding
    for j, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        # Validate the embedding
        if not embedder.validate_embedding_quality(embedding, chunk_text):
            logger.warning(f"Invalid embedding for chunk {j} of page {url}, skipping...")
            continue

        # Create chunk ID using URL and chunk index to prevent duplicates
        chunk_id = f"{url}#{j}"

        # Skip if already exists and reprocess flag is not set
        if not reprocess and not dry_run and storage.check_duplicate(chunk_id):
            logger.info(f"Chunk {chunk_id} already exists, skipping...")
            continue

        # Prepare metadata
        metadata = {
            'source_url': url,
            'page_title': content_data['page_title'],
            'chunk_index': j,
            'text_content': chunk_text,
            'created_at': time.time()
        }

        # Store in Qdrant if not in dry run mode
        if not dry_run:
            success = storage.save_chunk_to_qdrant(chunk_id, embedding, metadata)
            if success:
                logger.debug(f"Stored chunk {chunk_id} successfully")
                chunks_stored += 1
            else:
                logger.error(f"Failed to store chunk {chunk_id}")
        else:
            logger.info(f"Dry run: Would store chunk {chunk_id}")
            chunks_stored += 1

    return chunks_stored


def main():
    parser = argparse.ArgumentParser(
        description='Embedding Pipeline for Docusaurus Book Content'
    )
    parser.add_argument(
        '--url',
        type=str,
        default=settings.source_url,
        help='Base URL of the Docusaurus site to process (default from settings)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=settings.chunk_size,
        help='Size of text chunks for embedding (default from settings)'
    )
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=settings.chunk_overlap,
        help='Overlap between text chunks (default from settings)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run the pipeline without storing embeddings to Qdrant'
    )
    parser.add_argument(
        '--reprocess',
        action='store_true',
        help='Reprocess content even if it already exists in Qdrant'
    )

    args = parser.parse_args()

    logger.info(f"Starting embedding pipeline for URL: {args.url}")

    # Initialize components
    extractor = ContentExtractor()
    embedder = EmbeddingGenerator()
    storage = QdrantStorage()

    # Validate Qdrant connection before starting
    if not storage.validate_connection():
        logger.error("Cannot connect to Qdrant. Please check your configuration.")
        return

    # Create the collection in Qdrant
    if not args.dry_run:
        if not storage.create_collection():
            logger.error("Failed to create Qdrant collection. Exiting.")
            return

    # Get all URLs from the site
    urls = get_all_urls(args.url)
    logger.info(f"Discovered {len(urls)} URLs to process")

    # Process each URL
    total_chunks = 0
    processed_pages = 0

    start_time = time.time()

    for i, url in enumerate(urls):
        logger.info(f"Processing page {i+1}/{len(urls)}")

        chunks_from_page = process_single_page(
            url, extractor, embedder, storage,
            args.chunk_size, args.chunk_overlap,
            args.dry_run, args.reprocess
        )

        if chunks_from_page > 0:
            processed_pages += 1
            total_chunks += chunks_from_page

    end_time = time.time()
    duration = end_time - start_time

    logger.info(f"Pipeline completed!")
    logger.info(f"Processed {processed_pages} pages")
    logger.info(f"Created {total_chunks} content chunks")
    logger.info(f"Total duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")

    # Final validation: verify content is stored correctly
    if not args.dry_run:
        # Simple validation by checking collection size
        try:
            collection_info = storage.client.get_collection(settings.collection_name)
            logger.info(f"Final collection size: {collection_info.points_count} vectors")
        except Exception as e:
            logger.error(f"Could not verify collection size: {e}")


if __name__ == "__main__":
    main()