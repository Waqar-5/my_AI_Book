import logging
import sys
import os
from typing import List, Tuple
import re
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
import time
import random
from functools import wraps

# Add the backend directory to Python path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Function to setup a logger with specified name and level.
    """
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')

    if log_file:
        handler = logging.FileHandler(log_file)
    else:
        handler = logging.StreamHandler()
    
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def exponential_backoff(max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 60.0):
    """
    Decorator that implements exponential backoff for function retries.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    logging.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def get_all_urls(base_url: str) -> List[str]:
    """
    Discovers all URLs from a Docusaurus site using sitemap.xml.
    
    Args:
        base_url: The base URL of the Docusaurus site
        
    Returns:
        List of all discovered URLs
    """
    urls = set()
    
    # Try to get URLs from sitemap.xml
    sitemap_url = urljoin(base_url, 'sitemap.xml')
    try:
        response = requests.get(sitemap_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'xml')
            for loc in soup.find_all('loc'):
                url = loc.text.strip()
                if url.startswith(base_url):
                    urls.add(url)
    except Exception as e:
        logging.warning(f"Could not fetch sitemap.xml from {sitemap_url}: {e}")
    
    # If sitemap is not available or returns no URLs, try robots.txt
    if not urls:
        robots_url = urljoin(base_url, 'robots.txt')
        try:
            response = requests.get(robots_url)
            if response.status_code == 200:
                # Look for sitemap references in robots.txt
                for line in response.text.splitlines():
                    if line.lower().startswith('sitemap:'):
                        sitemap_ref = line.split(':', 1)[1].strip()
                        sitemap_url = urljoin(base_url, sitemap_ref)
                        
                        response = requests.get(sitemap_url)
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.content, 'xml')
                            for loc in soup.find_all('loc'):
                                url = loc.text.strip()
                                if url.startswith(base_url):
                                    urls.add(url)
        except Exception as e:
            logging.warning(f"Could not fetch robots.txt from {robots_url}: {e}")
    
    # Fallback: if still no URLs, we can try basic crawling from the main page
    # but for now we'll return what we have
    if not urls:
        urls.add(base_url)
    
    return list(urls)

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """
    Splits text into overlapping chunks to maintain semantic coherence.
    
    Args:
        text: The input text to chunk
        chunk_size: The maximum size of each chunk
        chunk_overlap: The overlap between consecutive chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If we're at the end of the text, take the remaining part
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to break at sentence boundary if possible
        chunk = text[start:end]
        last_sentence_end = -1
        
        # Look for sentence ending punctuation within the chunk
        for punct in ['.', '!', '?', ';']:
            last_idx = chunk.rfind(punct)
            if last_idx != -1 and last_idx > last_sentence_end:
                last_sentence_end = last_idx
        
        # If we found a sentence boundary and it's not too far from the end
        if last_sentence_end != -1 and last_sentence_end > chunk_size - chunk_overlap * 2:
            end = start + last_sentence_end + 1
            chunks.append(text[start:end])
            start = end - chunk_overlap  # Apply overlap
        else:
            # If no good sentence boundary, just take the chunk_size
            chunks.append(text[start:end])
            start = end - chunk_overlap
        
        # Ensure we make progress even if no sentence boundaries
        if start <= end - chunk_overlap:
            start = end - chunk_overlap
        else:
            start = end
    
    # Remove any empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]
    
    return chunks

def extract_text_from_html(html_content: str) -> str:
    """
    Extracts clean text from HTML content, removing navigation and UI elements.
    
    Args:
        html_content: HTML content to extract text from
        
    Returns:
        Clean text content
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
        script.decompose()
    
    # Attempt to find main content areas specific to Docusaurus
    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'container|main|content|doc')) or soup
    
    # Get text content and clean it up
    text = main_content.get_text(separator=' ')
    
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    return text