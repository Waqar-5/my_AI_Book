import logging
import sys
import os
from typing import List, Dict, Optional
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

# Add the backend directory to Python path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.helpers import extract_text_from_html, get_all_urls, setup_logger

logger = setup_logger(__name__)

class ContentExtractor:
    """
    Class for extracting content from Docusaurus-based book websites.
    """
    
    def __init__(self):
        self.session = requests.Session()
        # Set a user agent to avoid being blocked by some websites
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def extract_text_from_url(self, url: str) -> Optional[Dict[str, str]]:
        """
        Extract clean text from a single URL.

        Args:
            url: The URL to extract content from

        Returns:
            Dictionary with 'title' and 'content' keys, or None if extraction fails
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            # Extract title from the HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else 'No Title'

            # Extract clean text content
            text_content = extract_text_from_html(response.text)

            # Validate content - ensure it doesn't contain HTML tags, navigation, or UI elements
            if self._validate_extracted_content(text_content):
                # Log successful extraction
                logger.info(f"Successfully extracted content from {url} - Title: {title[:50]}...")

                return {
                    'source_url': url,
                    'page_title': title,
                    'raw_content': text_content
                }
            else:
                logger.warning(f"Content validation failed for {url} - potentially contains unwanted elements")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching URL {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return None

    def _validate_extracted_content(self, content: str) -> bool:
        """
        Validate that extracted content is clean and doesn't contain HTML elements.

        Args:
            content: The extracted content to validate

        Returns:
            Boolean indicating if content is valid
        """
        # Check if content is not empty
        if not content or len(content.strip()) < 10:
            return False

        # Check for common HTML tags that should have been removed
        html_tags = ['<html', '<body', '<div', '<nav', '<header', '<footer', '<script', '<style']
        for tag in html_tags:
            if tag in content.lower():
                return False  # Contains HTML tags

        # Content looks clean
        return True

    def extract_all_content(self, base_url: str) -> List[Dict[str, str]]:
        """
        Extract content from all pages discovered on the Docusaurus site.
        
        Args:
            base_url: The base URL of the Docusaurus site
            
        Returns:
            List of dictionaries with extracted content
        """
        logger.info(f"Starting content extraction from {base_url}")
        
        # Get all URLs from the site
        urls = get_all_urls(base_url)
        logger.info(f"Discovered {len(urls)} URLs to process")
        
        all_content = []
        failed_urls = []
        
        for i, url in enumerate(urls):
            logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")
            content = self.extract_text_from_url(url)
            
            if content:
                all_content.append(content)
            else:
                failed_urls.append(url)
        
        logger.info(f"Extraction completed. {len(all_content)} URLs processed successfully, {len(failed_urls)} failed")
        
        if failed_urls:
            logger.warning(f"Failed to extract content from the following URLs: {failed_urls}")
        
        return all_content

# Standalone function for backward compatibility with the original design
def extract_text_from_url(url: str) -> Optional[Dict[str, str]]:
    """
    Standalone function to extract clean text from a single URL.
    """
    extractor = ContentExtractor()
    return extractor.extract_text_from_url(url)

def get_all_urls(base_url: str) -> List[str]:
    """
    Standalone function to get all URLs from a Docusaurus site.
    """
    from src.utils.helpers import get_all_urls as helper_get_all_urls
    return helper_get_all_urls(base_url)  # Using the utility function