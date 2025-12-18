import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add the backend/src directory to the Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.extraction.content_extractor import ContentExtractor

class TestContentExtractor:
    def test_extract_text_from_html_basic(self):
        """Test basic HTML text extraction"""
        html_content = """
        <html>
            <head><title>Test Title</title></head>
            <body>
                <main>
                    <h1>Main Content</h1>
                    <p>This is a test paragraph.</p>
                    <nav>Navigation content</nav>
                    <footer>Footer content</footer>
                </main>
            </body>
        </html>
        """
        
        from src.utils.helpers import extract_text_from_html
        result = extract_text_from_html(html_content)
        
        # Should contain main content but not navigation or footer
        assert "Main Content" in result
        assert "test paragraph" in result
        # Navigation and footer should be excluded
        assert "Navigation content" not in result
        assert "Footer content" not in result
    
    @patch('requests.Session.get')
    def test_extract_text_from_url_success(self, mock_get):
        """Test successful content extraction from URL"""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <head><title>Test Page</title></head>
            <body><main><p>Test content</p></main></body>
        </html>
        """
        mock_get.return_value = mock_response
        
        extractor = ContentExtractor()
        result = extractor.extract_text_from_url("http://example.com")
        
        assert result is not None
        assert result['page_title'] == 'Test Page'
        assert 'Test content' in result['raw_content']
        assert result['source_url'] == 'http://example.com'
    
    @patch('requests.Session.get')
    def test_extract_text_from_url_failure(self, mock_get):
        """Test content extraction handles request failures"""
        # Mock a request exception
        mock_get.side_effect = Exception("Network error")
        
        extractor = ContentExtractor()
        result = extractor.extract_text_from_url("http://example.com")
        
        assert result is None