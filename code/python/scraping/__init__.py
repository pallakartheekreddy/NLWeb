"""
NLWeb Scraping Module

This module provides tools for web content extraction and processing, including:
- URL extraction from sitemaps
- Web page crawling and HTML extraction
- Schema.org markup extraction
- PDF text extraction and chunking
- Content processing for vector databases

Main Components:
- urlsFromSitemap: Extract URLs from XML sitemaps
- expBackOffCrawl: Robust web crawling with retry logic
- extractMarkup: Extract structured data from HTML
- pdf_extractor: Extract and chunk text from PDF files
"""

from .urlsFromSitemap import extract_urls_from_sitemap, process_site_or_sitemap, get_sitemaps_from_robots
from .expBackOffCrawl import SimpleCrawler
from .extractMarkup import process_directory, extract_schema_markup, extract_canonical_url

# PDF text extraction and chunking
from .pdf_extractor import (
    PDFTextExtractor,
    TextChunk,
    PDFStructure
)

__all__ = [
    'extract_urls_from_sitemap',
    'process_site_or_sitemap',
    'get_sitemaps_from_robots',
    'SimpleCrawler',
    'process_directory',
    'extract_schema_markup',
    'extract_canonical_url',
    # PDF extraction classes and functions
    'PDFTextExtractor',
    'TextChunk',
    'PDFStructure'
]