#!/usr/bin/env python3
"""
Test suite for PDF text extraction functionality.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add the scraping module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pdf_extractor import PDFTextExtractor, TextChunk, PDFStructure


class TestPDFTextExtractor(unittest.TestCase):
    """Test cases for PDF text extraction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = PDFTextExtractor(min_chunk_tokens=100, max_chunk_tokens=200)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_count_tokens(self):
        """Test token counting functionality."""
        # Test simple text
        text = "This is a simple test sentence."
        tokens = self.extractor.count_tokens(text)
        self.assertGreater(tokens, 0)
        self.assertIsInstance(tokens, int)
        
        # Test empty text
        self.assertEqual(self.extractor.count_tokens(""), 0)
        
        # Test longer text
        long_text = "This is a much longer text that should have more tokens. " * 10
        long_tokens = self.extractor.count_tokens(long_text)
        self.assertGreater(long_tokens, tokens)
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        # Test whitespace normalization
        messy_text = "This  has   multiple    spaces\n\n\n\nand\n  \nlines"
        cleaned = self.extractor._clean_text(messy_text)
        self.assertNotIn("   ", cleaned)  # No triple spaces
        self.assertNotIn("\n\n\n", cleaned)  # No triple newlines
        
        # Test empty text
        self.assertEqual(self.extractor._clean_text(""), "")
        
        # Test None input
        self.assertEqual(self.extractor._clean_text(None), "")
    
    def test_split_into_sentences(self):
        """Test sentence splitting functionality."""
        # Test basic sentence splitting
        text = "This is sentence one. This is sentence two! This is sentence three?"
        sentences = self.extractor.split_into_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertTrue(sentences[0].startswith("This is sentence one"))
        
        # Test abbreviations
        text_with_abbrev = "Dr. Smith went to St. Mary's hospital. He met Prof. Jones there."
        sentences = self.extractor.split_into_sentences(text_with_abbrev)
        # Should not split on "Dr." or "St." or "Prof."
        self.assertLess(len(sentences), 5)  # Should be 2 sentences, not more
        
        # Test empty text
        self.assertEqual(self.extractor.split_into_sentences(""), [])
    
    def test_detect_sections(self):
        """Test section detection functionality."""
        # Test text with clear headings
        text = """
        Introduction
        This is the introduction section.
        
        Chapter 1: Getting Started
        This is chapter one content.
        
        1.1 Basic Concepts
        This covers basic concepts.
        
        CONCLUSION
        This is the conclusion.
        """
        
        sections = self.extractor._detect_sections(text)
        self.assertGreater(len(sections), 0)
        
        # Check that sections are sorted by position
        if len(sections) > 1:
            positions = [s['start_pos'] for s in sections]
            self.assertEqual(positions, sorted(positions))
    
    def test_text_chunk_creation(self):
        """Test TextChunk creation and properties."""
        chunk = TextChunk(
            id="test-123",
            chunk_text="This is test chunk text.",
            page_number=1,
            metadata={"title": "Test Document", "section": "Introduction"},
            token_count=10
        )
        
        self.assertEqual(chunk.id, "test-123")
        self.assertEqual(chunk.page_number, 1)
        self.assertEqual(chunk.metadata["title"], "Test Document")
        self.assertEqual(chunk.token_count, 10)
    
    def test_pdf_structure_creation(self):
        """Test PDFStructure creation and properties."""
        structure = PDFStructure(
            title="Test Document",
            total_pages=10,
            sections=[{"title": "Chapter 1", "start_pos": 0}],
            metadata={"author": "Test Author"}
        )
        
        self.assertEqual(structure.title, "Test Document")
        self.assertEqual(structure.total_pages, 10)
        self.assertEqual(len(structure.sections), 1)
        self.assertEqual(structure.metadata["author"], "Test Author")
    
    def test_create_chunks_logic(self):
        """Test the chunk creation logic with mock data."""
        # Create mock pages data
        pages_data = [
            {
                'page_number': 1,
                'text': "This is the first sentence. This is the second sentence. This is a longer third sentence that contains more content.",
                'char_count': 150,
                'token_count': 30
            },
            {
                'page_number': 2, 
                'text': "This is on page two. It has multiple sentences as well. Each sentence adds to the content.",
                'char_count': 100,
                'token_count': 20
            }
        ]
        
        # Create mock PDF structure
        pdf_structure = PDFStructure(
            title="Test Document",
            total_pages=2,
            sections=[],
            metadata={}
        )
        
        # Create chunks
        chunks = self.extractor.create_chunks(pages_data, pdf_structure)
        
        # Verify chunks were created
        self.assertGreater(len(chunks), 0)
        
        # Verify chunk properties
        for chunk in chunks:
            self.assertIsInstance(chunk.id, str)
            self.assertGreater(len(chunk.chunk_text), 0)
            self.assertIn(chunk.page_number, [1, 2])
            self.assertIsInstance(chunk.metadata, dict)
            self.assertEqual(chunk.metadata['title'], "Test Document")
    
    def test_find_current_section(self):
        """Test section finding functionality."""
        sections = [
            {"title": "Introduction", "start_pos": 0},
            {"title": "Chapter 1: Getting Started", "start_pos": 100},
            {"title": "Conclusion", "start_pos": 200}
        ]
        
        # Test exact match
        text = "This is the Introduction section"
        result = self.extractor._find_current_section(text, sections)
        self.assertEqual(result, "Introduction")
        
        # Test case insensitive match
        text = "This is the INTRODUCTION section"
        result = self.extractor._find_current_section(text, sections)
        self.assertEqual(result, "Introduction")
        
        # Test no match
        text = "This is some other section"
        result = self.extractor._find_current_section(text, sections)
        self.assertIsNone(result)
    
    def test_save_chunks_to_json(self):
        """Test JSON saving functionality."""
        # Create test chunks
        chunks = [
            TextChunk(
                id="chunk-1",
                chunk_text="First chunk text.",
                page_number=1,
                metadata={"title": "Test Doc", "section": "Intro"},
                token_count=5
            ),
            TextChunk(
                id="chunk-2", 
                chunk_text="Second chunk text.",
                page_number=2,
                metadata={"title": "Test Doc", "section": "Chapter 1"},
                token_count=6
            )
        ]
        
        # Save to JSON
        output_path = os.path.join(self.temp_dir, "test_chunks.json")
        self.extractor.save_chunks_to_json(chunks, output_path)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_path))
        
        # Verify content
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 2)
        
        # Check first chunk
        chunk1 = data[0]
        self.assertEqual(chunk1['id'], 'chunk-1')
        self.assertEqual(chunk1['chunk_text'], 'First chunk text.')
        self.assertEqual(chunk1['page_number'], 1)
        self.assertEqual(chunk1['metadata']['title'], 'Test Doc')
        self.assertEqual(chunk1['metadata']['section'], 'Intro')
        
        # Check second chunk
        chunk2 = data[1]
        self.assertEqual(chunk2['id'], 'chunk-2')
        self.assertEqual(chunk2['page_number'], 2)
    
    def test_json_schema_compliance(self):
        """Test that output JSON matches required schema."""
        chunk = TextChunk(
            id="test-id",
            chunk_text="Test chunk content",
            page_number=5,
            metadata={"title": "Document Title", "section": "Chapter 2"},
            token_count=10
        )
        
        output_path = os.path.join(self.temp_dir, "schema_test.json")
        self.extractor.save_chunks_to_json([chunk], output_path)
        
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Verify schema structure
        self.assertEqual(len(data), 1)
        item = data[0]
        
        # Required fields
        self.assertIn('id', item)
        self.assertIn('chunk_text', item)
        self.assertIn('page_number', item)
        self.assertIn('metadata', item)
        
        # Metadata structure
        metadata = item['metadata']
        self.assertIn('title', metadata)
        self.assertIn('section', metadata)
        
        # Data types
        self.assertIsInstance(item['id'], str)
        self.assertIsInstance(item['chunk_text'], str)
        self.assertIsInstance(item['page_number'], int)
        self.assertIsInstance(item['metadata'], dict)
    
    @patch('pdfplumber.open')
    @patch('os.path.getsize')
    def test_extract_pdf_structure_mock(self, mock_getsize, mock_pdfplumber):
        """Test PDF structure extraction with mocked pdfplumber."""
        # Mock file size
        mock_getsize.return_value = 1000
        
        # Mock PDF object
        mock_pdf = Mock()
        mock_pdf.metadata = {'Title': 'Test Document', 'Author': 'Test Author'}
        
        # Mock page text extraction
        mock_page = Mock()
        mock_page.extract_text.return_value = "Chapter 1: Introduction\nThis is test content."
        mock_pdf.pages = [mock_page, mock_page, mock_page]  # 3 pages
        
        mock_pdfplumber.return_value.__enter__.return_value = mock_pdf
        
        # Test structure extraction
        structure = self.extractor.extract_pdf_structure("/fake/path.pdf")
        
        self.assertEqual(structure.title, 'Test Document')
        self.assertEqual(structure.total_pages, 3)
        self.assertEqual(structure.metadata['author'], 'Test Author')
    
    def test_process_pdf_file_validation(self):
        """Test PDF file validation."""
        # Test non-existent file
        with self.assertRaises(FileNotFoundError):
            self.extractor.process_pdf("/non/existent/file.pdf")
        
        # Test non-PDF file
        non_pdf_path = os.path.join(self.temp_dir, "test.txt")
        with open(non_pdf_path, 'w') as f:
            f.write("Not a PDF")
        
        with self.assertRaises(ValueError):
            self.extractor.process_pdf(non_pdf_path)


class TestPDFExtractorIntegration(unittest.TestCase):
    """Integration tests for PDF extractor."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.extractor = PDFTextExtractor()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_pdf(self):
        """Create a simple test PDF using reportlab if available."""
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
        except ImportError:
            self.skipTest("reportlab not available for PDF creation")
        
        pdf_path = os.path.join(self.temp_dir, "test_document.pdf")
        
        c = canvas.Canvas(pdf_path, pagesize=letter)
        
        # Page 1
        c.drawString(100, 750, "Test Document")
        c.drawString(100, 700, "Chapter 1: Introduction")
        c.drawString(100, 650, "This is the introduction to our test document.")
        c.drawString(100, 600, "It contains multiple sentences for testing.")
        c.showPage()
        
        # Page 2
        c.drawString(100, 750, "Chapter 2: Details")
        c.drawString(100, 700, "This chapter provides more detailed information.")
        c.drawString(100, 650, "We include various types of content here.")
        c.showPage()
        
        c.save()
        return pdf_path
    
    def test_end_to_end_processing(self):
        """Test complete end-to-end PDF processing."""
        try:
            pdf_path = self.create_test_pdf()
        except Exception:
            self.skipTest("Could not create test PDF")
        
        # Process the PDF
        output_path = self.extractor.process_pdf(pdf_path)
        
        # Verify output file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Verify JSON content
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertGreater(len(data), 0)
        
        # Verify schema compliance
        for item in data:
            self.assertIn('id', item)
            self.assertIn('chunk_text', item)
            self.assertIn('page_number', item)
            self.assertIn('metadata', item)
            
            metadata = item['metadata']
            self.assertIn('title', metadata)
            self.assertIn('section', metadata)


def create_sample_pdf_for_manual_testing():
    """
    Create a sample PDF for manual testing.
    This function can be called independently to generate test files.
    """
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
    except ImportError:
        print("reportlab not available. Install with: pip install reportlab")
        return None
    
    pdf_path = "/tmp/sample_test_document.pdf"
    
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    
    # Page 1 - Title and Introduction
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 100, "Sample Document for PDF Text Extraction")
    
    c.setFont("Helvetica", 12)
    y_pos = height - 150
    
    lines = [
        "Chapter 1: Introduction",
        "",
        "This is a sample document created specifically for testing the PDF text extraction",
        "functionality of the NLWeb system. The document contains multiple chapters, sections,",
        "and various types of content to thoroughly test the extraction and chunking algorithms.",
        "",
        "The text extraction tool should be able to:",
        "- Extract text while preserving document structure",
        "- Identify chapters and sections appropriately", 
        "- Split content into coherent chunks of 500-1000 tokens",
        "- Maintain sentence boundaries when creating chunks",
        "- Generate proper JSON output with the required schema",
        "",
        "This introduction serves as the first major section of the document and should",
        "be processed as part of the initial chunks. The content here is designed to be",
        "meaningful and substantial enough to test the tokenization limits."
    ]
    
    for line in lines:
        c.drawString(100, y_pos, line)
        y_pos -= 20
        if y_pos < 100:
            break
    
    c.showPage()
    
    # Page 2 - Chapter 2
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 100, "Chapter 2: Technical Details")
    
    c.setFont("Helvetica", 12)
    y_pos = height - 150
    
    lines = [
        "This chapter covers the technical implementation details of the PDF extraction system.",
        "The system uses pdfplumber for text extraction, which provides excellent support for",
        "preserving document structure and handling various PDF formats and encodings.",
        "",
        "2.1 Text Extraction Process",
        "",
        "The extraction process follows these steps:",
        "1. Load the PDF file using pdfplumber",
        "2. Extract metadata and document structure information",
        "3. Process each page to extract clean text content",
        "4. Identify sections and headings using pattern matching",
        "5. Split text into sentences while handling abbreviations",
        "6. Group sentences into appropriately sized chunks",
        "7. Generate JSON output with required metadata",
        "",
        "2.2 Tokenization and Chunking",
        "",
        "The system uses tiktoken for accurate token counting, ensuring that chunks fall within",
        "the specified range of 500-1000 tokens. Sentences are never broken across chunk boundaries,",
        "maintaining readability and coherence of the extracted content."
    ]
    
    for line in lines:
        c.drawString(100, y_pos, line)
        y_pos -= 20
        if y_pos < 100:
            break
    
    c.showPage()
    
    # Page 3 - Conclusion
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 100, "Chapter 3: Conclusion and Future Work")
    
    c.setFont("Helvetica", 12)
    y_pos = height - 150
    
    lines = [
        "This document demonstrates the capabilities of the PDF text extraction system.",
        "The implementation successfully handles various document structures and provides",
        "reliable chunking that respects sentence boundaries and token limits.",
        "",
        "Future enhancements could include:",
        "- Support for tables and complex layouts",
        "- Image and caption extraction",
        "- Enhanced section detection algorithms",
        "- Multi-language support and specialized tokenizers",
        "- Integration with vector databases for semantic search",
        "",
        "The current implementation provides a solid foundation for document processing",
        "in the NLWeb natural language interface system, enabling effective ingestion",
        "and retrieval of PDF content for AI-powered applications.",
        "",
        "This concludes our sample document. The text extraction system should process",
        "this content efficiently and produce well-structured JSON output that can be",
        "easily integrated into downstream processing pipelines."
    ]
    
    for line in lines:
        c.drawString(100, y_pos, line)
        y_pos -= 20
        if y_pos < 100:
            break
    
    c.save()
    print(f"Sample PDF created: {pdf_path}")
    return pdf_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run PDF extractor tests')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create a sample PDF for manual testing')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose test output')
    
    args = parser.parse_args()
    
    if args.create_sample:
        pdf_path = create_sample_pdf_for_manual_testing()
        if pdf_path:
            print(f"Sample PDF created at: {pdf_path}")
            print("You can now test the PDF extractor with:")
            print(f"python pdf_extractor.py {pdf_path}")
    else:
        # Run tests
        if args.verbose:
            verbosity = 2
        else:
            verbosity = 1
        
        unittest.main(argv=[''], verbosity=verbosity, exit=False)