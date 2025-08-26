#!/usr/bin/env python3
"""
PDF Text Extraction Tool for NLWeb

This module provides functionality to:
1. Load PDF files and extract text while preserving structure
2. Split text into coherent chunks of 500-1000 tokens without breaking sentences
3. Organize chunks into JSON with schema: {id, chunk_text, page_number, metadata: {title, section}}
4. Save the structured JSON output

The tool is designed to integrate with the existing NLWeb scraping pipeline.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from uuid import uuid4

# PDF processing library
try:
    import pdfplumber
except ImportError:
    print("pdfplumber not found. Installing...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber"])
        import pdfplumber
        print("pdfplumber installed successfully!")
    except Exception as e:
        print(f"Error installing pdfplumber: {e}")
        print("Please install manually with: pip install pdfplumber")
        sys.exit(1)

# Tokenization library
try:
    import tiktoken
except ImportError:
    print("tiktoken not found. Installing...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tiktoken"])
        import tiktoken
        print("tiktoken installed successfully!")
    except Exception as e:
        print(f"Error installing tiktoken: {e}")
        print("Please install manually with: pip install tiktoken")
        sys.exit(1)

# Import logging from the NLWeb framework
try:
    from misc.logger.logging_config_helper import get_configured_logger
    logger = get_configured_logger("pdf_extractor")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("pdf_extractor")


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    id: str
    chunk_text: str
    page_number: int
    metadata: Dict[str, Any]
    token_count: int = 0


@dataclass
class PDFStructure:
    """Represents the structure of a PDF document"""
    title: str
    total_pages: int
    sections: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class PDFTextExtractor:
    """
    Main class for extracting and chunking PDF text.
    """
    
    def __init__(self, 
                 min_chunk_tokens: int = 500,
                 max_chunk_tokens: int = 1000,
                 encoding_model: str = "cl100k_base"):
        """
        Initialize the PDF text extractor.
        
        Args:
            min_chunk_tokens: Minimum tokens per chunk
            max_chunk_tokens: Maximum tokens per chunk  
            encoding_model: Tokenizer model name for tiktoken
        """
        self.min_chunk_tokens = min_chunk_tokens
        self.max_chunk_tokens = max_chunk_tokens
        
        # Initialize tokenizer
        self.tokenizer = None
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_model)
        except Exception as e:
            logger.warning(f"Failed to load {encoding_model} encoding, using fallback method: {e}")
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as e2:
                logger.warning(f"Failed to load default encoding, using simple fallback: {e2}")
                self.tokenizer = None
        
        # Patterns for detecting structure
        self.heading_patterns = [
            re.compile(r'^(Chapter|CHAPTER)\s+\d+', re.MULTILINE),
            re.compile(r'^(\d+\.|\d+\.\d+\.)\s+[A-Z]', re.MULTILINE),
            re.compile(r'^[A-Z][A-Z\s]{5,}$', re.MULTILINE),  # ALL CAPS headings (reduced min length)
            re.compile(r'^\s*[IVX]+\.\s+[A-Z]', re.MULTILINE),  # Roman numerals
            re.compile(r'^\s*(Chapter|Section|Part)\s+\d+:', re.MULTILINE | re.IGNORECASE),  # More flexible chapters
        ]
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken or fallback method."""
        if not text or not text.strip():
            return 0
            
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Token counting failed, using fallback: {e}")
        
        # Fallback: simple estimation based on words and characters
        # OpenAI tokenizer roughly follows: 1 token ≈ 0.75 words ≈ 4 characters
        words = len(text.split())
        chars = len(text)
        
        # Use word-based estimation as primary, char-based as secondary
        word_estimate = int(words / 0.75)
        char_estimate = chars // 4
        
        # Return the average to be more conservative
        return max(1, (word_estimate + char_estimate) // 2)
    
    def extract_pdf_structure(self, pdf_path: str) -> PDFStructure:
        """
        Extract document structure and metadata from PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            PDFStructure object with document information
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Extract basic metadata
                metadata = pdf.metadata or {}
                title = metadata.get('Title', '') or os.path.splitext(os.path.basename(pdf_path))[0]
                total_pages = len(pdf.pages)
                
                # Extract text from first few pages to identify structure
                structure_text = ""
                for i, page in enumerate(pdf.pages[:5]):  # Look at first 5 pages
                    page_text = page.extract_text() or ""
                    structure_text += f"\n--- PAGE {i+1} ---\n{page_text}"
                
                # Detect sections using heading patterns
                sections = self._detect_sections(structure_text)
                
                return PDFStructure(
                    title=title,
                    total_pages=total_pages,
                    sections=sections,
                    metadata={
                        'author': metadata.get('Author', ''),
                        'subject': metadata.get('Subject', ''),
                        'creator': metadata.get('Creator', ''),
                        'creation_date': str(metadata.get('CreationDate', '')),
                        'file_path': pdf_path,
                        'file_size': os.path.getsize(pdf_path)
                    }
                )
                
        except Exception as e:
            logger.error(f"Failed to extract PDF structure from {pdf_path}: {e}")
            # Return minimal structure
            return PDFStructure(
                title=os.path.splitext(os.path.basename(pdf_path))[0],
                total_pages=0,
                sections=[],
                metadata={'file_path': pdf_path, 'error': str(e)}
            )
    
    def _detect_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect document sections using pattern matching.
        
        Args:
            text: Text to analyze for sections
            
        Returns:
            List of section dictionaries
        """
        sections = []
        
        # Find potential headings
        for pattern in self.heading_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                sections.append({
                    'title': match.group().strip(),
                    'start_pos': match.start(),
                    'pattern_type': pattern.pattern[:20] + '...'
                })
        
        # Sort by position and remove duplicates
        sections = sorted(sections, key=lambda x: x['start_pos'])
        
        # Remove overlapping sections (keep the first one)
        filtered_sections = []
        last_pos = -1
        for section in sections:
            if section['start_pos'] > last_pos + 10:  # Allow 10 char gap
                filtered_sections.append(section)
                last_pos = section['start_pos'] + len(section['title'])
        
        return filtered_sections
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with page-level granularity.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of page dictionaries with text and metadata
        """
        pages_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text
                    text = page.extract_text() or ""
                    
                    # Clean and normalize text
                    text = self._clean_text(text)
                    
                    if text.strip():  # Only include pages with text
                        pages_data.append({
                            'page_number': page_num,
                            'text': text,
                            'char_count': len(text),
                            'token_count': self.count_tokens(text),
                            'bbox': page.bbox if hasattr(page, 'bbox') else None
                        })
                        
                        logger.debug(f"Extracted {len(text)} characters from page {page_num}")
                
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            raise
        
        return pages_data
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple line breaks to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n ', '\n', text)  # Remove spaces at start of lines
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Missing spaces between words
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # Hyphenated words across lines
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, handling edge cases.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Basic sentence splitting with common abbreviations handling
        # This is simplified - for production use, consider using NLTK or spaCy
        
        # Patterns that should NOT be sentence boundaries
        abbreviations = r'(?:Mr|Mrs|Dr|Prof|Sr|Jr|vs|etc|Inc|Corp|Ltd|Co|St|Ave|Blvd|Ph\.D|M\.D|B\.A|M\.A)'
        
        # Replace abbreviations temporarily
        text = re.sub(f'({abbreviations})\\.', r'\1<ABBREV>', text, flags=re.IGNORECASE)
        
        # Split on sentence endings
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Restore abbreviations and clean up
        sentences = [re.sub(r'<ABBREV>', '.', s).strip() for s in sentences if s.strip()]
        
        return sentences
    
    def create_chunks(self, pages_data: List[Dict[str, Any]], 
                     pdf_structure: PDFStructure) -> List[TextChunk]:
        """
        Create text chunks from pages data.
        
        Args:
            pages_data: List of page data dictionaries
            pdf_structure: Document structure information
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        current_section = "Introduction"  # Default section
        
        for page_data in pages_data:
            page_num = page_data['page_number']
            page_text = page_data['text']
            
            # Update current section if we find a heading
            section_match = self._find_current_section(page_text, pdf_structure.sections)
            if section_match:
                current_section = section_match
            
            # Split page text into sentences
            sentences = self.split_into_sentences(page_text)
            
            # Group sentences into chunks
            current_chunk_sentences = []
            current_chunk_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = self.count_tokens(sentence)
                
                # Check if adding this sentence exceeds max tokens
                if (current_chunk_tokens + sentence_tokens > self.max_chunk_tokens and 
                    current_chunk_sentences and 
                    current_chunk_tokens >= self.min_chunk_tokens):
                    
                    # Create chunk from current sentences
                    chunk_text = " ".join(current_chunk_sentences)
                    chunk = TextChunk(
                        id=str(uuid4()),
                        chunk_text=chunk_text,
                        page_number=page_num,
                        metadata={
                            'title': pdf_structure.title,
                            'section': current_section,
                            'sentence_count': len(current_chunk_sentences),
                            'char_count': len(chunk_text)
                        },
                        token_count=current_chunk_tokens
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk
                    current_chunk_sentences = [sentence]
                    current_chunk_tokens = sentence_tokens
                else:
                    # Add sentence to current chunk
                    current_chunk_sentences.append(sentence)
                    current_chunk_tokens += sentence_tokens
            
            # Handle remaining sentences at end of page
            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                chunk = TextChunk(
                    id=str(uuid4()),
                    chunk_text=chunk_text,
                    page_number=page_num,
                    metadata={
                        'title': pdf_structure.title,
                        'section': current_section,
                        'sentence_count': len(current_chunk_sentences),
                        'char_count': len(chunk_text)
                    },
                    token_count=current_chunk_tokens
                )
                chunks.append(chunk)
                current_chunk_sentences = []
                current_chunk_tokens = 0
        
        logger.info(f"Created {len(chunks)} chunks from {len(pages_data)} pages")
        return chunks
    
    def _find_current_section(self, text: str, sections: List[Dict[str, Any]]) -> Optional[str]:
        """
        Find the current section based on text content.
        
        Args:
            text: Text to search for section headings
            sections: List of detected sections
            
        Returns:
            Section name if found, None otherwise
        """
        for section in sections:
            if section['title'].lower() in text.lower():
                return section['title']
        return None
    
    def save_chunks_to_json(self, chunks: List[TextChunk], output_path: str) -> None:
        """
        Save chunks to JSON file with the required schema.
        
        Args:
            chunks: List of TextChunk objects
            output_path: Path to save JSON file
        """
        # Convert chunks to the required schema format
        json_data = []
        for chunk in chunks:
            json_data.append({
                'id': chunk.id,
                'chunk_text': chunk.chunk_text,
                'page_number': chunk.page_number,
                'metadata': chunk.metadata
            })
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to JSON file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(chunks)} chunks to {output_path}")
            
            # Log statistics
            total_tokens = sum(chunk.token_count for chunk in chunks)
            avg_tokens = total_tokens / len(chunks) if chunks else 0
            logger.info(f"Statistics: {total_tokens} total tokens, {avg_tokens:.1f} avg per chunk")
            
        except Exception as e:
            logger.error(f"Failed to save chunks to {output_path}: {e}")
            raise
    
    def process_pdf(self, pdf_path: str, output_path: str = None) -> str:
        """
        Main method to process a PDF file end-to-end.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Path for output JSON file (optional)
            
        Returns:
            Path to the created JSON file
        """
        # Validate input
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"File must be a PDF: {pdf_path}")
        
        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_dir = os.path.dirname(pdf_path)
            output_path = os.path.join(output_dir, f"{base_name}_chunks.json")
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Step 1: Extract document structure
            pdf_structure = self.extract_pdf_structure(pdf_path)
            logger.info(f"Extracted structure: {pdf_structure.title} ({pdf_structure.total_pages} pages)")
            
            # Step 2: Extract text from all pages
            pages_data = self.extract_text_from_pdf(pdf_path)
            total_chars = sum(page['char_count'] for page in pages_data)
            logger.info(f"Extracted {total_chars} characters from {len(pages_data)} pages")
            
            # Step 3: Create chunks
            chunks = self.create_chunks(pages_data, pdf_structure)
            
            # Step 4: Save to JSON
            self.save_chunks_to_json(chunks, output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            raise


def main():
    """Command-line interface for the PDF extractor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract and chunk text from PDF files')
    parser.add_argument('pdf_path', help='Path to the PDF file')
    parser.add_argument('-o', '--output', help='Output JSON file path')
    parser.add_argument('--min-tokens', type=int, default=500, 
                       help='Minimum tokens per chunk (default: 500)')
    parser.add_argument('--max-tokens', type=int, default=1000,
                       help='Maximum tokens per chunk (default: 1000)')
    parser.add_argument('--encoding', default='cl100k_base',
                       help='Tokenizer encoding (default: cl100k_base)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create extractor
    extractor = PDFTextExtractor(
        min_chunk_tokens=args.min_tokens,
        max_chunk_tokens=args.max_tokens,
        encoding_model=args.encoding
    )
    
    try:
        # Process PDF
        output_path = extractor.process_pdf(args.pdf_path, args.output)
        print(f"Successfully processed PDF. Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()