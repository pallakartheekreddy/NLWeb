# PDF Text Extraction Tool

This module provides comprehensive PDF text extraction and chunking functionality for the NLWeb system.

## Overview

The PDF extractor loads PDF files, extracts text while preserving structure, splits the text into coherent chunks of 500–1000 tokens without breaking sentences, organizes the chunks into a JSON file using a specific schema, and saves the structured output.

## Features

- **PDF Loading**: Robust PDF file processing with error handling
- **Structure Preservation**: Extracts headings, sections, and document hierarchy
- **Smart Chunking**: Splits text at sentence boundaries within token limits
- **Metadata Extraction**: Captures page numbers, titles, sections for each chunk
- **JSON Schema Compliance**: Structured output with required fields
- **NLWeb Integration**: Seamless integration with existing pipeline

## Usage

### Basic Usage

```python
from scraping import PDFTextExtractor

# Create extractor
extractor = PDFTextExtractor()

# Process PDF
output_path = extractor.process_pdf('document.pdf')
print(f"Chunks saved to: {output_path}")
```

### Custom Settings

```python
# Custom token limits
extractor = PDFTextExtractor(
    min_chunk_tokens=300,
    max_chunk_tokens=800
)

# Process with custom output path
output_path = extractor.process_pdf('document.pdf', 'custom_chunks.json')
```

### Command Line Interface

```bash
# Basic usage
python pdf_extractor.py document.pdf

# With custom settings
python pdf_extractor.py document.pdf --min-tokens 300 --max-tokens 800 -o output.json

# Verbose output
python pdf_extractor.py document.pdf -v
```

## Output Schema

The tool generates JSON with the following schema:

```json
[
  {
    "id": "unique-chunk-id",
    "chunk_text": "The actual text content of the chunk...",
    "page_number": 1,
    "metadata": {
      "title": "Document Title",
      "section": "Chapter 1: Introduction",
      "sentence_count": 5,
      "char_count": 245
    }
  }
]
```

## Examples

See `example_pdf_usage.py` for comprehensive examples including:

1. Basic PDF processing
2. Custom tokenization settings  
3. Batch processing multiple files
4. Integration with NLWeb pipeline

Run examples with:
```bash
python example_pdf_usage.py 1  # Basic example
python example_pdf_usage.py --all  # All examples
```

## Testing

Run the test suite:
```bash
python -m pytest test_pdf_extractor.py -v
```

Create a sample PDF for testing:
```bash
python test_pdf_extractor.py --create-sample
```

## Dependencies

- `pdfplumber`: PDF text extraction
- `tiktoken`: Token counting (with fallback for offline use)
- `reportlab`: Test PDF generation (development only)

## Integration with NLWeb

The PDF extractor integrates seamlessly with the NLWeb scraping pipeline:

```python
from scraping import PDFTextExtractor

# Extract PDF content
extractor = PDFTextExtractor()
chunks = extractor.process_pdf('document.pdf')

# Convert to NLWeb format for vector database ingestion
# (See example_pdf_usage.py for complete integration example)
```

## Error Handling

The tool includes robust error handling for:

- Invalid or corrupted PDF files
- Network connectivity issues (tiktoken fallback)
- Missing dependencies (automatic installation prompts)
- File system permissions and paths

## Performance

- Processes typical documents in seconds
- Memory efficient streaming for large PDFs
- Configurable chunk sizes for different use cases
- Fallback tokenization when internet unavailable