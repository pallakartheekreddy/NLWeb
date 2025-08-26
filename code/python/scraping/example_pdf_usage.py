#!/usr/bin/env python3
"""
Example usage of the PDF text extraction tool.
Shows how to use the PDF extractor in various scenarios.
"""

import os
import sys
import json
from pathlib import Path

# Add the scraping module to the path
sys.path.insert(0, os.path.dirname(__file__))

from pdf_extractor import PDFTextExtractor


def example_basic_pdf_processing():
    """Basic example: extract and chunk text from a PDF file."""
    
    print("=== Basic PDF Processing Example ===\n")
    
    # Create a PDF extractor with custom settings
    extractor = PDFTextExtractor(
        min_chunk_tokens=300,  # Smaller chunks for this example
        max_chunk_tokens=600
    )
    
    # Check if sample PDF exists, create if not
    sample_pdf = "/tmp/sample_test_document.pdf"
    if not os.path.exists(sample_pdf):
        print("Creating sample PDF for demonstration...")
        try:
            from test_pdf_extractor import create_sample_pdf_for_manual_testing
            sample_pdf = create_sample_pdf_for_manual_testing()
        except ImportError:
            print("Could not create sample PDF. Please provide a PDF file path.")
            return
    
    print(f"Processing PDF: {sample_pdf}")
    
    try:
        # Process the PDF
        output_path = extractor.process_pdf(sample_pdf)
        
        print(f"✅ Successfully processed PDF!")
        print(f"📄 Output saved to: {output_path}")
        
        # Load and display results
        with open(output_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"\n📊 Processing Results:")
        print(f"   • Total chunks: {len(chunks)}")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"   • Chunk {i}: {len(chunk['chunk_text'])} chars, page {chunk['page_number']}")
            print(f"     Section: {chunk['metadata']['section']}")
            if i <= 2:  # Show first two chunks in detail
                print(f"     Preview: {chunk['chunk_text'][:100]}...")
            print()
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")


def example_custom_settings():
    """Example with custom tokenization and chunking settings."""
    
    print("=== Custom Settings Example ===\n")
    
    # Create extractor with different settings
    extractor = PDFTextExtractor(
        min_chunk_tokens=200,   # Smaller minimum
        max_chunk_tokens=800,   # Larger maximum
        encoding_model="cl100k_base"  # Specific encoding
    )
    
    sample_pdf = "/tmp/sample_test_document.pdf"
    if not os.path.exists(sample_pdf):
        print("Sample PDF not found. Run basic example first.")
        return
    
    print("Processing with custom settings:")
    print(f"• Min tokens per chunk: {extractor.min_chunk_tokens}")
    print(f"• Max tokens per chunk: {extractor.max_chunk_tokens}")
    
    try:
        # Process with custom output path
        output_path = os.path.join("/tmp", "custom_chunks.json")
        result_path = extractor.process_pdf(sample_pdf, output_path)
        
        # Compare with default settings
        with open(result_path, 'r') as f:
            custom_chunks = json.load(f)
        
        print(f"\n📊 Custom Processing Results:")
        print(f"   • Total chunks: {len(custom_chunks)}")
        
        # Calculate statistics
        token_counts = [extractor.count_tokens(chunk['chunk_text']) for chunk in custom_chunks]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        
        print(f"   • Average tokens per chunk: {avg_tokens:.1f}")
        print(f"   • Token range: {min(token_counts)}-{max(token_counts)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_batch_processing():
    """Example: process multiple PDF files in batch."""
    
    print("=== Batch Processing Example ===\n")
    
    # For this example, we'll process the same file multiple times with different settings
    sample_pdf = "/tmp/sample_test_document.pdf"
    if not os.path.exists(sample_pdf):
        print("Sample PDF not found. Run basic example first.")
        return
    
    # Different processing configurations
    configs = [
        {"name": "small_chunks", "min_tokens": 100, "max_tokens": 300},
        {"name": "medium_chunks", "min_tokens": 300, "max_tokens": 600},
        {"name": "large_chunks", "min_tokens": 500, "max_tokens": 1000},
    ]
    
    results = {}
    
    for config in configs:
        print(f"Processing with {config['name']} configuration...")
        
        extractor = PDFTextExtractor(
            min_chunk_tokens=config['min_tokens'],
            max_chunk_tokens=config['max_tokens']
        )
        
        try:
            output_path = f"/tmp/{config['name']}_output.json"
            extractor.process_pdf(sample_pdf, output_path)
            
            # Load and analyze results
            with open(output_path, 'r') as f:
                chunks = json.load(f)
            
            # Calculate statistics
            token_counts = [extractor.count_tokens(chunk['chunk_text']) for chunk in chunks]
            
            results[config['name']] = {
                'chunk_count': len(chunks),
                'avg_tokens': sum(token_counts) / len(token_counts) if token_counts else 0,
                'min_tokens': min(token_counts) if token_counts else 0,
                'max_tokens': max(token_counts) if token_counts else 0,
                'output_file': output_path
            }
            
            print(f"   ✅ {len(chunks)} chunks created")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[config['name']] = {'error': str(e)}
    
    # Display comparison
    print(f"\n📊 Batch Processing Comparison:")
    print(f"{'Configuration':<15} {'Chunks':<8} {'Avg Tokens':<12} {'Range':<15}")
    print("-" * 50)
    
    for name, result in results.items():
        if 'error' not in result:
            range_str = f"{result['min_tokens']}-{result['max_tokens']}"
            print(f"{name:<15} {result['chunk_count']:<8} {result['avg_tokens']:<12.1f} {range_str:<15}")
        else:
            print(f"{name:<15} {'ERROR':<8} {'':<12} {'':<15}")


def example_integration_with_nlweb():
    """Example: integrate PDF extraction with NLWeb data pipeline."""
    
    print("=== NLWeb Integration Example ===\n")
    
    sample_pdf = "/tmp/sample_test_document.pdf"
    if not os.path.exists(sample_pdf):
        print("Sample PDF not found. Run basic example first.")
        return
    
    print("Simulating NLWeb pipeline integration...")
    
    # Step 1: Extract PDF content
    extractor = PDFTextExtractor()
    
    try:
        # Process PDF and get structured data
        output_path = extractor.process_pdf(sample_pdf)
        
        with open(output_path, 'r') as f:
            chunks = json.load(f)
        
        print(f"✅ Extracted {len(chunks)} chunks from PDF")
        
        # Step 2: Convert to NLWeb format (simulated)
        nlweb_format = []
        
        for chunk in chunks:
            # Convert to format similar to existing scraped content
            nlweb_item = {
                'url': f"pdf://{os.path.basename(sample_pdf)}#page-{chunk['page_number']}",
                'content': chunk['chunk_text'],
                'metadata': {
                    'source_type': 'pdf',
                    'title': chunk['metadata']['title'],
                    'section': chunk['metadata']['section'],
                    'page_number': chunk['page_number'],
                    'chunk_id': chunk['id'],
                    'token_count': extractor.count_tokens(chunk['chunk_text'])
                }
            }
            nlweb_format.append(nlweb_item)
        
        # Step 3: Save in NLWeb-compatible format
        nlweb_output = "/tmp/nlweb_pdf_content.jsonl"
        with open(nlweb_output, 'w') as f:
            for item in nlweb_format:
                f.write(json.dumps(item) + '\n')
        
        print(f"✅ Converted to NLWeb format: {nlweb_output}")
        
        # Step 4: Display sample converted content
        print(f"\n📄 Sample NLWeb Format Entry:")
        sample_item = nlweb_format[0]
        print(json.dumps(sample_item, indent=2)[:500] + "...")
        
        print(f"\n🔗 Integration complete! The PDF content is now ready for:")
        print(f"   • Vector database ingestion")
        print(f"   • Embedding generation")
        print(f"   • Natural language search")
        print(f"   • AI-powered question answering")
        
    except Exception as e:
        print(f"❌ Integration error: {e}")


def main():
    """Run all examples or a specific one."""
    import argparse
    
    examples = {
        '1': ('Basic PDF processing', example_basic_pdf_processing),
        '2': ('Custom settings', example_custom_settings),
        '3': ('Batch processing', example_batch_processing),
        '4': ('NLWeb integration', example_integration_with_nlweb)
    }
    
    parser = argparse.ArgumentParser(description='PDF Extractor Examples')
    parser.add_argument('example', nargs='?', choices=examples.keys(),
                       help='Example number to run')
    parser.add_argument('--all', action='store_true',
                       help='Run all examples')
    
    args = parser.parse_args()
    
    if args.all:
        print("Running all PDF extractor examples...\n")
        for i, (name, func) in enumerate(examples.values(), 1):
            print(f"\n{'='*60}")
            func()
            if i < len(examples):
                input("\nPress Enter to continue to next example...")
    elif args.example:
        name, func = examples[args.example]
        print(f"Running: {name}\n")
        func()
    else:
        print("PDF Text Extractor Examples")
        print("=" * 40)
        print("\nAvailable examples:")
        for key, (name, _) in examples.items():
            print(f"{key}. {name}")
        
        print(f"\nUsage:")
        print(f"python {sys.argv[0]} <example_number>")
        print(f"python {sys.argv[0]} --all")
        print(f"\nExample: python {sys.argv[0]} 1")


if __name__ == "__main__":
    main()