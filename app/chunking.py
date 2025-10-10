"""
Document chunking utilities for optimal RAG performance
"""

import re
import tiktoken
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ChunkMetadata:
    """Metadata for document chunks"""
    total_tokens: int
    total_chars: int
    chunk_index: int = 0
    total_chunks: int = 1
    is_full_document: bool = True

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens using OpenAI's tokenizer"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback to cl100k_base (GPT-4 tokenizer)
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

def should_chunk_document(text: str, max_tokens: int = 500) -> bool:
    """Determine if document should be chunked based on token count"""
    token_count = count_tokens(text)
    return token_count > max_tokens

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Chunk text into overlapping segments
    
    Args:
        text: Text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position
        end = start + chunk_size
        
        # If we're not at the end, try to break at a good boundary
        if end < len(text):
            # Look for sentence boundaries first
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1
            else:
                # Look for paragraph boundaries
                para_end = text.rfind('\n\n', start, end)
                if para_end > start + chunk_size // 2:
                    end = para_end + 2
                else:
                    # Look for any newline
                    newline_end = text.rfind('\n', start, end)
                    if newline_end > start + chunk_size // 2:
                        end = newline_end + 1
                    # Otherwise use the original end position
        
        # Extract chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Calculate next start position with overlap
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    
    return chunks

def create_chunk_metadata(text: str, chunk_index: int = 0, total_chunks: int = 1) -> ChunkMetadata:
    """Create metadata for a text chunk"""
    return ChunkMetadata(
        total_tokens=count_tokens(text),
        total_chars=len(text),
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        is_full_document=(total_chunks == 1)
    )

def process_document_for_search(text: str, url: str, title: Optional[str] = None, 
                               max_tokens: int = 500, chunk_size: int = 1200, 
                               overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Process a document into search-ready chunks or return full document if small enough
    
    Args:
        text: Document text
        url: Document URL
        title: Document title
        max_tokens: Maximum tokens for full document inclusion
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters
    
    Returns:
        List of document/chunk objects with metadata
    """
    if not text.strip():
        return []
    
    # Check if document should be chunked
    if should_chunk_document(text, max_tokens):
        # Chunk the document
        chunks = chunk_text(text, chunk_size, overlap)
        
        results = []
        for i, chunk in enumerate(chunks):
            metadata = create_chunk_metadata(chunk, i, len(chunks))
            results.append({
                'text': chunk,
                'url': url,
                'title': title,
                'metadata': {
                    'total_tokens': metadata.total_tokens,
                    'total_chars': metadata.total_chars,
                    'chunk_index': metadata.chunk_index,
                    'total_chunks': metadata.total_chunks,
                    'is_full_document': False,
                    'original_doc_tokens': count_tokens(text),
                    'original_doc_chars': len(text)
                }
            })
        return results
    else:
        # Return full document
        metadata = create_chunk_metadata(text, 0, 1)
        return [{
            'text': text,
            'url': url,
            'title': title,
            'metadata': {
                'total_tokens': metadata.total_tokens,
                'total_chars': metadata.total_chars,
                'chunk_index': 0,
                'total_chunks': 1,
                'is_full_document': True,
                'original_doc_tokens': metadata.total_tokens,
                'original_doc_chars': metadata.total_chars
            }
        }]

def reconstruct_full_document(chunks: List[Dict[str, Any]]) -> Optional[str]:
    """
    Reconstruct full document from chunks if they're from the same document
    
    Args:
        chunks: List of chunk objects with metadata
    
    Returns:
        Full document text or None if chunks are from different documents
    """
    if not chunks:
        return None
    
    # Check if all chunks are from the same document
    first_url = chunks[0].get('url')
    if not all(chunk.get('url') == first_url for chunk in chunks):
        return None
    
    # Sort chunks by index
    sorted_chunks = sorted(chunks, key=lambda x: x.get('metadata', {}).get('chunk_index', 0))
    
    # Simple reconstruction - join with double newlines to indicate chunk boundaries
    return '\n\n'.join(chunk['text'] for chunk in sorted_chunks)