"""
Document splitting module.
Splits large documents into smaller chunks for better retrieval and processing.
"""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import settings


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for processing.
    
    Uses RecursiveCharacterTextSplitter which:
    - Tries to keep paragraphs, sentences, and words together
    - Splits on multiple separators in order of preference
    - Maintains context with overlap between chunks
    
    Args:
        docs: List of documents to split
        
    Returns:
        List[Document]: List of document chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # Try these in order
        is_separator_regex=False
    )
    
    chunks = splitter.split_documents(docs)
    
    # Add metadata about chunk position
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)
    
    return chunks


def get_chunk_info(chunks: List[Document]) -> dict:
    """
    Get summary information about document chunks.
    
    Args:
        chunks: List of document chunks
        
    Returns:
        dict: Chunk statistics
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0
        }
    
    chunk_sizes = [len(chunk.page_content) for chunk in chunks]
    
    return {
        "total_chunks": len(chunks),
        "avg_chunk_size": sum(chunk_sizes) // len(chunk_sizes),
        "min_chunk_size": min(chunk_sizes),
        "max_chunk_size": max(chunk_sizes)
    }