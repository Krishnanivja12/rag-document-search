"""
Vector store module.
Manages the FAISS vector database for semantic search.
"""
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from src.embeddings import get_embeddings
from config.settings import settings


def build_vector_store(chunks: List[Document]) -> FAISS:
    """
    Build a FAISS vector store from document chunks.
    
    FAISS (Facebook AI Similarity Search) is:
    - Fast and efficient for similarity search
    - Works entirely locally (no API needed)
    - Supports millions of vectors
    - Free and open-source
    
    Args:
        chunks: List of document chunks to index
        
    Returns:
        FAISS: Configured vector store instance
        
    Raises:
        ValueError: If chunks list is empty
    """
    if not chunks:
        raise ValueError("Cannot build vector store from empty chunks list")
    
    embeddings = get_embeddings()
    
    # Create FAISS index from documents
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    return vector_store


def get_retriever(
    vector_store: FAISS,
    k: int = None
) -> VectorStoreRetriever:
    """
    Create a retriever from the vector store.
    
    Args:
        vector_store: FAISS vector store instance
        k: Number of documents to retrieve (default from settings)
        
    Returns:
        VectorStoreRetriever: Configured retriever
    """
    if k is None:
        k = settings.RETRIEVER_K
    
    return vector_store.as_retriever(
        search_type="similarity",  # Can also use "mmr" for diversity
        search_kwargs={"k": k}
    )


def search_similar_docs(
    vector_store: FAISS,
    query: str,
    k: int = None
) -> List[Document]:
    """
    Search for similar documents using the vector store.
    
    Args:
        vector_store: FAISS vector store instance
        query: Search query
        k: Number of results to return (default from settings)
        
    Returns:
        List[Document]: Most similar documents
    """
    if k is None:
        k = settings.RETRIEVER_K
    
    return vector_store.similarity_search(query, k=k)


def search_with_scores(
    vector_store: FAISS,
    query: str,
    k: int = None
) -> List[tuple[Document, float]]:
    """
    Search for similar documents with similarity scores.
    
    Args:
        vector_store: FAISS vector store instance
        query: Search query
        k: Number of results to return (default from settings)
        
    Returns:
        List[tuple[Document, float]]: Documents with similarity scores
    """
    if k is None:
        k = settings.RETRIEVER_K
    
    return vector_store.similarity_search_with_score(query, k=k)