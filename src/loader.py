"""
Document loading module.
Handles loading documents from various sources: PDF, TXT files, and URLs.
"""
import os
import tempfile
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader
)


def load_documents(
    files: Optional[List] = None, 
    url: Optional[str] = None
) -> List[Document]:
    """
    Load documents from uploaded files or a URL.
    
    Args:
        files: List of uploaded file objects (PDF or TXT)
        url: Web URL to scrape content from
        
    Returns:
        List[Document]: List of loaded document objects
        
    Raises:
        ValueError: If neither files nor URL is provided
        Exception: If document loading fails
    """
    if not files and not url:
        raise ValueError("Either files or URL must be provided")
    
    docs: List[Document] = []
    
    # Load from uploaded files
    if files:
        docs.extend(_load_from_files(files))
    
    # Load from URL
    if url and url.strip().startswith("http"):
        docs.extend(_load_from_url(url.strip()))
    
    if not docs:
        raise ValueError("No documents were successfully loaded")
    
    return docs


def _load_from_files(files: List) -> List[Document]:
    """
    Load documents from uploaded files.
    
    Args:
        files: List of file objects from Streamlit uploader
        
    Returns:
        List[Document]: Loaded documents
    """
    docs: List[Document] = []
    
    for file in files:
        try:
            # Get file extension
            suffix = os.path.splitext(file.name)[1].lower()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file.getbuffer())
                temp_path = tmp.name
            
            # Load based on file type
            if suffix == ".pdf":
                loader = PyPDFLoader(temp_path)
                docs.extend(loader.load())
            elif suffix == ".txt":
                loader = TextLoader(temp_path, encoding="utf-8")
                docs.extend(loader.load())
            else:
                print(f"Unsupported file type: {suffix}")
                continue
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception:
                pass  # Ignore cleanup errors
                
        except Exception as e:
            print(f"Error loading file {file.name}: {str(e)}")
            continue
    
    return docs


def _load_from_url(url: str) -> List[Document]:
    """
    Load documents from a web URL.
    
    Args:
        url: Web URL to scrape
        
    Returns:
        List[Document]: Loaded documents
    """
    try:
        loader = WebBaseLoader(url)
        return loader.load()
    except Exception as e:
        print(f"Error loading URL {url}: {str(e)}")
        return []


def get_document_info(docs: List[Document]) -> dict:
    """
    Get summary information about loaded documents.
    
    Args:
        docs: List of documents
        
    Returns:
        dict: Document statistics
    """
    total_chars = sum(len(doc.page_content) for doc in docs)
    
    return {
        "total_documents": len(docs),
        "total_characters": total_chars,
        "avg_chars_per_doc": total_chars // len(docs) if docs else 0
    }