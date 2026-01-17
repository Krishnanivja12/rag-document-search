import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application configuration settings."""
    
    # OpenRouter API Configuration
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = os.getenv(
        "OPENROUTER_BASE_URL", 
        "https://openrouter.ai/api/v1"
    )
    USER_AGENT: str = os.getenv("USER_AGENT", "rag-openrouter-app/1.0")
    
    # Free models available on OpenRouter
    # For main LLM (text generation)
    LLM_MODEL: str = os.getenv(
        "LLM_MODEL",
        "nvidia/nemotron-3-nano-30b-a3b:free"  # Free model
        # Alternatives:
        # "google/gemma-2-9b-it:free"
        # "microsoft/phi-3-mini-128k-instruct:free"
        # "mistralai/mistral-7b-instruct:free"
    )
    
    # Temperature for LLM (0 = deterministic, 1 = creative)
    LLM_TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
    
    # Document Processing Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    
    # Retriever Settings
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "4"))  # Number of documents to retrieve
    
    # Embedding Model (Local - HuggingFace)
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Streamlit Configuration
    PAGE_TITLE: str = "RAG Document Assistant"
    PAGE_ICON: str = "📄"
    
    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        if not cls.OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )
    
    @classmethod
    def get_info(cls) -> dict:
        """Get configuration info for display."""
        return {
            "LLM Model": cls.LLM_MODEL,
            "Temperature": cls.LLM_TEMPERATURE,
            "Chunk Size": cls.CHUNK_SIZE,
            "Chunk Overlap": cls.CHUNK_OVERLAP,
            "Retriever K": cls.RETRIEVER_K,
            "Embedding Model": cls.EMBEDDING_MODEL,
        }


# Create a global settings instance
settings = Settings()
