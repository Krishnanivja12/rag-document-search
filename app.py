import streamlit as st
import time
from typing import List

from config.settings import settings
from src.loader import load_documents, get_document_info
from src.splitter import split_documents, get_chunk_info
from src.vector_store import build_vector_store, get_retriever
from src.rag_chain import run_rag
import os
from dotenv import load_dotenv


# Streamlit secrets से API key लें
if "OPENROUTER_API_KEY" in st.secrets:
    os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
    if "USER_AGENT" in st.secrets:
        os.environ["USER_AGENT"] = st.secrets["USER_AGENT"]
else:
    # Local development के लिए .env file use करें
    from dotenv import load_dotenv
    load_dotenv()

# Check करें कि API key मिली या नहीं
if not os.getenv("OPENROUTER_API_KEY"):
    st.error("⚠️ OPENROUTER_API_KEY not found. Please add it in Streamlit secrets.")
    st.stop()
# Validate settings on startup
try:
    settings.validate()
except ValueError as e:
    st.error(f"Configuration Error: {str(e)}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title=settings.PAGE_TITLE,
    page_icon=settings.PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f9ff;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📄 Document RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Chat with your documents using AI-powered search</div>', unsafe_allow_html=True)

# Session state initialization - MUST be before any widgets
def init_session_state():
    """Initialize all session state variables."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0
    
    if "document_loaded" not in st.session_state:
        st.session_state.document_loaded = False
    
    if "doc_info" not in st.session_state:
        st.session_state.doc_info = None
    
    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0
    
    if "url_input_key" not in st.session_state:
        st.session_state.url_input_key = 0

# Initialize session state
init_session_state()

# Sidebar
with st.sidebar:
    st.header("📁 Data Source")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.file_uploader_key}",
        help="You can upload multiple files at once"
    )
    
    st.markdown("**OR**")
    
    # URL input
    url = st.text_input(
        "Paste a web URL",
        key=f"url_input_{st.session_state.url_input_key}",
        placeholder="https://example.com/article",
        help="Enter a URL to scrape content from a webpage"
    )
    
    st.divider()
    
    # New document button
    if st.button("🔄 Start New Document", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.chat_history = []
        st.session_state.upload_key += 1
        st.session_state.file_uploader_key += 1
        st.session_state.url_input_key += 1
        st.session_state.document_loaded = False
        st.session_state.doc_info = None
        st.rerun()
    
    # Show document info if loaded
    if st.session_state.document_loaded and st.session_state.doc_info:
        st.divider()
        st.subheader("📊 Document Info")
        info = st.session_state.doc_info
        st.metric("Documents Loaded", info.get("total_documents", 0))
        st.metric("Total Chunks", info.get("total_chunks", 0))
        st.metric("Characters", f"{info.get('total_characters', 0):,}")
    
    # Settings expander
    with st.expander("⚙️ Settings", expanded=False):
        config_info = settings.get_info()
        for key, value in config_info.items():
            st.text(f"{key}: {value}")

# Main chat interface
st.divider()

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
question = st.chat_input("Ask a question about your documents...")

if question:
    # Validate input
    has_files = uploaded_files and len(uploaded_files) > 0
    has_url = url and url.strip() != ""
    
    if has_files and has_url:
        st.error("⚠️ Please use either files OR a URL, not both.")
        st.stop()
    
    if not has_files and not has_url:
        st.error("⚠️ Please upload a file or provide a URL first.")
        st.stop()
    
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner("🤔 Thinking..."):
            try:
                # Build vector store if not already done
                if st.session_state.vector_store is None:
                    with st.spinner("📚 Loading and processing documents..."):
                        # Load documents
                        docs = load_documents(
                            files=uploaded_files if has_files else None,
                            url=url.strip() if has_url else None
                        )
                        
                        doc_info = get_document_info(docs)
                        
                        # Split into chunks
                        chunks = split_documents(docs)
                        chunk_info = get_chunk_info(chunks)
                        
                        # Store combined info
                        st.session_state.doc_info = {**doc_info, **chunk_info}
                        
                        # Build vector store
                        st.session_state.vector_store = build_vector_store(chunks)
                        st.session_state.document_loaded = True
                
                # Retrieve relevant documents
                retriever = get_retriever(st.session_state.vector_store)
                relevant_docs = retriever.invoke(question)
                
                # Generate answer
                answer = run_rag(question, relevant_docs)
                
                # Simulate streaming effect
                streamed_text = ""
                words = answer.split()
                
                for i, word in enumerate(words):
                    streamed_text += word + " "
                    response_placeholder.markdown(streamed_text + "▌")
                    time.sleep(0.03)  # Adjust speed as needed
                
                # Final response without cursor
                response_placeholder.markdown(answer)
                
            except Exception as e:
                answer = f"❌ Error: {str(e)}\n\nPlease check your configuration and try again."
                response_placeholder.markdown(answer)
        
        # Add assistant message to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

# Welcome message if no chat history
if len(st.session_state.chat_history) == 0:
    st.info(f"""
    👋 **Welcome to RAG Document Assistant!**
            
    **Quick Start:** Upload a document or paste a URL in the sidebar, then ask questions!

    **Powered by:** {settings.LLM_MODEL.split('/')[1].replace(':free', '')} via OpenRouter
    """)
