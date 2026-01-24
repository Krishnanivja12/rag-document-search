# 📄 RAG Document Search System

A **Retrieval-Augmented Generation (RAG)** based application that enables users to upload documents and ask natural language questions, retrieving accurate and context-aware answers using semantic search and large language models.

🔗 **Live Demo:** https://rag-document-search.streamlit.app/  
🔗 **GitHub Repository:** https://github.com/Krishnanivja12/rag-document-search  

---

## 🚀 Features

- Upload and process documents (PDF / text-based files)
- Semantic search using vector embeddings
- Context-aware question answering using RAG architecture
- Efficient document chunking and retrieval
- Interactive web interface built with Streamlit
- End-to-end pipeline from ingestion to inference

---

## 🧠 System Architecture (High Level)

1. **Document Ingestion** – Uploaded documents are loaded and split into semantic chunks  
2. **Embedding Generation** – Text chunks are converted into vector embeddings  
3. **Vector Storage & Retrieval** – Relevant chunks are retrieved using similarity search  
4. **Generation** – Retrieved context is passed to an LLM to generate accurate answers  
5. **UI Layer** – Streamlit interface for document upload and querying  

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Frameworks / Libraries:** LangChain  
- **Embeddings & Retrieval:** Vector Databases, Semantic Search  
- **Frontend:** Streamlit  
- **LLM Integration:** API-based Language Models  

---

## 📦 Installation & Setup

### Clone the repository
```bash
git clone https://github.com/Krishnanivja12/rag-document-search.git
cd rag-document-search
```

Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```
Install dependencies
```bash
pip install -r requirements.txt
```
Environment Variables
This project requires an OpenRouter API key to access LLM services.
```env
OPENROUTER_API_KEY=Enter_your_openrouter_api_key
USER_AGENT=rag-openrouter-app/1.0
```
Run the application locally
```bash
streamlit run app.py
```
# 📊 Use Cases
Document-based question answering<br>
Resume or research paper search<br>
Internal knowledge base querying<br>
Legal, academic, or technical document exploration<br>

# 📈 Future Improvements
Support for multiple document formats<br>
Advanced retrieval evaluation metrics<br>
Response citation and source highlighting<br>
Authentication and user session management<br>
Scalable vector database integration

👤 Author
Krishna<br>
B.Tech Student | Data Science Intern<br>
Focus: AI, Machine Learning, LLMs, RAG Systems

