"""
RAG (Retrieval-Augmented Generation) chain module.
Combines retrieved documents with LLM to generate answers.
Uses OpenRouter's free models for cost-effective operation.
"""
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import settings


# System prompt for the RAG assistant
SYSTEM_PROMPT = """
You are an AI assistant answering questions strictly from the given context.

Rules:
- Use ONLY the provided context.
- Do NOT use outside knowledge.
- If the answer is not present, say:
  "I don't know based on the provided document."

Response format:
- Give ONE clear heading related to the question.
- On the next line, explain the answer in a clean paragraph.
- Do NOT use bullet points.
- Do NOT mention sources or documents.
- Keep the explanation clear and professional.
"""


def get_llm() -> ChatOpenAI:
    """
    Create and configure the LLM instance using OpenRouter.
    """
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.OPENROUTER_API_KEY,  # Fixed parameter name
        base_url=settings.OPENROUTER_BASE_URL,  # Fixed parameter name
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=1000,
        default_headers={
            "HTTP-Referer": "https://github.com/yourusername/rag-app",
            "X-Title": "RAG Document Assistant"
        }
    )


def format_context(docs: List[Document]) -> str:
    if not docs:
        return "No relevant context found."
    
    context_parts = []
    for i, doc in enumerate(docs, 1):
        # Add document with separator
        context_parts.append(f"--- Document {i} ---")
        context_parts.append(doc.page_content.strip())
        context_parts.append("")  # Empty line between documents
    
    return "\n".join(context_parts)


def run_rag(question: str, docs: List[Document]) -> str:
    context = format_context(docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", """Context:
{context}

Question: {question}

Please provide a clear and accurate answer based only on the context above.""")
    ])
    
    # Get LLM instance
    llm = get_llm()
    
    # Create the chain: prompt -> LLM -> output parser
    chain = prompt | llm | StrOutputParser()
    
    # Execute the chain
    try:
        answer = chain.invoke({
            "context": context,
            "question": question
        })
        return answer.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}\n\nPlease check your OpenRouter API key and try again."


def run_rag_with_sources(question: str, docs: List[Document]) -> dict:
    answer = run_rag(question, docs)
    
    return {
        "answer": answer,
        "sources": [
            {
                "content": doc.page_content[:200] + "...",  # First 200 chars
                "metadata": doc.metadata
            }
            for doc in docs
        ],
        "num_sources": len(docs)
    }