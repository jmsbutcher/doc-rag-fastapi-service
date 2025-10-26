"""
FastAPI service for RAG Q&A over FastAPI documentation.
"""
import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# Import our components
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ingestion.indexer import DocumentIndexer
from retrieval.hybrid_search import HybridSearcher

load_dotenv()


# Global state (loaded once at startup)
searcher: Optional[HybridSearcher] = None
llm_client: Optional[OpenAI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler to initialize indexes and LLM client on startup.

    This replaces the deprecated @app.on_event("startup") handler.
    """
    global searcher, llm_client

    print("Loading indexes...")
    indexer = DocumentIndexer()
    faiss_index, bm25_index, chunks = indexer.load_indexes()

    print("Initializing searcher...")
    searcher = HybridSearcher(faiss_index, bm25_index, chunks)

    print("Initializing LLM client...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    llm_client = OpenAI(api_key=api_key)

    print("✓ Service ready!")

    # Yield control back to FastAPI to start serving requests
    yield

    # Optional shutdown logic can go here (close clients, flush metrics, etc.)


# Initialize FastAPI app with lifespan handler
app = FastAPI(
    title="FastAPI Docs RAG Service",
    description="Intelligent Q&A over FastAPI documentation using RAG",
    version="0.1.0",
    lifespan=lifespan,
)


# Request/Response models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks")


class Source(BaseModel):
    source: str
    url: str
    relevance_score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Source]
    retrieval_method: str = "hybrid"


def generate_answer(query: str, context_chunks: List[Dict]) -> str:
    """
    Generate answer using LLM with retrieved context.
    
    Args:
        query: User question
        context_chunks: Retrieved document chunks
    
    Returns:
        Generated answer with citations
    """

    if llm_client is None:
        raise RuntimeError("LLM client not initialized")
    
    # Format context from chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get('source', 'Unknown')
        content = chunk.get('content', '')
        context_parts.append(f"[{i}] From {source}:\n{content}\n")
    
    context = "\n".join(context_parts)
    
    # Create prompt
    prompt = f"""
You are a helpful assistant answering questions about FastAPI based on official documentation.

Context from documentation:
{context}

Question: {query}

Instructions:
1. Answer the question based ONLY on the provided context
2. Be specific and cite sources using [1], [2], etc.
3. If the context doesn't contain enough information, say so
4. Keep answers concise but complete
5. Include code examples if they appear in the context

Answer:"""

    # Call LLM
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",  # Fast and cheap for testing
        messages=[
            {"role": "system", "content": "You are a FastAPI documentation expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Lower temperature for more factual answers
        max_tokens=500
    )
    
    answer = response.choices[0].message.content
    if answer is None:
        return "I apologize, but I couldn't generate a response to your question."
    return answer


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "FastAPI Docs RAG",
        "status": "running",
        "docs": "/docs"
    }


@app.post("/query", response_model=QueryResponse)
async def query_docs(request: QueryRequest):
    """
    Answer questions about FastAPI documentation.
    
    Args:
        request: Query request with question and optional parameters
    
    Returns:
        Answer with sources and citations
    """
    if searcher is None or llm_client is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Retrieve relevant chunks
        results = searcher.hybrid_search(
            query=request.query,
            top_k=request.top_k
        )
        
        # Generate answer
        answer = generate_answer(request.query, results)
        
        # Format sources
        sources = [
            Source(
                source=chunk['source'],
                url=chunk.get('url', ''),
                relevance_score=chunk['relevance_score']
            )
            for chunk in results
        ]
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# For testing locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
