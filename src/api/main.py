"""
FastAPI service for RAG Q&A over FastAPI documentation.
"""

import sys
import mangum
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.api_key import get_openai_key
from ingestion.indexer import DocumentIndexer
from retrieval.hybrid_search import HybridSearcher
from routing.query_router import QueryRouter


load_dotenv()


# Global state (loaded once at startup)
searcher: Optional[HybridSearcher] = None
router: Optional[QueryRouter] = None
llm_client: Optional[OpenAI] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler to initialize indexes and LLM client on startup.
    This replaces the deprecated @app.on_event("startup") handler.

    Load indexes and initialize the HybridSearcher and LLM client.
    """
    global searcher, router, llm_client

    print("Loading indexes...")
    indexer = DocumentIndexer()
    faiss_index, bm25_index, chunks = indexer.load_indexes()

    print("Initializing searcher...")
    searcher = HybridSearcher(faiss_index, bm25_index, chunks, use_reranking=True)

    print("Initializing query router...")
    router = QueryRouter(model="gpt-4o-mini")  # Use a fast/cheap model for classification

    print("Initializing LLM client...")
    llm_client = OpenAI(api_key=get_openai_key())

    print("CHECK! Service ready with agentic routing!")

    yield # <-- Yield control back to FastAPI to start serving requests

    # Optional shutdown logic can go here if needed (close clients, flush metrics, etc.)


# Initialize FastAPI app with lifespan handler
app = FastAPI(
    title="FastAPI Docs RAG Service",
    description="Intelligent Q&A over FastAPI documentation using RAG",
    version="0.2.0",
    lifespan=lifespan,
)

# AWS Lambda handler
handler = mangum.Mangum(app)


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
    query_type: str
    retrieval_stategy: str
    reasoning:str


def generate_answer(query: str, context_chunks: List[Dict], query_type: str) -> str:
    """
    Generate answer using LLM with retrieved context.
    
    Args:
        query: User question
        context_chunks: Retrieved document chunks
        query_type: Type of the query for prompt adaptation
    
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

    # Adapt prompt based on query type
    if query_type == "MULTI_PART":
        instruction_note = "This is a multi-part question. Address each part clearly."
    elif query_type == "COMPLEX":
        instruction_note = "This is an exploratory question. Provide comprehensive guidance."
    else:
        instruction_note = "Provide a direct, focused answer."
    
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
6. {instruction_note}

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
        "service": "FastAPI Docs RAG with Agentic Routing",
        "status": "running",
        "version": "0.2.0",
        "features": ["hybrid_search", "query_routing", "adaptive_retrieval"],
        "docs": "/docs"
    }


@app.post("/query", response_model=QueryResponse)
async def query_docs(request: QueryRequest):
    """
    Answer questions about FastAPI documentation with intelligent routing.
    
    The system automatically analyzes your query and selects the optimal
    retrieval strategy:
    - SIMPLE: Direct hybrid search
    - COMPLEX: HyDE preprocessing for better vocabulary matching
    - MULTI_PART: Query decomposition (coming soon)
    
    Args:
        request: Query request with question and optional parameters
    
    Returns:
        Answer with sources, query type, and routing information
    """
    if searcher is None or router is None or llm_client is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:

        # Route the query
        query_type, strategy, config = router.route_query(request.query)

        TESTING_WITHOUT_HYDE = False  # Set to True to bypass HyDE during testing

        # Retrieve relevant chunks
        if config["use_hyde"] and not TESTING_WITHOUT_HYDE:
            # COMPLEX queries: use HyDE preprocessing
            if searcher.use_reranking:
                print("[API] Using HyDE search with reranking for complex query")
                results = searcher.hyde_search_with_reranking(
                    query=request.query,
                    top_k=request.top_k,
                    retrieve_k=50
                )
            else:
                print("[API] Using HyDE search for complex query")
                results = searcher.hyde_search(
                    query=request.query,
                    top_k=request.top_k
                )
        else:
            # SIMPLE queries: direct hybrid search
            if searcher.use_reranking:
                print("[API] Using direct hybrid search with reranking")
                results = searcher.hybrid_search_with_reranking(
                    query=request.query,
                    top_k=request.top_k,
                    retrieve_k=50
                )
            else:
                print("[API] Using direct hybrid search")
                results = searcher.hybrid_search(
                    query=request.query,
                    top_k=request.top_k
                )
        
        # Generate answer
        answer = generate_answer(request.query, results, query_type.value)
        
        # Format sources
        sources = [
            Source(
                source=chunk['source'],
                url=chunk.get('url', ''),
                relevance_score=chunk['relevance_score']
            )
            for chunk in results
        ]

        # Generate reasoning
        classification = router.classify_query(request.query)
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            query_type=query_type.value,
            retrieval_stategy=strategy,
            reasoning = classification.reasoning
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# For testing locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
