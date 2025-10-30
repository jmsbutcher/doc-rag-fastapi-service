"""
Cross-encoder reranking for improved retrieval precision.
"""
from typing import List, Dict
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Reranks retrieved documents using cross-encoder for better relevance.
    
    Cross-encoders process query-document pairs jointly, capturing nuanced
    relevance that bi-encoders (used for initial retrieval) miss.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model for reranking
                       Default is fast, accurate model trained on MS MARCO
        """
        print(f"[Reranker] Loading model: {model_name}")
        self.model = CrossEncoder(model_name)
        print("[Reranker] Model loaded successfully")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank documents using cross-encoder scores.
        
        Args:
            query: Search query
            documents: List of document dicts with 'content' field
            top_k: Number of top documents to return
        
        Returns:
            Reranked list of top-k documents with updated scores
        """
        if not documents:
            return []
        
        # Prepare query-document pairs for cross-encoder
        pairs = [
            [query, doc['content']] 
            for doc in documents
        ]
        
        # Get cross-encoder scores
        # These are relevance scores (higher = more relevant)
        scores = self.model.predict(pairs)
        
        # Sort documents by cross-encoder score
        scored_docs = [
            (doc, float(score))
            for doc, score in zip(documents, scores)
        ]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k with updated scores
        reranked = []
        for doc, score in scored_docs[:top_k]:
            doc_copy = doc.copy()
            doc_copy['rerank_score'] = score
            # Keep original retrieval score as metadata
            doc_copy['original_score'] = doc.get('relevance_score', 0.0)
            reranked.append(doc_copy)
        
        return reranked
    
    def rerank_with_threshold(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Rerank and filter by minimum relevance score.
        
        Useful for ensuring quality - only return documents above threshold.
        
        Args:
            query: Search query
            documents: List of document dicts
            top_k: Maximum number of documents to return
            min_score: Minimum rerank score to include
        
        Returns:
            Filtered and reranked documents
        """
        reranked = self.rerank(query, documents, top_k=len(documents))
        
        # Filter by minimum score
        filtered = [
            doc for doc in reranked 
            if doc['rerank_score'] >= min_score
        ]
        
        return filtered[:top_k]


# Test the reranker
if __name__ == "__main__":
    print("Cross-Encoder Reranker Test\n")
    print("="*60)
    
    # Create some test documents
    test_query = "How do I add path parameters to my FastAPI route?"
    
    test_docs = [
        {
            "content": "Path parameters are declared using curly braces in the route path. For example: @app.get('/items/{item_id}') allows you to capture the item_id from the URL.",
            "source": "path-params.md",
            "relevance_score": 0.85
        },
        {
            "content": "Query parameters are added after the ? in the URL. They are optional by default in FastAPI and can be declared as function parameters.",
            "source": "query-params.md",
            "relevance_score": 0.82
        },
        {
            "content": "FastAPI automatically validates path parameters based on Python type hints. You can use int, str, float, or custom types.",
            "source": "path-validation.md",
            "relevance_score": 0.80
        },
        {
            "content": "Request body parameters are declared using Pydantic models. This allows for complex data validation and serialization.",
            "source": "body-params.md",
            "relevance_score": 0.75
        }
    ]
    
    # Initialize reranker
    reranker = CrossEncoderReranker()
    
    print(f"\nQuery: '{test_query}'")
    print("\n" + "-"*60)
    print("BEFORE RERANKING (top 3 by retrieval score):")
    print("-"*60)
    
    for i, doc in enumerate(test_docs[:4], 1):
        print(f"\n{i}. {doc['source']}")
        print(f"   Retrieval Score: {doc['relevance_score']:.3f}")
        print(f"   Content: {doc['content'][:80]}...")
    
    # Rerank
    print("\n" + "="*60)
    print("RERANKING...")
    print("="*60)
    
    reranked = reranker.rerank(test_query, test_docs, top_k=3)
    
    print("\n" + "-"*60)
    print("AFTER RERANKING (top 3 by cross-encoder score):")
    print("-"*60)
    
    for i, doc in enumerate(reranked, 1):
        print(f"\n{i}. {doc['source']}")
        print(f"   Rerank Score: {doc['rerank_score']:.3f}")
        print(f"   Original Score: {doc['original_score']:.3f}")
        print(f"   Content: {doc['content'][:80]}...")
    
    print("\n" + "="*60)
    print("\nNotice how reranking can change the order!")
    print("Cross-encoder captures query-document relevance better than")
    print("simple vector similarity or BM25 keyword matching.\n")
    print("="*60)
