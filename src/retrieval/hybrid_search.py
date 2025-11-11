"""
Hybrid search combining BM25 (keyword) and FAISS (semantic) retrieval.
Uses Reciprocal Rank Fusion to merge results.
"""
import numpy as np
from typing import List, Dict, Tuple

import faiss
from rank_bm25 import BM25Okapi
from openai import OpenAI

import os
import sys
# Add src directory to path so we can import from src package
# This allows running the script from the top-level directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.retrieval.reranker import CrossEncoderReranker
from src.api_key import get_openai_key


class HybridSearcher:
    """
    Combines BM25 and vector search with Reciprocal Rank Fusion.
    """
    
    def __init__(
        self,
        faiss_index: faiss.Index,
        bm25_index: BM25Okapi,
        chunks: List[Dict],
        embedding_model: str = "text-embedding-3-small",
        use_reranking: bool = True
    ):
        self.faiss_index = faiss_index
        self.bm25_index = bm25_index
        self.chunks = chunks
        self.embedding_model = embedding_model
        
        # Initialize OpenAI client for query embeddings
        self.client = OpenAI(api_key=get_openai_key())
        
        # Precompute tokenized chunks for BM25
        self.tokenized_chunks = [
            chunk['content'].lower().split()
            for chunk in chunks
        ]

        # Initialize reraker if enabled
        self.use_reranking = use_reranking
        self.reranker = None
        if self.use_reranking:
            print("[Searcher] Initializing CrossEncoderReranker...")
            self.reranker = CrossEncoderReranker()

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query string."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=[query]
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        
        # Normalize for cosine similarity via L2 distance
        faiss.normalize_L2(embedding.reshape(1, -1))
        
        return embedding
    
    def vector_search(self, query: str, top_k: int = 25) -> List[Tuple[int, float]]:
        """
        Semantic search using FAISS.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of (chunk_index, distance) tuples
        """
        # Embed query
        query_embedding = self.embed_query(query)
        
        # Search FAISS index
        if self.faiss_index is not None:
            # ! Ignore Pylance warning about missing arguments !
            distances, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1), 
                k=min(top_k, len(self.chunks))
            ) # type: ignore
        else:
            print("Error: FAISS index not loaded properly")
            exit(1)
        
        # Return as list of (index, distance) tuples
        results = [
            (int(idx), float(dist))
            for idx, dist in zip(indices[0], distances[0])
        ]
        
        return results
    
    def bm25_search(self, query: str, top_k: int = 25) -> List[Tuple[int, float]]:
        """
        Keyword search using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of (chunk_index, score) tuples
        """
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores for all documents
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return as list of (index, score) tuples
        results = [
            (int(idx), float(scores[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def reciprocal_rank_fusion(
        self,
        rankings: List[List[Tuple[int, float]]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Merge multiple rankings using Reciprocal Rank Fusion.
        
        RRF formula: score = sum(1 / (k + rank_i)) for each ranking i
        
        Args:
            rankings: List of rankings, each is a list of (doc_id, score) tuples
            k: Constant to prevent top results from dominating (default 60)
        
        Returns:
            Fused ranking as list of (doc_id, rrf_score) tuples
        """
        # Collect all document IDs
        all_doc_ids = set()
        for ranking in rankings:
            all_doc_ids.update(doc_id for doc_id, _ in ranking)
        
        # Calculate RRF scores
        rrf_scores = {}
        
        for doc_id in all_doc_ids:
            rrf_score = 0.0
            
            # Add contribution from each ranking
            for ranking in rankings:
                # Find rank of this doc in this ranking
                rank = None
                for pos, (ranked_doc_id, _) in enumerate(ranking):
                    if ranked_doc_id == doc_id:
                        rank = pos + 1  # 1-indexed
                        break
                
                # If doc appears in this ranking, add its contribution
                if rank is not None:
                    rrf_score += 1.0 / (k + rank)
            
            rrf_scores[doc_id] = rrf_score
        
        # Sort by RRF score (descending)
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_docs
    
    def _fuse_and_rank(
        self,
        query: str,
        top_k: int,
        bm25_k: int,
        vector_k: int
    ) -> List[Dict]:
        """
        Helper method to perform retrieval, fusion, and ranking.
        
        Args:
            query: Search query (or hypothetical document)
            top_k: Number of final results to return
            bm25_k: Number of results to retrieve from BM25
            vector_k: Number of results to retrieve from vector search
        
        Returns:
            List of chunk dicts with relevance scores
        """
        # Get results from both methods
        bm25_results = self.bm25_search(query, top_k=bm25_k)
        vector_results = self.vector_search(query, top_k=vector_k)
        
        # Fuse rankings
        fused_results = self.reciprocal_rank_fusion([bm25_results, vector_results])
        
        # Get top-k chunks with scores
        results = []
        for doc_id, rrf_score in fused_results[:top_k]:
            chunk = self.chunks[doc_id].copy()
            chunk['relevance_score'] = rrf_score
            results.append(chunk)
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        bm25_k: int = 25,
        vector_k: int = 25
    ) -> List[Dict]:
        """
        Perform hybrid search combining BM25 and vector search.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            bm25_k: Number of results to retrieve from BM25
            vector_k: Number of results to retrieve from vector search
        
        Returns:
            List of chunk dicts with relevance scores
        """
        return self._fuse_and_rank(query, top_k, bm25_k, vector_k)
    
    def hyde_search(
        self,
        query: str,
        top_k: int = 5,
        bm25_k: int = 25,
        vector_k: int = 25
    ) -> List[Dict]:
        """
        Perform hybrid search using HyDE (Hypothetical Document Embeddings).

        Generates a hypothetical answer, then searches with that instead of the query.
        Better for vague/exploratory queries.

        Args:
            query: Search query
            top_k: Number of final results to return
            bm25_k: Number of results to retrieve from BM25
            vector_k: Number of results to retrieve from vector search

        Returns:
            List of chunk dicts with relevance scores
        """
        from src.retrieval.hyde import HyDEGenerator

        # Generate hypothetical answer
        hyde_gen = HyDEGenerator()
        hypothetical_answer = hyde_gen.generate_hypothetical_answer(query)

        # Search using the hypothetical answer instead of the query
        print(f"[HyDE Search] Using hypothetical answer for retrieval.")

        return self._fuse_and_rank(hypothetical_answer, top_k, bm25_k, vector_k)
    
    def hybrid_search_with_reranking(
        self,
        query: str,
        top_k: int = 5,
        retrieve_k: int = 50  # Retrieve more, then rerank to top_k
    ) -> List[Dict]:
        """
        Hybrid search with cross-encoder reranking.
        
        Two-stage retrieval:
        1. Fast hybrid search retrieves top-50 candidates
        2. Slow cross-encoder reranks to select best top-5
        
        Args:
            query: Search query
            top_k: Number of final results after reranking
            retrieve_k: Number of candidates to retrieve before reranking
        
        Returns:
            List of reranked chunk dicts
        """
        # Stage 1: Retrieve candidates with hybrid search
        candidates = self.hybrid_search(
            query=query,
            top_k=retrieve_k
        )
        
        # Stage 2: Rerank candidates
        if self.use_reranking and self.reranker and len(candidates) > 0:
            print(f"[Search] Reranking {len(candidates)} candidates to top-{top_k}")
            reranked = self.reranker.rerank(query, candidates, top_k=top_k)
            return reranked
        else:
            # No reranking - just return top-k from hybrid search
            return candidates[:top_k]
    
    def hyde_search_with_reranking(
        self,
        query: str,
        top_k: int = 5,
        retrieve_k: int = 50
    ) -> List[Dict]:
        """
        HyDE search with cross-encoder reranking.
        
        Args:
            query: Search query
            top_k: Number of final results after reranking
            retrieve_k: Number of candidates to retrieve before reranking
        
        Returns:
            List of reranked chunk dicts
        """

        candidates = self.hyde_search(query, top_k, retrieve_k // 2)

        # Rerank using ORIGINAL query (not hypothetical answer)
        # This is important - we want relevance to the actual question
        if self.use_reranking and self.reranker and len(candidates) > 0:
            print(f"[HyDE Search] Reranking to top-{top_k}")
            reranked = self.reranker.rerank(query, candidates, top_k=top_k)
            return reranked
        else:
            return candidates[:top_k]
    

    

# Test the hybrid searcher
if __name__ == "__main__":
    import os
    import sys  
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    from src.ingestion.indexer import DocumentIndexer
    
    print("Hybrid Search Test\n")
    
    # Load indexes
    indexer = DocumentIndexer()
    faiss_index, bm25_index, chunks = indexer.load_indexes()
    
    # Initialize searcher
    searcher = HybridSearcher(faiss_index, bm25_index, chunks)
    

    # Test queries using hybrid search
    test_queries = [
        "How do I add path parameters?",
        "What is dependency injection in FastAPI?",
        "How to handle 404 errors?",
        "async functions with database connections"
    ]
    
    print("="*60)
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 60)
        
        # Perform hybrid search
        results = searcher.hybrid_search(query, top_k=3)
        
        # Display results
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['source']}")
            print(f"   Score: {result['relevance_score']:.4f}")
            print(f"   Preview: {result['content'][:120]}...")
        
        print("\n" + "="*60)



    # Compare with individual methods (Hybrid vs BM25-only vs Vector-only)
    print("\n" + "="*60)
    print("COMPARISON: Hybrid vs BM25 vs Vector")
    print("="*60)

    query = "How do I handle HTTP 404 errors?"

    print(f"\nQuery: '{query}'")

    # BM25 only
    print("\n--- BM25 Only (Top 3) ---")
    bm25_results = searcher.bm25_search(query, top_k=3)
    for i, (idx, score) in enumerate(bm25_results, 1):
        print(f"{i}. {chunks[idx]['source']}: {score:.3f}")

    # Vector only  
    print("\n--- Vector Only (Top 3) ---")
    vector_results = searcher.vector_search(query, top_k=3)
    for i, (idx, score) in enumerate(vector_results, 1):
        print(f"{i}. {chunks[idx]['source']}: {score:.3f}")

    # Hybrid
    print("\n--- Hybrid (Top 3) ---")
    hybrid_results = searcher.hybrid_search(query, top_k=3)
    for i, result in enumerate(hybrid_results, 1):
        print(f"{i}. {result['source']}: {result['relevance_score']:.4f}")
