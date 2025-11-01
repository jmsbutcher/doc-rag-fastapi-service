
"""
Run retrieval on test queries and save results for evaluation.

This compares 4 strategies to show incremental improvements:
1. Baseline: Hybrid search (BM25 + vector), no reranking, no routing
2. +Reranking: Hybrid search with cross-encoder reranking
3. +Routing: Hybrid search with reranking + agentic routing (no HyDE)
4. +HyDE: Full system (routing with HyDE for COMPLEX queries)
"""

import os
import sys
import json

# # This allows running the script from the top-level directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.indexer import DocumentIndexer
from src.retrieval.hybrid_search import HybridSearcher
from src.routing.query_router import QueryRouter


def load_test_queries(path: str = "evaluation/ground_truth.json") -> list:
    """Load test queries from ground truth file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return [q["query"] for q in data["queries"]]


# Global state (loaded once)
_indexer = None
_faiss_index = None
_bm25_index = None
_chunks = None


def get_indexes():
    """Load indexes once and reuse."""
    global _indexer, _faiss_index, _bm25_index, _chunks
    
    if _indexer is None:
        print("Loading indexes...")
        _indexer = DocumentIndexer()
        _faiss_index, _bm25_index, _chunks = _indexer.load_indexes()
        print(f"Loaded {len(_chunks)} chunks")
    
    return _faiss_index, _bm25_index, _chunks


def _get_searcher(use_reranking: bool = False) -> HybridSearcher:
    """
    Helper to get initialized HybridSearcher with proper type checking.
    Centralizes initialization logic and eliminates Pylance warnings.
    
    Args:
        use_reranking: Whether to enable cross-encoder reranking
        
    Returns:
        Initialized HybridSearcher instance
    """
    faiss_index, bm25_index, chunks = get_indexes()
    
    # Type assertions to satisfy Pylance (indexes are loaded, never None here)
    assert faiss_index is not None, "FAISS index not loaded"
    assert bm25_index is not None, "BM25 index not loaded"
    assert chunks is not None, "Chunks not loaded"
    
    return HybridSearcher(
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        chunks=chunks,
        use_reranking=use_reranking
    )


def _get_router() -> QueryRouter:
    """
    Helper to get initialized QueryRouter with proper type checking.
    
    Returns:
        Initialized QueryRouter instance
    """
    return QueryRouter(model="gpt-4o-mini")


def strategy_1_baseline(query: str, top_k: int = 10) -> list:
    """
    Strategy 1: Baseline
    - Hybrid search (BM25 + vector)
    - NO reranking
    - NO routing
    """
    # Get searcher WITHOUT reranking
    searcher = _get_searcher(use_reranking=False)
    
    # Direct hybrid search
    results = searcher.hybrid_search(query, top_k=top_k)
    
    # Extract chunk IDs from results
    return [result.get('chunk_id', result['source']) for result in results]


def strategy_2_with_reranking(query: str, top_k: int = 10) -> list:
    """
    Strategy 2: Add Reranking
    - Hybrid search (BM25 + vector)
    - WITH cross-encoder reranking
    - NO routing
    """
    # Get searcher WITH reranking
    searcher = _get_searcher(use_reranking=True)
    
    # Hybrid search with reranking (retrieve 50, rerank to top_k)
    results = searcher.hybrid_search_with_reranking(
        query, 
        top_k=top_k,
        retrieve_k=50
    )
    
    return [result.get('chunk_id', result['source']) for result in results]


def strategy_3_with_routing(query: str, top_k: int = 10) -> list:
    """
    Strategy 3: Add Routing (but no HyDE yet)
    - Hybrid search (BM25 + vector)
    - WITH cross-encoder reranking
    - WITH routing (but force direct search, no HyDE)
    
    This isolates routing logic without HyDE complexity.
    """
    # Get searcher WITH reranking
    searcher = _get_searcher(use_reranking=True)
    
    # Use router to classify query
    router = _get_router()
    query_type, strategy, config = router.route_query(query)
    
    # But always use direct search (no HyDE) to isolate routing impact
    results = searcher.hybrid_search_with_reranking(
        query, 
        top_k=top_k,
        retrieve_k=50
    )
    
    return [result.get('chunk_id', result['source']) for result in results]


def strategy_4_full_system(query: str, top_k: int = 10) -> list:
    """
    Strategy 4: Full System
    - Hybrid search (BM25 + vector)
    - WITH cross-encoder reranking
    - WITH routing
    - WITH HyDE for COMPLEX queries
    
    This is your complete system.
    """
    # Get searcher WITH reranking
    searcher = _get_searcher(use_reranking=True)
    
    # Use router to classify and route query
    router = _get_router()
    query_type, strategy, config = router.route_query(query)
    
    # Route based on query type
    if config["use_hyde"]:
        # COMPLEX query: use HyDE
        print(f"  → {query_type.value}: Using HyDE")
        results = searcher.hyde_search_with_reranking(
            query,
            top_k=top_k,
            retrieve_k=50
        )
    else:
        # SIMPLE query: direct search
        print(f"  → {query_type.value}: Direct search")
        results = searcher.hybrid_search_with_reranking(
            query,
            top_k=top_k,
            retrieve_k=50
        )
    
    return [result.get('chunk_id', result['source']) for result in results]


def run_strategy(
    strategy_name: str,
    strategy_fn,
    queries: list,
    top_k: int = 10
):
    """
    Run a retrieval strategy on all queries and save results.
    
    Args:
        strategy_name: Name for output file
        strategy_fn: Function taking (query, top_k) -> chunk_ids
        queries: List of test queries
        top_k: Number of results to retrieve
    """
    results = []
    
    print("\n" + "="*60)
    print(f"STRATEGY: {strategy_name}")
    print("="*60)
    
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query[:60]}...")
        
        try:
            retrieved_chunk_ids = strategy_fn(query, top_k)
            results.append({
                "query": query,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "num_retrieved": len(retrieved_chunk_ids)
            })
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "query": query,
                "retrieved_chunk_ids": [],
                "num_retrieved": 0,
                "error": str(e)
            })
    
    # Save results
    output_path = f"evaluation/retrieval_results_{strategy_name}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved to {output_path}")
    return results


def main():
    """Run retrieval evaluation on all test queries."""
    
    print("\n" + "="*70)
    print("RETRIEVAL EVALUATION - Incremental Feature Comparison")
    print("="*70)
    print("\nComparing 4 strategies:")
    print("  1. Baseline: Hybrid search only")
    print("  2. +Reranking: Add cross-encoder reranking")
    print("  3. +Routing: Add query classification")
    print("  4. +HyDE: Full system with HyDE for complex queries")
    print()
    
    # Load test queries
    queries = load_test_queries()
    print(f"Loaded {len(queries)} test queries\n")
    
    # Run all 4 strategies
    run_strategy("1_baseline", strategy_1_baseline, queries, top_k=10)
    run_strategy("2_reranking", strategy_2_with_reranking, queries, top_k=10)
    run_strategy("3_routing", strategy_3_with_routing, queries, top_k=10)
    run_strategy("4_full", strategy_4_full_system, queries, top_k=10)
    
    print("\n" + "="*70)
    print("✓ All strategies complete!")
    print("="*70)
    print("\nResults saved to evaluation/retrieval_results_*.json")
    print("\nNext step: Run retrieval_evaluator.py to compute metrics")


if __name__ == "__main__":
    main()