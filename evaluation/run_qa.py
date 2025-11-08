"""
Run full Q&A pipeline (retrieval + generation) and save results for RAGAS evaluation.

This generates qa_results.json files for each strategy needed by ragas_evaluator.py
"""

import os
import sys
import json
import time
from typing import List

# # This allows running the script from the top-level directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api_key import get_openai_key
from src.ingestion.indexer import DocumentIndexer
from src.retrieval.hybrid_search import HybridSearcher
from src.routing.query_router import QueryRouter
from openai import OpenAI

# Initialize global LLM client
client = OpenAI(api_key=get_openai_key())


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


def generate_answer(query: str, contexts: List[str]) -> str:
    """
    Generate answer using LLM with retrieved contexts.
    
    Args:
        query: User question
        contexts: List of relevant text chunks
    
    Returns:
        Generated answer
    """
    # Combine contexts
    context_text = "\n\n".join([
        f"[Context {i+1}]\n{ctx}" 
        for i, ctx in enumerate(contexts)
    ])
    
    # Create prompt
    prompt = f"""You are a FastAPI documentation expert. Answer the question using ONLY the provided context.

Context:
{context_text}

Question: {query}

Instructions:
1. Answer directly and concisely
2. Use information ONLY from the provided context
3. If the context doesn't contain enough information, say so
4. Include code examples if they appear in the context

Answer:"""
    
    # Call LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a FastAPI documentation expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    
    answer = response.choices[0].message.content
    return answer if answer is not None else ""


def run_qa_strategy_1(query: str, top_k: int = 5) -> dict:
    """Strategy 1: Baseline (no reranking, no routing)"""
    # Get searcher WITHOUT reranking
    searcher = _get_searcher(use_reranking=False)
    
    results = searcher.hybrid_search(query, top_k=top_k)
    contexts = [r['content'] for r in results]
    answer = generate_answer(query, contexts)
    
    return {
        "query": query,
        "answer": answer,
        "contexts": contexts
    }


def run_qa_strategy_2(query: str, top_k: int = 5) -> dict:
    """Strategy 2: With reranking"""
    # Get searcher WITH reranking
    searcher = _get_searcher(use_reranking=True)
    
    results = searcher.hybrid_search_with_reranking(
        query, 
        top_k=top_k,
        retrieve_k=50
    )
    contexts = [r['content'] for r in results]
    answer = generate_answer(query, contexts)
    
    return {
        "query": query,
        "answer": answer,
        "contexts": contexts
    }


def run_qa_strategy_3(query: str, top_k: int = 5) -> dict:
    """Strategy 3: With routing (but no HyDE)"""
    # Get searcher WITH reranking
    searcher = _get_searcher(use_reranking=True)
    
    # Classify but don't use HyDE
    router = _get_router()
    query_type, strategy, config = router.route_query(query)
    
    results = searcher.hybrid_search_with_reranking(
        query, 
        top_k=top_k,
        retrieve_k=50
    )
    contexts = [r['content'] for r in results]
    answer = generate_answer(query, contexts)
    
    return {
        "query": query,
        "answer": answer,
        "contexts": contexts,
        "query_type": query_type.value
    }


def run_qa_strategy_4(query: str, top_k: int = 5) -> dict:
    """Strategy 4: Full system (routing + HyDE)"""
    # Get searcher WITH reranking
    searcher = _get_searcher(use_reranking=True)
    
    # Use router to classify and route query
    router = _get_router()
    query_type, strategy, config = router.route_query(query)
    
    # Route based on query type
    if config["use_hyde"]:
        results = searcher.hyde_search_with_reranking(
            query,
            top_k=top_k,
            retrieve_k=50
        )
    else:
        results = searcher.hybrid_search_with_reranking(
            query,
            top_k=top_k,
            retrieve_k=50
        )
    
    contexts = [r['content'] for r in results]
    answer = generate_answer(query, contexts)
    
    return {
        "query": query,
        "answer": answer,
        "contexts": contexts,
        "query_type": query_type.value
    }


def run_strategy(strategy_name: str, strategy_fn, queries: list):
    """
    Run Q&A strategy on all queries and save results.
    
    Args:
        strategy_name: Name for output file
        strategy_fn: Function taking (query) -> result dict
        queries: List of test queries
    """
    print("\n" + "="*60)
    print(f"STRATEGY: {strategy_name}")
    print("="*60)
    
    results = []
    total_time = 0
    
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query[:60]}...")
        
        start = time.time()
        try:
            result = strategy_fn(query)
            elapsed = time.time() - start
            total_time += elapsed
            
            result["latency_seconds"] = elapsed
            results.append(result)
            
            print(f"  CHECK! Generated answer ({elapsed:.2f}s)")
            
        except Exception as e:
            print(f"  X ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "query": query,
                "answer": "",
                "contexts": [],
                "error": str(e)
            })
    
    # Save results
    output_path = f"evaluation/results/qa_results_{strategy_name}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    avg_latency = total_time / len(queries) if queries else 0
    print(f"\nCHECK! Saved to {output_path}")
    print(f"  Average latency: {avg_latency:.2f}s")
    
    return results


def main():
    """Run Q&A pipeline on all test queries for all strategies."""
    
    print("\n" + "="*70)
    print("Q&A EVALUATION - Incremental Feature Comparison")
    print("="*70)
    print("\nRunning full Q&A (retrieval + generation) for 4 strategies:")
    print("  1. Baseline: Hybrid search only")
    print("  2. +Reranking: Add cross-encoder reranking")
    print("  3. +Routing: Add query classification")
    print("  4. +HyDE: Full system with HyDE for complex queries")
    print()
    
    # Load test queries
    queries = load_test_queries()
    print(f"Loaded {len(queries)} test queries\n")
    
    # Run all 4 strategies
    run_strategy("1_baseline", run_qa_strategy_1, queries)
    run_strategy("2_reranking", run_qa_strategy_2, queries)
    run_strategy("3_routing", run_qa_strategy_3, queries)
    run_strategy("4_full", run_qa_strategy_4, queries)
    
    print("\n" + "="*70)
    print("CHECK! Q&A complete! Results saved to evaluation/results/")
    print("="*70)
    print("\nNext step: Run ragas_evaluator.py to compute generation metrics")


if __name__ == "__main__":
    main()
