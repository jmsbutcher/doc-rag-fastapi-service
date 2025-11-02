"""
Retrieval Evaluator - Computes IR (Information Retrieval) metrics using 
ground truth labels.

Metrics:
- Precision@K: Proportion of relevant docs in top-K
- Recall@K: Proportion of all relevant docs retrieved in top-K
- NDCG@K: Normalized Discounted Cumulative Gain (considers ranking order)
- MRR: Mean Reciprocal Rank (position of first relevant result)
"""

import json
import numpy as np
from typing import Dict, List
import math


def load_ground_truth(path: str = "evaluation/ground_truth.json") -> Dict:
    """Load manually labeled ground truth data."""
    with open(path, 'r') as f:
        return json.load(f)


def dcg_at_k(relevances: List[int], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at rank k.
    
    DCG = sum(rel_i / log2(i+1)) for i=1 to k
    Higher relevance scores and earlier positions contribute more.
    """
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg_at_k(relevances: List[int], k: int) -> float:
    """
    Compute Normalized DCG at rank k.
    
    NDCG = DCG / IDCG (ideal DCG)
    Normalizes to [0, 1] by dividing by best possible DCG.
    """
    actual_dcg = dcg_at_k(relevances, k)
    ideal_dcg = dcg_at_k(sorted(relevances, reverse=True), k)
    
    if ideal_dcg == 0:
        return 0.0
    
    return actual_dcg / ideal_dcg


def precision_at_k(relevances: List[int], k: int) -> float:
    """
    Compute Precision@K.
    
    Precision@K = (# relevant docs in top-K) / K
    Measures what fraction of retrieved docs are relevant.
    """
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    
    # Count docs with relevance >= 1 (somewhat or highly relevant)
    relevant_count = sum(1 for rel in relevances if rel >= 1)
    return relevant_count / len(relevances)


def recall_at_k(relevances: List[int], all_relevant_count: int, k: int) -> float:
    """
    Compute Recall@K.
    
    Recall@K = (# relevant docs in top-K) / (total # relevant docs)
    Measures what fraction of all relevant docs were retrieved.
    """
    if all_relevant_count == 0:
        return 0.0
    
    relevances = relevances[:k]
    relevant_count = sum(1 for rel in relevances if rel >= 1)
    return relevant_count / all_relevant_count


def mrr(relevances: List[int]) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    MRR = 1 / (rank of first relevant document)
    Higher is better (1.0 = relevant doc at position 1).
    """
    for idx, rel in enumerate(relevances):
        if rel >= 1:  # First relevant doc
            return 1.0 / (idx + 1)
    return 0.0  # No relevant docs found


def evaluate_retrieval(
    retrieved_chunk_ids: List[str],
    ground_truth_labels: Dict[str, int],
    k_values: List[int] = [3, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate retrieval performance using ground truth labels.
    
    Args:
        retrieved_chunk_ids: List of chunk IDs in retrieval order
        ground_truth_labels: Dict mapping chunk_id -> relevance (0, 1, 2)
        k_values: List of K values to compute metrics for
    
    Returns:
        Dictionary of metric_name -> score
    """
    # Map retrieved chunks to relevance scores
    relevances = [
        ground_truth_labels.get(chunk_id, 0) 
        for chunk_id in retrieved_chunk_ids
    ]
    
    # Count total relevant docs in ground truth
    total_relevant = sum(1 for rel in ground_truth_labels.values() if rel >= 1)
    
    metrics = {}
    
    # Compute metrics at different K values
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(relevances, k)
        metrics[f"recall@{k}"] = recall_at_k(relevances, total_relevant, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(relevances, k)
    
    # MRR doesn't need K
    metrics["mrr"] = mrr(relevances)
    
    return metrics


def evaluate_all_queries(
    ground_truth_path: str = "evaluation/ground_truth.json",
    retrieval_results_path: str = "evaluation/retrieval_results.json"
) -> Dict:
    """
    Evaluate retrieval across all queries in ground truth.
    
    Expects retrieval_results.json with format:
    {
        "query": "...",
        "retrieved_chunk_ids": ["chunk_1", "chunk_2", ...]
    }
    """
    # Load ground truth
    gt_data = load_ground_truth(ground_truth_path)
    
    # Load retrieval results (you'll generate this)
    with open(retrieval_results_path, 'r') as f:
        retrieval_results = json.load(f)
    
    # Build lookup: query -> ground truth labels
    query_to_labels = {}
    for query_data in gt_data["queries"]:
        query = query_data["query"]
        labels = {
            item["chunk_id"]: item["relevance_label"]
            for item in query_data["labeled_chunks"]
        }
        query_to_labels[query] = labels
    
    # Evaluate each query
    all_metrics = []
    for result in retrieval_results:
        query = result["query"]
        retrieved_ids = result["retrieved_chunk_ids"]
        
        if query not in query_to_labels:
            print(f"Warning: Query not in ground truth: {query}")
            continue
        
        metrics = evaluate_retrieval(
            retrieved_chunk_ids=retrieved_ids,
            ground_truth_labels=query_to_labels[query],
            k_values=[3, 5, 10]
        )
        
        all_metrics.append({
            "query": query,
            "metrics": metrics
        })
    
    # Compute average metrics
    avg_metrics = {}
    if all_metrics:
        metric_names = all_metrics[0]["metrics"].keys()
    
        for metric_name in metric_names:
            scores = [q["metrics"][metric_name] for q in all_metrics]
            avg_metrics[metric_name] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores)
            }
    
    return {
        "per_query_metrics": all_metrics,
        "average_metrics": avg_metrics,
        "num_queries": len(all_metrics)
    }


def compare_strategies(strategy_names: list = ["1_baseline", "2_reranking", "3_routing", "4_full"]):
    """
    Compare multiple retrieval strategies and print comparison table.
    
    Args:
        strategy_names: List of strategy names matching file suffixes
    """
    print("\n" + "="*80)
    print("RETRIEVAL STRATEGY COMPARISON")
    print("="*80)
    
    all_results = {}
    
    # Load and evaluate each strategy
    for strategy in strategy_names:
        results_path = f"evaluation/retrieval_results_{strategy}.json"
        print(f"\nEvaluating {strategy}...")
        
        try:
            results = evaluate_all_queries(retrieval_results_path=results_path)
            all_results[strategy] = results
            
            # Save individual results
            metrics_path = f"evaluation/retrieval_metrics_{strategy}.json"
            with open(metrics_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"CHECK! Saved metrics to {metrics_path}")
            
        except FileNotFoundError:
            print(f"X File not found: {results_path}")
            continue
    
    # Print comparison table
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("COMPARISON TABLE")
        print("="*80)
        
        # Define metrics to compare
        key_metrics = ["precision@5", "recall@5", "ndcg@10", "mrr"]
        
        # Print header
        header = f"{'Metric':<15}"
        for strategy in strategy_names:
            if strategy in all_results:
                header += f" | {strategy:<12}"
        print(header)
        print("-" * len(header))
        
        # Print each metric
        for metric in key_metrics:
            row = f"{metric:<15}"
            baseline_value = None
            
            for i, strategy in enumerate(strategy_names):
                if strategy not in all_results:
                    row += f" | {'N/A':<12}"
                    continue
                
                value = all_results[strategy]["average_metrics"][metric]["mean"]
                
                if i == 0:
                    # Baseline - no improvement shown
                    baseline_value = value
                    row += f" | {value:>7.3f}     "
                else:
                    # Calculate improvement vs baseline
                    if baseline_value and baseline_value > 0:
                        improvement = ((value - baseline_value) / baseline_value) * 100
                        row += f" | {value:>7.3f} ({improvement:+.1f}%)"
                    else:
                        row += f" | {value:>7.3f}     "
            
            print(row)
        
        print("="*80)
    
    return all_results



if __name__ == "__main__":
    compare_strategies(["1_baseline", "2_reranking", "3_routing", "4_full"])
