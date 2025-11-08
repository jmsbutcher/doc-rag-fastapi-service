"""
RAGAS Evaluator - Measures generation quality using RAGAS framework.

Metrics:
- Faithfulness: Are claims in the answer supported by retrieved context?
- Answer Relevancy: Does the answer address the query?
- Context Relevancy: Is retrieved context relevant to the query?

RAGAS uses LLM-as-judge to evaluate these without ground truth answers.
"""
import sys
import json
from langchain_openai import ChatOpenAI
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from api_key import get_openai_key
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy
)


def load_qa_results(path: str = "evaluation/results/qa_results.json") -> List[Dict]:
    """
    Load Q&A results for evaluation.
    
    Expected format:
    {
        "query": "...",
        "answer": "...",
        "contexts": ["...", "..."],  # Retrieved chunks
        "ground_truth": "..." (optional)
    }
    """
    with open(path, 'r') as f:
        return json.load(f)


def prepare_ragas_dataset(qa_results: List[Dict]) -> Dataset:
    """
    Convert Q&A results to RAGAS dataset format.
    
    RAGAS expects:
    - question: str
    - answer: str  
    - contexts: List[str]
    - ground_truth: str (optional, for answer correctness)
    """
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
    }
    
    # Add ground truth if available
    has_ground_truth = any("ground_truth" in r for r in qa_results)
    if has_ground_truth:
        data["ground_truth"] = []
    
    for result in qa_results:
        data["question"].append(result["query"])
        data["answer"].append(result["answer"])
        data["contexts"].append(result["contexts"])
        
        if has_ground_truth:
            data["ground_truth"].append(result.get("ground_truth", ""))
    
    return Dataset.from_dict(data)


def evaluate_generation(
    qa_results_path: str = "evaluation/results/qa_results.json",
    output_path: str = "evaluation/results/generation_metrics.json"
) -> Dict:
    """
    Evaluate generation quality using RAGAS.
    
    Returns dictionary with per-query and average metrics.
    """
    # Load Q&A results
    qa_results = load_qa_results(qa_results_path)
    print(f"Loaded {len(qa_results)} Q&A results")
    
    # Prepare RAGAS dataset
    dataset = prepare_ragas_dataset(qa_results)
    
    # Define metrics to compute (no ground truth required)
    metrics = [
        faithfulness,        # Claims supported by context?
        answer_relevancy,    # Answer addresses query?
    ]
    
    print("\nRunning RAGAS evaluation...")
    print("This will make LLM calls - may take 1-2 minutes")
    print("-" * 60)
    
    # Run evaluation
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=ChatOpenAI(api_key=get_openai_key(), model="gpt-4o-mini") # type: ignore
    )
    
    # RAGAS 0.3.x: Result object has direct attribute access
    # Try multiple access patterns for compatibility
    try:
        # Pattern 1: Direct dictionary access
        results_dict = {
            'faithfulness': list(result['faithfulness']), # type: ignore
            'answer_relevancy': list(result['answer_relevancy']) # type: ignore
        }
    except (TypeError, KeyError):
        try:
            # Pattern 2: Access via .scores attribute
            results_dict = {
                'faithfulness': list(result.scores['faithfulness']), # type: ignore
                'answer_relevancy': list(result.scores['answer_relevancy']) # type: ignore
            }
        except (AttributeError, KeyError):
            # Pattern 3: Direct attribute access
            results_dict = {
                'faithfulness': [result.faithfulness] if hasattr(result, 'faithfulness') else [0], # type: ignore
                'answer_relevancy': [result.answer_relevancy] if hasattr(result, 'answer_relevancy') else [0] # type: ignore
            }
    
    # Build per-query results
    per_query = []
    num_queries = len(qa_results)
    
    for i in range(num_queries):
        query_metrics = {
            "query": qa_results[i]["query"],
            "faithfulness": float(results_dict.get("faithfulness", [0])[i]) if i < len(results_dict.get("faithfulness", [])) else 0.0,
            "answer_relevancy": float(results_dict.get("answer_relevancy", [0])[i]) if i < len(results_dict.get("answer_relevancy", [])) else 0.0,
        }
        per_query.append(query_metrics)
    
    # Compute averages
    import numpy as np
    avg_metrics = {
        "faithfulness": {
            "mean": float(np.mean(results_dict.get("faithfulness", [0]))),
            "std": float(np.std(results_dict.get("faithfulness", [0]))),
        },
        "answer_relevancy": {
            "mean": float(np.mean(results_dict.get("answer_relevancy", [0]))),
            "std": float(np.std(results_dict.get("answer_relevancy", [0]))),
        }
    }
    
    output = {
        "average_metrics": avg_metrics,
        "per_query_metrics": per_query,
        "num_queries": len(qa_results)
    }
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nCHECK! Saved results to {output_path}")
    
    return output


def print_results(results: Dict):
    """Pretty print RAGAS results."""
    print("\n" + "="*60)
    print("GENERATION EVALUATION RESULTS (RAGAS)")
    print("="*60)
    
    print(f"\nEvaluated {results['num_queries']} queries\n")
    
    print("Average Metrics:")
    print("-" * 60)
    for metric, stats in results["average_metrics"].items():
        print(f"{metric:20s}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    print("\n" + "="*60)
    print("\nMetric Interpretation:")
    print("  Faithfulness       : Claims supported by context (target: >=0.95)")
    print("  Answer Relevancy   : Answer addresses query (target: >=0.85)")
    print("\nNote: Using 2 metrics that don't require ground truth answers")
    print("="*60)


def compare_generation_strategies(strategy_names: list = ["1_baseline", "2_reranking", "3_routing", "4_full"]):
    """
    Compare generation quality across multiple strategies.
    
    Args:
        strategy_names: List of strategy names matching file suffixes
    """
    print("\n" + "="*80)
    print("GENERATION QUALITY COMPARISON (RAGAS)")
    print("="*80)
    
    all_results = {}
    
    # Evaluate each strategy
    for strategy in strategy_names:
        qa_path = f"evaluation/results/qa_results_{strategy}.json"
        metrics_path = f"evaluation/results/generation_metrics_{strategy}.json"
        
        print(f"\nEvaluating {strategy}...")
        
        try:
            results = evaluate_generation(
                qa_results_path=qa_path,
                output_path=metrics_path
            )
            all_results[strategy] = results
            
        except FileNotFoundError:
            print(f"X File not found: {qa_path}")
            continue
    
    # Print comparison table
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("COMPARISON TABLE")
        print("="*80)
        
        # Define metrics to compare
        key_metrics = ["faithfulness", "answer_relevancy"]
        
        # Print header
        header = f"{'Metric':<20}"
        for strategy in strategy_names:
            if strategy in all_results:
                header += f" | {strategy:<12}"
        print(header)
        print("-" * len(header))
        
        # Print each metric
        for metric in key_metrics:
            row = f"{metric:<20}"
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
        print("\nProduction Targets:")
        print("  Faithfulness: >=0.95 (hallucination rate <5%)")
        print("  Answer Relevancy: >=0.85")
        print("  Context Relevancy: >=0.75")
        print("="*80)
    
    return all_results


if __name__ == "__main__":
    # Compare all 4 strategies
    compare_generation_strategies(["1_baseline", "2_reranking", "3_routing", "4_full"])