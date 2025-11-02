"""
Main Evaluation Script - Orchestrates full evaluation pipeline.

This script:
1. Runs retrieval on test queries (4 strategies: baseline, +reranking, +routing, +HyDE)
2. Computes retrieval metrics using ground truth
3. Runs full Q&A pipeline for all strategies
4. Computes generation metrics with RAGAS
5. Generates comprehensive comparison report

Run with:
    python evaluation/run_evaluation.py
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def run_step(step_name: str, script_path: str):
    """Run evaluation step and handle errors."""
    print("\n" + "="*70)
    print(f"STEP: {step_name}")
    print("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR in {step_name}:")
        print(e.stdout)
        print(e.stderr)
        return False


def load_json(path: str) -> Dict | None:
    """Load JSON file safely."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def generate_report():
    """Generate comprehensive evaluation report."""
    
    print("\n" + "="*70)
    print("GENERATING COMPARISON REPORT")
    print("="*70)
    
    # Define all strategies
    strategies = ["1_baseline", "2_reranking", "3_routing", "4_full"]
    strategy_names = {
        "1_baseline": "Baseline (no reranking)",
        "2_reranking": "+ Reranking",
        "3_routing": "+ Routing",
        "4_full": "+ HyDE (Full)"
    }
    
    # Load all results
    retrieval_metrics = {}
    generation_metrics = {}
    
    for strategy in strategies:
        retrieval_path = f"evaluation/results/retrieval_metrics_{strategy}.json"
        generation_path = f"evaluation/results/generation_metrics_{strategy}.json"
        
        retrieval_metrics[strategy] = load_json(retrieval_path)
        generation_metrics[strategy] = load_json(generation_path)
    
    # Create report
    report = []
    report.append("# RAG System Evaluation Report\n")
    report.append(f"#### Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Executive Summary
    report.append("## Executive Summary\n\n")
    report.append("This report compares incremental improvements across 4 retrieval strategies:\n\n")
    for strategy in strategies:
        report.append(f"- **{strategy}**: {strategy_names[strategy]}\n")
    report.append("\n")
    
    # Retrieval Metrics Comparison
    report.append("## Retrieval Metrics\n\n")
    report.append("| Metric | Baseline | +Reranking | +Routing | +HyDE | Best Improvement |\n")
    report.append("|--------|----------|------------|----------|-------|------------------|\n")
    
    key_retrieval_metrics = ["precision@5", "recall@5", "ndcg@10", "mrr"]
    
    for metric in key_retrieval_metrics:
        row_values = []
        baseline_val = None
        max_improvement = 0
        
        for i, strategy in enumerate(strategies):
            if retrieval_metrics[strategy]:
                val = retrieval_metrics[strategy]["average_metrics"][metric]["mean"]
                row_values.append(val)
                
                if i == 0:
                    baseline_val = val
                else:
                    improvement = ((val - baseline_val) / baseline_val) * 100
                    max_improvement = max(max_improvement, improvement)
            else:
                row_values.append(None)
        
        row = f"| {metric:12s} |"
        for val in row_values:
            if val is not None:
                row += f" {val:.3f} |"
            else:
                row += " N/A |"
        row += f" +{max_improvement:.1f}% |\n"
        report.append(row)
    
    report.append("\n")
    
    # Generation Metrics Comparison
    report.append("## Generation Metrics (RAGAS)\n\n")
    report.append("| Metric | Baseline | +Reranking | +Routing | +HyDE | Best Improvement |\n")
    report.append("|--------|----------|------------|----------|-------|------------------|\n")
    
    key_generation_metrics = ["faithfulness", "answer_relevancy"]
    
    for metric in key_generation_metrics:
        row_values = []
        baseline_val = None
        max_improvement = 0
        
        for i, strategy in enumerate(strategies):
            if generation_metrics[strategy]:
                val = generation_metrics[strategy]["average_metrics"][metric]["mean"]
                row_values.append(val)
                
                if i == 0:
                    baseline_val = val
                else:
                    improvement = ((val - baseline_val) / baseline_val) * 100
                    max_improvement = max(max_improvement, improvement)
            else:
                row_values.append(None)
        
        row = f"| {metric:17s} |"
        for val in row_values:
            if val is not None:
                row += f" {val:.3f} |"
            else:
                row += " N/A |"
        row += f" +{max_improvement:.1f}% |\n"
        report.append(row)
    
    report.append("\n")
    
    # Feature Impact Analysis
    report.append("## Feature Impact Analysis\n\n")
    
    if retrieval_metrics["1_baseline"] and retrieval_metrics["2_reranking"]:
        recall_baseline = retrieval_metrics["1_baseline"]["average_metrics"]["recall@5"]["mean"]
        recall_reranking = retrieval_metrics["2_reranking"]["average_metrics"]["recall@5"]["mean"]
        reranking_impact = ((recall_reranking - recall_baseline) / recall_baseline) * 100
        
        report.append(f"**Reranking Impact**: +{reranking_impact:.1f}% improvement in Recall@5\n")
        report.append(f"- Cross-encoder reranking improved retrieval precision by selecting ")
        report.append(f"the most relevant documents from the initial hybrid search results.\n\n")
    
    if retrieval_metrics["2_reranking"] and retrieval_metrics["4_full"]:
        recall_reranking = retrieval_metrics["2_reranking"]["average_metrics"]["recall@5"]["mean"]
        recall_full = retrieval_metrics["4_full"]["average_metrics"]["recall@5"]["mean"]
        routing_hyde_impact = ((recall_full - recall_reranking) / recall_reranking) * 100
        
        report.append(f"**Routing + HyDE Impact**: +{routing_hyde_impact:.1f}% improvement in Recall@5\n")
        report.append(f"- Agentic routing with HyDE for complex queries improved vocabulary matching ")
        report.append(f"between user questions and documentation.\n\n")
    
    # Production Targets
    report.append("## Production Targets\n\n")
    report.append("According to 2025 RAG best practices:\n\n")
    report.append("**Retrieval Metrics:**\n")
    report.append("- Recall@5: >=0.80\n")
    report.append("- Precision@5: >=0.60\n")
    report.append("- NDCG@10: >=0.75\n")
    report.append("- MRR: >=0.70\n\n")
    report.append("**Generation Metrics:**\n")
    report.append("- Faithfulness: >=0.95 (hallucination rate <5%)\n")
    report.append("- Answer Relevancy: >=0.85\n")
    
    # Key Insights
    report.append("## Key Insights\n\n")
    
    # Check if we meet production targets
    if retrieval_metrics["4_full"]:
        recall = retrieval_metrics["4_full"]["average_metrics"]["recall@5"]["mean"]
        precision = retrieval_metrics["4_full"]["average_metrics"]["precision@5"]["mean"]
        
        if recall >= 0.80 and precision >= 0.60:
            report.append("CHECK! **Retrieval metrics meet production targets**\n\n")
        else:
            report.append("[!] **Retrieval metrics below production targets**\n")
            if recall < 0.80:
                report.append(f"  - Recall@5: {recall:.3f} (target: >=0.80)\n")
            if precision < 0.60:
                report.append(f"  - Precision@5: {precision:.3f} (target: >=0.60)\n")
            report.append("\n")
    
    if generation_metrics["4_full"]:
        faithfulness = generation_metrics["4_full"]["average_metrics"]["faithfulness"]["mean"]
        relevancy = generation_metrics["4_full"]["average_metrics"]["answer_relevancy"]["mean"]
        
        if faithfulness >= 0.95 and relevancy >= 0.85:
            report.append("CHECK! **Generation metrics meet production targets**\n\n")
        else:
            report.append("[!] **Generation metrics below production targets**\n")
            if faithfulness < 0.95:
                report.append(f"  - Faithfulness: {faithfulness:.3f} (target: >=0.95)\n")
            if relevancy < 0.85:
                report.append(f"  - Answer Relevancy: {relevancy:.3f} (target: >=0.85)\n")
            report.append("\n")
    
    
    # Save report
    report_path = f"evaluation/EVALUATION_REPORT_{time.strftime('%Y-%m-%d_%H-%M-%S')}.md"
    with open(report_path, 'w') as f:
        f.writelines(report)
    
    print(f"\nCHECK! Report saved to {report_path}\n")
    
    # Print to console
    print("\n" + "".join(report))


def main():
    """Run complete evaluation pipeline."""
    
    start_time = time.time()
    
    print("\n" + "="*70)
    print("RAG SYSTEM EVALUATION PIPELINE")
    print("="*70)
    print("\nThis will:")
    print("  1. Run retrieval on 10 test queries (4 strategies)")
    print("  2. Compute retrieval metrics using ground truth labels")
    print("  3. Run full Q&A pipeline (retrieval + generation)")
    print("  4. Compute generation metrics using RAGAS")
    print("  5. Generate comprehensive comparison report")
    print("\nStrategies being compared:")
    print("  - Baseline: Hybrid search only")
    print("  - +Reranking: Add cross-encoder reranking")
    print("  - +Routing: Add query classification")
    print("  - +HyDE: Full system with HyDE for complex queries")
    print("\nEstimated time: 5-8 minutes\n")
    
    input("Press Enter to start...")
    
    # Step 1: Run retrieval
    if not run_step(
        "1. Run Retrieval (All 4 Strategies)",
        "evaluation/run_retrieval.py"
    ):
        print("\nX Failed at retrieval step")
        return
    
    # Step 2: Evaluate retrieval
    if not run_step(
        "2. Compute Retrieval Metrics",
        "evaluation/retrieval_evaluator.py"
    ):
        print("\nX Failed at retrieval evaluation step")
        return
    
    # Step 3: Run Q&A pipeline
    if not run_step(
        "3. Run Q&A Pipeline (All 4 Strategies)",
        "evaluation/run_qa.py"
    ):
        print("\nX Failed at Q&A step")
        return
    
    # Step 4: Evaluate generation with RAGAS
    if not run_step(
        "4. Compute Generation Metrics (RAGAS)",
        "evaluation/ragas_evaluator.py"
    ):
        print("\nX Failed at generation evaluation step")
        return
    
    # Step 5: Generate report
    generate_report()
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print("\nResults saved to evaluation/results/")
    print("  - retrieval_results_*.json (4 strategies)")
    print("  - retrieval_metrics_*.json (4 strategies)")
    print("  - qa_results_*.json (4 strategies)")
    print("  - generation_metrics_*.json (4 strategies)")
    print("  - EVALUATION_REPORT.md (comprehensive comparison)")
    print("="*70)


if __name__ == "__main__":
    main()
