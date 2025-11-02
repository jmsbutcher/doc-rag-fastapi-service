# RAG System Evaluation Report
#### Generated: 2025-11-02 12:42:27
## Executive Summary

This report compares incremental improvements across 4 retrieval strategies:

- **1_baseline**: Baseline (no reranking)
- **2_reranking**: + Reranking
- **3_routing**: + Routing
- **4_full**: + HyDE (Full)

## Retrieval Metrics

| Metric | Baseline | +Reranking | +Routing | +HyDE | Best Improvement |
|--------|----------|------------|----------|-------|------------------|
| precision@5  | 0.640 | 0.680 | 0.680 | 0.700 | +9.4% |
| recall@5     | 0.313 | 0.366 | 0.366 | 0.375 | +19.9% |
| ndcg@10      | 0.808 | 0.858 | 0.858 | 0.915 | +13.2% |
| mrr          | 0.779 | 0.817 | 0.817 | 0.933 | +19.9% |

## Generation Metrics (RAGAS)

| Metric | Baseline | +Reranking | +Routing | +HyDE | Best Improvement |
|--------|----------|------------|----------|-------|------------------|
| faithfulness      | 0.874 | 0.845 | 0.795 | 0.825 | +0.0% |
| answer_relevancy  | 0.660 | 0.752 | nan | 0.943 | +42.8% |

## Feature Impact Analysis

**Reranking Impact**: +17.1% improvement in Recall@5
- Cross-encoder reranking improved retrieval precision by selecting the most relevant documents from the initial hybrid search results.

**Routing + HyDE Impact**: +2.4% improvement in Recall@5
- Agentic routing with HyDE for complex queries improved vocabulary matching between user questions and documentation.

## Production Targets

According to 2025 RAG best practices:

**Retrieval Metrics:**
- Recall@5: >=0.80
- Precision@5: >=0.60
- NDCG@10: >=0.75
- MRR: >=0.70

**Generation Metrics:**
- Faithfulness: >=0.95 (hallucination rate <5%)
- Answer Relevancy: >=0.85
## Key Insights

[!] **Retrieval metrics below production targets**
  - Recall@5: 0.375 (target: >=0.80)

[!] **Generation metrics below production targets**
  - Faithfulness: 0.825 (target: >=0.95)

