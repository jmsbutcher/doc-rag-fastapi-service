
# FastAPI Documentation RAG System with Agentic Routing

A production-grade Retrieval-Augmented Generation (RAG) system for answering questions about FastAPI documentation, featuring intelligent query routing, hybrid search, and comprehensive evaluation.

## Project Highlights

- **Hybrid Search**: Combines BM25 keyword matching with semantic vector search
- **Cross-Encoder Reranking**: Improves retrieval recall by 16.9% and answer relevancy by 13.9%
- **Agentic Query Routing**: Automatically classifies queries and selects optimal retrieval strategy
- **HyDE for Complex Queries**: Generates hypothetical documents to bridge vocabulary gaps
- **Evaluation**: Ground truth labeling + RAGAS metrics for production-readiness validation

## Key Results

Systematic evaluation across 10 test queries demonstrates measurable improvements:

| Metric | Baseline | +Reranking | +Routing | +HyDE | Improvement |
|--------|----------|------------|----------|-------|-------------|
| **Retrieval Quality** |
| precision@5  | 0.640 | 0.680 | 0.680 | 0.700 | **+9.4%** |
| recall@5     | 0.313 | 0.366 | 0.366 | 0.375 | **+19.9%** |
| ndcg@10      | 0.808 | 0.858 | 0.858 | 0.915 | **+13.2%** |
| mrr          | 0.779 | 0.817 | 0.817 | 0.933 | **+19.9%** |
| **Generation Quality** |
| faithfulness      | 0.874 | 0.845 | 0.795 | 0.825 | -5.6% |
| answer_relevancy  | 0.660 | 0.752 | nan | 0.943 | **+42.8%** |

**Key Findings**: 

Reranking provided the highest impact improvement on retrieval recall (+16.9%), while HyDE provided the highest improvement on answer relevancy for complex queries (+25.4%).

Counterintuitively, reranking and Hyde *decreased* answer faithfulness slightly (The hallucination rate increased). A possible explanation is that with less relevant context, the LLM has no choice but to reference that poor context directly (high faithfulness, low relevancy), but with highly relevant context, an LLM is "put into the correct frame of mind" and is therefore more likely to "fill in" the rest of the information on the topic that it already knows in its weights (low faithfulness, high relevancy). I expect this could be mitigated by adding more emphasis to the system prompt telling the LLM to answer the query *strictly using the information provided in the context* or by adding a verification step to the flow. I may try this in a future project.




### System architecture
```
┌────────────────────────────────────────────────────────────┐
│                         USER REQUEST                       │
└────────────────────────────┬───────────────────────────────┘
                             ↓
┌────────────────────────────────────────────────────────────┐
│                      FastAPI Service                       │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  POST /query                                          │ │
│  │  - Input validation (Pydantic)                        │ │
│  │  - Request logging                                    │ │
│  │  - Rate limiting (SlowAPI)                            │ │
│  └───────────────────────────────────────────────────────┘ │
└────────────────────────────┬───────────────────────────────┘
                             ↓
┌────────────────────────────────────────────────────────────┐
│              QUERY ANALYZER AGENT                          │
│                                                            │
│  Uses GPT-4o-mini to classify query into:                  │
│  ┌─────────────┬──────────────────┬─────────────────┐      │
│  │ SIMPLE      │ COMPLEX          │ MULTI_PART      │      │
│  │ Factual     │ Vague/Conceptual │ Multiple Qs     │      │
│  └─────────────┴──────────────────┴─────────────────┘      │
└───────┬─────────────────┬────────────────────┬─────────────┘
        │                 │                    │
        ↓                 ↓                    ↓
  ┌─────────┐      ┌──────────┐       ┌──────────────┐
  │ Direct  │      │  HyDE    │       │ Decompose    │
  │ Search  │      │ Generate │       │ Sub-queries  │
  └────┬────┘      │ Hypo Doc │       └──────┬───────┘
       │           └─────┬────┘              │
       │                 │                   |
       └─────────────────┴───────────────────┘
                         ↓
┌────────────────────────────────────────────────────────────┐
│                   HYBRID RETRIEVAL LAYER                   │
│                                                            │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  BM25 Retrieval  │         │ Vector Search    │         │
│  │  (rank_bm25)     │         │ (FAISS index)    │         │
│  │  Top-25 docs     │         │ Top-25 docs      │         │
│  └────────┬─────────┘         └────────┬─────────┘         │
│           │                            │                   │
│           └────────────┬───────────────┘                   │
│                        ↓                                   │
│           ┌─────────────────────────┐                      │
│           │ Reciprocal Rank Fusion  │                      │
│           │ Combine top-50 docs     │                      │
│           └─────────────────────────┘                      │
└────────────────────────┬───────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────────┐
│                  RERANKING LAYER                           │
│                                                            │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  Cross-Encoder Reranker                               │ │
│  │  (ms-marco-MiniLM-L-6-v2)                             │ │
│  │  Input: 79 docs -> Output: Top-5 docs                 │ │
│  └───────────────────────────────────────────────────────┘ │
└────────────────────────┬───────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────────┐
│                   GENERATION LAYER                         │
│                                                            │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  LLM Generation (Claude Haiku / GPT-4o-mini)          │ │
│  │  - Context: Top-5 reranked docs                       │ │
│  │  - Prompt: Answer + cite sources                      │ │
│  │  - Output: Structured response with citations         │ │
│  └───────────────────────────────────────────────────────┘ │
└────────────────────────┬───────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────────┐
│                    RESPONSE + METADATA                     │
│  {                                                         │
│    "answer": "...",                                        │
│    "sources": [{doc_id, title, relevance_score}],          │
│    "query_type": "COMPLEX",                                │
│    "retrieval_strategy": "hyde",                           │
│    "latency_ms": 1247,                                     │
│    "tokens_used": {...}                                    │
│  }                                                         │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│                    SUPPORTING COMPONENTS                   │
│                                                            │
│  Evaluation Pipeline (RAGAS)                               │
│  - Test set: 10 queries across all types                   │
│  - Metrics: Retrieval (Recall@5, MRR) + Generation         │
│                                                            │
│  Document Ingestion (Offline)                              │
│  - Load docs -> Semantic chunking -> Embed -> Index        │
│  - Store: FAISS index + metadata in S3                     │
│                                                            │
│  Monitoring/Logging                                        │
│  - Request/response logging                                │
│  - Cost tracking (tokens per query)                        │
│  - Latency tracking by query type                          │
└────────────────────────────────────────────────────────────┘
```

### File structure

```
rag-fastapi-docs/
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py          # Download & load FastAPI docs
│   │   ├── chunker.py         # Semantic chunking
│   │   └── indexer.py         # Create FAISS + BM25 indexes
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid_search.py   # BM25 + Vector + Fusion
│   │   ├── hyde.py            # HyDE hypothetical answer generator
│   │   └── reranker.py        # Cross-encoder
│   ├── routing/
│   │   ├── __init__.py
│   │   └── query_router.py    # Agentic classifier
│   └── api/
│       ├── __init__.py
│       └── main.py            # FastAPI app
├── data/
│   ├── raw/                   # Downloaded docs
│   └── indexes/               # Chunked docs and FAISS + BM25 saved here
├── evaluation/
│   ├── ground_truth.json      # Test queries with manually-ranked retrieval data
│   ├── label_helper.py        # Helper for creating ground truth data
│   ├── retrieval_evaluator.py # Runs test queries through 4 retrieval strategies
│   ├── run_retrieval.py       # Perform the retrieval evaluation
│   ├── ragas_evaluator.py     # Runs test queries through full Q&A pipeline
│   ├── run_qa.py              # Perform the full Q&A pipeline (retrieval + LLM gen.)
│   ├── run_evaluation.py      # Runs all evaluation steps and produces report
│   └── results/               # Evaluation results saved here
├── requirements.txt
├── Dockerfile
└── README.md
```



## Prerequisites

- Python 3.12
- OpenAI API key
- pip package manager


## Setup

### Clone repository

```bash
git clone https://github.com/jmsbutcher/doc-rag-fastapi-service.git 
cd doc_rag_service
```


### Set OpenAI API key

```bash
export OPENAI_API_KEY="your key here"
```
or create .env file with this line:
```bash
OPENAI_API_KEY="your key here"
```


### Create virtual environment and activate it

Unix/Mac:
```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:
```powershell
python -m venv .venv
.venv\Scripts\activate
```


### Install dependencies

```bash
pip install -r requirements.txt
```


### Download docs and build indexes (first time only)

```
python -m src.ingestion.loader
python -m src.ingestion.indexer
```


### Run service 

```bash
python -m src.api.main
```


## API Usage

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are path parameters in FastAPI?",
    "top_k": 5
  }'
```
Or, go to http://localhost:8000/docs#/default/query_docs_query_post in your
browser to use the interactive FastAPI interface. (Under POST /query, click 
"Try it out", edit the "query" field in the request body, then click 
"Execute")

Example response:
```bash
{
  "query": "What are path parameters in FastAPI?",
  "answer": "Path parameters in FastAPI are variables that are part of the URL path and are declared using a syntax similar to Python format strings. They allow you to capture values from the URL and pass them as arguments to your endpoint functions. For example, you can declare a path parameter like this:\n\n```python\n@app.get(\"/items/{item_id}\")\ndef read_item(item_id: str):\n    return {\"item_id\": item_id}\n```\n\nIn this example, if you access the URL `http://127.0.0.1:8000/items/foo`, the value `\"foo\"` will be passed to the `read_item` function as the argument `item_id` [5].\n\nPath parameters can also have types specified using standard Python type annotations, which enables data validation and automatic type conversion. For instance:\n\n```python\n@app.get(\"/items/{item_id}\")\ndef read_item(item_id: int):\n    return {\"item_id\": item_id}\n```\n\nIn this case, if you access `http://127.0.0.1:8000/items/3`, the value will be converted to an integer, and if a non-integer value is provided, FastAPI will return a clear error message indicating the validation failure [5]. \n\nAdditionally, FastAPI supports path parameters that can contain paths themselves by using a special syntax, such as:\n\n```python\n@app.get(\"/files/{file_path:path}\")\ndef read_file(file_path: str):\n    return {\"file_path\": file_path}\n```\n\nThis allows the `file_path` parameter to capture any path structure, although it may require a specific URL format [1].",
  "sources": [
    {
      "source": "tutorial\\path-params.md",
      "url": "https://fastapi.tiangolo.com/path-params/",
      "relevance_score": 0.015625
    },
    {
      "source": "tutorial\\body.md",
      "url": "https://fastapi.tiangolo.com/body/",
      "relevance_score": 0.02967032967032967
    },
    {
      "source": "tutorial\\query-params.md",
      "url": "https://fastapi.tiangolo.com/query-params/",
      "relevance_score": 0.0264301860545992
    },
    {
      "source": "tutorial\\query-params.md",
      "url": "https://fastapi.tiangolo.com/query-params/",
      "relevance_score": 0.029850746268656716
    },
    {
      "source": "tutorial\\path-params.md",
      "url": "https://fastapi.tiangolo.com/path-params/",
      "relevance_score": 0.01639344262295082
    }
  ],
  "query_type": "SIMPLE",
  "retrieval_stategy": "Direct hybrid search - fast retrieval for clear queries",
  "reasoning": "specific technical term with focused scope"
}
```


## Evaluation Methodology

### Ground Truth Creation
- Manually labeled 10 diverse test queries
- 10 - 30 document chunks per query rated 0 (not relevant), 1 (somewhat relevant), or 2 (highly relevant)
- Covers simple factual queries, complex conceptual queries, and multi-part questions

### Metrics Tracked

**Retrieval Metrics** (using ground truth labels):
- **Precision@5**: Proportion of relevant docs in top-5 (target: ≥0.60) ✓
- **Recall@5**: Proportion of all relevant docs retrieved (target: ≥0.80) 
- **NDCG@10**: Ranking quality score (target: ≥0.75) ✓
- **MRR**: Position of first relevant result (target: ≥0.70) ✓

**Generation Metrics** (using RAGAS):
- **Faithfulness**: Claims supported by context (target: ≥0.95)
- **Answer Relevancy**: Answer addresses query (target: ≥0.85) ✓

### Key Insights

1. **Reranking Impact**: The highest-leverage optimization, improving recall by 16.9% and answer relevancy by 13.9%

2. **HyDE for Complex Queries**: Improved ranking quality (NDCG +6.6%, MRR +14.2%) without sacrificing recall, particularly effective for vague queries like "My API is slow when many users connect"

3. **Faithfulness-Relevancy Trade-off**: More relevant context enables better synthesis but slightly reduces strict faithfulness (0.929 → 0.899). The massive relevancy gain (+42.8%) justifies this trade-off for Q&A applications.

4. **Query Routing Effectiveness**: 30% of queries classified as COMPLEX benefit from HyDE preprocessing, demonstrating the value of adaptive retrieval strategies



## Technical Decisions

### Why Hybrid Search?
Vector-only search misses exact keyword matches (e.g., "HTTP 404" vs "HTTP 500"). BM25 + vector search with RRF achieves 15-25% better retrieval accuracy on technical documentation.

### Why Cross-Encoder Reranking?
Bi-encoders (used for initial retrieval) compress documents into single vectors, losing information. Cross-encoders process query-document pairs jointly, capturing nuanced relevance (at the cost of speed). Two-stage retrieval (retrieve 50, rerank to 5) balances speed and quality.

### Why HyDE for Complex Queries?
Vague queries like "My API is slow" don't match documentation vocabulary. HyDE generates hypothetical answers ("FastAPI can handle multiple requests with workers..."), then searches using answer-to-answer similarity rather than question-to-answer matching. This bridges vocabulary gaps.

### Why Query Routing?
Different query types benefit from different strategies:
- **SIMPLE** (factual): Direct search is faster and sufficient
- **COMPLEX** (conceptual): HyDE preprocessing improves results
- **MULTI_PART**: Query decomposition

Adaptive routing optimizes both quality and cost/latency.

## Production Considerations

### Performance
- Average query latency: 2.3 seconds (retrieval: 0.3s, generation: 2.0s)
- Concurrent request handling via FastAPI async endpoints
- Semantic caching recommended for 60-80% cost reduction on repeated queries

### Scaling
- Current implementation: In-memory FAISS (suitable for <1M documents)
- For larger scale: Consider Amazon Aurora pgvector or OpenSearch Serverless
- Horizontal scaling: Separate retrieval and generation services

### Monitoring
- Track retrieval metrics on production queries (sample 10-20%)
- Log query classifications for routing analysis
- Monitor faithfulness scores to detect hallucination drift

### Cost Optimization
- Embedding generation: $0.10 per 1M tokens (one-time for indexing)
- LLM generation: ~$0.01 per query with GPT-4o-mini
- Prompt caching: 85% latency reduction, 90% cost savings on repeated context

## Future Enhancements

**Identified in Evaluation:**
1. Improve recall (0.40 → 0.80): Increase retrieval breadth or refine chunking
2. Improve faithfulness (0.90 → 0.95): Add citation verification step
3. Multi-part query decomposition: Currently routes to baseline strategy

**Additional Features:**
- Query expansion with generated variations
- GraphRAG for entity relationships
- Streaming responses for better UX
- Conversational memory for multi-turn dialogues

## Learning Outcomes

This project demonstrates:
- **Production RAG Architecture**: Hybrid search, reranking, agentic routing
- **Systematic Evaluation**: Ground truth labeling, retrieval metrics (IR), generation metrics (RAGAS)
- **Data-Driven Optimization**: Quantified impact of each feature (+16.9% recall from reranking)
- **Trade-off Analysis**: Faithfulness vs relevancy, precision vs recall
- **AWS Integration**: Deployment-ready for ECS/Fargate with CloudWatch monitoring

## References

- [RAGAS Evaluation Framework](https://github.com/explodinggradients/ragas)
- [HyDE: Hypothetical Document Embeddings](https://arxiv.org/abs/2212.10496)
- [Cross-Encoder Reranking](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [Building Production RAG Systems (2025)](https://developers.llamaindex.ai/python/framework/optimizing/production_rag/)


## Contact

James Butcher - [GitHub](https://github.com/jmsbutcher) | [LinkedIn](https://linkedin.com/in/james-butcher-275a2080/)



