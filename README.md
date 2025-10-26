

### System architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                         USER REQUEST                            │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Service                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  POST /query                                               │ │
│  │  - Input validation (Pydantic)                             │ │
│  │  - Request logging                                         │ │
│  │  - Rate limiting (SlowAPI)                                 │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              QUERY ANALYZER AGENT (Your AI Hook!)               │
│                                                                 │
│  Uses GPT-4o-mini to classify query into:                       │
│  ┌─────────────┬──────────────────┬─────────────────┐           │
│  │ SIMPLE      │ COMPLEX          │ MULTI_PART      │           │
│  │ Factual     │ Vague/Conceptual │ Multiple Qs     │           │
│  └─────────────┴──────────────────┴─────────────────┘           │
└───────┬─────────────────┬────────────────────┬──────────────────┘
        │                 │                    │
        ↓                 ↓                    ↓
  ┌─────────┐      ┌──────────┐       ┌──────────────┐
  │ Direct  │      │  HyDE    │       │ Decompose    │
  │ Search  │      │ Generate │       │ Sub-queries  │
  └────┬────┘      │ Hypo Doc │       └──────┬───────┘
       │           └─────┬────┘              │
       │                 │              (Parallel)
       └─────────────────┴───────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   HYBRID RETRIEVAL LAYER                        │
│                                                                 │
│  ┌──────────────────┐              ┌──────────────────┐         │
│  │  BM25 Retrieval  │              │ Vector Search    │         │
│  │  (rank_bm25)     │              │ (FAISS index)    │         │
│  │  Top-25 docs     │              │ Top-25 docs      │         │
│  └────────┬─────────┘              └────────┬─────────┘         │
│           │                                 │                   │
│           └────────────┬────────────────────┘                   │
│                        ↓                                        │
│           ┌─────────────────────────┐                           │
│           │ Reciprocal Rank Fusion  │                           │
│           │ Combine top-50 docs     │                           │
│           └─────────────────────────┘                           │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                  RERANKING LAYER                                │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Cross-Encoder Reranker                                 │    │
│  │  (ms-marco-MiniLM-L-6-v2)                               │    │
│  │  Input: 50 docs → Output: Top-5 docs                    │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   GENERATION LAYER                              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  LLM Generation (Claude Haiku / GPT-4o-mini)            │    │
│  │  - Context: Top-5 reranked docs                         │    │
│  │  - Prompt: Answer + cite sources                        │    │
│  │  - Output: Structured response with citations           │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    RESPONSE + METADATA                          │
│  {                                                              │
│    "answer": "...",                                             │
│    "sources": [{doc_id, title, relevance_score}],               │
│    "query_type": "COMPLEX",                                     │
│    "retrieval_strategy": "hyde",                                │
│    "latency_ms": 1247,                                          │
│    "tokens_used": {...}                                         │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    SUPPORTING COMPONENTS                        │
│                                                                 │
│  Evaluation Pipeline (RAGAS)                                    │
│  - Test set: 50 queries across all types                        │
│  - Metrics: Retrieval (Recall@5, MRR) + Generation              │
│                                                                 │
│  Document Ingestion (Offline)                                   │
│  - Load docs → Semantic chunking → Embed → Index                │
│  - Store: FAISS index + metadata in S3                          │
│                                                                 │
│  Monitoring/Logging                                             │
│  - Request/response logging                                     │
│  - Cost tracking (tokens per query)                             │
│  - Latency tracking by query type                               │
└─────────────────────────────────────────────────────────────────┘
```

### File structure

```
rag-fastapi-docs/
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py           # Download & load FastAPI docs
│   │   ├── chunker.py          # Semantic chunking
│   │   └── indexer.py          # Create FAISS + BM25 indexes
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid_search.py    # BM25 + Vector + Fusion
│   │   └── reranker.py         # Cross-encoder
│   ├── routing/
│   │   ├── __init__.py
│   │   └── query_router.py     # Agentic classifier
│   ├── generation/
│   │   ├── __init__.py
│   │   └── llm.py              # LLM generation
│   └── api/
│       ├── __init__.py
│       └── main.py             # FastAPI app
├── data/
│   ├── raw/                    # Downloaded docs
│   ├── processed/              # Chunked docs
│   └── indexes/                # FAISS + BM25 saved here
├── tests/
│   └── test_queries.py
├── evaluation/
│   ├── test_set.json
│   └── evaluate.py
├── requirements.txt
├── Dockerfile
└── README.md
```


### Semanitc Chunking

Implemented semantic chunking with 100-token overlap to prevent information loss at boundaries. Trade-off: some chunks start mid-sentence, but retrieval accuracy improved significantly (see evaluation metrics).


