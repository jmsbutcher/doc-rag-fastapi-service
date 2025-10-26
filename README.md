
### File structure

```
rag-fastapi-docs/
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py          # Download & load FastAPI docs
│   │   ├── chunker.py          # Semantic chunking
│   │   └── indexer.py          # Create FAISS + BM25 indexes
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid_search.py    # BM25 + Vector + Fusion
│   │   └── reranker.py         # Cross-encoder (Week 2)
│   ├── routing/
│   │   ├── __init__.py
│   │   └── query_router.py     # Your agentic classifier (Week 2)
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


