"""
Create and manage FAISS vector index and BM25 keyword index.
"""
import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import faiss
from rank_bm25 import BM25Okapi

load_dotenv()


class DocumentIndexer:
    """
    Creates dual indexes for hybrid search:
    - FAISS for vector similarity search
    - BM25 for keyword search
    """
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        embedding_dim: int = 1536,
        index_dir: str = "data/indexes"
    ):
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=api_key)
        
        # Will hold our indexes
        self.faiss_index = None
        self.bm25_index = None
        self.chunks = []
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        Uses batching to stay under rate limits.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts per API call
        
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            print(f"Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
            
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                
                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"Error embedding batch {i//batch_size + 1}: {e}")
                # Add zero vectors for failed batch
                all_embeddings.extend([[0.0] * self.embedding_dim] * len(batch))
        
        return np.array(all_embeddings, dtype=np.float32)
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """
        Build FAISS index from embeddings.
        Uses L2 (Euclidean) distance, which works well with normalized embeddings.
        
        Args:
            embeddings: numpy array of embeddings
        
        Returns:
            FAISS index
        """
        # Debug: Check embeddings shape
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Embeddings dtype: {embeddings.dtype}")
        
        # Ensure embeddings are float32 and 2D
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        embeddings = embeddings.astype(np.float32)

        # Normalize embeddings (important for cosine similarity via L2)
        faiss.normalize_L2(embeddings)
        
        # Create index
        # IndexFlatL2 = brute force exact search (fine for <100k vectors)
        index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add vectors to index
        # ! Ignore Pylance warning about missing arguments !
        index.add(embeddings) # type: ignore

        print(f"✓ Built FAISS index with {index.ntotal} vectors")
        
        return index
    
    def build_bm25_index(self, chunks: List[Dict]) -> BM25Okapi:
        """
        Build BM25 index for keyword search.
        
        Args:
            chunks: List of chunk dicts with 'content' key
        
        Returns:
            BM25 index
        """
        # Tokenize documents (simple whitespace split)
        # In production, you'd use proper tokenization
        tokenized_docs = [
            chunk['content'].lower().split()
            for chunk in chunks
        ]
        
        # Create BM25 index
        bm25 = BM25Okapi(tokenized_docs)
        
        print(f"✓ Built BM25 index with {len(tokenized_docs)} documents")
        
        return bm25
    
    def index_documents(self, chunks: List[Dict]) -> Tuple[faiss.Index, BM25Okapi]:
        """
        Create both FAISS and BM25 indexes from chunks.
        
        Args:
            chunks: List of chunk dicts with 'content' and metadata
        
        Returns:
            Tuple of (faiss_index, bm25_index)
        """
        print(f"\nIndexing {len(chunks)} chunks...\n")
        
        # Store chunks for later retrieval
        self.chunks = chunks
        
        # Extract text content for embedding
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.generate_embeddings(texts)
        
        # Build FAISS index
        print("\nBuilding FAISS index...")
        self.faiss_index = self.build_faiss_index(embeddings)
        
        # Build BM25 index
        print("\nBuilding BM25 index...")
        self.bm25_index = self.build_bm25_index(chunks)
        
        print(f"\n{'='*60}")
        print("Indexing complete!")
        print(f"{'='*60}\n")
        
        return self.faiss_index, self.bm25_index
    
    def save_indexes(self, prefix: str = "fastapi_docs"):
        """
        Save indexes and chunks to disk.
        
        Args:
            prefix: Filename prefix for saved files
        """
        if self.faiss_index is None or self.bm25_index is None:
            raise ValueError("No indexes to save. Run index_documents() first.")
        
        # Save FAISS index
        faiss_path = self.index_dir / f"{prefix}_faiss.index"
        faiss.write_index(self.faiss_index, str(faiss_path))
        print(f"✓ Saved FAISS index to {faiss_path}")
        
        # Save BM25 index
        bm25_path = self.index_dir / f"{prefix}_bm25.pkl"
        with open(bm25_path, 'wb') as f:
            pickle.dump(self.bm25_index, f)
        print(f"✓ Saved BM25 index to {bm25_path}")
        
        # Save chunks metadata
        chunks_path = self.index_dir / f"{prefix}_chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved chunks metadata to {chunks_path}")
    
    def load_indexes(self, prefix: str = "fastapi_docs") -> Tuple[faiss.Index, BM25Okapi, List[Dict]]:
        """
        Load indexes and chunks from disk.
        
        Args:
            prefix: Filename prefix for saved files
        
        Returns:
            Tuple of (faiss_index, bm25_index, chunks)
        """
        # Load FAISS index
        faiss_path = self.index_dir / f"{prefix}_faiss.index"
        self.faiss_index = faiss.read_index(str(faiss_path))
        print(f"✓ Loaded FAISS index from {faiss_path}")
        
        # Load BM25 index
        bm25_path = self.index_dir / f"{prefix}_bm25.pkl"
        with open(bm25_path, 'rb') as f:
            self.bm25_index = pickle.load(f)
        print(f"✓ Loaded BM25 index from {bm25_path}")
        
        # Load chunks
        chunks_path = self.index_dir / f"{prefix}_chunks.json"
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        print(f"✓ Loaded {len(self.chunks)} chunks from {chunks_path}")
        
        return self.faiss_index, self.bm25_index, self.chunks


# Test the indexer
if __name__ == "__main__":
    from loader import FastAPIDocsLoader
    from chunker import SemanticChunker
    
    print("FastAPI Documentation Indexer\n")
    
    # Load and chunk documents
    print("Step 1: Loading documents...")
    loader = FastAPIDocsLoader()
    docs = loader.load_documents()
    print(f"✓ Loaded {len(docs)} documents\n")
    
    print("Step 2: Chunking documents...")
    chunker = SemanticChunker(chunk_size=600, chunk_overlap=100)
    chunks = chunker.chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks\n")
    
    print("Step 3: Creating indexes...")
    indexer = DocumentIndexer()
    
    # Check if indexes already exist
    index_files_exist = all([
        (indexer.index_dir / "fastapi_docs_faiss.index").exists(),
        (indexer.index_dir / "fastapi_docs_bm25.pkl").exists(),
        (indexer.index_dir / "fastapi_docs_chunks.json").exists()
    ])
    
    if index_files_exist:
        print("Indexes already exist. Loading from disk...\n")
        indexer.load_indexes()
    else:
        print("Building new indexes...\n")
        indexer.index_documents(chunks)
        indexer.save_indexes()
    
    print(f"\n{'='*60}")
    print("Setup complete! Indexes ready for search.")
    print(f"{'='*60}\n")
    
    # Quick test search
    print("Testing with a sample query...\n")
    test_query = "How do I add path parameters?"
    
    # Generate query embedding
    query_embedding = indexer.generate_embeddings([test_query])[0]
    
    # Search FAISS (get top 5)
    if indexer.faiss_index is not None:
        # ! Ignore Pylance warning about missing arguments !
        distances, indices = indexer.faiss_index.search(
            query_embedding.reshape(1, -1), 
            k=5
        ) # type: ignore
    else:
        print("Error: FAISS index not loaded properly")
        exit(1)
    
    print(f"Query: '{test_query}'")
    print(f"\nTop 3 results from FAISS:")
    for i, idx in enumerate(indices[0][:3]):
        chunk = indexer.chunks[idx]
        print(f"\n{i+1}. {chunk['source']} (distance: {distances[0][i]:.3f})")
        print(f"   {chunk['content'][:150]}...")