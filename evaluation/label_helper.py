"""
Helper script to assist with manual ground truth labeling.
"""
import json
from pathlib import Path
import sys
import os

# Add src directory to path so we can import from src package
# This allows running the script from the top-level directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.indexer import DocumentIndexer
from src.retrieval.hybrid_search import HybridSearcher


def search_for_labeling(query: str, top_k: int = 30):
    """
    Search and display results for manual labeling.
    
    Args:
        query: Query to search for
        top_k: Number of results to show for labeling
    """
    # Load indexes
    print("Loading indexes...")
    indexer = DocumentIndexer()
    faiss_index, bm25_index, chunks = indexer.load_indexes()
    
    # Initialize searcher (without reranking for raw results)
    searcher = HybridSearcher(faiss_index, bm25_index, chunks, use_reranking=False)
    
    # Search
    print(f"\nSearching for: '{query}'")
    print("="*80)
    
    results = searcher.hybrid_search(query, top_k=top_k)
    
    # Display results for labeling
    print(f"\nTop {len(results)} results to label:\n")
    
    for i, result in enumerate(results, 1):
        print(f"\n{'='*80}")
        print(f"[{i}] Chunk ID: {result['chunk_id']}")
        print(f"Source: {result['source']}")
        print(f"Relevance Score: {result['relevance_score']:.4f}")
        #print(f"\nContent:\n{result['content'][:1000]}...")
        formatted_content = result['content'].replace("\n\n", "__n__")
        formatted_content = formatted_content.replace("\n", " ")
        formatted_content = formatted_content.replace("__n__", "\n")
        print(f"\nContent:\n{formatted_content}")
        print(f"\n{'='*80}")
        
        # Ask for label
        while True:
            label = input("\nRelevance (2=Highly, 1=Somewhat, 0=Not, s=Skip): ").strip().lower()
            if label in ['0', '1', '2', 's']:
                break
            print("Invalid input. Use 0, 1, 2, or s")
        
        if label == 's':
            continue
        
        # Store label
        result['relevance_label'] = int(label)
        
        # Save progress
        yield result


def create_ground_truth_file(query: str, labeled_results: list, output_file: str):
    """Save labeled results to ground truth file."""
    ground_truth = {
        "query": query,
        "labeled_chunks": [
            {
                "chunk_id": r['chunk_id'],
                "source": r['source'],
                "relevance_label": r.get('relevance_label', 0)
            }
            for r in labeled_results
            if 'relevance_label' in r
        ]
    }
    
    # Load existing ground truth if it exists
    output_path = Path(output_file)
    if output_path.exists():
        with open(output_path, 'r') as f:
            all_ground_truth = json.load(f)
    else:
        all_ground_truth = {"queries": []}
    
    # Add or update this query
    existing_query = next(
        (q for q in all_ground_truth['queries'] if q['query'] == query),
        None
    )
    
    if existing_query:
        existing_query.update(ground_truth)
    else:
        all_ground_truth['queries'].append(ground_truth)
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(all_ground_truth, f, indent=2)
    
    print(f"\n✓ Saved ground truth to {output_file}")


# Interactive labeling session
if __name__ == "__main__":
    print("Ground Truth Labeling Tool")
    print("="*80)
    
    # Example query - you'll do this for each of your 10 queries
    query = input("\nEnter query to label: ").strip()
    
    if not query:
        query = "What are path parameters in FastAPI?"
        print(f"Using example query: '{query}'")
    
    labeled_results = []
    
    print("\n\nInstructions:")
    print("- Read each chunk carefully")
    print("- Label as 2 (Highly Relevant), 1 (Somewhat), 0 (Not Relevant)")
    print("- Press 's' to skip")
    print("- Ground truth is saved automatically as you go\n")
    
    input("Press Enter to start labeling...")
    
    for result in search_for_labeling(query, top_k=30):
        labeled_results.append(result)
        
        # Ask if they want to continue
        if len(labeled_results) >= 10:
            cont = input("\nContinue labeling more chunks? (y/n): ").strip().lower()
            if cont != 'y':
                break
    
    # Save ground truth
    output_file = "evaluation/ground_truth.json"
    create_ground_truth_file(query, labeled_results, output_file)
    
    print(f"\n✓ Labeled {len(labeled_results)} chunks for query: '{query}'")
    print(f"Summary:")
    print(f"  Highly Relevant (2): {sum(1 for r in labeled_results if r.get('relevance_label') == 2)}")
    print(f"  Somewhat Relevant (1): {sum(1 for r in labeled_results if r.get('relevance_label') == 1)}")
    print(f"  Not Relevant (0): {sum(1 for r in labeled_results if r.get('relevance_label') == 0)}")