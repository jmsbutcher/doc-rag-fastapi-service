"""
Agentic query router that classifies queries and selects optimal retrieval strategy.
"""
from enum import Enum
from typing import Dict, Tuple
from openai import OpenAI
from pydantic import BaseModel
from src.api_key import get_openai_key


class QueryType(str, Enum):
    """Query classification types."""
    SIMPLE = "SIMPLE"
    COMPLEX = "COMPLEX"
    MULTI_PART = "MULTI_PART"


class QueryClassification(BaseModel):
    """Result of query classification."""
    query_type: QueryType
    reasoning: str


class QueryRouter:
    """
    Intelligent query router that analyzes queries and selects retrieval strategies.
    
    This is the "agentic" component - it makes decisions about how to process queries.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize router with LLM for classification.
        
        Args:
            model: LLM model to use for classification (cheap/fast model is fine)
        """
        self.client = OpenAI(api_key=get_openai_key())
        self.model = model
        
        self.classifier_prompt = """You are a query analyzer for a technical documentation system.

Classify each query as SIMPLE, COMPLEX, or MULTI_PART to optimize retrieval strategy.

SIMPLE - Direct factual lookups with clear terminology:
- Uses specific technical terms or concepts
- Single, focused question with clear scope
- User shows familiarity with the domain
Examples:
- "What is the syntax for Python list comprehension?"
- "How do I install pytest?"
- "What does HTTP 404 mean?"

COMPLEX - Vague or exploratory queries that need terminology bridging:
- Describes problems/symptoms rather than solutions
- Uses casual language instead of technical terms
- Exploratory "how do I..." questions without specific technical direction
- Would benefit from restating in technical terminology first
Examples:
- "My API is slow when many people use it at once"
- "How do I make my application more secure?"
- "Best way to organize code in large projects?"

MULTI_PART - Multiple distinct questions or comparisons:
- Asks two or more separate questions
- Requests comparisons ("vs", "difference between")
- Has clearly separable sub-goals
Examples:
- "What is OAuth and how do I implement it in Python?"
- "Difference between REST and GraphQL, and when to use each?"
- "Explain async/await, promises, and callbacks"

Query: {query}

Respond with ONLY: CLASSIFICATION | brief reason
Example: SIMPLE | specific technical term with focused scope"""
    
    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify a query into SIMPLE, COMPLEX, or MULTI_PART.
        
        Args:
            query: User query to classify
        
        Returns:
            QueryClassification with type and reasoning
        """
        # Format prompt with query
        prompt = self.classifier_prompt.format(query=query)
        
        try:
            # Call LLM for classification
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise query classifier."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=100
            )

            # Parse response
            result = response.choices[0].message.content.strip() # type: ignore
    
            # Expected format: "CLASSIFICATION | reason"
            if "|" in result:
                classification, reasoning = result.split("|", 1)
                classification = classification.strip().upper()
                reasoning = reasoning.strip()
            else:
                # Fallback if format is wrong
                classification = result.split()[0].upper()
                reasoning = result
            
            # Validate classification
            try:
                query_type = QueryType(classification)
            except ValueError:
                # Default to SIMPLE if classification is invalid
                query_type = QueryType.SIMPLE
                reasoning = f"Invalid classification '{classification}', defaulted to SIMPLE"
            
            return QueryClassification(
                query_type=query_type,
                reasoning=reasoning
            )
        
        except Exception as e:
            # Fallback to SIMPLE on error
            print(f"Classification error: {e}")
            return QueryClassification(
                query_type=QueryType.SIMPLE,
                reasoning=f"Error during classification: {str(e)}, defaulted to SIMPLE"
            )
    
    def route_query(self, query: str) -> Tuple[QueryType, str, Dict]:
        """
        Classify query and return routing decision.
        
        Args:
            query: User query
        
        Returns:
            Tuple of (query_type, strategy_description, routing_config)
        """
        # Classify the query
        classification = self.classify_query(query)
        
        # Determine routing strategy based on classification
        if classification.query_type == QueryType.SIMPLE:
            strategy = "Direct hybrid search - fast retrieval for clear queries"
            config = {
                "use_hyde": False,
                "use_decomposition": False,
                "retrieval_k": 25,
                "final_k": 5
            }
        
        elif classification.query_type == QueryType.COMPLEX:
            strategy = "HyDE preprocessing - generate hypothetical answer for better retrieval"
            config = {
                "use_hyde": True,
                "use_decomposition": False,
                "retrieval_k": 25,
                "final_k": 5
            }
        
        else:  # MULTI_PART
            strategy = "Query decomposition - break into sub-queries for comprehensive coverage"
            config = {
                "use_hyde": False,
                "use_decomposition": True,
                "retrieval_k": 15,  # Less per sub-query
                "final_k": 5
            }
        
        print(f"\n[ROUTER] Query Type: {classification.query_type}")
        print(f"[ROUTER] Reasoning: {classification.reasoning}")
        print(f"[ROUTER] Strategy: {strategy}\n")
        
        return classification.query_type, strategy, config


# Test the router
if __name__ == "__main__":
    print("Query Router Test\n")
    print("="*60)
    
    router = QueryRouter()
    
    # Test queries across all categories
    test_queries = [
        # SIMPLE queries
        "What is a path parameter in FastAPI?",
        "How do I install FastAPI?",
        "What does status code 200 mean?",
        
        # COMPLEX queries
        "My API keeps timing out when users upload files, what's wrong?",
        "How do I make my FastAPI app faster?",
        "Best practices for structuring FastAPI projects?",
        
        # MULTI_PART queries
        "What's the difference between path parameters and query parameters?",
        "How do I use OAuth2 and what are the security considerations?",
        "Explain dependencies, dependency injection, and how to test them"
    ]
    

    print(f"\nQuery: '{test_queries[0]}'")
    print("-" * 60)
    query_type, strategy, config = router.route_query(test_queries[0])
    print(f"Config: {config}")
    print("=" * 60)
