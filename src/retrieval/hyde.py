"""
HyDE (Hypothetical Document Embeddings) for improved retrieval on vague queries.

HyDE works by generating a hypothetical answer to the user's query. This helps
retrieve the best documents by matching answer-to-answer similarity instead of
question-to-answer similarity. This is because the RAG documents are indexed
with *solution/answer* language, not question language.


Example 1:
"My API is really slow when lots of people use it at the same time, what should I do?"


-------------------------
without HyDE:

    If your FastAPI application becomes unresponsive when many users connect 
    simultaneously, it could be due to several factors related to how your 
    application is handling concurrent connections. Here are some potential 
    solutions and considerations:

    1. **WebSocket Connections**: If you are using WebSockets, ensure that 
        you are properly handling multiple connections. FastAPI allows you to 
        manage WebSocket connections, but if you're using in-memory storage 
        for clients, it may not scale well. For example, if you are broadcasting 
        messages to all connected clients, ensure that your implementation can 
        handle multiple clients efficiently.

        You can manage disconnections and handle multiple clients using the 
        `WebSocketDisconnect` exception. Here is a simple example of how to 
        handle disconnections:

        ```python
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect

        app = FastAPI()
        clients = []

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            clients.append(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    for client in clients:
                        await client.send_text(f"Message: {data}")
            except WebSocketDisconnect:
                clients.remove(websocket)
                for client in clients:
                    await client.send_text("A client has left the chat")
        ```

        This example demonstrates how to manage multiple WebSocket connections 
        and handle disconnections properly, which can help maintain 
        responsiveness under load [2].

    2. **Asynchronous Processing**: Ensure that your FastAPI application is 
        utilizing asynchronous programming effectively. FastAPI is built on 
        Starlette, which is an asynchronous framework. If your endpoints are 
        blocking (e.g., long-running synchronous tasks), they can cause the 
        application to become unresponsive. Use `async` and `await` for I/O-bound 
        operations to keep the event loop free to handle other requests.

    3. **Scaling**: If your application is still unresponsive under load, consider 
        scaling your application horizontally by running multiple instances of your 
        FastAPI app. You can use tools like Docker, Kubernetes, or Uvicorn with a 
        process manager like Gunicorn to manage multiple worker processes. For 
        example, you can run Uvicorn with multiple workers like this:

        ```bash
        uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
        ```

        This command runs your FastAPI application with 4 worker processes, 
        allowing it to ...


-------------------------
with HyDE:

    "If your FastAPI application is slow when handling many concurrent users, there
      are several strategies you can implement to improve performance:

    1. **Use Multiple Worker Processes**: FastAPI can handle multiple requests
        concurrently with a single process, but to scale effectively, especially
        on multi-core servers, you should run multiple worker processes. This 
        allows your application to handle more requests simultaneously. You can 
        start your FastAPI application with multiple workers using the `--workers` 
        option when running Uvicorn. For example:

    ```bash
    uvicorn main:app --workers 4
    ```

    This command will start 4 worker processes, distributing incoming requests
      among them [1].

    2. **Optimize Resource Usage**: Ensure that your application is not consuming
        excessive memory or CPU resources. If you are using a lot of memory (for 
        instance, with machine learning models), monitor your memory usage and adjust
        the number of containers or processes accordingly [3].

    3. **Deploy in Containers**: If you are using Docker or Kubernetes, consider
        deploying multiple containers, each running a single process. This can help
        in scaling your application horizontally. You can also use an Init Container
        in Kubernetes to prepare your environment before starting the main worker
        containers [3].

    4. **Utilize Asynchronous Programming**: If your application performs I/O-bound 
        operations (like database queries or API calls), consider using asynchronous 
        programming with `async def` for your path operations and dependencies. This 
        allows your application to handle other requests while waiting for I/O 
        operations to complete, improving overall responsiveness [5].

    5. **Use a Load Balancer**: If you have multiple instances of your application 
        running, use a load balancer to distribute traffic evenly across your
        instances. This can help manage high traffic more efficiently.

    6. **Monitor and Profile Performance**: Use profiling tools to identify 
        bottlenecks in your application. This can help you understand which parts 
        of your code are slowing down your API and allow you to optimize them.

    7. **Optimize Database Queries**: If your API interacts with a database, ensure 
        that your queries are optimized. Use indexing, avoid N+1 query problems, 
        and consider caching strategies to reduce database load.

    8. **Consider Caching Responses**: Implement caching for frequently requested 
    data to reduce the load on your application and improve response times.

    By combining these strategies, you can significantly enhance the performance of 
    your FastAPI application under heavy load."

"""
from typing import List
from openai import OpenAI
from api_key import get_openai_key


class HyDEGenerator:
    """
    Generates hypothetical documents for queries to improve retrieval.
    
    The key insight: search answer-to-answer similarity instead of 
    question-to-answer similarity.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize HyDE generator.
        
        Args:
            model: LLM to use for generating hypothetical answers
        """
        self.client = OpenAI(api_key=get_openai_key())
        self.model = model
        
        # Prompt for generating hypothetical answers
        self.hyde_prompt = """
You are a FastAPI documentation expert. Given a user's question, generate a 
hypothetical answer that might appear in FastAPI documentation.

The answer should:
1. Use technical terminology that would appear in docs
2. Be specific and detailed (2-3 paragraphs)
3. Include relevant concepts and keywords
4. Sound like it came from official documentation

Do NOT try to be accurate - just generate plausible documentation text that 
uses the right terminology.

Question: {query}

Hypothetical documentation answer:"""
    
    def generate_hypothetical_answer(self, query: str) -> str:
        """
        Generate a hypothetical answer for a query.
        
        Args:
            query: User's question
        
        Returns:
            Hypothetical answer text
        """
        prompt = self.hyde_prompt.format(query=query)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a technical documentation writer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Higher temp for more diverse hypothetical text
                max_tokens=300
            )

            hypothetical_answer = response.choices[0].message.content.strip()  # type: ignore

            print(f"[HyDE] Generated hypothetical answer ({len(hypothetical_answer)} chars)")
            
            return hypothetical_answer
        
        except Exception as e:
            print(f"[HyDE] Error generating hypothetical answer: {e}")
            # Fallback to original query
            return query
    
    def generate_multiple_hypotheses(self, query: str, n: int = 3) -> List[str]:
        """
        Generate multiple hypothetical answers for diversity.
        
        Args:
            query: User's question
            n: Number of hypothetical answers to generate
        
        Returns:
            List of hypothetical answer texts
        """
        hypotheses = []
        
        for i in range(n):
            print(f"[HyDE] Generating hypothesis {i+1}/{n}...")
            hypothesis = self.generate_hypothetical_answer(query)
            hypotheses.append(hypothesis)
        
        return hypotheses


# Test HyDE
if __name__ == "__main__":
    print("HyDE Generator Test\n")
    print("="*60)
    
    hyde = HyDEGenerator()
    
    # Test queries - especially vague/complex ones where HyDE helps
    test_queries = [
        "My API is slow when many users connect at once",
        "How do I make my FastAPI application more secure?",
        "What's the best way to handle errors in production?"
    ]
    
    for query in test_queries:
        print(f"\nOriginal Query: '{query}'")
        print("-" * 60)
        
        # Generate hypothetical answer
        hypo = hyde.generate_hypothetical_answer(query)
        
        print(f"\nHypothetical Answer:\n{hypo}")
        print("\n" + "="*60)
