
import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

import os
from dotenv import load_dotenv

load_dotenv()


data = {
    "question": ["What is 2+2?"],
    "answer": ["2+2 equals 4"],
    "contexts": [["The sum of 2 and 2 is 4"]],
}

dataset = Dataset.from_dict(data)
result = evaluate(
    dataset, 
    metrics=[faithfulness, answer_relevancy],
    llm=ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
)

print("\n=== Results ===")

# Method 1: Dictionary access
print("Faithfulness score:", result['faithfulness'])
print("Answer Relevancy score:", result['answer_relevancy'])

# Method 2: Convert to DataFrame
print("\n=== DataFrame View ===")
df = result.to_pandas()
print(df)

# Method 3: Get all scores as dict
print("\n=== All Scores ===")
print(result.scores)