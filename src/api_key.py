""" Single file for initializing and obtaining API keys. """

import boto3
import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Cache the key to avoid SSM call on every request
_openai_api_key = None

def get_openai_key() -> str:

    global _openai_api_key

    # 1. Return cached key if available
    if _openai_api_key is not None:
        return _openai_api_key
    
    # 2. Try to retrieve key from environment variable (for local dev)
    _openai_api_key = os.getenv("OPENAI_API_KEY")
    if _openai_api_key is not None:
        return _openai_api_key
    
    # 3. Try to retrieve OpenAI API key from AWS SSM Parameter Store (for Lambda deployment)
    client = boto3.client('ssm')
    response = client.get_parameter(
        Name='/lambda/doc-rag-fastapi-service/openai-api-key',
        WithDecryption=True
    )
    _openai_api_key = response['Parameter']['Value']
    if _openai_api_key is not None:
        return _openai_api_key
    

    raise ValueError("OpenAI API key not found")
