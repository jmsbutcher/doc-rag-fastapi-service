"""
Load FastAPI documentation from the web.
"""
import os
import requests
from pathlib import Path
from typing import List, Dict
from bs4 import BeautifulSoup
import markdown


class FastAPIDocsLoader:
    """Load FastAPI documentation files from GitHub."""
    
    def __init__(self, docs_dir: str = "data/raw"):
        self.docs_dir = Path(docs_dir)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        # FastAPI docs are markdown files in their GitHub repo
        self.docs_base_url = "https://raw.githubusercontent.com/tiangolo/fastapi/master/docs/en/docs"
        
        # Core documentation pages to download
        self.doc_pages = [
            "index.md",
            "tutorial/index.md",
            "tutorial/metadata.md",
            "tutorial/first-steps.md",
            "tutorial/path-params.md",
            "tutorial/query-params.md",
            "tutorial/body.md",
            "tutorial/response-model.md",
            "tutorial/testing.md",
            "tutorial/dependencies/index.md",
            "advanced/additional-responses.md",
            "advanced/additional-status-codes.md",
            "advanced/advanced-dependencies.md",
            "advanced/async-tests.md",
            "advanced/behind-a-proxy.md",
            "advanced/custom-response.md",
            "advanced/dataclasses.md",
            "advanced/events.md",
            "advanced/generate-clients.md",
            "advanced/index.md",
            "advanced/middleware.md",
            "advanced/openapi-callbacks.md",
            "advanced/openapi-webhooks.md",
            "advanced/path-operation-advanced-configuration.md",
            "advanced/response-change-status-code.md",
            "advanced/response-directly.md",
            "advanced/response-headers.md",
            "advanced/settings.md",
            "advanced/sub-applications.md",
            "advanced/templates.md",
            "advanced/testing-dependencies.md",
            "advanced/websockets.md",
            "advanced/wsgi.md",
            "advanced/security/index.md",
            "advanced/security/http-basic-auth.md",
            "advanced/security/oauth2-scopes.md",
            "how-to/conditional-openapi.md",
            "how-to/configure-swagger-ui.md",
            "how-to/custom-docs-ui-assets.md",
            "how-to/custom-request-and-route.md",
            "how-to/extending-openapi.md",
            "how-to/general.md",
            "how-to/graphql.md",
            "how-to/index.md",
            "how-to/migrate-from-pydantic-v1-to-pydantic-v2.md",
            "how-to/separate-openapi-schemas.md",
            "how-to/testing-database.md",
            "deployment/index.md",
            "deployment/cloud.md",
            "deployment/concepts.md",
            "deployment/docker.md",
            "deployment/https.md",
            "deployment/manually.md",
            "deployment/server-workers.md",
            "deployment/versions.md",
            "fastapi-cli.md", # Command Line Interface
            "features.md", # Features overview
            "help-fastapi.md", # Troubleshooting
            "project-generation.md", # Full stack FastAPI template
            "release-notes.md", # Info about features and bug fixes
        ]
    
    def download_docs(self) -> List[Path]:
        """Download documentation files if not already cached."""
        downloaded_files = []
        
        for doc_page in self.doc_pages:
            local_path = self.docs_dir / doc_page
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if already downloaded (caching)
            if local_path.exists():
                print(f"✓ Already cached: {doc_page}")
                downloaded_files.append(local_path)
                continue
            
            url = f"{self.docs_base_url}/{doc_page}"
            print(f"Downloading: {url}")
            
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                with open(local_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                downloaded_files.append(local_path)
                print(f"✓ Downloaded: {doc_page}")
                
            except requests.RequestException as e:
                print(f"✗ Failed to download {doc_page}: {e}")
                continue
        
        return downloaded_files
    
    def load_documents(self) -> List[Dict[str, str]]:
        """
        Load documents into memory with metadata.
        
        Returns:
            List of dicts with 'content', 'source', and 'url' keys
        """
        files = self.download_docs()
        documents = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Convert markdown to HTML, then extract plain text
                # This removes markdown syntax while preserving structure
                html = markdown.markdown(content)
                text = BeautifulSoup(html, 'html.parser').get_text(separator='\n')
                
                # Clean up excessive whitespace
                text = '\n'.join(
                    line.strip() for line in text.split('\n') if line.strip())
                
                # Extract title from first line if it's a header
                lines = text.split('\n')
                title = lines[0] if lines else file_path.stem
                
                documents.append({
                    'content': text,
                    'title': title,
                    'source': str(file_path.relative_to(self.docs_dir)),
                    'url': f"https://fastapi.tiangolo.com/{file_path.stem}/"
                })
                
            except Exception as e:
                print(f"✗ Failed to process {file_path}: {e}")
                continue
        
        return documents


# Test the loader
if __name__ == "__main__":
    print("FastAPI Documentation Loader\n")
    
    loader = FastAPIDocsLoader()
    docs = loader.load_documents()
    
    print(f"\n{'='*60}")
    print(f"Loaded {len(docs)} documents")
    print(f"{'='*60}\n")
    
    # Show preview of first document
    if docs:
        first_doc = docs[0]
        print(f"Title: {first_doc['title']}")
        print(f"Source: {first_doc['source']}")
        print(f"URL: {first_doc['url']}")
        print(f"\nContent preview (first 300 chars):")
        print(f"{first_doc['content'][:300]}...")
        print(f"\nTotal content length: {len(first_doc['content'])} characters")
