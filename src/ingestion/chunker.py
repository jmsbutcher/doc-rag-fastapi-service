"""
Semantic chunking for FastAPI documentation.
Splits documents intelligently while preserving context.
"""
from typing import List, Dict, Any
import re
import tiktoken


class SemanticChunker:
    """
    Chunk documents with semantic awareness.
    Preserves markdown structure and code blocks.
    """
    
    def __init__(
        self,
        chunk_size: int = 600,  # tokens
        chunk_overlap: int = 100,  # tokens
        encoding_name: str = "cl100k_base"  # OpenAI's tokenizer
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using OpenAI's tokenizer."""
        return len(self.encoding.encode(text))
    
    def split_by_headers(self, text: str) -> List[Dict[str, str]]:
        """
        Split text by markdown headers to preserve document structure.
        Returns list of sections with their headers.
        """
        # Pattern matches markdown headers (# Header, ## Header, etc.)
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        sections = []
        current_section = {'header': '', 'content': ''}
        
        for line in text.split('\n'):
            header_match = re.match(header_pattern, line)
            
            if header_match:
                # Save previous section if it has content
                if current_section['content'].strip():
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))  # Count #'s
                header_text = header_match.group(2)
                current_section = {
                    'header': header_text,
                    'level': level,
                    'content': line + '\n'
                }
            else:
                current_section['content'] += line + '\n'
        
        # Add final section
        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections
    
    def chunk_text(self, text: str, max_tokens: int) -> List[str]:
        """
        Recursively chunk text by paragraphs, then sentences, then words.
        Preserves semantic boundaries as much as possible.
        """
        # If text fits in max_tokens, return as-is
        if self.count_tokens(text) <= max_tokens:
            return [text.strip()]
        
        chunks = []
        
        # Try splitting by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        if len(paragraphs) > 1:
            current_chunk = ""
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # Check if adding this paragraph exceeds limit
                test_chunk = current_chunk + "\n\n" + para if current_chunk else para
                
                if self.count_tokens(test_chunk) <= max_tokens:
                    current_chunk = test_chunk
                else:
                    # Current chunk is full, save it
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                    # Start new chunk with this paragraph
                    # If single paragraph is too large, split it further
                    if self.count_tokens(para) > max_tokens:
                        chunks.extend(self.chunk_text(para, max_tokens))
                        current_chunk = ""
                    else:
                        current_chunk = para
            
            # Add remaining chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        else:
            # No paragraph breaks, try splitting by sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            if len(sentences) > 1:
                current_chunk = ""
                
                for sentence in sentences:
                    test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                    
                    if self.count_tokens(test_chunk) <= max_tokens:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
            
            else:
                # Last resort: split by words (shouldn't happen often)
                words = text.split()
                current_chunk = ""
                
                for word in words:
                    test_chunk = current_chunk + " " + word if current_chunk else word
                    
                    if self.count_tokens(test_chunk) <= max_tokens:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = word
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        return chunks
    
    def create_chunks_with_overlap(
        self,
        text: str,
        metadata: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Create overlapping chunks from text with metadata.
        
        Args:
            text: Document text to chunk
            metadata: Metadata to attach to each chunk (source, url, etc.)
        
        Returns:
            List of chunk dicts with content and metadata
        """
        # First split by headers to preserve structure
        sections = self.split_by_headers(text)
        
        all_chunks = []
        chunk_id = 0
        
        for section in sections:
            section_text = section['content']
            header = section.get('header', '')
            
            # Skip empty sections
            if not section_text.strip():
                continue
            
            # Chunk this section
            section_chunks = self.chunk_text(section_text, self.chunk_size)
            
            # Add overlap between chunks
            for i, chunk_text in enumerate(section_chunks):
                # For chunks after the first, add overlap from previous chunk
                if i > 0 and self.chunk_overlap > 0:
                    prev_chunk = section_chunks[i - 1]
                    
                    # Take last N tokens from previous chunk as overlap
                    prev_tokens = self.encoding.encode(prev_chunk)
                    overlap_tokens = prev_tokens[-self.chunk_overlap:]
                    overlap_text = self.encoding.decode(overlap_tokens)
                    
                    # Prepend overlap
                    chunk_text = overlap_text + "\n\n" + chunk_text
                
                # Create chunk with metadata
                chunk = {
                    'chunk_id': f"{metadata.get('source', 'doc')}_{chunk_id}",
                    'content': chunk_text.strip(),
                    'header': header,
                    'section_index': i,
                    'total_sections': len(section_chunks),
                    **metadata  # Include all original metadata
                }
                
                all_chunks.append(chunk)
                chunk_id += 1
        
        return all_chunks
    
    def chunk_documents(
        self,
        documents: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document dicts with 'content' and metadata
        
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            content = doc['content']
            
            # Extract metadata (everything except content)
            metadata = {k: v for k, v in doc.items() if k != 'content'}
            
            # Create chunks for this document
            doc_chunks = self.create_chunks_with_overlap(content, metadata)
            all_chunks.extend(doc_chunks)
            
            print(f"✓ Chunked {metadata.get('source', 'document')}: "
                  f"{len(doc_chunks)} chunks")
        
        return all_chunks


# Test the chunker
if __name__ == "__main__":
    from loader import FastAPIDocsLoader
    
    print("FastAPI Documentation Chunker\n")
    
    # Load documents
    loader = FastAPIDocsLoader()
    docs = loader.load_documents()
    print(f"Loaded {len(docs)} documents\n")
    
    # Chunk documents
    chunker = SemanticChunker(
        chunk_size=600,
        chunk_overlap=100
    )
    
    chunks = chunker.chunk_documents(docs)
    
    print(f"\n{'='*60}")
    print(f"Created {len(chunks)} total chunks")
    print(f"{'='*60}\n")
    
    # Show statistics
    chunk_sizes = [chunker.count_tokens(chunk['content']) for chunk in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes)
    min_size = min(chunk_sizes)
    max_size = max(chunk_sizes)
    
    print(f"Chunk size statistics:")
    print(f"  Average: {avg_size:.0f} tokens")
    print(f"  Min: {min_size} tokens")
    print(f"  Max: {max_size} tokens")
    
    # Show a few sample chunks
    for i in range(30, min(33, len(chunks))):
        print(f"\nSample chunk:")
        print(f"  ID: {chunks[i]['chunk_id']}")
        print(f"  Source: {chunks[i]['source']}")
        print(f"  Header: {chunks[i].get('header', 'N/A')}")
        print(f"  Tokens: {chunker.count_tokens(chunks[i]['content'])}")
        print(f"\n  Content preview:")
        print(f"\n------------------------------")
        print(f"  {chunks[i]['content']}...")
        print(f"\n------------------------------")
