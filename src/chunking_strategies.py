"""Different text chunking strategies for document processing."""

from typing import List, Dict, Any
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import SentenceTransformerEmbeddings
import logging

logger = logging.getLogger(__name__)

class ChunkingManager:
    """Manages different chunking strategies for text documents."""
    
    def __init__(self):
        self.embedding_model = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
    
    def fixed_size_chunking(self, text: str, chunk_size: int = 1000, 
                           chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Split text into fixed-size chunks."""
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n"
        )
        
        chunks = splitter.split_text(text)
        return [{"text": chunk, "strategy": "fixed_size"} for chunk in chunks]
    
    def recursive_chunking(self, text: str, chunk_size: int = 1000,
                          chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Split text recursively by different separators."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )
        
        chunks = splitter.split_text(text)
        return [{"text": chunk, "strategy": "recursive"} for chunk in chunks]
    
    def semantic_chunking(self, text: str, buffer_size: int = 1) -> List[Dict[str, Any]]:
        """Split text based on semantic similarity."""
        try:
            splitter = SemanticChunker(
                embeddings=self.embedding_model,
                buffer_size=buffer_size,
                breakpoint_threshold_type="percentile"
            )
            
            chunks = splitter.split_text(text)
            return [{"text": chunk, "strategy": "semantic"} for chunk in chunks]
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            # Fallback to recursive chunking
            return self.recursive_chunking(text)
    
    def sentence_chunking(self, text: str, chunk_size: int = 1000,
                         chunk_overlap: int = 100) -> List[Dict[str, Any]]:
        """Split text by sentences with size constraints."""
        splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=chunk_size // 4  # Approximate tokens
        )
        
        chunks = splitter.split_text(text)
        return [{"text": chunk, "strategy": "sentence"} for chunk in chunks]
    
    def chunk_documents(self, documents: List[Dict[str, Any]], 
                       strategy: str, **kwargs) -> List[Dict[str, Any]]:
        """Chunk multiple documents using the specified strategy."""
        all_chunks = []
        
        for doc in documents:
            text = doc["text"]
            metadata = doc.get("metadata", {})
            
            if strategy == "fixed_size":
                chunks = self.fixed_size_chunking(text, **kwargs)
            elif strategy == "recursive":
                chunks = self.recursive_chunking(text, **kwargs)
            elif strategy == "semantic":
                chunks = self.semantic_chunking(text, **kwargs)
            elif strategy == "sentence":
                chunks = self.sentence_chunking(text, **kwargs)
            else:
                logger.warning(f"Unknown chunking strategy: {strategy}")
                chunks = self.recursive_chunking(text)
            
            # Add metadata to chunks
            for i, chunk in enumerate(chunks):
                chunk["metadata"] = {
                    **metadata,
                    "chunk_id": f"{doc.get('filename', 'doc')}_{i}",
                    "chunk_strategy": strategy
                }
                all_chunks.append(chunk)
        
        return all_chunks
    
    def get_chunking_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for the chunking process."""
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk["text"]) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "total_characters": sum(chunk_lengths)
        }