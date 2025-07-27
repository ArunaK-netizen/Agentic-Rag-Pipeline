from typing import List, Dict, Any
import logging
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np

logger = logging.getLogger(__name__)

class ChunkingManager:

    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def fixed_size_chunking(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Split text into fixed-size chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append({"text": text[start:end], "strategy": "fixed_size"})
            start += chunk_size - chunk_overlap
        return chunks

    def recursive_chunking(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """Split text using recursive separators."""
        separators = ["\n\n", "\n", " ", ""]
        for sep in separators:
            if sep and sep in text:
                splits = text.split(sep)
                break
        else:
            splits = [text]

        chunks = []
        chunk = ""
        for part in splits:
            if len(chunk) + len(part) + len(sep) <= chunk_size:
                chunk += part + sep
            else:
                chunks.append({"text": chunk.strip(), "strategy": "recursive"})
                chunk = part + sep
        if chunk:
            chunks.append({"text": chunk.strip(), "strategy": "recursive"})
        return chunks

    def semantic_chunking(self, text: str, buffer_size: int = 1) -> List[Dict[str, Any]]:
        """Split text based on semantic similarity using embeddings."""
        try:
            sentences = re.split(r'(\.|\?|!)\s+', text)
            if not sentences:
                return self.recursive_chunking(text)

            embeddings = self.embedding_model.encode(sentences)
            chunks = []
            current_chunk = [sentences[0]]
            current_embeds = [embeddings[0]]

            for i in range(1, len(sentences)):
                similarity = util.cos_sim(embeddings[i], np.mean(current_embeds, axis=0)).item()
                if similarity < 0.75:
                    chunks.append({"text": " ".join(current_chunk).strip(), "strategy": "semantic"})
                    current_chunk = [sentences[i]]
                    current_embeds = [embeddings[i]]
                else:
                    current_chunk.append(sentences[i])
                    current_embeds.append(embeddings[i])
            if current_chunk:
                chunks.append({"text": " ".join(current_chunk).strip(), "strategy": "semantic"})

            return chunks
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            return self.recursive_chunking(text)

    def sentence_chunking(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
        """Split text by sentences with length limits."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                chunks.append({"text": current_chunk.strip(), "strategy": "sentence"})
                current_chunk = sentence + " "
        if current_chunk:
            chunks.append({"text": current_chunk.strip(), "strategy": "sentence"})

        return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]], strategy: str, **kwargs) -> List[Dict[str, Any]]:
        """Chunk multiple documents using specified strategy."""
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
                logger.warning(f"Unknown chunking strategy: {strategy}, defaulting to recursive.")
                chunks = self.recursive_chunking(text)

            for i, chunk in enumerate(chunks):
                chunk["metadata"] = {
                    **metadata,
                    "chunk_id": f"{doc.get('filename', 'doc')}_{i}",
                    "chunk_strategy": strategy
                }
                all_chunks.append(chunk)

        return all_chunks

    def get_chunking_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
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
