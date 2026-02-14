from typing import List, Dict, Any
import logging
import re
import numpy as np
import streamlit as st
import os

try:
    import google.generativeai as genai
except ImportError:
    genai = None

logger = logging.getLogger(__name__)

class ChunkingManager:

    def __init__(self):
        if genai is not None:
            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                self.model = genai.embed_content
            except Exception as e:
                logger.warning(f"Gemini embedding not available: {e}")
                self.model = None
        else:
            logger.warning("google.generativeai not installed; embedding unavailable")
            self.model = None

    def fixed_size_chunking(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append({"text": text[start:end], "strategy": "fixed_size"})
            start += chunk_size - chunk_overlap
        print(f"[DEBUG] Fixed size chunking: {len(chunks)} chunks created")
        return chunks

    def recursive_chunking(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
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

    def _gemini_embed(self, text: str) -> np.ndarray:
        try:
            response = self.model(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return np.array(response["embedding"])
        except Exception as e:
            logger.error(f"Gemini embedding error: {e}")
            return np.zeros(768)

    def semantic_chunking(self, text: str, buffer_size: int = 1) -> List[Dict[str, Any]]:
        try:
            sentences = re.split(r'(?<=[.?!])\s+', text)
            if not sentences:
                return self.recursive_chunking(text)

            embeddings = [self._gemini_embed(sent) for sent in sentences]
            chunks = []
            current_chunk = [sentences[0]]
            current_embeds = [embeddings[0]]

            for i in range(1, len(sentences)):
                sim = self._cosine_similarity(embeddings[i], np.mean(current_embeds, axis=0))
                if sim < 0.75:
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

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def sentence_chunking(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
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
        all_chunks = []
        for doc in documents:
            text = doc["text"]
            metadata = doc.get("metadata", {})

            if strategy == "fixed_size":
                chunks = self.fixed_size_chunking(text, **kwargs)
            elif strategy == "recursive":
                chunks = self.recursive_chunking(text, **kwargs)
            elif strategy == "semantic":
                chunks = self.semantic_chunking(text, buffer_size=kwargs.get("buffer_size", 1))
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
