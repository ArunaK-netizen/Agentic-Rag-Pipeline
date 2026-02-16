"""Configuration settings for the RAG pipeline."""

import os
from typing import Dict, Any

# OCR Configuration
ENABLE_LOCAL_OCR = False

# Vector Database Configurations
VECTOR_DB_CONFIGS = {
    "qdrant": {
        "name": "Qdrant",
        "description": "Vector similarity search engine",
        "host": "localhost",
        "port": 6333
    },
    "pinecone": {
        "name": "Pinecone",
        "description": "Managed vector database",
        "api_key": os.getenv("PINECONE_API_KEY"),
        "environment": os.getenv("PINECONE_ENV", "us-west1-gcp")
    }
}

# Chunking Strategy Configurations
CHUNKING_STRATEGIES = {
    "fixed_size": {
        "name": "Fixed Size",
        "description": "Split text into fixed-size chunks",
        "chunk_size": 1000,
        "chunk_overlap": 200
    },
    "recursive": {
        "name": "Recursive Character",
        "description": "Split recursively by characters",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "separators": ["\n\n", "\n", " ", ""]
    },
    "semantic": {
        "name": "Semantic",
        "description": "Split based on semantic similarity",
        "buffer_size": 1,
        "breakpoint_threshold_type": "percentile"
    },
    "sentence": {
        "name": "Sentence-based",
        "description": "Split by sentences",
        "chunk_size": 1000,
        "chunk_overlap": 100
    }
}

# Search Strategy Configurations
SEARCH_STRATEGIES = {
    "vector": {
        "name": "Vector Search",
        "description": "Pure vector similarity search",
        "search_type": "similarity"
    },
    "semantic": {
        "name": "Semantic Search",
        "description": "Enhanced semantic understanding",
        "search_type": "mmr",
        "k": 5,
        "fetch_k": 20
    },
    "hybrid": {
        "name": "Hybrid Search",
        "description": "Combines keyword and vector search",
        "alpha": 0.5  # Balance between keyword and vector
    }
}

# Embedding Models
EMBEDDING_MODELS = {
    "sentence_transformers": {
        "name": "all-MiniLM-L6-v2",
        "dimension": 384
    },
    "openai": {
        "name": "text-embedding-ada-002",
        "dimension": 1536
    }
}

# Default settings
DEFAULT_SETTINGS = {
    "chunk_strategy": "recursive",
    "search_strategy": "semantic",
    "vector_db": "chroma",
    "embedding_model": "sentence_transformers",
    "top_k": 5
}

# Control whether local OCR (EasyOCR) is allowed. Set via env var `ENABLE_LOCAL_OCR=1`
ENABLE_LOCAL_OCR = os.getenv("ENABLE_LOCAL_OCR", "0").lower() in ("1", "true", "yes")