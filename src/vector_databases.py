"""Vector database implementations for different providers."""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging
import numpy as np
import json
import os

# Import specific database clients
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except ImportError:
    QdrantClient = None

try:
    import pinecone
except ImportError:
    pinecone = None

try:
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.core.credentials import AzureKeyCredential
except ImportError:
    SearchClient = None

logger = logging.getLogger(__name__)

class VectorDatabaseInterface(ABC):
    """Abstract interface for vector databases."""
    
    @abstractmethod
    def create_index(self, dimension: int, index_name: str = "default"):
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def delete_index(self):
        pass

class ChromaDBManager(VectorDatabaseInterface):
    """ChromaDB vector database implementation."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        if chromadb is None:
            raise ImportError("ChromaDB not installed")
        
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None
    
    def create_index(self, dimension: int, index_name: str = "default"):
        try:
            self.collection = self.client.get_or_create_collection(
                name=index_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.error(f"Error creating ChromaDB index: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        if self.collection is None:
            raise ValueError("Index not created")
        
        ids = [doc["metadata"]["chunk_id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.collection is None:
            raise ValueError("Index not created")
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        search_results = []
        for i in range(len(results["documents"][0])):
            search_results.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i]  # Convert distance to similarity
            })
        
        return search_results
    
    def delete_index(self):
        if self.collection:
            self.client.delete_collection(self.collection.name)

class FAISSManager(VectorDatabaseInterface):
    """FAISS vector database implementation."""
    
    def __init__(self, index_file: str = "./faiss_index"):
        if faiss is None:
            raise ImportError("FAISS not installed")
        
        self.index_file = index_file
        self.index = None
        self.documents = []
        self.dimension = None
    
    def create_index(self, dimension: int, index_name: str = "default"):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product similarity
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        if self.index is None:
            raise ValueError("Index not created")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(documents)
        
        # Save index and documents
        faiss.write_index(self.index, f"{self.index_file}.index")
        with open(f"{self.index_file}.docs", 'w') as f:
            json.dump(self.documents, f)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            raise ValueError("Index not created")
        
        # Normalize query
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        search_results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                search_results.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": float(distances[0][i])
                })
        
        return search_results
    
    def delete_index(self):
        for ext in ['.index', '.docs']:
            file_path = f"{self.index_file}{ext}"
            if os.path.exists(file_path):
                os.remove(file_path)

class QdrantManager(VectorDatabaseInterface):
    """Qdrant vector database implementation."""
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        if QdrantClient is None:
            raise ImportError("Qdrant client not installed")
        
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = None
    
    def create_index(self, dimension: int, index_name: str = "default"):
        self.collection_name = index_name
        
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
            )
        except Exception as e:
            logger.error(f"Error creating Qdrant collection: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        if self.collection_name is None:
            raise ValueError("Index not created")
        
        points = []
        for i, doc in enumerate(documents):
            points.append(PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload={
                    "text": doc["text"],
                    "metadata": doc["metadata"]
                }
            ))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.collection_name is None:
            raise ValueError("Index not created")
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        
        results = []
        for point in search_result:
            results.append({
                "text": point.payload["text"],
                "metadata": point.payload["metadata"],
                "score": point.score
            })
        
        return results
    
    def delete_index(self):
        if self.collection_name:
            self.client.delete_collection(self.collection_name)

class VectorDatabaseFactory:
    """Factory for creating vector database instances."""
    
    @staticmethod
    def create_database(db_type: str, **kwargs) -> VectorDatabaseInterface:
        if db_type == "chroma":
            return ChromaDBManager(**kwargs)
        elif db_type == "faiss":
            return FAISSManager(**kwargs)
        elif db_type == "qdrant":
            return QdrantManager(**kwargs)
        elif db_type == "pinecone":
            # Placeholder for Pinecone implementation
            raise NotImplementedError("Pinecone implementation pending")
        elif db_type == "azure":
            # Placeholder for Azure AI Search implementation
            raise NotImplementedError("Azure AI Search implementation pending")
        else:
            raise ValueError(f"Unsupported database type: {db_type}")