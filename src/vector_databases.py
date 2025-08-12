"""Vector database implementations for different providers."""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging
import numpy as np
import json
import os

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

# For loading .env file without dotenv dependency
def load_env_variables():
    """Load environment variables from .env file manually"""
    env_path = '.env'
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value

# Azure AI Search imports (already in your code)
try:
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        SearchIndex, SearchField, SearchFieldDataType, VectorSearch,
        VectorSearchProfile, HnswAlgorithmConfiguration, SimpleField,
        SearchableField
    )
    from azure.search.documents.models import VectorizedQuery
    from azure.core.credentials import AzureKeyCredential
except ImportError:
    SearchClient = None

logger = logging.getLogger(__name__)

class AzureAISearchManager(VectorDatabaseInterface):
    """Azure AI Search vector database implementation."""
    
    def __init__(self, service_endpoint: str = None, api_key: str = None, 
                 index_name: str = "default-vector-index"):
        if SearchClient is None:
            raise ImportError("Azure Search Documents not installed. Run: pip install azure-search-documents")
        
        # Load environment variables if not provided
        if not service_endpoint or not api_key:
            load_env_variables()
            
        self.service_endpoint = service_endpoint or os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_SEARCH_API_KEY")
        self.index_name = index_name
        
        if not self.service_endpoint or not self.api_key:
            raise ValueError("Azure Search service endpoint and API key must be provided via parameters or .env file")
        
        self.credential = AzureKeyCredential(self.api_key)
        self.index_client = SearchIndexClient(self.service_endpoint, self.credential)
        self.search_client = None
        self.dimension = None
    
    def create_index(self, dimension: int, index_name: str = "default"):
        """Create Azure AI Search index with vector field configuration."""
        self.dimension = dimension
        self.index_name = index_name
        
        try:
            # Define the index schema
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SearchField(
                    name="content_vector",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=dimension,
                    vector_search_profile_name="vector-profile"
                ),
                SearchableField(name="metadata", type=SearchFieldDataType.String)
            ]
            
            # Configure vector search
            vector_search = VectorSearch(
                profiles=[
                    VectorSearchProfile(
                        name="vector-profile",
                        algorithm_configuration_name="vector-algorithm"
                    )
                ],
                algorithms=[
                    HnswAlgorithmConfiguration(name="vector-algorithm")
                ]
            )
            
            # Create the index
            index = SearchIndex(
                name=self.index_name,
                fields=fields,
                vector_search=vector_search
            )
            
            # Create or update the index
            self.index_client.create_or_update_index(index)
            
            # Initialize search client
            self.search_client = SearchClient(
                self.service_endpoint, 
                self.index_name, 
                self.credential
            )
            
            logger.info(f"Successfully created Azure AI Search index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error creating Azure AI Search index: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add documents with embeddings to the Azure AI Search index."""
        if self.search_client is None:
            raise ValueError("Index not created. Call create_index() first.")
        
        try:
            # Prepare documents for Azure AI Search
            search_documents = []
            for i, doc in enumerate(documents):
                # Create a unique ID if not present
                doc_id = doc.get("metadata", {}).get("chunk_id", f"doc_{i}")
                
                search_doc = {
                    "id": str(doc_id),
                    "content": doc.get("text", ""),
                    "content_vector": embeddings[i].tolist(),
                    "metadata": str(doc.get("metadata", {}))  # Convert dict to string for storage
                }
                search_documents.append(search_doc)
            
            # Upload documents in batches
            batch_size = 100  # Azure AI Search batch limit
            for i in range(0, len(search_documents), batch_size):
                batch = search_documents[i:i + batch_size]
                result = self.search_client.upload_documents(documents=batch)
                
                # Check for any failures
                for doc_result in result:
                    if not doc_result.succeeded:
                        logger.warning(f"Failed to index document {doc_result.key}: {doc_result.error_message}")
            
            logger.info(f"Successfully added {len(documents)} documents to Azure AI Search index")
            
        except Exception as e:
            logger.error(f"Error adding documents to Azure AI Search: {e}")
            raise
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform vector similarity search in Azure AI Search."""
        if self.search_client is None:
            raise ValueError("Index not created. Call create_index() first.")
        
        try:
            # Create vector query
            vector_query = VectorizedQuery(
                vector=query_embedding.tolist(),
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            
            # Execute the search
            results = self.search_client.search(
                search_text=None,  # Pure vector search
                vector_queries=[vector_query],
                select=["id", "content", "metadata"],
                top=top_k
            )
            
            # Format results
            search_results = []
            for result in results:
                try:
                    # Parse metadata back from string
                    metadata = eval(result.get("metadata", "{}")) if result.get("metadata") else {}
                except:
                    metadata = {"raw_metadata": result.get("metadata", "")}
                
                search_results.append({
                    "text": result.get("content", ""),
                    "metadata": metadata,
                    "score": result.get("@search.score", 0.0)
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error performing vector search in Azure AI Search: {e}")
            raise
    
    def hybrid_search(self, query_text: str, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search combining text and vector search."""
        if self.search_client is None:
            raise ValueError("Index not created. Call create_index() first.")
        
        try:
            # Create vector query
            vector_query = VectorizedQuery(
                vector=query_embedding.tolist(),
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            
            # Execute hybrid search
            results = self.search_client.search(
                search_text=query_text,  # Text search component
                vector_queries=[vector_query],  # Vector search component
                select=["id", "content", "metadata"],
                top=top_k
            )
            
            # Format results
            search_results = []
            for result in results:
                try:
                    metadata = eval(result.get("metadata", "{}")) if result.get("metadata") else {}
                except:
                    metadata = {"raw_metadata": result.get("metadata", "")}
                
                search_results.append({
                    "text": result.get("content", ""),
                    "metadata": metadata,
                    "score": result.get("@search.score", 0.0)
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error performing hybrid search in Azure AI Search: {e}")
            raise
    
    def delete_index(self):
        """Delete the Azure AI Search index."""
        try:
            if self.index_name:
                self.index_client.delete_index(self.index_name)
                self.search_client = None
                logger.info(f"Successfully deleted Azure AI Search index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error deleting Azure AI Search index: {e}")
            raise


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
            return AzureAISearchManager(**kwargs)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")