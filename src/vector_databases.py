from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging
import numpy as np
import json
import os
import uuid
import streamlit as st
import re

def _normalize_pinecone_name(name: str, fallback: str = "default") -> str:
    if not name or not isinstance(name, str):
        name = fallback
    name = name.lower()
    name = re.sub(r"[^a-z0-9-]+", "-", name)   # only a-z, 0-9, and '-'
    name = re.sub(r"-{2,}", "-", name)         # collapse multiple '-'
    name = name.strip("-") or fallback
    # enforce a reasonable max length (Pinecone currently allows up to 45-48 safely)
    return name[:48]



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import streamlit as st

def load_env_variables():
    env_vars = {
        "PINECONE_API_KEY": st.secrets["PINECONE_API_KEY"],
        "PINECONE_REGION": st.secrets["PINECONE_REGION"],
        "PINECONE_CLOUD": st.secrets["PINECONE_CLOUD"],
        "AZURE_SEARCH_ENDPOINT": st.secrets["AZURE_SEARCH_ENDPOINT"],
        "AZURE_SEARCH_API_KEY": st.secrets["AZURE_SEARCH_API_KEY"],
        "GEMINI_API_KEY": st.secrets["GEMINI_API_KEY"],
        "QDRANT_URL": st.secrets["QDRANT_URL"],
        "QDRANT_API_KEY": st.secrets["QDRANT_API_KEY"]
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    return env_vars


def json_dumps_safe(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return json.dumps({"raw": str(obj)})

def json_loads_safe(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {"raw": s}

# Optional imports guarded
try:
    import chromadb
except Exception:
    chromadb = None

try:
    import faiss
except Exception:
    faiss = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
except Exception:
    QdrantClient = None

try:
    from pinecone import Pinecone, ServerlessSpec, PodSpec
except Exception:
    Pinecone = None
    ServerlessSpec = None
    PodSpec = None


class VectorDatabaseInterface(ABC):
    @abstractmethod
    def create_index(self, dimension: int, index_name: str = "default"):
        ...

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        ...

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def delete_index(self):
        ...

    def healthcheck(self) -> bool:
        return True


class ChromaDBManager(VectorDatabaseInterface):
    def __init__(self, persist_directory: Optional[str] = None, index_name_default: str = "default"):
        if chromadb is None:
            raise ImportError("ChromaDB not installed")
        load_env_variables()
        self.persist_directory = persist_directory or st.secrets.get("CHROMA_PERSIST_DIR", "./chroma_data")
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = None
        self.index_name_default = index_name_default

    def create_index(self, dimension: int, index_name: str = None):
        idx = index_name or self.index_name_default
        try:
            self.collection = self.client.get_or_create_collection(
                name=idx,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.exception("Error creating ChromaDB index")
            raise

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        if self.collection is None:
            raise ValueError("Index not created")
        ids = [str(doc.get("metadata", {}).get("chunk_id", f"id_{i}")) for i, doc in enumerate(documents)]
        texts = [doc.get("text", "") for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        self.collection.add(
            embeddings=embeddings.astype(float).tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.collection is None:
            raise ValueError("Index not created")
        res = self.collection.query(query_embeddings=[query_embedding.astype(float).tolist()], n_results=top_k)
        out = []
        for i in range(len(res.get("documents", [[]])[0])):
            dist = res["distances"][i]
            out.append({
                "text": res["documents"][i],
                "metadata": res["metadatas"][i],
                "score": 1 - float(dist)
            })
        return out

    def delete_index(self):
        if self.collection:
            name = self.collection.name
            self.client.delete_collection(name)
            self.collection = None

    def healthcheck(self) -> bool:
        try:
            self.client.heartbeat()
            return True
        except Exception:
            return False


class FAISSManager(VectorDatabaseInterface):
    def __init__(self, index_file: str = "./faiss_index"):
        if faiss is None:
            raise ImportError("FAISS not installed")
        self.index_file = index_file
        self.index = None
        self.documents: List[Dict[str, Any]] = []
        self.dimension = None
        self._load_if_exists()

    def _load_if_exists(self):
        idx_path = f"{self.index_file}.index"
        docs_path = f"{self.index_file}.docs"
        if os.path.exists(idx_path):
            try:
                self.index = faiss.read_index(idx_path)
                self.dimension = self.index.d
            except Exception:
                self.index = None
        if os.path.exists(docs_path):
            try:
                with open(docs_path, "r") as f:
                    self.documents = json.load(f)
            except Exception:
                self.documents = []

    def create_index(self, dimension: int, index_name: str = "default"):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        if self.index is None:
            raise ValueError("Index not created")
        emb = embeddings.astype("float32")
        faiss.normalize_L2(emb)
        self.index.add(emb)
        self.documents.extend(documents)
        faiss.write_index(self.index, f"{self.index_file}.index")
        with open(f"{self.index_file}.docs", "w") as f:
            json.dump(self.documents, f, ensure_ascii=False)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            raise ValueError("Index not created")
        q = query_embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(q)
        distances, indices = self.index.search(q, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    "text": doc.get("text", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": float(distances[i])
                })
        return results

    def delete_index(self):
        for ext in [".index", ".docs"]:
            p = f"{self.index_file}{ext}"
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        self.index = None
        self.documents = []
        self.dimension = None

    def healthcheck(self) -> bool:
        return self.index is not None or (os.path.exists(f"{self.index_file}.index") and os.path.exists(f"{self.index_file}.docs"))


class QdrantManager(VectorDatabaseInterface):
    """Qdrant vector database implementation with env-based config."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        prefer_grpc: Optional[bool] = None,
        grpc_port: Optional[int] = None,
        timeout: int = 30,
    ):
        if QdrantClient is None:
            raise ImportError("Qdrant client not installed")

        # Load .env so QDRANT_URL and QDRANT_API_KEY are available in non-local deploys
        load_env_variables()

        # Resolve configuration from parameters or environment
        env_url = st.secrets["QDRANT_URL"]
        env_api_key = st.secrets["QDRANT_API_KEY"]
        env_host = st.secrets["QDRANT_HOST", "localhost"]
        env_port = int(st.secrets["QDRANT_PORT", "6333"])
        env_prefer_grpc = st.secrets["QDRANT_PREFER_GRPC", ""].lower()
        env_grpc_port = int(st.secrets["QDRANT_GRPC_PORT", "6334"])

        resolved_url = url or env_url
        resolved_api_key = api_key or env_api_key
        resolved_host = host or env_host
        resolved_port = int(port or env_port)
        resolved_prefer_grpc = prefer_grpc if prefer_grpc is not None else (env_prefer_grpc in ("1", "true", "yes"))
        resolved_grpc_port = int(grpc_port or env_grpc_port)

        # Initialize client:
        # - If URL is provided, it implies REST endpoint (http://...:6333). prefer_grpc is honored but requires gRPC availability.
        # - If URL is not provided, use host/port with optional gRPC configuration.
        if resolved_url:
            # Example: QDRANT_URL=http://localhost:6333 or https://xxxx.eu-central.aws.cloud.qdrant.io
            self.client = QdrantClient(
                url=resolved_url,
                api_key=resolved_api_key,
                prefer_grpc=resolved_prefer_grpc,
                timeout=timeout,
            )
        else:
            # Local/docker: ensure ports are mapped: 6333 for REST, 6334 for gRPC
            self.client = QdrantClient(
                host=resolved_host,
                port=resolved_port,
                grpc_port=resolved_grpc_port,
                api_key=resolved_api_key,
                prefer_grpc=resolved_prefer_grpc,
                timeout=timeout,
            )

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

    def _make_point_id(self, doc: Dict[str, Any], fallback_int: int):
        """
        Qdrant requires point IDs to be uint64 or UUID.
        Use deterministic UUID v5 if 'chunk_id' exists, else fallback to int.
        """
        import uuid
        meta = doc.get("metadata", {}) or {}
        chunk_id = meta.get("chunk_id")
        if chunk_id:
            # Deterministic UUID from a namespace + chunk_id so upserts overwrite
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"qdrant://chunk/{chunk_id}"))
        # If no chunk_id, use numeric fallback (uint64)
        return int(fallback_int)

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        if self.collection_name is None:
            raise ValueError("Index not created")

        points = []
        for i, doc in enumerate(documents):
            point_id = self._make_point_id(doc, i)
            payload = {
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}) or {}
            }
            points.append(PointStruct(
                id=point_id,
                vector=embeddings[i].astype(float).tolist(),
                payload=payload
            ))

        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.collection_name is None:
            raise ValueError("Index not created")

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.astype(float).tolist(),
            limit=top_k
        )

        results = []
        for point in search_result:
            results.append({
                "text": point.payload.get("text", ""),
                "metadata": point.payload.get("metadata", {}),
                "score": float(point.score)
            })

        return results

    def delete_index(self):
        if self.collection_name:
            self.client.delete_collection(self.collection_name)
            self.collection_name = None

class PineconeManager(VectorDatabaseInterface):
    def __init__(
        self,
        index_name_default: str = "default",
        dimension_default: Optional[int] = None,
        metric: str = "cosine",
        cloud: Optional[str] = None,      # e.g., "aws", "gcp", "azure"
        region: Optional[str] = None,     # e.g., "us-west-2", "us-east-1", "eu-central-1"
        # Legacy pods (optional)
        pod_env: Optional[str] = None,    # e.g., "us-west1-gcp"
        pod_spec_size: str = "p1.x1",     # pod size if using pods
    ):
        if Pinecone is None:
            raise ImportError("Pinecone not installed")
        load_env_variables()
        api_key = st.secrets["PINECONE_API_KEY"]
        if not api_key:
            raise ValueError("PINECONE_API_KEY missing")

        self.pc = Pinecone(api_key=api_key)
        self.index_name_default = index_name_default
        self.metric = metric
        self.dimension_default = dimension_default

        # Prefer serverless; fall back to pod if explicit
        self.cloud = cloud or st.secrets["PINECONE_CLOUD"]          # "aws" | "gcp" | "azure"
        self.region = region or st.secrets["PINECONE_REGION"]       # "us-west-2" etc.
        self.pod_env = pod_env or st.secrets["PINECONE_ENVIRONMENT"]  # legacy pod env
        self.pod_spec_size = pod_spec_size

        self.index = None

    
    def _ensure_index(self, idx: str, dimension: int):
        idx = _normalize_pinecone_name(idx)
        existing = self.pc.list_indexes()
        existing_names = [i.name for i in getattr(existing, "indexes", [])] or getattr(existing, "names", lambda: [])()
        if idx not in existing_names:
            if self.cloud and self.region:
                self.pc.create_index(
                    name=idx,
                    dimension=dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                )
            elif self.pod_env:
                self.pc.create_index(
                    name=idx,
                    dimension=dimension,
                    metric=self.metric,
                    spec=PodSpec(environment=self.pod_env, pod_type=self.pod_spec_size, pods=1, replicas=1),
                )
            else:
                raise ValueError("Provide serverless (PINECONE_CLOUD + PINECONE_REGION) or pod (PINECONE_ENVIRONMENT) config.")
        self.index = self.pc.Index(idx)

    def create_index(self, dimension: int, index_name: str = None):
        idx = _normalize_pinecone_name(index_name or self.index_name_default or "default")
        if not dimension and not self.dimension_default:
            raise ValueError("Pinecone requires an embedding dimension")
        dim = int(dimension or self.dimension_default)
        self._ensure_index(idx, dim)
    
    
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        if self.index is None:
            raise ValueError("Index not created")
        vectors = []
        for i, doc in enumerate(documents):
            vid = str(doc.get("metadata", {}).get("chunk_id", f"id_{i}"))
            vectors.append({
                "id": vid,
                "values": embeddings[i].astype(float).tolist(),
                "metadata": {"text": doc.get("text", ""), **(doc.get("metadata", {}) or {})}
            })
        # upsert in chunks
        batch = 100
        for i in range(0, len(vectors), batch):
            self.index.upsert(vectors=vectors[i:i+batch])

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            raise ValueError("Index not created")
        res = self.index.query(
            vector=query_embedding.astype(float).tolist(),
            top_k=top_k,
            include_metadata=True
        )
        out = []
        matches = getattr(res, "matches", None) or []
        for m in matches:
            md = getattr(m, "metadata", {}) or {}
            out.append({
                "text": md.get("text", ""),
                "metadata": {k: v for k, v in md.items() if k != "text"},
                "score": float(getattr(m, "score", 0.0) or 0.0)
            })
        return out

    def delete_index(self):
        if self.index:
            name = self.index.name if hasattr(self.index, "name") else getattr(self.index, "_name", None)
            self.index = None
            if name:
                self.pc.delete_index(name)

    def healthcheck(self) -> bool:
        try:
            _ = self.pc.list_indexes()
            return True
        except Exception:
            return False




class VectorDatabaseFactory:
    @staticmethod
    def create_database(db_type: str, **kwargs) -> VectorDatabaseInterface:
        db_type = (db_type or "").lower()
        if db_type == "chroma":
            return ChromaDBManager(**kwargs)
        if db_type == "faiss":
            return FAISSManager(**kwargs)
        if db_type == "qdrant":
            return QdrantManager(**kwargs)
        if db_type == "pinecone":
            return PineconeManager(**kwargs)
        raise ValueError(f"Unsupported database type: {db_type}")
