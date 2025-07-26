"""Different search strategies for retrieving relevant documents."""

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class SearchManager:
    """Manages different search strategies for document retrieval."""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode query text into embeddings."""
        return self.embedding_model.encode([query])[0]
    
    def encode_documents(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """Encode document texts into embeddings."""
        texts = [doc["text"] for doc in documents]
        return self.embedding_model.encode(texts)
    
    def vector_search(self, vector_db, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Pure vector similarity search."""
        query_embedding = self.encode_query(query)
        results = vector_db.search(query_embedding, top_k)
        
        # Add search strategy info
        for result in results:
            result["search_strategy"] = "vector"
        
        return results
    
    def semantic_search(self, vector_db, query: str, top_k: int = 5, 
                       fetch_k: int = 20) -> List[Dict[str, Any]]:
        """Enhanced semantic search with re-ranking."""
        # First, get more results than needed
        query_embedding = self.encode_query(query)
        initial_results = vector_db.search(query_embedding, min(fetch_k, top_k * 4))
        
        if len(initial_results) <= top_k:
            for result in initial_results:
                result["search_strategy"] = "semantic"
            return initial_results
        
        # Re-rank based on semantic similarity
        result_texts = [result["text"] for result in initial_results]
        result_embeddings = self.embedding_model.encode(result_texts)
        
        # Calculate similarities
        similarities = np.dot(result_embeddings, query_embedding)
        
        # Sort by similarity and take top_k
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        
        reranked_results = []
        for idx in sorted_indices:
            result = initial_results[idx]
            result["score"] = float(similarities[idx])
            result["search_strategy"] = "semantic"
            reranked_results.append(result)
        
        return reranked_results
    
    def hybrid_search(self, vector_db, query: str, top_k: int = 5, 
                     alpha: float = 0.5) -> List[Dict[str, Any]]:
        """Hybrid search combining keyword and vector search."""
        # Vector search component
        vector_results = self.vector_search(vector_db, query, top_k * 2)
        
        # Keyword search component (simplified)
        keyword_scores = self._calculate_keyword_scores(query, vector_results)
        
        # Combine scores
        combined_results = []
        for i, result in enumerate(vector_results):
            vector_score = result.get("score", 0.0)
            keyword_score = keyword_scores.get(i, 0.0)
            
            # Weighted combination
            combined_score = alpha * vector_score + (1 - alpha) * keyword_score
            
            result["score"] = combined_score
            result["search_strategy"] = "hybrid"
            result["vector_score"] = vector_score
            result["keyword_score"] = keyword_score
            combined_results.append(result)
        
        # Sort by combined score and return top_k
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        return combined_results[:top_k]
    
    def _calculate_keyword_scores(self, query: str, results: List[Dict[str, Any]]) -> Dict[int, float]:
        """Calculate keyword-based scores for results."""
        query_terms = set(query.lower().split())
        scores = {}
        
        for i, result in enumerate(results):
            text = result["text"].lower()
            text_terms = set(text.split())
            
            # Simple term frequency scoring
            common_terms = query_terms.intersection(text_terms)
            score = len(common_terms) / len(query_terms) if query_terms else 0.0
            scores[i] = score
        
        return scores
    
    def search_with_strategy(self, vector_db, query: str, strategy: str, 
                           top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Search using the specified strategy."""
        if strategy == "vector":
            return self.vector_search(vector_db, query, top_k)
        elif strategy == "semantic":
            return self.semantic_search(vector_db, query, top_k, **kwargs)
        elif strategy == "hybrid":
            return self.hybrid_search(vector_db, query, top_k, **kwargs)
        else:
            logger.warning(f"Unknown search strategy: {strategy}")
            return self.vector_search(vector_db, query, top_k)
    
    def get_search_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for search results."""
        if not results:
            return {}
        
        scores = [result.get("score", 0.0) for result in results]
        
        return {
            "total_results": len(results),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "search_strategy": results[0].get("search_strategy", "unknown")
        }