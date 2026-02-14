"""Main RAG pipeline orchestrating all components."""

from typing import List, Dict, Any, Optional
import logging
import time
from .pdf_processor import PDFProcessor
from .chunking_strategies import ChunkingManager
from .vector_databases import VectorDatabaseFactory
from .search_strategies import SearchManager
from .config import VECTOR_DB_CONFIGS, CHUNKING_STRATEGIES, SEARCH_STRATEGIES
import os
import tempfile
import fitz  
import google.generativeai
import streamlit as st

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main RAG pipeline class that orchestrates the entire process."""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.chunking_manager = ChunkingManager()
        self.search_manager = SearchManager()
        self.vector_db = None
        self.processed_documents = []
        self.chunks = []
        self.pipeline_stats = {}
    
    def extract_text_from_pdf(self, file) -> str:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name
            doc = fitz.open(tmp_path)
            text = ""
            for i, page in enumerate(doc):
                page_text = page.get_text()
                print(f"[DEBUG] Page {i+1} of {file.name}: {len(page_text)} chars")
                text += page_text
            return text
        except Exception as e:
            print(f"[ERROR] Could not extract from {file.name}: {e}")
            return ""
        finally:
            file.seek(0)  # always reset

    def process_documents(self, uploaded_files, chunking_strategy: str = "recursive", 
                        **chunking_kwargs) -> Dict[str, Any]:
        start_time = time.time()
        print(f"[INFO] Starting document processing: {len(uploaded_files)} files")

        # Use the centralized PDFProcessor which handles multiple formats and OCR/VLM fallbacks
        try:
            docs = self.pdf_processor.process_uploaded_files(uploaded_files)
        except Exception as e:
            print(f"[ERROR] PDFProcessor failed: {e}")
            return {"error": str(e)}

        processed_documents = []
        skipped_files = []

        for d in docs:
            text = d.get("text", "")
            name = d.get("filename") or d.get("metadata", {}).get("source", "unknown")
            if text and text.strip():
                processed_documents.append({"text": text, "name": name})
                print(f"[SUCCESS] {name}: extracted {len(text.strip())} characters")
            else:
                skipped_files.append(name)
                print(f"[WARNING] Skipped {name}: no extractable text")

        if not processed_documents:
            print(f"[FAIL] No documents processed. Skipped files: {skipped_files}")
            return {"error": "No documents were successfully processed"}

        print(f"[INFO] Chunking {len(processed_documents)} documents...")
        chunks = self.chunking_manager.chunk_documents(processed_documents, chunking_strategy, **chunking_kwargs)
        chunk_stats = self.chunking_manager.get_chunking_stats(chunks)

        processing_time = time.time() - start_time
        self.processed_documents = processed_documents
        self.chunks = chunks
        self.pipeline_stats["processing"] = {
            "processing_time": processing_time,
            "num_documents": len(processed_documents),
            "chunking_strategy": chunking_strategy,
            **chunk_stats
        }

        print(f"[DONE] Processed {len(processed_documents)} documents into {len(chunks)} chunks")
        return {
            "success": True,
            "num_documents": len(processed_documents),
            "num_chunks": len(chunks),
            "processing_time": processing_time,
            "chunk_stats": chunk_stats,
            "skipped_files": skipped_files
        }
    
    def setup_vector_database(self, db_type: str, index_name: str = "rag_index", 
                             **db_kwargs) -> Dict[str, Any]:
        """Set up and populate the vector database."""
        if not self.chunks:
            return {"error": "No chunks available. Process documents first."}
        
        start_time = time.time()
        
        try:
            # Create vector database
            logger.info(f"Setting up {db_type} vector database...")
            db_config = VECTOR_DB_CONFIGS.get(db_type, {})
            all_kwargs = {**db_config, **db_kwargs}
            allowed_keys = ["persist_directory"]  # whatever args ChromaDBManager actually accepts
            init_kwargs = {k: v for k, v in all_kwargs.items() if k in allowed_keys}

            self.vector_db = VectorDatabaseFactory.create_database(
                db_type,
                **init_kwargs
            )



            
            # Create index
            self.vector_db.create_index(
                dimension=self.search_manager.embedding_dimension,
                index_name=index_name
            )
            
            # Generate embeddings
            logger.info("Generating embeddings for chunks...")
            embeddings = self.search_manager.encode_documents(self.chunks)
            
            # Add documents to vector database
            logger.info("Adding documents to vector database...")
            self.vector_db.add_documents(self.chunks, embeddings)
            
            indexing_time = time.time() - start_time
            
            self.pipeline_stats["indexing"] = {
                "indexing_time": indexing_time,
                "vector_db_type": db_type,
                "embedding_dimension": self.search_manager.embedding_dimension,
                "num_vectors": len(self.chunks)
            }
            
            return {
                "success": True,
                "vector_db_type": db_type,
                "indexing_time": indexing_time,
                "num_vectors": len(self.chunks)
            }
            
        except Exception as e:
            logger.error(f"Error setting up vector database: {e}")
            return {"error": str(e)}
    
    def search_documents(self, query: str, search_strategy: str = "semantic", 
                        top_k: int = 5, **search_kwargs) -> Dict[str, Any]:
        """Search for relevant documents."""
        if not self.vector_db:
            return {"error": "Vector database not set up. Set up database first."}
        
        start_time = time.time()
        
        try:
            logger.info(f"Searching with {search_strategy} strategy...")
            results = self.search_manager.search_with_strategy(
                self.vector_db, 
                query, 
                search_strategy, 
                top_k, 
                **search_kwargs
            )
            
            search_time = time.time() - start_time
            search_stats = self.search_manager.get_search_stats(results)
            
            self.pipeline_stats["search"] = {
                "search_time": search_time,
                "query": query,
                **search_stats
            }
            
            return {
                "success": True,
                "results": results,
                "search_time": search_time,
                "search_stats": search_stats
            }
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return {"error": str(e)}
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics and summary."""
        return {
            "pipeline_stats": self.pipeline_stats,
            "current_config": {
                "num_documents": len(self.processed_documents),
                "num_chunks": len(self.chunks),
                "vector_db_active": self.vector_db is not None
            }
        }
    
    def reset_pipeline(self):  
        """Reset the pipeline state."""
        if self.vector_db:
            try:
                self.vector_db.delete_index()
            except Exception as e:
                logger.warning(f"Error deleting index: {e}")
        
        self.vector_db = None
        self.processed_documents = []
        self.chunks = []
        self.pipeline_stats = {}
        
        logger.info("Pipeline reset successfully")
        
        
    def generate_answer_with_gemini(self, query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an answer using Gemini Pro LLM based on retrieved document chunks.
        Args:
            query (str): The user's question.
            search_results (List[Dict[str, Any]]): Retrieved chunks from the vector database.
        Returns:
            Dict[str, Any]: {"success": True, "answer": ...} or {"error": ...}
        """
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            if not api_key:
                logger.error("GEMINI_API_KEY not found in secrets.")
                return {"error": "GEMINI_API_KEY not set in secrets."}
            google.generativeai.configure(api_key=api_key)

            # Concatenate retrieved document texts
            context = "\n\n".join([chunk.get("text", "") for chunk in search_results if chunk.get("text")])

            # Build prompt
            prompt = (
                "You are an expert assistant. Use the following context from documents to answer the user's question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                "Answer:"
            )

            # Generate answer using Gemini Pro
            model = google.generativeai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            answer = response.text if hasattr(response, "text") else str(response)

            return {"success": True, "answer": answer.strip()}
        except Exception as e:
            logger.error(f"Error generating answer with Gemini: {e}")
            return {"error": str(e)}
