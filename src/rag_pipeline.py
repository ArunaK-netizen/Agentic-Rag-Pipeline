"""Main RAG pipeline orchestrating all components."""

from typing import List, Dict, Any, Optional
import logging
import time
from .pdf_processor import PDFProcessor
from .chunking_strategies import ChunkingManager
from .vector_databases import VectorDatabaseFactory
from .search_strategies import SearchManager
from .config import VECTOR_DB_CONFIGS, CHUNKING_STRATEGIES, SEARCH_STRATEGIES

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
    
    def extract_text_from_pdf(file) -> str:
        """Extract text from a PDF file object using PyMuPDF."""
        try:
            file_bytes = file.read()
            file.seek(0)  # reset stream pointer
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from {getattr(file, 'name', 'unnamed')}: {e}")
            return ""


    def process_documents(self, uploaded_files, chunking_strategy: str = "recursive", 
                        **chunking_kwargs) -> Dict[str, Any]:
        """Process uploaded PDF documents."""
        start_time = time.time()
        processed_documents = []
        skipped_files = []

        logger.info(f"Processing {len(uploaded_files)} uploaded files...")

        # 1. Extract text from PDFs
        for file in uploaded_files:
            try:
                text = self.extract_text_from_pdf(file)
                if text.strip():
                    processed_documents.append({"text": text, "name": file.name})
                else:
                    skipped_files.append((file.name, "No extractable text"))
            except Exception as e:
                skipped_files.append((file.name, str(e)))

        if not processed_documents:
            return {"error": "No documents were successfully processed"}

        # 2. Chunk documents
        logger.info(f"Chunking {len(processed_documents)} documents using '{chunking_strategy}' strategy...")
        chunks = self.chunking_manager.chunk_documents(
            processed_documents,
            chunking_strategy,
            **chunking_kwargs
        )

        processing_time = time.time() - start_time
        chunk_stats = self.chunking_manager.get_chunking_stats(chunks)

        # Save for downstream usage
        self.processed_documents = processed_documents
        self.chunks = chunks
        self.pipeline_stats["processing"] = {
            "processing_time": processing_time,
            "num_documents": len(processed_documents),
            "chunking_strategy": chunking_strategy,
            **chunk_stats
        }

        return {
            "success": True,
            "num_documents": len(processed_documents),
            "num_chunks": len(chunks),
            "processing_time": processing_time,
            "chunk_stats": chunk_stats,
            "skipped_files": skipped_files  # <-- for frontend if you want to show it
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