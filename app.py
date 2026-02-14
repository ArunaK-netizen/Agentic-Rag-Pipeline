import streamlit as st
import pandas as pd
import logging
import sys
import os
from typing import Dict, Any, List
import io
from src.rag_pipeline import RAGPipeline
from src.comparison_table import ComparisonAnalyzer
from src.config import VECTOR_DB_CONFIGS, CHUNKING_STRATEGIES, SEARCH_STRATEGIES, ENABLE_LOCAL_OCR
from src.gemini_vlm import is_gemini_available

# Pre-download EasyOCR models to avoid timeout on first use
@st.cache_resource
def load_ocr_model():
    """Pre-cache EasyOCR model to avoid timeout during document processing."""
    try:
        import easyocr
        return easyocr.Reader(['en'], gpu=False)
    except Exception as e:
        logger.warning(f"EasyOCR initialization failed: {e}")
        return None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Trigger OCR model loading early only when local OCR is enabled
if ENABLE_LOCAL_OCR:
    load_ocr_model()

st.set_page_config(
    page_title="RAG Pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Show Gemini Vision status in the sidebar
try:
    gemini_status = is_gemini_available()
    if gemini_status.get("available"):
        st.sidebar.success(f"Gemini Vision: available — {gemini_status.get('reason')}")
    else:
        st.sidebar.warning(f"Gemini Vision: unavailable — {gemini_status.get('reason')}")
except Exception as _e:
    st.sidebar.info("Gemini Vision: status unknown")

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    
    if 'comparison_analyzer' not in st.session_state:
        st.session_state.comparison_analyzer = ComparisonAnalyzer()
    
    if 'experiment_results' not in st.session_state:
        st.session_state.experiment_results = []
    
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    
    if 'database_setup' not in st.session_state:
        st.session_state.database_setup = False

def render_sidebar():
    st.sidebar.header("Configuration")

    st.sidebar.subheader("Document Source")
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents",
        type=["pdf", "png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp", "txt"],
        accept_multiple_files=True,
        help="Upload PDFs, images, or text files. Unprocessable files will automatically use OCR"
    )
    filenames = []
    total_size_mb = 0

    if uploaded_files:
        for f in uploaded_files:
            filenames.append(f.name)
            total_size_mb += f.size / (1024 * 1024)
        st.sidebar.success(f"{len(uploaded_files)} file(s) uploaded ({total_size_mb:.2f} MB)")
        with st.sidebar.expander("Preview Uploaded Files"):
            for name in filenames[:5]:
                st.write(f"• {name}")
            if len(filenames) > 5:
                st.write(f"... and {len(filenames) - 5} more")
    else:
        st.sidebar.info("No files uploaded yet.")

    st.sidebar.subheader("Chunking Strategy")
    chunking_strategy = st.sidebar.selectbox(
        "Select chunking strategy",
        options=list(CHUNKING_STRATEGIES.keys()),
        format_func=lambda x: CHUNKING_STRATEGIES[x]["name"],
        help="Choose how to split documents into chunks"
    )
    
    with st.sidebar.expander("Chunking Parameters"):
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)
    
    st.sidebar.subheader("Vector Database")
    vector_db = st.sidebar.selectbox(
        "Select vector database",
        options=list(VECTOR_DB_CONFIGS.keys()),
        format_func=lambda x: VECTOR_DB_CONFIGS[x]["name"],
        help="Choose the vector database for storing embeddings"
    )
    
    st.sidebar.subheader("Search Strategy")
    search_strategy = st.sidebar.selectbox(
        "Select search strategy",
        options=list(SEARCH_STRATEGIES.keys()),
        format_func=lambda x: SEARCH_STRATEGIES[x]["name"],
        help="Choose the search method for retrieval"
    )
    
    with st.sidebar.expander("Search Parameters"):
        top_k = st.slider("Number of Results", 1, 20, 5)
        if search_strategy == "hybrid":
            alpha = st.slider("Vector/Keyword Balance", 0.0, 1.0, 0.5)
        else:
            alpha = 0.5
    
    return {
        'uploaded_files': uploaded_files,
        'chunking_strategy': chunking_strategy,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'vector_db': vector_db,
        'search_strategy': search_strategy,
        'top_k': top_k,
        'alpha': alpha
    }
def process_documents_section(config: Dict[str, Any]):
    """Handle document processing section."""
    st.header("Document Processing")
    
    if config['uploaded_files']:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info(f"Uploaded {len(config['uploaded_files'])} file(s)")


        
        with col2:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    result = st.session_state.rag_pipeline.process_documents(
                        config['uploaded_files'],
                        config['chunking_strategy'],
                        chunk_size=config['chunk_size'],
                        chunk_overlap=config['chunk_overlap']
                    )
                
                if result.get('success'):
                    st.session_state.documents_processed = True
                    st.success("Documents processed successfully!")
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Documents", result['num_documents'])
                    with col2:
                        st.metric("Chunks", result['num_chunks'])
                    with col3:
                        st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                    with col4:
                        st.metric("Avg Chunk Length", f"{result['chunk_stats']['avg_chunk_length']:.0f}")
                else:
                    st.error(f"Error processing documents: {result.get('error')}")
    else:
        st.info("Please upload PDF files to get started.")

def setup_database_section(config: Dict[str, Any]):
    """Handle vector database setup section."""
    st.header("Vector Database Setup")
    
    if not st.session_state.documents_processed:
        st.warning("Please process documents first before setting up the database.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info(f"Selected Database: {VECTOR_DB_CONFIGS[config['vector_db']]['name']}")
        st.write(VECTOR_DB_CONFIGS[config['vector_db']]['description'])
    
    with col2:
        if st.button("🔧 Setup Database", type="primary"):
            with st.spinner("Setting up vector database..."):
                result = st.session_state.rag_pipeline.setup_vector_database(
                    config['vector_db']
                )
            
            if result.get('success'):
                st.session_state.database_setup = True
                st.success("Database setup successful!")
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Database Type", result['vector_db_type'])
                with col2:
                    st.metric("Vectors", result['num_vectors'])
                with col3:
                    st.metric("Indexing Time", f"{result['indexing_time']:.2f}s")
            else:
                st.error(f"Error setting up database: {result.get('error')}")

    """Handle document search section."""
    st.header("Document Search")
    
    if not st.session_state.database_setup:
        st.warning("Please setup the vector database first before searching.")
        return
    
    # Query input
    query = st.text_input(
        "Enter your search query:",
        placeholder="What would you like to know about the documents?",
        help="Enter a natural language question or keywords"
    )
    
    if query:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("Search", type="primary"):
                with st.spinner("Searching documents..."):
                    search_kwargs = {}
                    if config['search_strategy'] == "hybrid":
                        search_kwargs['alpha'] = config['alpha']
                    
                    result = st.session_state.rag_pipeline.search_documents(
                        query,
                        config['search_strategy'],
                        config['top_k'],
                        **search_kwargs
                    )
                
                if result.get('success'):
                    st.success(f"Found {len(result['results'])} relevant documents")
                    
                    # Store experiment results
                    experiment_data = {
                        'query': query,
                        'chunking_strategy': config['chunking_strategy'],
                        'vector_db_type': config['vector_db'],
                        'search_strategy': config['search_strategy'],
                        'num_results': len(result['results']),
                        'search_time': result['search_time'],
                        'avg_score': result['search_stats']['avg_score'],
                        'processing_time': st.session_state.rag_pipeline.pipeline_stats.get('processing', {}).get('processing_time', 0),
                        'indexing_time': st.session_state.rag_pipeline.pipeline_stats.get('indexing', {}).get('indexing_time', 0),
                        'num_chunks': st.session_state.rag_pipeline.pipeline_stats.get('processing', {}).get('total_chunks', 0),
                        'avg_chunk_length': st.session_state.rag_pipeline.pipeline_stats.get('processing', {}).get('avg_chunk_length', 0)
                    }
                    
                    st.session_state.experiment_results.append(experiment_data)
                    st.session_state.comparison_analyzer.add_experiment(experiment_data)
                    
                    # Display search metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Search Time", f"{result['search_time']:.3f}s")
                    with col2:
                        st.metric("Results Found", len(result['results']))
                    with col3:
                        st.metric("Avg Score", f"{result['search_stats']['avg_score']:.3f}")
                    
                    # Display results
                    st.subheader("Search Results")
                    for i, doc in enumerate(result['results']):
                        with st.expander(f"Result {i+1} (Score: {doc.get('score', 0):.3f})"):
                            st.write(doc['text'][:500] + "..." if len(doc['text']) > 500 else doc['text'])
                            st.json(doc.get('metadata', {}))
                else:
                    st.error(f"Search error: {result.get('error')}")

    """Handle strategy comparison section."""
    st.header("Strategy Comparison")
    
    if not st.session_state.experiment_results:
        st.info("Run some experiments to see comparisons!")
        return
    
    # Create tabs for different comparisons - Fix: unpack the tuple
    tab1, = st.tabs(["Performance"])  # Note the comma after tab1
    
    with tab1:
        # Performance comparison table
        comparison_df = st.session_state.comparison_analyzer.generate_strategy_comparison_table()
        st.subheader("Performance Comparison")
        st.dataframe(comparison_df, use_container_width=True)
        

def gemini_chat_section(config: Dict[str, Any]):
    """Section for LLM-powered Q&A using Gemini Pro."""
    st.header("💬 Gemini LLM Chatbot (Ask Questions About Your Documents)")

    if not st.session_state.database_setup:
        st.warning("Please setup the vector database first before chatting with Gemini.")
        return

    # User input for chatbot
    user_query = st.text_input(
        "Ask a question to Gemini about your documents:",
        placeholder="e.g., Summarize the main findings from the documents.",
        key="gemini_chat_query"
    )

    if user_query:
        if st.button("Get Gemini Answer", key="gemini_chat_button"):
            with st.spinner("Retrieving relevant context and generating answer with Gemini..."):
                # Retrieve top-k relevant chunks
                search_kwargs = {}
                if config['search_strategy'] == "hybrid":
                    search_kwargs['alpha'] = config['alpha']

                search_result = st.session_state.rag_pipeline.search_documents(
                    user_query,
                    config['search_strategy'],
                    config['top_k'],
                    **search_kwargs
                )

                if not search_result.get("success"):
                    st.error(f"Retrieval error: {search_result.get('error')}")
                    return

                # Generate answer with Gemini
                gemini_result = st.session_state.rag_pipeline.generate_answer_with_gemini(
                    user_query,
                    search_result["results"]
                )

                if gemini_result.get("success"):
                    st.success("Gemini's Answer:")
                    st.markdown(f"> {gemini_result['answer']}")
                else:
                    st.error(f"Gemini error: {gemini_result.get('error')}")


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.markdown('<h1 class="main-header">RAG Pipeline</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Render sidebar and get configuration
    config = render_sidebar()
    
    # Main content sections
    process_documents_section(config)
    st.markdown("---")
    
    setup_database_section(config)
    st.markdown("---")

    
    gemini_chat_section(config)
    st.markdown("---")
    
    
    # Footer
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            Built with Streamlit • RAG Pipeline with Multiple Strategies
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()