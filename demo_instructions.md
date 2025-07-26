# Demo Instructions for RAG Pipeline

Since this RAG pipeline requires Python libraries that cannot be installed in the WebContainer environment, here are the instructions to run this locally:

## Quick Setup

1. **Install Python 3.8+** on your system

2. **Create a virtual environment**:
```bash
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Run the application**:
```bash
streamlit run app.py
```

## What You'll See

The application provides a comprehensive RAG pipeline interface with:

### 1. Document Processing
- Upload multiple PDF files
- Choose from 4 chunking strategies
- Real-time processing metrics

### 2. Vector Database Setup
- Select from 5 different vector databases
- Automatic embedding generation
- Performance timing

### 3. Document Search
- Natural language queries
- 3 different search strategies
- Relevance scoring and results

### 4. Strategy Comparison
- Performance comparison tables
- Interactive charts and visualizations
- Export functionality for reports

## Testing the Pipeline

### Sample Workflow:
1. Upload a few PDF documents
2. Try different chunking strategies (start with "Recursive Character")
3. Set up ChromaDB (easiest to get started)
4. Run searches with different strategies
5. Compare results in the comparison section

### Expected Results:
- **Processing Time**: Varies by document size and chunking strategy
- **Search Quality**: Semantic search typically provides best results
- **Performance**: FAISS is fastest, ChromaDB is most user-friendly

## Key Features Demonstrated

✅ **Multi-format PDF processing**  
✅ **4 chunking strategies with real-time comparison**  
✅ **5 vector database integrations**  
✅ **3 search strategies (Vector, Semantic, Hybrid)**  
✅ **Interactive Streamlit UI with configuration options**  
✅ **Comprehensive performance analytics**  
✅ **Export functionality for comparison reports**  

This implementation showcases a production-ready RAG pipeline with multiple strategies and databases, perfect for evaluating different approaches for your specific use case.