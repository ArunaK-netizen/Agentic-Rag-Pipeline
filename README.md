# Agentic RAG Pipeline with Streamlit UI

A comprehensive Retrieval-Augmented Generation (RAG) pipeline with multiple chunking strategies, vector databases, and search methods.

## Features

### 📄 Document Processing
- **PDF Text Extraction**: Extract text from uploaded PDF files
- **Multiple File Support**: Process multiple PDFs simultaneously
- **Metadata Preservation**: Maintain document metadata throughout the pipeline

### ✂️ Chunking Strategies
- **Fixed Size**: Split documents into equal-sized chunks
- **Recursive Character**: Split recursively using multiple separators
- **Semantic**: Split based on semantic similarity using embeddings
- **Sentence-based**: Split by sentences with size constraints

### 🗄️ Vector Databases
- **Qdrant**: Vector similarity search engine
- **Pinecone**: Managed cloud vector database (planned)
- **Azure AI Search**: Microsoft's cognitive search service (planned)

### 🔍 Search Strategies
- **Vector Search**: Pure semantic similarity search
- **Semantic Search**: Enhanced vector search with re-ranking
- **Hybrid Search**: Combines keyword and vector search

### 📊 Analysis & Comparison
- **Performance Metrics**: Track processing, indexing, and search times
- **Strategy Comparison**: Compare different approaches side-by-side
- **Export Functionality**: Generate comprehensive comparison reports

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd rag-pipeline
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables** (create `.env` file):
```env
# Optional: OpenAI API key for enhanced embeddings
OPENAI_API_KEY=your_openai_api_key

# Optional: Pinecone configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENV=us-west1-gcp

# Optional: Azure AI Search configuration
AZURE_SEARCH_ENDPOINT=https://your-service.search.windows.net
AZURE_SEARCH_API_KEY=your_azure_search_api_key
```

## Usage

1. **Start the Streamlit application**:
```bash
streamlit run app.py
```

2. **Upload PDF documents** using the sidebar file uploader

3. **Configure your pipeline**:
   - Select chunking strategy
   - Choose vector database
   - Pick search strategy
   - Adjust parameters as needed

4. **Process documents** to extract text and create chunks

5. **Set up vector database** to store embeddings

6. **Search documents** using natural language queries

7. **Compare strategies** in the comparison section

## Architecture

### Core Components

- **PDFProcessor**: Handles PDF text extraction
- **ChunkingManager**: Implements different chunking strategies
- **VectorDatabaseFactory**: Creates and manages vector database instances
- **SearchManager**: Handles different search strategies
- **RAGPipeline**: Orchestrates the entire pipeline
- **ComparisonAnalyzer**: Generates comparison reports and visualizations

### Supported Configurations

#### Chunking Strategies
| Strategy | Description | Best For |
|----------|-------------|----------|
| Fixed Size | Equal-sized chunks | Simple, uniform processing |
| Recursive | Multi-separator splitting | Structured documents |
| Semantic | Similarity-based splitting | High-quality retrieval |
| Sentence | Sentence boundary splitting | Question-answering |

#### Vector Databases
| Database | Type | Persistence | Scalability | Cost |
|----------|------|-------------|-------------|------|
| Qdrant | Server | Yes | High | Free/Paid |
| Pinecone | Cloud | Yes | Very High | Paid |

#### Search Strategies
| Strategy | Description | Latency | Best For |
|----------|-------------|---------|----------|
| Vector | Pure similarity search | Low | Semantic queries |
| Semantic | Enhanced with re-ranking | Medium | Complex queries |
| Hybrid | Keyword + vector | High | Mixed query types |

## Configuration Options

### OCR Runtime Configuration
- **`ENABLE_LOCAL_OCR`**: Set to `1`, `true`, or `yes` to enable local EasyOCR model usage. Default is disabled to avoid heavy model downloads during deployment. To enable local OCR set the environment variable in your deployment or `.env` file:
```env
ENABLE_LOCAL_OCR=1
```

### Gemini Vision Configuration
To enable Gemini Vision API (recommended over local OCR in deployments) set one or both of the following:

- `GEMINI_API_KEY` — your Gemini/Google Generative AI API key. Prefer storing this in your deployment secrets or `.streamlit/secrets.toml` for Streamlit Cloud.
- `GEMINI_VISION_ENDPOINT` — optional custom REST endpoint that accepts JSON payloads with base64 images (used when SDK is not present).

Examples (in your `.env` or deployment settings):
```env
GEMINI_API_KEY=sk-...your_key...
# Optional: custom REST endpoint
GEMINI_VISION_ENDPOINT=https://your-custom-gemini-endpoint.example.com/v1/vision:analyze
```

Notes:
- The app will prefer the `google.generativeai` SDK if installed and the API key is present. Otherwise it will call `GEMINI_VISION_ENDPOINT` if configured.
- If Gemini is available, local OCR (EasyOCR) is disabled by default unless you set `ENABLE_LOCAL_OCR=1`.


### Chunking Parameters
- **Chunk Size**: 500-2000 characters
- **Chunk Overlap**: 0-500 characters
- **Separators**: Customizable for recursive splitting

### Search Parameters
- **Top K**: Number of results to return (1-20)
- **Alpha**: Balance between keyword and vector search (hybrid only)
- **Fetch K**: Initial results for re-ranking (semantic search)

## Performance Considerations

- **Qdrant**: Excellent for production with advanced filtering needs
- **Semantic Chunking**: Higher quality but computationally expensive
- **Hybrid Search**: Most flexible but slower than pure vector search

## Limitations

- PDF text extraction quality depends on document format
- Semantic chunking requires significant computational resources
- Some vector databases require additional setup (Qdrant server, cloud accounts)
- Large documents may require memory optimization

## Future Enhancements

- [ ] Support for more document formats (Word, PowerPoint, etc.)
- [ ] Advanced metadata filtering
- [ ] Custom embedding model integration
- [ ] Distributed processing capabilities
- [ ] Real-time document updates
- [ ] Advanced visualization dashboards

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or contributions, please open an issue on the GitHub repository.