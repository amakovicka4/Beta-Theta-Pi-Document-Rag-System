# RAG Pipeline for Document Q&A with Gemini API

I made this RAG Pipeline to answer questions about my fraternities rules & schedule using natural language. It allows my executive team to quickly find search results as opposed to spending time searching through the documents.

## Features

- **Multi-format Document Parsing**: Handles PDF and Excel files with metadata extraction
- **Intelligent Chunking**: Semantic text chunking with configurable overlap for optimal context preservation
- **Vector Search**: ChromaDB-based vector storage with sentence-transformer embeddings
- **Claude Integration**: Conversational AI powered by Gemini Flash 2.5
- **Interactive Chat**: Stream responses in real-time with conversation history
- **Persistent Storage**: Vector database persists between sessions

## Installation

### Prerequisites

- Python 3.8 or higher
- Gemini API Key

### Setup Steps

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure API key**:

# Edit .env and add your API key
# GEMINI_API_KEY=your_api_key_here
```

3. **Add your documents**:
   - Place PDF and Excel files in the `Files/` directory
   - The pipeline will automatically parse and index them

## Usage
python rag_pipeline.py

### Embedding Model
- **Model**: all-MiniLM-L6-v2
- **Dimensions**: 384
- **Performance**: Fast, efficient, good quality
- **Use case**: Semantic similarity search

### Vector Database
- **Database**: ChromaDB
- **Similarity**: Cosine distance
- **Persistence**: Local file-based storage
- **Scalability**: Handles thousands of chunks efficiently

### Chunking Strategy
- **Method**: Recursive hierarchical splitting
- **Separators**: Sections → Paragraphs → Sentences → Words
- **Token counting**: tiktoken (cl100k_base)
- **Overlap**: Maintains context across chunk boundaries

## License

This project is provided as-is for use with Beta Theta Pi documentation.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the configuration options
3. Examine the example usage in each module's `__main__` section
