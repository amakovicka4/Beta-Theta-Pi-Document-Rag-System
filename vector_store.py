"""
Vector Store for Document Embeddings
Uses ChromaDB for efficient similarity search with Google's embedding API
"""

import chromadb
from chromadb.config import Settings
from google import genai
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
from dotenv import load_dotenv

class VectorStore:
    """
    Vector database for storing and retrieving document chunks
    Uses ChromaDB with Google's gemini-embedding-001 model
    """

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "chroma_db",
        embedding_model: str = "gemini-embedding-001"
    ):
        """
        Initialize vector store with Google embedding model and persistence

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector database
            embedding_model: Google embedding model to use
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model

        # Load environment variables
        load_dotenv()

        # Create persist directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize Google client for embeddings
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set in .env file")

        print(f"Initializing Google embedding model: {embedding_model}")
        self.genai_client = genai.Client(api_key=api_key)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        print(f"Vector store initialized. Collection: {collection_name}")
        print(f"Current document count: {self.collection.count()}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using Google's API

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []

        # Process in batches of 100 (API limit)
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            result = self.genai_client.models.embed_content(
                model=self.embedding_model,
                contents=batch
            )

            for embedding in result.embeddings:
                embeddings.append(embedding.values)

        return embeddings

    def add_chunks(self, chunks: List) -> None:
        """
        Add document chunks to the vector store

        Args:
            chunks: List of Chunk objects from chunker
        """
        if not chunks:
            print("No chunks to add")
            return

        print(f"\nAdding {len(chunks)} chunks to vector store...")

        # Prepare data for batch insertion
        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Convert metadata values to strings (ChromaDB requirement)
        cleaned_metadatas = []
        for metadata in metadatas:
            cleaned = {}
            for key, value in metadata.items():
                if isinstance(value, (list, dict)):
                    cleaned[key] = str(value)
                elif value is None:
                    cleaned[key] = ""
                else:
                    cleaned[key] = str(value)
            cleaned_metadatas.append(cleaned)

        # Generate embeddings
        print("Generating embeddings with Google API...")
        embeddings = self.embed_texts(texts)

        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))

            self.collection.add(
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                ids=ids[i:end_idx],
                metadatas=cleaned_metadatas[i:end_idx]
            )

            print(f"Added batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}")

        print(f"Successfully added {len(chunks)} chunks to vector store")
        print(f"Total documents in collection: {self.collection.count()}")

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents using semantic similarity

        Args:
            query: Search query text
            n_results: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            Dictionary containing results with documents, metadatas, and distances
        """
        # Generate query embedding
        query_embedding = self.embed_texts([query])[0]

        # Search the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )

        return results

    def reset_collection(self) -> None:
        """Delete and recreate the collection (useful for rebuilding index)"""
        print(f"Resetting collection: {self.collection_name}")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("Collection reset complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            "collection_name": self.collection_name,
            "total_documents": self.collection.count(),
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_model
        }


if __name__ == "__main__":
    # Test the vector store
    from document_parser import DocumentParser
    from chunker import DocumentChunker

    # Parse and chunk documents
    parser = DocumentParser()
    documents = parser.parse_all_documents()

    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk_documents(documents)

    # Create and populate vector store
    vector_store = VectorStore()

    # Reset if needed (uncomment to start fresh)
    # vector_store.reset_collection()

    # Add chunks
    vector_store.add_chunks(chunks)

    # Test search
    print("\n" + "="*50)
    print("Testing search functionality")
    print("="*50)

    test_query = "What are the risk management policies?"
    results = vector_store.search(test_query, n_results=3)

    print(f"\nQuery: {test_query}")
    print(f"\nTop {len(results['documents'][0])} results:")

    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        print(f"\n--- Result {i} (Distance: {distance:.4f}) ---")
        print(f"Source: {metadata.get('file_name', 'unknown')}")
        print(f"Chunk: {metadata.get('chunk_index', 'N/A')}/{metadata.get('total_chunks', 'N/A')}")
        print(f"Preview: {doc[:200]}...")
