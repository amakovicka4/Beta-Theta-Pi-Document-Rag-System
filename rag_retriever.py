"""
RAG Retrieval System
Handles intelligent document retrieval and context preparation for Claude
"""

from typing import List, Dict, Any, Optional
from vector_store import VectorStore
import json


class RAGRetriever:
    """
    Retrieval-Augmented Generation system that:
    1. Retrieves relevant document chunks based on queries
    2. Ranks and filters results
    3. Formats context for Claude
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        relevance_threshold: float = 1.5  # Lower is more similar for cosine distance
    ):
        """
        Initialize RAG retriever

        Args:
            vector_store: Initialized VectorStore instance
            top_k: Number of top chunks to retrieve
            relevance_threshold: Maximum distance threshold for relevance
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold

    def retrieve(
        self,
        query: str,
        n_results: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query

        Args:
            query: User's question or search query
            n_results: Number of results to retrieve (defaults to top_k)
            filter_metadata: Optional metadata filters

        Returns:
            List of relevant chunks with metadata and relevance scores
        """
        n_results = n_results or self.top_k

        # Search vector store
        results = self.vector_store.search(
            query=query,
            n_results=n_results,
            filter_metadata=filter_metadata
        )

        # Process and format results
        retrieved_chunks = []

        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            # Filter by relevance threshold
            if distance > self.relevance_threshold:
                continue

            chunk_info = {
                'text': doc,
                'metadata': metadata,
                'relevance_score': 1 - distance,  # Convert distance to similarity score
                'distance': distance,
                'source': metadata.get('file_name', 'unknown'),
                'chunk_index': metadata.get('chunk_index', 'N/A'),
                'total_chunks': metadata.get('total_chunks', 'N/A')
            }

            retrieved_chunks.append(chunk_info)

        return retrieved_chunks

    def format_context_for_claude(
        self,
        chunks: List[Dict[str, Any]],
        include_metadata: bool = True
    ) -> str:
        """
        Format retrieved chunks into a context string for Claude

        Args:
            chunks: List of retrieved chunk dictionaries
            include_metadata: Whether to include source metadata

        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant information found in the documents."

        context_parts = []

        # Group chunks by source for better organization
        chunks_by_source = {}
        for chunk in chunks:
            source = chunk['source']
            if source not in chunks_by_source:
                chunks_by_source[source] = []
            chunks_by_source[source].append(chunk)

        # Format context
        context_parts.append("RELEVANT INFORMATION FROM DOCUMENTS:\n")
        context_parts.append("=" * 80 + "\n")

        for source, source_chunks in chunks_by_source.items():
            context_parts.append(f"\nSource: {source}\n")
            context_parts.append("-" * 80 + "\n")

            for i, chunk in enumerate(source_chunks, 1):
                if include_metadata:
                    context_parts.append(
                        f"[Excerpt {i} - Chunk {chunk['chunk_index']}/{chunk['total_chunks']} - "
                        f"Relevance: {chunk['relevance_score']:.2%}]\n"
                    )

                context_parts.append(f"{chunk['text']}\n\n")

        return "".join(context_parts)

    def retrieve_and_format(
        self,
        query: str,
        n_results: Optional[int] = None,
        include_metadata: bool = True
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve relevant chunks and format them for Claude

        Args:
            query: User's question
            n_results: Number of results to retrieve
            include_metadata: Whether to include metadata in context

        Returns:
            Tuple of (formatted_context, retrieved_chunks)
        """
        chunks = self.retrieve(query, n_results)
        context = self.format_context_for_claude(chunks, include_metadata)

        return context, chunks

    def get_retrieval_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about retrieved chunks

        Args:
            chunks: List of retrieved chunks

        Returns:
            Dictionary with retrieval statistics
        """
        if not chunks:
            return {"num_chunks": 0, "sources": []}

        sources = list(set(chunk['source'] for chunk in chunks))
        avg_relevance = sum(chunk['relevance_score'] for chunk in chunks) / len(chunks)

        return {
            "num_chunks": len(chunks),
            "sources": sources,
            "num_sources": len(sources),
            "avg_relevance_score": avg_relevance,
            "min_relevance": min(chunk['relevance_score'] for chunk in chunks),
            "max_relevance": max(chunk['relevance_score'] for chunk in chunks)
        }


if __name__ == "__main__":
    # Test the retriever
    print("Initializing RAG Retriever...")

    vector_store = VectorStore()
    retriever = RAGRetriever(vector_store, top_k=5)

    # Test queries
    test_queries = [
        "What are the risk management policies?",
        "Who are the executive contacts?",
        "What is the housing agreement about?"
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)

        context, chunks = retriever.retrieve_and_format(query, n_results=3)

        # Print stats
        stats = retriever.get_retrieval_stats(chunks)
        print(f"\nRetrieval Stats:")
        print(json.dumps(stats, indent=2))

        # Print formatted context
        print(f"\n{context}")
