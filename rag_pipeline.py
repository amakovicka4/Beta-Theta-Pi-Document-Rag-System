"""
Complete RAG Pipeline with Gemini Integration
Main application for conversational question-answering with document retrieval
"""

import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
from document_parser import DocumentParser
from chunker import DocumentChunker
from vector_store import VectorStore
from rag_retriever import RAGRetriever


class RAGPipeline:
    """
    Complete RAG pipeline that integrates:
    - Document parsing and chunking
    - Vector storage and retrieval
    - Gemini API for conversational responses
    """

    def __init__(
        self,
        # api_key: Optional[str] = None,
        files_directory: str = "Files",
        persist_directory: str = "chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k_results: int = 5,
        gemini_model: str = "gemini-2.5-flash"
    ):
        """
        Initialize the RAG pipeline

        Args:
            api_key: Gemini API key (or set GEMINI_API_KEY env var)
            files_directory: Directory containing documents to parse
            persist_directory: Directory for vector database persistence
            chunk_size: Size of text chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
            top_k_results: Number of chunks to retrieve for context
            gemini_model: Gemini model to use
        """
        # Load environment variables
        load_dotenv()

        # Initialize Gemini client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set in .env file")

        self.client = genai.Client(api_key=api_key)
        self.gemini_model = gemini_model

        # Initialize components
        self.parser = DocumentParser(files_directory)
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = VectorStore(persist_directory=persist_directory)
        self.retriever = RAGRetriever(self.vector_store, top_k=top_k_results)

        # Track conversation history
        self.conversation_history: List[types.Content] = []

        print("RAG Pipeline initialized successfully!")

    def index_documents(self, reset: bool = False) -> None:
        """
        Parse, chunk, and index all documents in the files directory

        Args:
            reset: If True, reset the vector store before indexing
        """
        print("\n" + "="*80)
        print("INDEXING DOCUMENTS")
        print("="*80)

        if reset:
            self.vector_store.reset_collection()

        # Check if already indexed
        if self.vector_store.collection.count() > 0 and not reset:
            print(f"Vector store already contains {self.vector_store.collection.count()} chunks")
            print("Use reset=True to reindex documents")
            return

        # Parse documents
        print("\nStep 1: Parsing documents...")
        documents = self.parser.parse_all_documents()

        if not documents:
            print("No documents found to index!")
            return

        # Chunk documents
        print("\nStep 2: Chunking documents...")
        chunks = self.chunker.chunk_documents(documents)

        # Add to vector store
        print("\nStep 3: Adding chunks to vector store...")
        self.vector_store.add_chunks(chunks)

        print("\n" + "="*80)
        print("INDEXING COMPLETE")
        print("="*80)
        print(f"Total documents indexed: {len(documents)}")
        print(f"Total chunks created: {len(chunks)}")
        print(f"Vector store count: {self.vector_store.collection.count()}")

    def ask(
        self,
        question: str,
        include_context: bool = True,
        stream: bool = False,
        max_tokens: int = 4096
    ) -> str:
        """
        Ask a question and get an answer from Gemini with RAG context

        Args:
            question: User's question
            include_context: Whether to retrieve and include document context
            stream: Whether to stream the response
            max_tokens: Maximum tokens in Gemini's response

        Returns:
            Gemini's response
        """
        # Retrieve relevant context
        context = ""
        retrieved_chunks = []

        if include_context:
            context, retrieved_chunks = self.retriever.retrieve_and_format(question)

            # Print retrieval stats
            stats = self.retriever.get_retrieval_stats(retrieved_chunks)
            if stats['num_chunks'] > 0:
                print(f"\nRetrieved {stats['num_chunks']} relevant chunks from {stats['num_sources']} documents")
                print(f"Average relevance: {stats['avg_relevance_score']:.2%}")
                print(f"Sources: {', '.join(stats['sources'])}")
            else:
                print("\nNo relevant chunks found in documents")

        # Build system prompt
        system_prompt = self._build_system_prompt(context if include_context else None)

        # Build contents with conversation history + new question
        contents = list(self.conversation_history)
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=question)]
        ))

        # Generate response
        print(f"\nAsking Gemini ({self.gemini_model})...\n")

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            temperature=0.7
        )

        if stream:
            response_text = self._stream_response(contents, config)
        else:
            response = self.client.models.generate_content(
                model=self.gemini_model,
                contents=contents,
                config=config
            )
            response_text = response.text

        # Update conversation history
        self.conversation_history.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=question)]
        ))
        self.conversation_history.append(types.Content(
            role="model",
            parts=[types.Part.from_text(text=response_text)]
        ))

        return response_text

    def _build_system_prompt(self, context: Optional[str] = None) -> str:
        """Build system prompt for Gemini"""
        base_prompt = """You are a helpful AI assistant with access to a collection of documents.
Your role is to answer questions accurately based on the provided context.

Guidelines:
- Answer questions using ONLY the information provided in the context
- If the context doesn't contain relevant information, clearly state that
- Be concise but comprehensive in your answers
- Cite specific sources when possible
- If asked about something not in the context, acknowledge the limitation"""

        if context:
            return f"{base_prompt}\n\n{context}"

        return base_prompt

    def _stream_response(self, contents: List[types.Content], config: types.GenerateContentConfig) -> str:
        """Stream Gemini's response"""
        full_response = ""

        for chunk in self.client.models.generate_content_stream(
            model=self.gemini_model,
            contents=contents,
            config=config
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                full_response += chunk.text

        print()  # New line after streaming
        return full_response

    def chat(self, stream: bool = True) -> None:
        """
        Start an interactive chat session

        Args:
            stream: Whether to stream responses
        """
        print("\n" + "="*80)
        print("RAG CHAT SESSION")
        print("="*80)
        print("Ask questions about your documents. Type 'quit', 'exit', or 'q' to end.")
        print("Type 'clear' to clear conversation history.")
        print("Type 'stats' to see vector store statistics.")
        print("="*80 + "\n")

        while True:
            try:
                # Get user input
                question = input("\nYou: ").strip()

                if not question:
                    continue

                # Handle commands
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                if question.lower() == 'clear':
                    self.conversation_history = []
                    print("\nConversation history cleared.")
                    continue

                if question.lower() == 'stats':
                    stats = self.vector_store.get_stats()
                    print(f"\nVector Store Stats:")
                    print(f"  Total chunks: {stats['total_documents']}")
                    print(f"  Collection: {stats['collection_name']}")
                    continue

                # Get answer
                print("\nAssistant: ", end="" if stream else "")
                answer = self.ask(question, stream=stream)

                if not stream:
                    print(answer)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")

    def reset_conversation(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared")


if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RAGPipeline()

    # Index documents (only needed first time or when documents change)
    pipeline.index_documents(reset=False)

    # Start interactive chat
    pipeline.chat(stream=True)
