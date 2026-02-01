"""
Intelligent Document Chunker
Implements semantic chunking strategies for optimal retrieval
"""

from typing import List, Dict, Any
import re
from dataclasses import dataclass
import tiktoken


@dataclass
class Chunk:
    """Represents a document chunk with metadata"""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str


class DocumentChunker:
    """
    Advanced chunking strategy that:
    1. Preserves semantic boundaries (paragraphs, sections)
    2. Maintains context with overlap
    3. Optimizes for embedding model constraints
    4. Tracks source metadata
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize chunker with configurable parameters

        Args:
            chunk_size: Target size for each chunk in tokens
            chunk_overlap: Number of overlapping tokens between chunks
            separators: List of separators to split on (in priority order)
            encoding_name: Tokenizer encoding to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

        # Default separators prioritize semantic boundaries
        self.separators = separators or [
            "\n\n\n",  # Multiple newlines (major sections)
            "\n\n",    # Paragraphs
            "\n",      # Lines
            ". ",      # Sentences
            " ",       # Words
            ""         # Characters (fallback)
        ]

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text"""
        return len(self.encoding.encode(text))

    def split_text_by_separator(self, text: str, separator: str) -> List[str]:
        """Split text by separator while preserving the separator"""
        if separator == "":
            return list(text)

        splits = text.split(separator)
        # Rejoin with separator except for last element
        result = []
        for i, split in enumerate(splits[:-1]):
            result.append(split + separator)
        if splits[-1]:  # Add last element if not empty
            result.append(splits[-1])

        return [s for s in result if s.strip()]

    def merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        Merge small splits into chunks of appropriate size with overlap
        """
        chunks = []
        current_chunk = []
        current_size = 0

        for split in splits:
            split_size = self.count_tokens(split)

            # If single split exceeds chunk_size, add it as its own chunk
            if split_size > self.chunk_size:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                chunks.append(split)
                continue

            # Check if adding this split would exceed chunk_size
            if current_size + split_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(separator.join(current_chunk))

                # Start new chunk with overlap
                overlap_text = separator.join(current_chunk)
                overlap_tokens = self.count_tokens(overlap_text)

                # Keep removing from start until we're under overlap limit
                while overlap_tokens > self.chunk_overlap and len(current_chunk) > 1:
                    current_chunk.pop(0)
                    overlap_text = separator.join(current_chunk)
                    overlap_tokens = self.count_tokens(overlap_text)

                # If still too large, start fresh
                if overlap_tokens > self.chunk_overlap:
                    current_chunk = []
                    current_size = 0

                current_size = overlap_tokens

            current_chunk.append(split)
            current_size += split_size

        # Add final chunk
        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks using hierarchical separators
        """
        # Start with full text
        splits = [text]

        # Try each separator in priority order
        for separator in self.separators:
            new_splits = []
            for split in splits:
                # Only split if chunk is too large
                if self.count_tokens(split) > self.chunk_size:
                    new_splits.extend(self.split_text_by_separator(split, separator))
                else:
                    new_splits.append(split)
            splits = new_splits

        # Merge splits into appropriately sized chunks with overlap
        final_chunks = self.merge_splits(splits, " ")

        return final_chunks

    def chunk_document(self, document, doc_id: str = None) -> List[Chunk]:
        """
        Chunk a document into smaller pieces with metadata

        Args:
            document: Document object from parser
            doc_id: Optional document identifier

        Returns:
            List of Chunk objects
        """
        text_chunks = self.split_text(document.content)

        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            # Create enhanced metadata
            chunk_metadata = {
                **document.metadata,  # Include all original metadata
                'chunk_index': i,
                'total_chunks': len(text_chunks),
                'chunk_size_tokens': self.count_tokens(chunk_text),
                'source_doc_type': document.doc_type
            }

            # Generate unique chunk ID
            chunk_id = f"{doc_id or document.metadata.get('file_name', 'unknown')}_{i}"

            chunks.append(Chunk(
                text=chunk_text,
                metadata=chunk_metadata,
                chunk_id=chunk_id
            ))

        return chunks

    def chunk_documents(self, documents: List) -> List[Chunk]:
        """
        Chunk multiple documents

        Args:
            documents: List of Document objects

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []

        for doc_idx, document in enumerate(documents):
            print(f"Chunking document {doc_idx + 1}/{len(documents)}: {document.metadata.get('file_name', 'unknown')}")

            doc_chunks = self.chunk_document(document, f"doc_{doc_idx}")
            all_chunks.extend(doc_chunks)

            print(f"  Created {len(doc_chunks)} chunks")

        print(f"\nTotal chunks created: {len(all_chunks)}")
        return all_chunks


if __name__ == "__main__":
    # Test the chunker
    from document_parser import DocumentParser

    parser = DocumentParser()
    documents = parser.parse_all_documents()

    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk_documents(documents)

    # Display sample chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n{'='*50}")
        print(f"Chunk {i + 1} (ID: {chunk.chunk_id})")
        print(f"Tokens: {chunk.metadata['chunk_size_tokens']}")
        print(f"Text preview: {chunk.text[:200]}...")
