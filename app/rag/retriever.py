"""
RAG Retriever for fetching relevant context from the codebase index.

Uses Vertex AI embeddings and FAISS for semantic search.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import logging

try:
    import faiss
except ImportError:
    faiss = None

from .embeddings import VertexAIEmbeddings
from .indexer import CodebaseIndex, IndexedChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """Context retrieved for a file being reviewed."""
    similar_chunks: List[Tuple[IndexedChunk, float]] = field(default_factory=list)
    related_chunks: List[IndexedChunk] = field(default_factory=list)

    def format_for_prompt(self, max_tokens: int = 2000) -> str:
        """
        Format retrieved context for injection into AI prompt.

        Args:
            max_tokens: Approximate token budget for context

        Returns:
            Formatted context string
        """
        sections = []
        estimated_chars = 0
        max_chars = max_tokens * 4  # Rough estimate

        # Section 1: Similar Code Patterns
        if self.similar_chunks:
            similar_section = "## Similar Code in Codebase\n"
            for chunk, score in self.similar_chunks[:3]:
                if score < 0.5:
                    continue

                chunk_text = f"### {chunk.file_path}:{chunk.start_line}"
                if chunk.name:
                    chunk_text += f" ({chunk.name})"
                chunk_text += f" [similarity: {score:.2f}]\n```\n"

                content = chunk.content
                if len(content) > 400:
                    content = content[:400] + "\n# ... (truncated)"
                chunk_text += content + "\n```\n"

                if estimated_chars + len(chunk_text) > max_chars * 0.7:
                    break

                similar_section += chunk_text
                estimated_chars += len(chunk_text)

            if len(similar_section) > 30:
                sections.append(similar_section)

        # Section 2: Related Code from same files
        if self.related_chunks and estimated_chars < max_chars * 0.9:
            related_section = "## Related Code\n"
            for chunk in self.related_chunks[:2]:
                chunk_text = f"### {chunk.file_path}:{chunk.start_line}"
                if chunk.name:
                    chunk_text += f" ({chunk.name})"
                chunk_text += "\n```\n"

                content = chunk.content
                if len(content) > 300:
                    content = content[:300] + "\n# ... (truncated)"
                chunk_text += content + "\n```\n"

                if estimated_chars + len(chunk_text) > max_chars:
                    break

                related_section += chunk_text
                estimated_chars += len(chunk_text)

            if len(related_section) > 20:
                sections.append(related_section)

        return "\n".join(sections) if sections else ""


class RAGRetriever:
    """
    Retrieves relevant context for code review using semantic search.
    """

    def __init__(self, codebase_index: CodebaseIndex):
        """
        Initialize the retriever.

        Args:
            codebase_index: Pre-built FAISS index with code chunks
        """
        self.index = codebase_index
        self.embeddings = VertexAIEmbeddings()

    def retrieve_context(
        self,
        file_path: str,
        file_content: str,
        diff_content: str,
        top_k: int = 5
    ) -> RetrievedContext:
        """
        Retrieve relevant context for reviewing a file.

        Args:
            file_path: Path of the file being reviewed
            file_content: Full content of the file
            diff_content: Git diff for the file
            top_k: Number of similar chunks to retrieve

        Returns:
            RetrievedContext with similar code
        """
        context = RetrievedContext()

        # Search for similar code using the diff
        query = diff_content if diff_content else file_content[:2000]
        context.similar_chunks = self._search_similar(
            query=query,
            exclude_file=file_path,
            top_k=top_k
        )

        # Get other chunks from the same file for context
        chunk_indices = self.index.file_to_chunks.get(file_path, [])
        for idx in chunk_indices[:3]:
            if idx < len(self.index.chunks):
                context.related_chunks.append(self.index.chunks[idx])

        return context

    def _search_similar(
        self,
        query: str,
        exclude_file: Optional[str] = None,
        top_k: int = 5
    ) -> List[Tuple[IndexedChunk, float]]:
        """
        Search for similar code chunks using FAISS.

        Args:
            query: Query text (code or diff)
            exclude_file: File to exclude from results
            top_k: Number of results to return

        Returns:
            List of (chunk, similarity_score) tuples
        """
        if not query or self.index.index.ntotal == 0:
            return []

        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_text(query)
            query_array = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_array)

            # Search FAISS index
            search_k = top_k * 3 if exclude_file else top_k
            distances, indices = self.index.index.search(query_array, search_k)

            # Convert results to chunks
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.index.chunks):
                    continue

                chunk = self.index.chunks[idx]

                # Skip chunks from the file being reviewed
                if exclude_file and chunk.file_path == exclude_file:
                    continue

                similarity = float(dist)
                results.append((chunk, similarity))

                if len(results) >= top_k:
                    break

            return results

        except Exception as e:
            logger.error(f"Error searching similar code: {e}")
            return []

    def search_by_function_name(
        self,
        function_name: str,
        top_k: int = 5
    ) -> List[Tuple[IndexedChunk, float]]:
        """
        Search for chunks by function/class name.

        Args:
            function_name: Name to search for
            top_k: Number of results

        Returns:
            List of matching chunks with scores
        """
        # Try exact name match first
        exact_matches = []
        for chunk in self.index.chunks:
            if chunk.name and function_name.lower() in chunk.name.lower():
                exact_matches.append((chunk, 1.0))

        if exact_matches:
            return exact_matches[:top_k]

        # Fall back to semantic search
        return self._search_similar(
            query=f"function {function_name}",
            top_k=top_k
        )
