"""
RAG (Retrieval-Augmented Generation) module for AI PR Reviewer.

Uses Vertex AI embeddings and FAISS for semantic code search.
"""

from .embeddings import VertexAIEmbeddings
from .code_parser import CodeParser, CodeChunk, ImportInfo, ImportType
from .indexer import CodebaseIndexer, CodebaseIndex, IndexedChunk
from .retriever import RAGRetriever, RetrievedContext

__all__ = [
    "VertexAIEmbeddings",
    "CodeParser",
    "CodeChunk",
    "ImportInfo",
    "ImportType",
    "CodebaseIndexer",
    "CodebaseIndex",
    "IndexedChunk",
    "RAGRetriever",
    "RetrievedContext",
]
