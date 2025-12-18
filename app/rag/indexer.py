"""
Codebase indexer for building FAISS index of repository code.

Walks the repository, parses code files, chunks them at function/method level,
generates embeddings using Vertex AI, and builds a searchable FAISS index.
"""

import os
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging

try:
    import faiss
except ImportError:
    faiss = None

from .embeddings import VertexAIEmbeddings
from .code_parser import CodeParser

logger = logging.getLogger(__name__)


@dataclass
class IndexedChunk:
    """Metadata for an indexed code chunk."""
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str  # 'function', 'class', 'import', 'block', 'file'
    name: Optional[str]  # Function/class name if applicable
    language: str
    content: str  # The actual code content


@dataclass
class CodebaseIndex:
    """Container for the FAISS index and associated metadata."""
    index: object  # faiss.Index
    chunks: List[IndexedChunk]
    import_graph: Dict[str, Set[str]]  # file -> set of imported files
    reverse_import_graph: Dict[str, Set[str]]  # file -> set of files that import it
    file_to_chunks: Dict[str, List[int]]  # file_path -> chunk indices


# Directories and files to skip during indexing
SKIP_DIRS = {
    '.git', '.svn', '.hg',
    'node_modules', 'vendor', 'venv', '.venv', 'env', '.env',
    '__pycache__', '.pytest_cache', '.mypy_cache',
    'build', 'dist', 'target', 'out',
    '.idea', '.vscode', '.vs',
    'coverage', '.coverage', 'htmlcov',
}

SKIP_FILES = {
    '.DS_Store', 'Thumbs.db',
    'package-lock.json', 'yarn.lock', 'poetry.lock',
}

# File extensions to index
INDEXABLE_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.mjs',
    '.java', '.kt', '.go', '.rs',
    '.c', '.cpp', '.cc', '.h', '.hpp',
    '.rb', '.php', '.cs', '.swift',
}


class CodebaseIndexer:
    """
    Indexes a codebase for RAG retrieval using Vertex AI embeddings.
    """

    MAX_CHUNK_LINES = 100

    def __init__(self, max_chunk_lines: int = 100):
        """
        Initialize the indexer.

        Args:
            max_chunk_lines: Maximum lines per chunk
        """
        if faiss is None:
            raise ImportError(
                "faiss-cpu is required for indexing. "
                "Install with: pip install faiss-cpu"
            )

        self.embeddings = VertexAIEmbeddings()
        self.max_chunk_lines = max_chunk_lines
        self.parser = CodeParser(default_chunk_size=max_chunk_lines)

    def _should_skip_path(self, path: str) -> bool:
        """Check if a path should be skipped during indexing."""
        parts = path.split(os.sep)
        for part in parts:
            if part in SKIP_DIRS:
                return True
        filename = os.path.basename(path)
        return filename in SKIP_FILES

    def _should_index_file(self, filepath: str) -> bool:
        """Check if a file should be indexed."""
        if self._should_skip_path(filepath):
            return False
        _, ext = os.path.splitext(filepath)
        return ext.lower() in INDEXABLE_EXTENSIONS

    def _collect_files(self, repo_path: str) -> List[str]:
        """Collect all indexable files in the repository."""
        files = []
        for root, dirs, filenames in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for filename in filenames:
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, repo_path)

                if self._should_index_file(rel_path):
                    files.append(rel_path)

        return files

    def _read_file_safe(self, repo_path: str, filepath: str) -> Optional[str]:
        """Safely read a file, handling encoding issues."""
        full_path = os.path.join(repo_path, filepath)
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(full_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception:
                return None
        except Exception:
            return None

    def _chunk_file_by_functions(
        self,
        filepath: str,
        content: str,
        language: str
    ) -> List[IndexedChunk]:
        """Chunk a file at function/method level."""
        lines = content.split('\n')
        total_lines = len(lines)

        # For small files, keep as single chunk
        if total_lines <= 30:
            return [IndexedChunk(
                file_path=filepath,
                start_line=1,
                end_line=total_lines,
                chunk_type='file',
                name=os.path.basename(filepath),
                language=language,
                content=content
            )]

        # Get function/class definitions
        definitions = self.parser.extract_definitions(content, language)

        if not definitions:
            return [IndexedChunk(
                file_path=filepath,
                start_line=1,
                end_line=total_lines,
                chunk_type='file',
                name=os.path.basename(filepath),
                language=language,
                content=content
            )]

        chunks = []
        sorted_defs = sorted(definitions, key=lambda d: d['line'])

        # Imports and file header
        first_def_line = sorted_defs[0]['line']
        if first_def_line > 1:
            header_content = '\n'.join(lines[:first_def_line - 1])
            if header_content.strip():
                chunks.append(IndexedChunk(
                    file_path=filepath,
                    start_line=1,
                    end_line=first_def_line - 1,
                    chunk_type='imports',
                    name='imports',
                    language=language,
                    content=header_content
                ))

        # Create chunk for each function/class
        for i, defn in enumerate(sorted_defs):
            start_line = defn['line']

            if i + 1 < len(sorted_defs):
                end_line = sorted_defs[i + 1]['line'] - 1
            else:
                end_line = total_lines

            chunk_lines = lines[start_line - 1:end_line]
            chunk_content = '\n'.join(chunk_lines)

            chunks.append(IndexedChunk(
                file_path=filepath,
                start_line=start_line,
                end_line=end_line,
                chunk_type=defn['type'],
                name=defn['name'],
                language=language,
                content=chunk_content[:5000]  # Limit chunk size
            ))

        return chunks

    def index_repository(
        self,
        repo_path: str = ".",
        show_progress: bool = True
    ) -> CodebaseIndex:
        """
        Index an entire repository at function/method level.

        Args:
            repo_path: Path to the repository root
            show_progress: Whether to log progress messages

        Returns:
            CodebaseIndex containing the FAISS index and metadata
        """
        repo_path = os.path.abspath(repo_path)

        if show_progress:
            logger.info("Collecting files...")

        files = self._collect_files(repo_path)

        if show_progress:
            logger.info(f"Found {len(files)} files to index")

        self.parser.set_repo_files(set(files))

        if show_progress:
            logger.info("Chunking files at function level...")

        all_chunks: List[IndexedChunk] = []
        file_to_chunks: Dict[str, List[int]] = {}
        chunk_texts: List[str] = []

        for filepath in files:
            content = self._read_file_safe(repo_path, filepath)
            if content is None or not content.strip():
                continue

            _, ext = os.path.splitext(filepath)
            language = self.parser.detect_language(ext)

            file_chunks = self._chunk_file_by_functions(filepath, content, language)
            file_chunk_indices = []

            for chunk in file_chunks:
                chunk_idx = len(all_chunks)
                file_chunk_indices.append(chunk_idx)
                all_chunks.append(chunk)

                # Prepare text for embedding
                embed_text = f"File: {filepath}\n"
                if chunk.name:
                    embed_text += f"{chunk.chunk_type}: {chunk.name}\n"
                embed_text += chunk.content
                chunk_texts.append(embed_text)

            if file_chunk_indices:
                file_to_chunks[filepath] = file_chunk_indices

        if show_progress:
            logger.info(f"Created {len(all_chunks)} function-level chunks")
            logger.info("Generating embeddings with Vertex AI...")

        embeddings = self.embeddings.embed_batch(chunk_texts)

        if show_progress:
            logger.info("Building FAISS index...")

        dimension = len(embeddings[0]) if embeddings else 768
        index = faiss.IndexFlatIP(dimension)

        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)

        if show_progress:
            logger.info(f"Index built with {index.ntotal} vectors")

        return CodebaseIndex(
            index=index,
            chunks=all_chunks,
            import_graph={},
            reverse_import_graph={},
            file_to_chunks=file_to_chunks
        )

    def index_files_from_github(
        self,
        files: List[Dict],
        get_content_func
    ) -> CodebaseIndex:
        """
        Index files fetched from GitHub API.

        Args:
            files: List of file info dicts from GitHub
            get_content_func: Function to get file content by path

        Returns:
            CodebaseIndex
        """
        logger.info(f"Indexing {len(files)} files from GitHub...")

        all_chunks: List[IndexedChunk] = []
        file_to_chunks: Dict[str, List[int]] = {}
        chunk_texts: List[str] = []

        for file_info in files:
            filepath = file_info.get("filename", "")
            if not self._should_index_file(filepath):
                continue

            content = get_content_func(filepath)
            if not content:
                continue

            _, ext = os.path.splitext(filepath)
            language = self.parser.detect_language(ext)

            file_chunks = self._chunk_file_by_functions(filepath, content, language)
            file_chunk_indices = []

            for chunk in file_chunks:
                chunk_idx = len(all_chunks)
                file_chunk_indices.append(chunk_idx)
                all_chunks.append(chunk)

                embed_text = f"File: {filepath}\n"
                if chunk.name:
                    embed_text += f"{chunk.chunk_type}: {chunk.name}\n"
                embed_text += chunk.content
                chunk_texts.append(embed_text)

            if file_chunk_indices:
                file_to_chunks[filepath] = file_chunk_indices

        if not chunk_texts:
            logger.warning("No chunks to index")
            return CodebaseIndex(
                index=faiss.IndexFlatIP(768),
                chunks=[],
                import_graph={},
                reverse_import_graph={},
                file_to_chunks={}
            )

        logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
        embeddings = self.embeddings.embed_batch(chunk_texts)

        dimension = len(embeddings[0]) if embeddings else 768
        index = faiss.IndexFlatIP(dimension)

        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)

        logger.info(f"Index built with {index.ntotal} vectors")

        return CodebaseIndex(
            index=index,
            chunks=all_chunks,
            import_graph={},
            reverse_import_graph={},
            file_to_chunks=file_to_chunks
        )
