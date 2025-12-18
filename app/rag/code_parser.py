"""
Code parser for extracting imports, definitions, and chunking code files.

Distinguishes between:
- Local codebase imports → Retrieved from repo via RAG
- Standard library imports → Retrieved from pre-embedded stdlib docs
- Third-party imports → Noted but not retrieved (could add PyPI docs later)
"""

import re
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum


class ImportType(Enum):
    """Classification of import source."""
    LOCAL = "local"          # From this codebase
    STDLIB = "stdlib"        # Python standard library
    THIRD_PARTY = "third_party"  # External packages (pip installed)


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""
    content: str
    start_line: int
    end_line: int
    chunk_type: str  # 'function', 'class', 'import', 'block'
    name: Optional[str] = None


@dataclass
class ImportInfo:
    """Represents an import with classification."""
    raw_import: str          # Original import string
    module_name: str         # Top-level module name
    full_path: str           # Full import path
    import_type: ImportType  # LOCAL, STDLIB, or THIRD_PARTY
    is_relative: bool        # True for relative imports (., ..)


class CodeParser:
    """Language-aware code parser for extracting structure and chunking."""

    # Language detection by extension
    LANGUAGE_MAP = {
        'py': 'python',
        'js': 'javascript',
        'jsx': 'javascript',
        'ts': 'typescript',
        'tsx': 'typescript',
        'java': 'java',
        'kt': 'kotlin',
        'swift': 'swift',
        'go': 'go',
        'rs': 'rust',
        'c': 'c',
        'cpp': 'cpp',
        'h': 'c',
        'hpp': 'cpp',
        'rb': 'ruby',
        'php': 'php',
        'cs': 'csharp',
    }

    # Python standard library modules (comprehensive list)
    PYTHON_STDLIB = {
        # Built-in types and functions (always available)
        'builtins',
        # Text Processing
        'string', 're', 'difflib', 'textwrap', 'unicodedata', 'stringprep',
        # Binary Data
        'struct', 'codecs',
        # Data Types
        'datetime', 'zoneinfo', 'calendar', 'collections', 'heapq', 'bisect',
        'array', 'weakref', 'types', 'copy', 'pprint', 'reprlib', 'enum',
        'graphlib',
        # Numeric and Math
        'numbers', 'math', 'cmath', 'decimal', 'fractions', 'random', 'statistics',
        # Functional Programming
        'itertools', 'functools', 'operator',
        # File and Directory Access
        'pathlib', 'os', 'fileinput', 'stat', 'filecmp', 'tempfile', 'glob',
        'fnmatch', 'linecache', 'shutil',
        # Data Persistence
        'pickle', 'copyreg', 'shelve', 'marshal', 'dbm', 'sqlite3',
        # Data Compression
        'zlib', 'gzip', 'bz2', 'lzma', 'zipfile', 'tarfile',
        # File Formats
        'csv', 'configparser', 'tomllib', 'netrc', 'plistlib',
        # Cryptographic
        'hashlib', 'hmac', 'secrets',
        # OS Services
        'os', 'io', 'time', 'argparse', 'getopt', 'logging', 'getpass',
        'curses', 'platform', 'errno', 'ctypes',
        # Concurrent Execution
        'threading', 'multiprocessing', 'concurrent', 'subprocess', 'sched',
        'queue', 'contextvars',
        # Networking
        'asyncio', 'socket', 'ssl', 'select', 'selectors', 'signal',
        # Internet Data Handling
        'email', 'json', 'mailbox', 'mimetypes', 'base64', 'binascii',
        'quopri',
        # HTML/XML Processing
        'html', 'xml',
        # Internet Protocols
        'webbrowser', 'wsgiref', 'urllib', 'http', 'ftplib', 'poplib',
        'imaplib', 'smtplib', 'uuid', 'socketserver', 'xmlrpc', 'ipaddress',
        # Multimedia
        'wave', 'colorsys',
        # Internationalization
        'gettext', 'locale',
        # Program Frameworks
        'turtle', 'cmd', 'shlex',
        # GUI
        'tkinter',
        # Development Tools
        'typing', 'pydoc', 'doctest', 'unittest', 'test',
        # Debugging and Profiling
        'bdb', 'faulthandler', 'pdb', 'timeit', 'trace', 'tracemalloc',
        # Software Packaging
        'ensurepip', 'venv', 'zipapp',
        # Python Runtime
        'sys', 'sysconfig', 'builtins', 'warnings', 'dataclasses',
        'contextlib', 'abc', 'atexit', 'traceback', 'gc', 'inspect',
        'site',
        # Importing
        'importlib', 'zipimport', 'pkgutil', 'modulefinder', 'runpy',
        # Python Language
        'ast', 'symtable', 'token', 'keyword', 'tokenize', 'tabnanny',
        'pyclbr', 'py_compile', 'compileall', 'dis', 'pickletools',
    }

    # Common third-party packages (to distinguish from local imports)
    COMMON_THIRD_PARTY = {
        # Web frameworks
        'flask', 'django', 'fastapi', 'starlette', 'tornado', 'aiohttp',
        'bottle', 'pyramid', 'sanic',
        # Data science
        'numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'plotly',
        'sklearn', 'scikit-learn', 'tensorflow', 'torch', 'keras',
        'xgboost', 'lightgbm', 'statsmodels',
        # HTTP/API
        'requests', 'httpx', 'urllib3', 'aiohttp', 'httplib2',
        # Database
        'sqlalchemy', 'pymongo', 'redis', 'psycopg2', 'mysql', 'pymysql',
        'asyncpg', 'motor', 'peewee', 'tortoise',
        # Testing
        'pytest', 'nose', 'mock', 'hypothesis', 'faker', 'factory_boy',
        # Utilities
        'click', 'typer', 'pydantic', 'attrs', 'marshmallow', 'cerberus',
        'python-dotenv', 'dotenv', 'tqdm', 'rich', 'colorama',
        # Async
        'trio', 'anyio', 'uvloop', 'gevent', 'eventlet', 'celery',
        # Cloud/AWS
        'boto3', 'botocore', 'google', 'azure',
        # Serialization
        'yaml', 'pyyaml', 'toml', 'msgpack', 'orjson', 'ujson',
        # OpenAI and AI
        'openai', 'anthropic', 'langchain', 'llama_index', 'transformers',
        'sentence_transformers', 'tiktoken', 'faiss',
        # Image processing
        'PIL', 'pillow', 'cv2', 'opencv',
        # Other common
        'jinja2', 'beautifulsoup4', 'bs4', 'lxml', 'scrapy', 'selenium',
    }

    # Import patterns for Python
    PYTHON_IMPORT_PATTERNS = [
        # import module
        (r'^import\s+([\w.]+)(?:\s+as\s+\w+)?', False),
        # from module import ...
        (r'^from\s+([\w.]+)\s+import', False),
        # from .relative import ... (relative)
        (r'^from\s+(\.+[\w.]*)\s+import', True),
    ]

    # Function/class definition patterns
    DEFINITION_PATTERNS = {
        'python': {
            'function': r'^(\s*)def\s+(\w+)\s*\(',
            'class': r'^(\s*)class\s+(\w+)',
            'async_function': r'^(\s*)async\s+def\s+(\w+)\s*\(',
        },
        'javascript': {
            'function': r'(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()',
            'class': r'class\s+(\w+)',
        },
        'typescript': {
            'function': r'(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()',
            'class': r'class\s+(\w+)',
            'interface': r'interface\s+(\w+)',
        },
        'java': {
            'method': r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(',
            'class': r'(?:public|private)?\s*class\s+(\w+)',
        },
        'go': {
            'function': r'^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(',
            'struct': r'^type\s+(\w+)\s+struct',
        },
    }

    def __init__(self, repo_files: Set[str] = None, default_chunk_size: int = 50):
        """
        Initialize parser.

        Args:
            repo_files: Set of file paths in the repository (for identifying local imports)
            default_chunk_size: Default number of lines per chunk
        """
        self.repo_files = repo_files or set()
        self.default_chunk_size = default_chunk_size
        # Build lookup for faster local import detection
        self._local_modules = self._build_local_module_set()

    def set_repo_files(self, repo_files: Set[str]):
        """Update the set of known repository files."""
        self.repo_files = repo_files
        self._local_modules = self._build_local_module_set()

    def _build_local_module_set(self) -> Set[str]:
        """Build a set of potential local module names from repo files."""
        modules = set()
        for filepath in self.repo_files:
            if filepath.endswith('.py'):
                # Convert path to potential module name
                # e.g., "src/utils/helpers.py" -> "src.utils.helpers", "utils.helpers", "helpers"
                parts = filepath.replace('/', '.').replace('\\', '.').replace('.py', '')
                # Add all possible import variations
                path_parts = parts.split('.')
                for i in range(len(path_parts)):
                    modules.add('.'.join(path_parts[i:]))
                # Also handle __init__.py
                if parts.endswith('.__init__'):
                    package = parts.replace('.__init__', '')
                    package_parts = package.split('.')
                    for i in range(len(package_parts)):
                        modules.add('.'.join(package_parts[i:]))
        return modules

    def detect_language(self, file_extension: str) -> str:
        """Detect programming language from file extension."""
        ext = file_extension.lstrip('.').lower()
        return self.LANGUAGE_MAP.get(ext, 'unknown')

    def classify_import(self, module_name: str, is_relative: bool) -> ImportType:
        """
        Classify an import as LOCAL, STDLIB, or THIRD_PARTY.

        Args:
            module_name: The imported module name
            is_relative: Whether it's a relative import

        Returns:
            ImportType classification
        """
        # Relative imports are always local
        if is_relative:
            return ImportType.LOCAL

        # Get top-level module name
        top_module = module_name.split('.')[0]

        # Check if it's in Python stdlib
        if top_module in self.PYTHON_STDLIB:
            return ImportType.STDLIB

        # Check if it's a known third-party package
        if top_module.lower() in self.COMMON_THIRD_PARTY:
            return ImportType.THIRD_PARTY

        # Check if it matches a local module
        if module_name in self._local_modules or top_module in self._local_modules:
            return ImportType.LOCAL

        # Default to third-party for unknown imports
        # (better to be conservative - local imports should be in repo_files)
        return ImportType.THIRD_PARTY

    def extract_imports(self, code: str, language: str = 'python') -> List[ImportInfo]:
        """
        Extract and classify all imports from code.

        Args:
            code: Source code content
            language: Programming language (currently only Python fully supported)

        Returns:
            List of ImportInfo objects with classifications
        """
        if language != 'python':
            # For non-Python, return basic extraction without classification
            return self._extract_imports_basic(code, language)

        imports = []
        seen = set()

        for pattern, is_relative_pattern in self.PYTHON_IMPORT_PATTERNS:
            for match in re.finditer(pattern, code, re.MULTILINE):
                import_str = match.group(1)

                if import_str in seen:
                    continue
                seen.add(import_str)

                # Determine if relative
                is_relative = is_relative_pattern or import_str.startswith('.')

                # Clean up module name for relative imports
                clean_module = import_str.lstrip('.')
                top_module = clean_module.split('.')[0] if clean_module else ''

                # Classify the import
                import_type = self.classify_import(import_str, is_relative)

                imports.append(ImportInfo(
                    raw_import=import_str,
                    module_name=top_module,
                    full_path=clean_module,
                    import_type=import_type,
                    is_relative=is_relative
                ))

        return imports

    def _extract_imports_basic(self, code: str, language: str) -> List[ImportInfo]:
        """Basic import extraction for non-Python languages."""
        # Simplified - treat all as THIRD_PARTY for now
        imports = []
        # Could add language-specific patterns here
        return imports

    def get_imports_by_type(
        self,
        code: str,
        language: str = 'python'
    ) -> Tuple[List[ImportInfo], List[ImportInfo], List[ImportInfo]]:
        """
        Extract imports grouped by type.

        Returns:
            Tuple of (local_imports, stdlib_imports, third_party_imports)
        """
        all_imports = self.extract_imports(code, language)

        local = [i for i in all_imports if i.import_type == ImportType.LOCAL]
        stdlib = [i for i in all_imports if i.import_type == ImportType.STDLIB]
        third_party = [i for i in all_imports if i.import_type == ImportType.THIRD_PARTY]

        return local, stdlib, third_party

    def extract_definitions(self, code: str, language: str) -> List[Dict]:
        """
        Extract function and class definitions from code.

        Args:
            code: Source code content
            language: Programming language

        Returns:
            List of definition dicts with name, type, and line number
        """
        patterns = self.DEFINITION_PATTERNS.get(language, {})
        definitions = []
        lines = code.split('\n')

        for def_type, pattern in patterns.items():
            for line_num, line in enumerate(lines, 1):
                match = re.search(pattern, line)
                if match:
                    groups = match.groups()
                    name = next((g for g in groups if g and not g.isspace()), None)
                    if name:
                        definitions.append({
                            'name': name,
                            'type': def_type,
                            'line': line_num,
                        })

        return definitions

    def chunk_code(
        self,
        code: str,
        language: str,
        max_lines: int = None,
        overlap: int = 5
    ) -> List[CodeChunk]:
        """
        Split code into semantic chunks.

        Args:
            code: Source code content
            language: Programming language
            max_lines: Maximum lines per chunk
            overlap: Overlapping lines between chunks

        Returns:
            List of CodeChunk objects
        """
        if max_lines is None:
            max_lines = self.default_chunk_size

        lines = code.split('\n')
        total_lines = len(lines)

        if total_lines <= max_lines:
            return [CodeChunk(
                content=code,
                start_line=1,
                end_line=total_lines,
                chunk_type='file'
            )]

        definitions = self.extract_definitions(code, language)

        if definitions:
            return self._chunk_by_definitions(lines, definitions, max_lines, overlap)

        return self._chunk_fixed_size(lines, max_lines, overlap)

    def _chunk_by_definitions(
        self,
        lines: List[str],
        definitions: List[Dict],
        max_lines: int,
        overlap: int
    ) -> List[CodeChunk]:
        """Chunk code based on function/class definitions."""
        chunks = []
        total_lines = len(lines)
        sorted_defs = sorted(definitions, key=lambda d: d['line'])

        # Header/imports chunk
        if sorted_defs and sorted_defs[0]['line'] > 1:
            header_end = sorted_defs[0]['line'] - 1
            header_content = '\n'.join(lines[:header_end])
            if header_content.strip():
                chunks.append(CodeChunk(
                    content=header_content,
                    start_line=1,
                    end_line=header_end,
                    chunk_type='import'
                ))

        # Chunk by definitions
        for i, defn in enumerate(sorted_defs):
            start_line = defn['line']
            end_line = sorted_defs[i + 1]['line'] - 1 if i + 1 < len(sorted_defs) else total_lines

            chunk_lines = lines[start_line - 1:end_line]
            content = '\n'.join(chunk_lines)

            if len(chunk_lines) > max_lines:
                sub_chunks = self._chunk_fixed_size(
                    chunk_lines, max_lines, overlap, base_line=start_line
                )
                for sc in sub_chunks:
                    sc.name = defn['name']
                    sc.chunk_type = defn['type']
                chunks.extend(sub_chunks)
            else:
                chunks.append(CodeChunk(
                    content=content,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type=defn['type'],
                    name=defn['name']
                ))

        return chunks

    def _chunk_fixed_size(
        self,
        lines: List[str],
        max_lines: int,
        overlap: int,
        base_line: int = 1
    ) -> List[CodeChunk]:
        """Split lines into fixed-size chunks with overlap."""
        chunks = []
        total_lines = len(lines)
        step = max(1, max_lines - overlap)

        i = 0
        while i < total_lines:
            end_idx = min(i + max_lines, total_lines)
            chunk_lines = lines[i:end_idx]
            content = '\n'.join(chunk_lines)

            chunks.append(CodeChunk(
                content=content,
                start_line=base_line + i,
                end_line=base_line + end_idx - 1,
                chunk_type='block'
            ))

            i += step
            if end_idx >= total_lines:
                break

        return chunks

    def get_file_summary(self, code: str, language: str) -> str:
        """Generate a brief summary of a code file."""
        local, stdlib, _ = self.get_imports_by_type(code, language)
        definitions = self.extract_definitions(code, language)

        summary_parts = []

        if local:
            local_names = [i.module_name for i in local[:3]]
            summary_parts.append(f"Local: {', '.join(local_names)}")

        if stdlib:
            stdlib_names = [i.module_name for i in stdlib[:3]]
            summary_parts.append(f"Stdlib: {', '.join(stdlib_names)}")

        if definitions:
            by_type = {}
            for d in definitions:
                by_type.setdefault(d['type'], []).append(d['name'])
            for def_type, names in list(by_type.items())[:2]:
                summary_parts.append(f"{def_type}: {', '.join(names[:3])}")

        return '; '.join(summary_parts) if summary_parts else "No structure detected"
