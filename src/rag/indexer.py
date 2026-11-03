"""
indexer.py — fetch a GitHub repo's Python source files via the API,
chunk them by top-level function/class, embed with sentence-transformers,
and persist to ChromaDB.

Usage:
    from src.rag.indexer import index_repo

    index_repo("scikit-learn/scikit-learn", token=os.getenv("GITHUB_TOKEN"))

The ChromaDB collection is named after the repo slug (slashes replaced with
double-underscores) so multiple repos can be indexed without collision.
"""

import ast
import base64
import contextlib
import contextvars
import hashlib
import os
import re
import textwrap
import time
from dataclasses import dataclass
from typing import Optional

import chromadb
import requests
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chromadb")
EMBED_MODEL  = os.getenv("EMBED_MODEL",  "all-MiniLM-L6-v2")
_RETRY_SLEEP = 2.0   # seconds; doubles on each retry
MAX_FILES    = int(os.getenv("INDEX_MAX_FILES", "500"))   # guard against huge repos
MAX_CHUNK_TOKENS = 400   # rough chars-based limit; keeps chunks focused

_embed_model: Optional[SentenceTransformer] = None

# The commit the index is read at. None means "the repo as it is today".
# The evaluation sets it to a PR's base commit, so a pre-2023 PR is checked
# against the code that existed then, not against code written after it.
_INDEX_REF: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "pr_rakshaka_index_ref", default=None)


@contextlib.contextmanager
def use_index_ref(ref: Optional[str]):
    """Within this block, every index lookup, build and check uses the repo at `ref`."""
    token = _INDEX_REF.set(ref or None)
    try:
        yield
    finally:
        _INDEX_REF.reset(token)

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

_TRIPLE_QUOTED = re.compile(r'(?s)[rRuUbB]{0,2}("""|' + "'''" + r').*?\1')

def _strip_docstrings_and_comments(code: str) -> str:
    """Regex fallback for code that won't parse (diff fragments)."""
    code = _TRIPLE_QUOTED.sub("", code)
    kept = [
        ln.rstrip() for ln in code.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    return "\n".join(kept)


def normalize_for_embedding(code: str) -> str:
    """Drop docstrings/comments/formatting so index and query are comparable."""
    code = textwrap.dedent(code)
    # second candidate closes a dangling "def f(...):" with a stub body
    for candidate in (code, code + "\n    pass"):
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
                body = node.body
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    node.body = body[1:] or [ast.Pass()]
        return ast.unparse(tree)   # unparse also drops comments
    return _strip_docstrings_and_comments(code)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    repo: str           # "owner/repo"
    filepath: str       # "sklearn/utils/_bunch.py"
    symbol: str         # "class Bunch" or "def fit" or "<module>"
    source: str         # the actual source text of this chunk
    start_line: int
    end_line: int

    @property
    def chunk_id(self) -> str:
        """Stable ID: sha256 of repo + filepath + symbol."""
        raw = f"{self.repo}::{self.filepath}::{self.symbol}::{self.start_line}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def text_for_embedding(self) -> str:
        return normalize_for_embedding(self.source)

# ---------------------------------------------------------------------------
# Step 1 — list Python files in the repo via GitHub tree API
# ---------------------------------------------------------------------------

def _get(url: str, headers: dict, params: Optional[dict] = None,
         timeout: int = 30, attempts: int = 4) -> requests.Response:
    """requests.get with retries for timeouts, dropped connections, 5xx and rate limits.
    Raises the last error when every attempt fails (so the index stays marked partial)."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            limited = resp.status_code == 429 or (
                resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0")
            if resp.status_code >= 500 or limited:
                raise requests.HTTPError(f"{resp.status_code} from {url}")
            return resp
        except requests.RequestException as exc:
            last = exc
            if i < attempts - 1:
                time.sleep(_RETRY_SLEEP * (2 ** i))
    raise last  # type: ignore[misc]


def _list_python_files(repo: str, token: Optional[str],
                       ref: Optional[str] = None) -> list[str]:
    """Return list of .py file paths for the repo at `ref` (default: its default branch)."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    if ref:
        branch = ref
    else:
        # get default branch
        meta = _get(f"https://api.github.com/repos/{repo}", headers, timeout=30)
        meta.raise_for_status()
        branch = meta.json().get("default_branch", "main")

    # fetch full recursive tree
    tree_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
    resp = _get(tree_url, headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    paths = [
        item["path"]
        for item in data.get("tree", [])
        if item["type"] == "blob"
        and item["path"].endswith(".py")
        and not _is_noise_file(item["path"])
    ]
    return paths[:MAX_FILES]


def _is_noise_file(path: str) -> bool:
    """Skip test files, migrations, generated code, setup boilerplate."""
    noise_patterns = [
        r"test_", r"_test\.py", r"/tests/", r"/test/",
        r"conftest\.py", r"setup\.py", r"setup_cfg",
        r"migrations/", r"__pycache__", r"\.egg",
        r"docs/", r"examples/", r"benchmarks/",
    ]
    return any(re.search(p, path) for p in noise_patterns)


# ---------------------------------------------------------------------------
# Step 2 — fetch a single file's source via GitHub contents API
# ---------------------------------------------------------------------------

def _fetch_file(repo: str, path: str, token: Optional[str],
                ref: Optional[str] = None) -> Optional[str]:
    """Return decoded source text, or None if fetch fails / file is binary."""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    resp = _get(f"https://api.github.com/repos/{repo}/contents/{path}", headers,
                params={"ref": ref} if ref else None, timeout=30)
    if resp.status_code != 200:
        return None

    data = resp.json()
    if data.get("encoding") != "base64":
        return None

    try:
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Step 3 — chunk source into top-level functions and classes via AST
# ---------------------------------------------------------------------------

def _iter_defs(body, prefix: str = ""):
    """Yield (symbol_label, node) for functions, classes, and methods."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield f"def {prefix}{node.name}", node
        elif isinstance(node, ast.ClassDef):
            yield f"class {prefix}{node.name}", node
            yield from _iter_defs(node.body, prefix=f"{prefix}{node.name}.")

def _chunk_file(repo: str, filepath: str, source: str) -> list[CodeChunk]:
    """
    Parse the file with the AST and extract each top-level function/class
    as a separate chunk.  Falls back to the whole file (truncated) if AST
    parsing fails (e.g. syntax errors, Python 2).
    """
    chunks: list[CodeChunk] = []
    lines = source.splitlines(keepends=True)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # fallback: treat whole file as one chunk
        truncated = source[:MAX_CHUNK_TOKENS * 4]   # rough char limit
        chunks.append(CodeChunk(
            repo=repo, filepath=filepath,
            symbol="<module>",
            source=truncated,
            start_line=1, end_line=len(lines),
        ))
        return chunks

    for symbol_name, node in _iter_defs(tree.body):
        start = node.lineno - 1
        end   = node.end_lineno

        chunk_source = textwrap.dedent("".join(lines[start:end]))

        if len(chunk_source.strip().splitlines()) < 3:
            continue

        if len(chunk_source) > MAX_CHUNK_TOKENS * 4:
            chunk_source = chunk_source[:MAX_CHUNK_TOKENS * 4] + "\n    # ... (truncated)"

        chunks.append(CodeChunk(
            repo=repo, filepath=filepath,
            symbol=symbol_name,
            source=chunk_source,
            start_line=node.lineno,
            end_line=node.end_lineno,
        ))

    if not chunks:
        # file has no top-level defs (e.g. pure constants) — index as module
        truncated = source[:MAX_CHUNK_TOKENS * 4]
        chunks.append(CodeChunk(
            repo=repo, filepath=filepath,
            symbol="<module>",
            source=truncated,
            start_line=1, end_line=len(lines),
        ))

    return chunks


# ---------------------------------------------------------------------------
# Step 4 — embed chunks and upsert into ChromaDB
# ---------------------------------------------------------------------------

def _collection_name(repo: str, ref: Optional[str] = None) -> str:
    """ChromaDB collection names: no slashes. A pinned commit gets its own collection."""
    ref = ref or _INDEX_REF.get()
    name = repo.replace("/", "__")
    return f"{name}__at_{ref[:10]}" if ref else name


def _partial_name(repo: str, ref: Optional[str] = None) -> str:
    """Marker collection that exists only while an index build is unfinished."""
    return _collection_name(repo, ref) + "__partial"


def _get_or_create_collection(repo: str, ref: Optional[str] = None) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=_collection_name(repo, ref),
        metadata={"hnsw:space": "cosine"},
    )


def _upsert_chunks(collection: chromadb.Collection, chunks: list[CodeChunk]) -> None:
    if not chunks:
        return

    model = _get_embed_model()
    texts = [c.text_for_embedding for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    collection.upsert(
        ids        = [c.chunk_id for c in chunks],
        embeddings = embeddings,
        documents  = [c.source for c in chunks],
        metadatas  = [
            {
                "repo":       c.repo,
                "filepath":   c.filepath,
                "symbol":     c.symbol,
                "start_line": c.start_line,
                "end_line":   c.end_line,
            }
            for c in chunks
        ],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def index_repo(
    repo: str,
    token: Optional[str] = None,
    force_reindex: bool = False,
    ref: Optional[str] = None,
) -> dict:
    """
    Index a GitHub repo into ChromaDB.

    Args:
        repo:          "owner/repo" e.g. "scikit-learn/scikit-learn"
        token:         GitHub personal access token (optional, raises rate limit)
        force_reindex: if True, delete the collection first and rebuild from scratch
        ref:           commit SHA to index the repo at (default: the current default
                       branch, or the commit set by `use_index_ref`)

    Returns:
        {"repo": repo, "files_indexed": N, "chunks_indexed": M}
    """
    ref = ref or _INDEX_REF.get()
    collection = _get_or_create_collection(repo, ref)

    if force_reindex:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        client.delete_collection(_collection_name(repo, ref))
        collection = _get_or_create_collection(repo, ref)
        existing_ids: set[str] = set()
    else:
        # avoid re-embedding files we already have
        existing_meta = collection.get(include=["metadatas"])
        existing_ids  = {
            m["filepath"] for m in existing_meta.get("metadatas") or []
        }

    store = chromadb.PersistentClient(path=CHROMA_PATH)
    store.get_or_create_collection(name=_partial_name(repo, ref))   # removed when the build finishes

    print(f"[indexer] Listing Python files in {repo}" + (f" at {ref[:10]}" if ref else "") + "...")
    py_files = _list_python_files(repo, token, ref)
    print(f"[indexer] Found {len(py_files)} .py files (cap: {MAX_FILES})")

    files_indexed = 0
    chunks_indexed = 0

    for i, path in enumerate(py_files):
        if path in existing_ids:
            continue   # already indexed, skip

        source = _fetch_file(repo, path, token, ref)
        if not source:
            continue

        chunks = _chunk_file(repo, path, source)
        _upsert_chunks(collection, chunks)

        files_indexed  += 1
        chunks_indexed += len(chunks)

        if (i + 1) % 50 == 0:
            print(f"[indexer]   {i+1}/{len(py_files)} files processed...")

    try:
        store.delete_collection(_partial_name(repo, ref))
    except Exception:  # noqa: BLE001 - marker already gone
        pass
    print(f"[indexer] Done. {files_indexed} files, {chunks_indexed} chunks indexed.")
    return {
        "repo":           repo,
        "files_indexed":  files_indexed,
        "chunks_indexed": chunks_indexed,
    }

# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    repo_arg = sys.argv[1] if len(sys.argv) > 1 else "psf/requests"
    ref_arg = sys.argv[2] if len(sys.argv) > 2 else None     # optional commit SHA
    token_arg = os.getenv("GITHUB_TOKEN")
    result = index_repo(repo_arg, token=token_arg, ref=ref_arg)
    print(result)
    