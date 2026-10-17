"""
retriever.py — given a code diff chunk (or any code snippet), retrieve
the top-k most semantically similar functions/classes from the indexed
ChromaDB collection for a given repo.

Used by the LLM checks to answer "does this already exist in the codebase?"

Usage:
    from src.rag.retriever import retrieve_similar

    results = retrieve_similar(
        repo="scikit-learn/scikit-learn",
        query_code=diff_patch,   # the raw unified diff text for a changed file
        top_k=3,
    )
    for r in results:
        print(r.similarity, r.filepath, r.symbol)
        print(r.source)
"""

import os
from dataclasses import dataclass

import chromadb
from sentence_transformers import SentenceTransformer

from src.rag.indexer import (
    CHROMA_PATH,
    EMBED_MODEL,
    _collection_name,
    _get_embed_model,
    normalize_for_embedding,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    repo: str
    filepath: str
    symbol: str           # "def calculate_similarity" or "class Bunch"
    source: str           # the existing code in the codebase
    start_line: int
    end_line: int
    similarity: float     # cosine similarity 0–1; higher = more similar

    @property
    def is_likely_duplicate(self) -> bool:
        """Cosine similarity above 0.92 = almost certainly the same thing."""
        return self.similarity >= 0.92

    @property
    def is_possible_reuse(self) -> bool:
        """0.75–0.92 = semantically very similar; worth reviewing."""
        return 0.75 <= self.similarity < 0.92

    def as_context_block(self) -> str:
        """
        Format for injection into an LLM prompt.
        Gives the LLM exactly what it needs: where it is and what it does.
        """
        label = "⚠️ LIKELY DUPLICATE" if self.is_likely_duplicate else "similar existing code"
        return (
            f"[{label}]\n"
            f"File: {self.filepath}  (lines {self.start_line}–{self.end_line})\n"
            f"Symbol: {self.symbol}\n"
            f"Similarity: {self.similarity:.2f}\n"
            f"```python\n{self.source}\n```"
        )


# ---------------------------------------------------------------------------
# Core retrieval
# ---------------------------------------------------------------------------

def retrieve_similar(
    repo: str,
    query_code: str,
    top_k: int = 3,
    min_similarity: float = 0.60,
) -> list[RetrievedChunk]:
    """
    Retrieve the top-k most similar code chunks from the indexed repo.

    Args:
        repo:            "owner/repo" — must already be indexed
        query_code:      the diff patch text (or any code snippet) to match against
        top_k:           max number of results to return
        min_similarity:  filter out results below this cosine similarity

    Returns:
        List of RetrievedChunk sorted by similarity descending.
        Empty list if the repo is not indexed or no matches above threshold.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col_name = _collection_name(repo)

    try:
        collection = client.get_collection(col_name)
    except Exception:
        # repo not indexed yet
        return []

    if collection.count() == 0:
        return []

    model = _get_embed_model()
    query_embedding = model.encode([normalize_for_embedding(query_code)], 
                                   show_progress_bar=False).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[RetrievedChunk] = []

    docs      = results.get("documents",  [[]])[0]
    metas     = results.get("metadatas",  [[]])[0]
    distances = results.get("distances",  [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        # ChromaDB cosine distance is 1 - similarity when space="cosine"
        similarity = 1.0 - dist

        if similarity < min_similarity:
            continue

        chunks.append(RetrievedChunk(
            repo       = meta["repo"],
            filepath   = meta["filepath"],
            symbol     = meta["symbol"],
            source     = doc,
            start_line = int(meta["start_line"]),
            end_line   = int(meta["end_line"]),
            similarity = round(similarity, 4),
        ))

    return sorted(chunks, key=lambda c: c.similarity, reverse=True)


def retrieve_for_pr_file(
    repo: str,
    diff_patch: str,
    filename: str,
    top_k: int = 3,
) -> list[RetrievedChunk]:
    """
    Convenience wrapper used by the check modules.
    Strips diff metadata (+/-/@@) and retrieves similar code.

    Args:
        repo:       "owner/repo"
        diff_patch: raw unified diff patch from GitHub API (the `patch` field)
        filename:   the file being changed — used to build better query context
        top_k:      max results
    """
    # strip diff markers — we want to embed the code, not the diff syntax
    clean_lines = []
    for line in diff_patch.splitlines():
        if line.startswith("@@") or line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("+") or line.startswith("-"):
            clean_lines.append(line[1:])   # strip the +/- prefix
        else:
            clean_lines.append(line)

    clean_code = "\n".join(clean_lines)

    # prepend filename for better embedding context
    query = f"# {filename}\n{clean_code}"

    return retrieve_similar(repo, query, top_k=top_k)


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # usage: python retriever.py "owner/repo" "def my_function(): ..."
    repo_arg  = sys.argv[1] if len(sys.argv) > 1 else "psf/requests"
    query_arg = sys.argv[2] if len(sys.argv) > 2 else "def get(url, params=None): pass"

    results = retrieve_similar(repo_arg, query_arg, top_k=3)

    if not results:
        print("No similar code found (repo not indexed or no matches above threshold).")
    else:
        for r in results:
            print(r.as_context_block())
            print()