"""
tools.py — the tools the review agent can call.

Rules that make the agent trustworthy and the evaluation fair:

  * Reads are pinned. `ref` can only be "head" (the PR's last commit) or "base"
    (where it branched). The model cannot name a branch or a later commit, so it
    can never read a fix that was added after the reviewer looked.
  * `pr_comments` exists only when the context says `allow_comments` (a live
    review). In an evaluation the reviewer's comment IS the answer; leaving the
    tool out is what stops the agent from reading it.
  * Tool output is data. It is length-capped, and a failure is returned to the
    model as text ("error: ...") so one bad call never crashes the run.
  * Tools only read. Nothing here writes to GitHub or runs the PR's code.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable, Optional

from src.agent.github import GitHubReader, ReaderError
from src.ingestion.pr_parser import PullRequest
from src.llm.prompts import is_test_file

MAX_OUTPUT_CHARS = 6000
MAX_FILE_LINES = 200
MAX_MATCHES = 40


@dataclass
class AgentContext:
    pr: PullRequest
    allow_comments: bool = False                       # live review only, never in evaluation
    # keyword search over the code index at the base commit: (query, k) -> hits
    index_search: Optional[Callable[[str, int], list[dict]]] = None
    # semantic search over the same index: (code or description, k) -> hits
    similar_search: Optional[Callable[[str, int], list[dict]]] = None


def _clean_path(path: str) -> str:
    p = (path or "").strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")
    if not p or ".." in PurePosixPath(p).parts:
        raise ValueError(f"not a repository path: {path!r}")
    return p


def _clip(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit("\n", 1)[0] + f"\n... (output cut at {limit} characters)"


class RepoTools:
    def __init__(self, ctx: AgentContext, reader: GitHubReader):
        self.ctx, self.reader = ctx, reader
        self.seen_paths: set[str] = set()     # every repo path a tool has shown the model
        self.seen_chunks: dict[str, dict] = {}  # "path::symbol" -> hit, for citing what it found
        self._handlers = {"read_file": self.read_file, "find_tests": self.find_tests,
                          "pr_context": self.pr_context}
        if ctx.index_search:
            self._handlers["search_code"] = self.search_code
        if ctx.similar_search:
            self._handlers["find_similar"] = self.find_similar
        if ctx.allow_comments:
            self._handlers["pr_comments"] = self.pr_comments

    # -- plumbing ----------------------------------------------------------

    def specs(self) -> list[dict]:
        """Tool definitions in the OpenAI function-calling format both backends accept."""
        def fn(name, description, props, required=()):
            return {"type": "function", "function": {
                "name": name, "description": description,
                "parameters": {"type": "object", "properties": props, "required": list(required)}}}
        specs = [
            fn("read_file",
               "Read a file from the repository. Returns numbered lines. Use `pattern` (a regular "
               "expression) to see only matching lines with 2 lines of context, which is cheaper "
               "than reading a whole file. ref 'head' is the PR's last commit, 'base' is before the PR.",
               {"path": {"type": "string"},
                "ref": {"type": "string", "enum": ["head", "base"]},
                "start_line": {"type": "integer"}, "end_line": {"type": "integer"},
                "pattern": {"type": "string"}}, ["path"]),
            fn("find_tests",
               "List test files that probably cover a source file: tests changed by this PR plus "
               "test files in the repo whose name and folder match. A name match is a lead, not "
               "proof: read the file to confirm it exercises the code.",
               {"source_path": {"type": "string"}}, ["source_path"]),
            fn("pr_context", "The PR title, description and changed files (test files marked).", {}),
        ]
        if "search_code" in self._handlers:
            specs.append(fn("search_code",
                            "Keyword search over the functions and classes in the repository as it was "
                            "before this PR (test files are not indexed).",
                            {"query": {"type": "string"}, "max_results": {"type": "integer"}}, ["query"]))
        if "find_similar" in self._handlers:
            specs.append(fn("find_similar",
                            "Find existing functions and classes whose meaning is close to a piece of "
                            "code or to a plain-language description of what it does (semantic search "
                            "over the repository as it was before this PR). Best for 'does something "
                            "like this already exist?'.",
                            {"query": {"type": "string"}, "max_results": {"type": "integer"}}, ["query"]))
        if "pr_comments" in self._handlers:
            specs.append(fn("pr_comments", "Recent comments on this PR.", {}))
        return specs

    def call(self, name: str, args: dict) -> str:
        handler = self._handlers.get(name)
        if handler is None:
            return f"error: unknown tool '{name}'. Available: {', '.join(self._handlers)}"
        try:
            return _clip(handler(**(args or {})))
        except TypeError as exc:
            return f"error: bad arguments for {name}: {exc}"
        except (ValueError, ReaderError) as exc:
            return f"error: {exc}"

    # -- tools -------------------------------------------------------------

    def _ref_sha(self, ref: str) -> str:
        if ref not in ("head", "base"):
            raise ValueError("ref must be 'head' or 'base'")
        sha = self.ctx.pr.head_sha if ref == "head" else self.ctx.pr.base_sha
        if not sha:
            raise ValueError(f"the '{ref}' commit is not known for this PR")
        return sha

    def read_file(self, path: str, ref: str = "head", start_line: int = 1,
                  end_line: Optional[int] = None, pattern: Optional[str] = None) -> str:
        path, sha = _clean_path(path), self._ref_sha(ref)
        text = self.reader.file_text(self.ctx.pr.repo, path, sha)
        if text is None:
            hint = " (a file the PR adds does not exist at 'base')" if ref == "base" else ""
            return f"{path} does not exist at {ref} ({sha[:10]}){hint}."
        self.seen_paths.add(path)
        lines = text.splitlines()
        head = f"{path} @ {ref} ({sha[:10]}), {len(lines)} lines"
        if pattern:
            if len(pattern) > 200:
                raise ValueError("pattern is too long")
            try:
                rx = re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid regular expression: {exc}") from exc
            hits = [i for i, ln in enumerate(lines) if rx.search(ln)]
            if not hits:
                return f"{head}\nno line matches /{pattern}/"
            hit_set = set(hits[:MAX_MATCHES])
            keep = sorted({j for i in hit_set for j in range(max(0, i - 2), min(len(lines), i + 3))})
            shown, prev = [], -2
            for j in keep:
                if shown and j != prev + 1:
                    shown.append("   ...")
                shown.append(f"{j + 1:5d}{'>' if j in hit_set else ' '} {lines[j]}")
                prev = j
            more = f"\n({len(hits) - MAX_MATCHES} more matches not shown)" if len(hits) > MAX_MATCHES else ""
            return f"{head}\n" + "\n".join(shown) + more
        start = max(1, int(start_line or 1))
        end = min(len(lines), int(end_line) if end_line else start + MAX_FILE_LINES - 1,
                  start + MAX_FILE_LINES - 1)
        body = "\n".join(f"{i:5d}  {lines[i - 1]}" for i in range(start, end + 1))
        more = f"\n(lines {end + 1}-{len(lines)} not shown; ask for them with start_line)" if end < len(lines) else ""
        return f"{head}, showing {start}-{end}\n{body}{more}"

    def find_tests(self, source_path: str) -> str:
        src = _clean_path(source_path)
        pr = self.ctx.pr
        stem, parent = PurePosixPath(src).stem, PurePosixPath(src).parent.parts
        changed = [f.filename for f in pr.files if is_test_file(f.filename)]
        tree = self.reader.tree(pr.repo, self._ref_sha("head"))
        scored = []
        for p in tree["paths"]:
            if not p.endswith(".py") or not is_test_file(p):
                continue
            name = PurePosixPath(p).stem
            if name in (f"test_{stem}", f"{stem}_test"):
                score = 3
            elif len(stem) >= 4 and stem != "__init__" and stem in name:
                score = 1
            else:
                continue
            score += len(set(parent[-2:]) & set(PurePosixPath(p).parent.parts))
            scored.append((-score, len(p), p))
        found = [p for _, _, p in sorted(scored)][:8]
        lines = [f"test files changed in this PR ({len(changed)}):"]
        lines += [f"  {p}" for p in changed] or ["  (none)"]
        lines.append(f"test files in the repo that match '{stem}' by name and folder ({len(found)}):")
        lines += [f"  {p}" + ("  (also changed in this PR)" if p in changed else "") for p in found] or ["  (none)"]
        if tree.get("truncated"):
            lines.append("note: the repository file list was cut off by GitHub, so this list may be incomplete.")
        lines.append("A missing name match does not prove the code is untested: tests can live in a "
                     "differently named file. Check with read_file(pattern=...) before concluding.")
        return "\n".join(lines)

    def pr_context(self) -> str:
        pr = self.ctx.pr
        rows = [f"  {'[test]' if is_test_file(f.filename) else '      '} {f.filename} "
                f"({f.status}, +{f.additions}/-{f.deletions})" for f in pr.files]
        body = (pr.body or "").strip() or "(no description)"
        return (f"title: {pr.title}\nrepo: {pr.repo}#{pr.number}\n"
                f"description:\n{_clip(body, 1500)}\nchanged files ({len(pr.files)}):\n" + "\n".join(rows))

    def search_code(self, query: str, max_results: int = 5) -> str:
        if not (query or "").strip():
            raise ValueError("query is empty")
        hits = self.ctx.index_search(query.strip(), max(1, min(int(max_results or 5), 10)))
        if not hits:
            return f"no indexed function or class contains '{query}'."
        return self._show_hits(hits, 600)

    def find_similar(self, query: str, max_results: int = 5) -> str:
        if not (query or "").strip():
            raise ValueError("query is empty")
        hits = self.ctx.similar_search(query.strip(), max(1, min(int(max_results or 5), 8)))
        if not hits:
            return "nothing in the index is close to that."
        return self._show_hits(hits, 1200)

    def _show_hits(self, hits: list[dict], snippet_chars: int) -> str:
        for h in hits:
            self.seen_paths.add(h["path"])
            self.seen_chunks[f"{h['path']}::{h['symbol']}"] = h
        return "\n\n".join(f"{h['path']}  {h['symbol']}  (lines {h['start_line']}-{h['end_line']})\n"
                           f"{_clip(h['snippet'], snippet_chars)}" for h in hits)

    def pr_comments(self) -> str:
        comments = self.reader.comments(self.ctx.pr.repo, self.ctx.pr.number)[-10:]
        if not comments:
            return "no comments."
        return "\n".join(f"{c['user']}" + (f" on {c['path']}" if c["path"] else "") + f": {_clip(c['body'], 500)}"
                         for c in comments)


def make_index_search(repo: str, ref: str) -> Optional[Callable[[str, int], list[dict]]]:
    """Keyword search over the ChromaDB index of `repo` at `ref`, or None if there is none."""
    try:
        import chromadb
        from src.rag.indexer import CHROMA_PATH, _collection_name
        collection = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(_collection_name(repo, ref))
        if collection.count() == 0:
            return None
    except Exception:  # noqa: BLE001 - no chromadb, no index, corrupt store: the tool is simply absent
        return None

    def search(query: str, k: int) -> list[dict]:
        got = collection.get(where_document={"$contains": query}, limit=k,
                             include=["documents", "metadatas"])
        return [{"path": m["filepath"], "symbol": m["symbol"], "start_line": m["start_line"],
                 "end_line": m["end_line"], "snippet": d}
                for d, m in zip(got.get("documents") or [], got.get("metadatas") or [])]
    return search


def make_similar_search(repo: str, ref: Optional[str] = None) -> Optional[Callable[[str, int], list[dict]]]:
    """Semantic search over the index (the same one check 2 uses), or None if there is none.
    With ref=None the index in force for this run is used (see use_index_ref)."""
    try:
        import chromadb
        from src.rag.indexer import CHROMA_PATH, _collection_name
        if chromadb.PersistentClient(path=CHROMA_PATH).get_collection(_collection_name(repo, ref)).count() == 0:
            return None
    except Exception:  # noqa: BLE001 - no chromadb, no index: the tool is simply absent
        return None

    def search(query: str, k: int) -> list[dict]:
        from src.rag.retriever import retrieve_similar
        return [{"path": c.filepath, "symbol": c.symbol, "start_line": c.start_line,
                 "end_line": c.end_line, "snippet": c.source, "similarity": c.similarity}
                for c in retrieve_similar(repo, query, top_k=k, min_similarity=0.5)]
    return search
