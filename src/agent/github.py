"""
github.py — what the agent may read from GitHub, and nothing else.

Every read is pinned to a commit sha chosen by the caller, never to a branch
name: during an evaluation the agent must see the repository as it was when the
reviewer looked, not as it is today (which contains the fix).
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Optional

import requests

from src.agent.cache import MISSING, Cache

API = "https://api.github.com"


class ReaderError(RuntimeError):
    """A read failed for a reason that says nothing about the file (rate limit, network)."""


class GitHubReader:
    def __init__(self, token: Optional[str] = None, *, cache: Optional[Cache] = None,
                 get: Optional[Callable[[str, Optional[dict], bool], tuple[int, Any]]] = None,
                 attempts: int = 3, sleep: float = 1.5):
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.cache = cache or Cache()
        self._get = get or self._http
        self.attempts, self.sleep = attempts, sleep

    def _http(self, path: str, params: Optional[dict], raw: bool) -> tuple[int, Any]:
        headers = {"Accept": "application/vnd.github.raw+json" if raw
                   else "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        status = 0
        for i in range(self.attempts):
            try:
                r = requests.get(f"{API}{path}", headers=headers, params=params, timeout=30)
            except requests.RequestException:
                if i == self.attempts - 1:
                    raise
                time.sleep(self.sleep * 2 ** i)
                continue
            status = r.status_code
            if status in (200, 404):
                return status, (r.content if raw else (r.json() if status == 200 else None))
            if status in (403, 429) or status >= 500:
                time.sleep(self.sleep * 2 ** i)
                continue
            return status, None
        return status, None

    def _call(self, path: str, params: Optional[dict] = None, raw: bool = False):
        try:
            return self._get(path, params, raw)
        except requests.RequestException as exc:
            raise ReaderError(f"network error reading {path}: {exc.__class__.__name__}") from exc

    def file_text(self, repo: str, path: str, sha: str) -> Optional[str]:
        """Text of `path` at commit `sha`; None if the file is not there (or is binary)."""
        key = f"file|{repo}|{sha}|{path}"
        hit = self.cache.get(key)
        if hit is not MISSING:
            return hit
        status, body = self._call(f"/repos/{repo}/contents/{path}", {"ref": sha}, raw=True)
        if status == 404:
            text = None
        elif status == 200:
            try:
                text = body.decode("utf-8")
            except UnicodeDecodeError:
                text = None
        else:
            raise ReaderError(f"GitHub answered {status} for {path} (not cached)")
        self.cache.set(key, text)
        return text

    def tree(self, repo: str, sha: str) -> dict:
        """{'paths': [...], 'truncated': bool}: every file path in the repo at `sha`."""
        key = f"tree|{repo}|{sha}"
        hit = self.cache.get(key)
        if hit is not MISSING:
            return hit
        status, body = self._call(f"/repos/{repo}/git/trees/{sha}", {"recursive": "1"})
        if status != 200 or not isinstance(body, dict):
            raise ReaderError(f"GitHub answered {status} for the file tree at {sha[:10]}")
        value = {"paths": [t["path"] for t in body.get("tree", []) if t.get("type") == "blob"],
                 "truncated": bool(body.get("truncated"))}
        self.cache.set(key, value)
        return value

    def comments(self, repo: str, number: int) -> list[dict]:
        """Review and conversation comments, oldest first. Not cached: they change.
        For live reviews only; see AgentContext.allow_comments."""
        out = []
        for path in (f"/repos/{repo}/pulls/{number}/comments", f"/repos/{repo}/issues/{number}/comments"):
            status, body = self._call(path, {"per_page": 50})
            if status == 200 and isinstance(body, list):
                out += [{"user": (c.get("user") or {}).get("login", ""), "body": c.get("body", ""),
                         "path": c.get("path", ""), "at": c.get("created_at", "")} for c in body]
        return sorted(out, key=lambda c: c["at"])
