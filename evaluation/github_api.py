"""
github_api.py — a tiny GitHub REST client for the evaluation scripts.

"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

import requests

API = "https://api.github.com"


class GitHub:
    def __init__(self, token: Optional[str] = None, *,
                 get: Optional[Callable[..., dict]] = None, pause: float = 2.2):
        self.token = token or os.getenv("GITHUB_TOKEN")
        self._get = get or self._http_get
        self.pause = pause            # seconds between search calls

    def _http_get(self, path: str, params: Optional[dict] = None):
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        for attempt in range(4):
            r = requests.get(f"{API}{path}", headers=headers, params=params, timeout=30)
            limited = r.status_code in (403, 429) and (
                r.headers.get("X-RateLimit-Remaining") == "0" or "retry-after" in r.headers
                or "secondary rate limit" in r.text.lower())
            if limited:
                wait = float(r.headers.get("retry-after") or 0)
                reset = r.headers.get("X-RateLimit-Reset")
                if not wait and reset:
                    wait = max(0.0, float(reset) - time.time()) + 1
                time.sleep(min(wait or 30 * (attempt + 1), 120))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"GitHub kept rate-limiting {path}")

    def get(self, path: str, params: Optional[dict] = None):
        return self._get(path, params)

    def search_prs(self, query: str, *, per_page: int = 50, page: int = 1,
                   sort: str = "created", order: str = "desc") -> list[dict]:
        """Items of /search/issues for a query (PR search qualifiers go in `query`)."""
        time.sleep(self.pause)
        data = self.get("/search/issues", {"q": query, "per_page": per_page, "page": page,
                                           "sort": sort, "order": order})
        return list(data.get("items", []))

    def pr(self, repo: str, number: int) -> dict:
        return self.get(f"/repos/{repo}/pulls/{number}")

    def pr_files(self, repo: str, number: int) -> list[dict]:
        out, page = [], 1
        while True:
            batch = self.get(f"/repos/{repo}/pulls/{number}/files", {"per_page": 100, "page": page})
            out += batch
            if len(batch) < 100 or page >= 5:
                return out
            page += 1

    def pr_comments(self, repo: str, number: int) -> list[dict]:
        """Inline review comments and conversation comments, oldest first."""
        a = self.get(f"/repos/{repo}/pulls/{number}/comments", {"per_page": 100})
        b = self.get(f"/repos/{repo}/issues/{number}/comments", {"per_page": 100})
        return sorted(list(a) + list(b), key=lambda c: c.get("created_at", ""))
