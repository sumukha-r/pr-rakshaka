"""
cache.py — a tiny key/value cache for the agent's GitHub reads.

Only commit-addressed reads go in here (file text at a sha, a tree at a sha):
those never change, so a cached copy can never be stale. That is what keeps an
agent loop, which asks for many files, inside GitHub's and Groq's free limits
across repeated runs. With no directory it is an in-memory dict for one run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

MISSING = object()


class Cache:
    def __init__(self, directory: Optional[str] = None):
        self.dir = Path(directory) if directory else None
        self._mem: dict[str, Any] = {}

    def _path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.dir / h[:2] / f"{h}.json"       # type: ignore[operator]

    def get(self, key: str) -> Any:
        if key in self._mem:
            return self._mem[key]
        if self.dir:
            try:
                value = json.loads(self._path(key).read_text(encoding="utf-8"))["v"]
            except (OSError, ValueError, KeyError):
                return MISSING
            self._mem[key] = value
            return value
        return MISSING

    def set(self, key: str, value: Any) -> None:
        self._mem[key] = value
        if self.dir:
            try:
                path = self._path(key)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"k": key, "v": value}), encoding="utf-8")
            except OSError:
                pass            # a cache that cannot write is just a slower run
