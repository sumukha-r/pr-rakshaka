"""
github_comment.py — post the review as ONE pull-request comment, and update it on a re-run (M7).

    python -m src.integrations.github_comment --repo owner/repo --pr 12 --file review.md

The comment starts with a hidden marker. On a re-run the tool finds its own earlier comment and
edits it, so a PR never collects a pile of near-identical reviews. Only comments written by a bot
account are touched, so a person cannot make the tool overwrite their comment by copying the marker.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import requests

MARKER = "<!-- pr-rakshaka-review -->"
MAX_PAGES = 5                    # 500 comments is more than any PR we review should have


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


def find_own_comment(repo: str, number: int, token: str, *, http=requests, api: str = "") -> Optional[int]:
    api = api or os.getenv("GITHUB_API_URL", "https://api.github.com")
    for page in range(1, MAX_PAGES + 1):
        r = http.get(f"{api}/repos/{repo}/issues/{number}/comments", headers=_headers(token),
                     params={"per_page": 100, "page": page}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        for c in batch:
            if MARKER in (c.get("body") or "") and (c.get("user") or {}).get("type") == "Bot":
                return int(c["id"])
        if len(batch) < 100:
            break
    return None


def upsert_comment(repo: str, number: int, body: str, token: str, *, http=requests, api: str = "") -> str:
    """Create or update the review comment. Returns 'created' or 'updated'."""
    api = api or os.getenv("GITHUB_API_URL", "https://api.github.com")
    text = body if body.startswith(MARKER) else f"{MARKER}\n{body}"
    existing = find_own_comment(repo, number, token, http=http, api=api)
    if existing:
        r = http.patch(f"{api}/repos/{repo}/issues/comments/{existing}", headers=_headers(token),
                       json={"body": text}, timeout=30)
        verb = "updated"
    else:
        r = http.post(f"{api}/repos/{repo}/issues/{number}/comments", headers=_headers(token),
                      json={"body": text}, timeout=30)
        verb = "created"
    r.raise_for_status()
    return verb


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Post or update the pr-rakshaka review comment.")
    ap.add_argument("--repo", required=True, help="owner/repo")
    ap.add_argument("--pr", required=True, type=int)
    ap.add_argument("--file", required=True, help="Markdown file with the review")
    args = ap.parse_args(argv)
    token = os.getenv("GITHUB_TOKEN", "")
    if not token:
        print("error: GITHUB_TOKEN is not set", file=sys.stderr)
        return 2
    try:
        body = open(args.file, encoding="utf-8").read()
        print(f"comment {upsert_comment(args.repo, args.pr, body, token)} on {args.repo}#{args.pr}")
    except (OSError, requests.RequestException) as exc:
        print(f"error: could not post the comment: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
