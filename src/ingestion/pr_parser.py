"""
pr_parser.py — fetch a GitHub PR and return a structured dict.
No auth token needed for public repos (60 req/hr unauthenticated).
"""

import os
import re
import requests
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PRFile:
    filename: str
    status: str          # added | modified | deleted | renamed
    additions: int
    deletions: int
    patch: Optional[str]  # raw unified diff; None if binary file


@dataclass
class PullRequest:
    repo: str            # e.g. "scikit-learn/scikit-learn"
    number: int
    title: str
    author: str
    state: str           # open | closed
    merged: bool
    body: str
    additions: int
    deletions: int
    changed_files: int
    files: list[PRFile] = field(default_factory=list)
    head_sha: str = ""    # commit of the PR branch; lets checks read whole files


def parse_pr_url(url: str) -> tuple[str, int]:
    """
    Extract repo and PR number from a GitHub PR URL.
    Accepts:
      https://github.com/owner/repo/pull/123
      owner/repo#123
    """
    match = re.search(r"github\.com/([^/]+/[^/]+)/pull/(\d+)", url)
    if match:
        return match.group(1), int(match.group(2))
    # fallback: "owner/repo#123"
    match = re.match(r"([^#]+)#(\d+)", url)
    if match:
        return match.group(1), int(match.group(2))
    raise ValueError(f"Cannot parse PR URL: {url}")


def fetch_pr(repo: str, number: int, token: Optional[str] = None) -> PullRequest:
    """
    Fetch PR metadata + file list from GitHub REST API.
    token: optional personal access token to raise rate limit to 5000/hr.
    """
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    base = f"https://api.github.com/repos/{repo}/pulls/{number}"

    # --- metadata ---
    resp = requests.get(base, headers=headers, timeout=15)
    resp.raise_for_status()
    meta = resp.json()

    # --- files (paginated; most PRs fit in one page of 30) ---
    files = []
    page = 1
    while True:
        r = requests.get(
            f"{base}/files",
            headers=headers,
            params={"per_page": 100, "page": page},
            timeout=15,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for f in batch:
            files.append(PRFile(
                filename=f["filename"],
                status=f["status"],
                additions=f["additions"],
                deletions=f["deletions"],
                patch=f.get("patch"),   # binary files have no patch
            ))
        page += 1

    return PullRequest(
        repo=repo,
        number=meta["number"],
        title=meta["title"],
        author=meta["user"]["login"],
        state=meta["state"],
        merged=meta.get("merged", False),
        body=meta.get("body") or "",
        additions=meta["additions"],
        deletions=meta["deletions"],
        changed_files=meta["changed_files"],
        files=files,
        head_sha=(meta.get("head") or {}).get("sha", ""),
    )


def fetch_file_text(repo: str, path: str, ref: str,
                    token: Optional[str] = None) -> Optional[str]:
    """
    Source text of `path` at commit `ref`, or None when it cannot be read
    (missing, binary, rate-limited, offline). Callers must treat None as
    "no extra context", never as an error.
    """
    if not ref:
        return None
    headers = {"Accept": "application/vnd.github.raw+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(f"https://api.github.com/repos/{repo}/contents/{path}",
                         headers=headers, params={"ref": ref}, timeout=15)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        return r.content.decode("utf-8")
    except UnicodeDecodeError:
        return None


def fetch_source_at(repo: str, path: str, ref: str) -> Optional[str]:
    """fetch_file_text with the token from $GITHUB_TOKEN: the default whole-file reader."""
    return fetch_file_text(repo, path, ref, token=os.getenv("GITHUB_TOKEN"))


def fetch_pr_from_url(url: str, token: Optional[str] = None) -> PullRequest:
    """Convenience wrapper — takes a full GitHub PR URL."""
    repo, number = parse_pr_url(url)
    return fetch_pr(repo, number, token)


if __name__ == "__main__":
    # quick smoke test
    pr = fetch_pr_from_url(
        "https://github.com/scikit-learn/scikit-learn/pull/35137"
    )
    print(f"PR #{pr.number}: {pr.title}")
    print(f"Author: {pr.author} | Merged: {pr.merged}")
    print(f"Files changed ({pr.changed_files}):")
    for f in pr.files:
        print(f"  [{f.status:8s}] {f.filename}")