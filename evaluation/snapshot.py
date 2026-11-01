"""
snapshot.py — a pull request as it looked when the reviewer commented.

A reviewer who asks for a test is looking at a diff without one. By the time the
PR was merged the author has usually added it, so the merged diff would be
labelled 'finding' but the tool would rightly stay silent. To avoid that, every
label records `snapshot_sha`, the commit the comment was made on, and the tool
is run on the diff up to that commit. This also removes the worry that the
repository has changed since: nothing here depends on today's code.
"""

from __future__ import annotations

from typing import Optional

from evaluation.github_api import GitHub
from src.ingestion.pr_parser import PRFile, PullRequest


def snapshot_sha_for(match: dict, commits: list[dict]) -> str:
    """The commit a reviewer comment applies to.

    Inline comments carry it. For a conversation comment, take the newest PR
    commit made at or before the moment the comment was written."""
    if match.get("commit_id"):
        return match["commit_id"]
    when = match.get("created_at", "")
    best, best_date = "", ""
    for c in commits:
        date = ((c.get("commit") or {}).get("committer") or {}).get("date", "")
        if date and date <= when and date >= best_date:
            best, best_date = c.get("sha", ""), date
    return best


def build_pr(gh: GitHub, label: dict) -> PullRequest:
    """PullRequest for a label: the PR's diff from its base up to `snapshot_sha`."""
    head = label.get("snapshot_sha") or ""
    data = gh.compare(label["repo"], label["base_sha"], head)
    files = [PRFile(filename=f["filename"], status=f.get("status", "modified"),
                    additions=f.get("additions", 0), deletions=f.get("deletions", 0),
                    patch=f.get("patch")) for f in data.get("files", [])]
    return PullRequest(repo=label["repo"], number=label["number"], title=label.get("title", ""),
                       author="", state="closed", merged=True, body="",
                       additions=sum(f.additions for f in files),
                       deletions=sum(f.deletions for f in files),
                       changed_files=len(files), files=files, head_sha=head)


def added_lines_of(pr: PullRequest, filename: str) -> list[str]:
    """The lines the PR adds to one file, in order (no '+')."""
    for f in pr.files:
        if f.filename == filename and f.patch:
            return [ln[1:] for ln in f.patch.splitlines()
                    if ln.startswith("+") and not ln.startswith("+++")]
    return []
