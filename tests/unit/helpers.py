"""Shared builders for unit tests: diffs, PRs and files. No network, no LLM."""

import difflib

from src.ingestion.pr_parser import PRFile, PullRequest


def make_diff(old: str, new: str) -> str:
    """Unified diff the way GitHub's `patch` field looks (no ---/+++ header)."""
    lines = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=3))
    return "\n".join(lines[2:]) + "\n"


def hunk(*lines: str, start: int = 1) -> str:
    """One hunk. Pass each line with its diff prefix: '+', ' ' or '-'."""
    n = sum(1 for ln in lines if ln[:1] in "+ ")
    return f"@@ -{start},{len(lines)} +{start},{n} @@\n" + "\n".join(lines) + "\n"


def make_file(name: str, patch: str, status: str = "modified") -> PRFile:
    rows = patch.splitlines()
    return PRFile(
        filename=name, status=status,
        additions=sum(1 for r in rows if r.startswith("+") and not r.startswith("+++")),
        deletions=sum(1 for r in rows if r.startswith("-") and not r.startswith("---")),
        patch=patch,
    )


def make_pr(*files: PRFile, title: str = "Test PR") -> PullRequest:
    return PullRequest(repo="demo/demo", number=1, title=title, author="demo",
                       state="open", merged=False, body="", additions=0, deletions=0,
                       changed_files=len(files), files=list(files))
