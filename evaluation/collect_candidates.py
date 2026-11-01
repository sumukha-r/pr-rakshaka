"""
collect_candidates.py — search GitHub for PRs worth labelling, write a CSV.

    python -m evaluation.collect_candidates --repos scikit-learn/scikit-learn psf/requests \\
        --since 2019-01-01 --until 2022-10-31 --out evaluation/candidates.csv

"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

from evaluation.github_api import GitHub
from evaluation.snapshot import snapshot_sha_for
from src.llm.prompts import is_test_file

# What a reviewer says when each check would be right. Short phrases, matched
# case-insensitively as plain text.
PHRASES = {
    "tests": ["add a test", "add tests", "needs a test", "needs tests", "non-regression test",
              "regression test", "test for this", "unit test", "no test"],
    "smell": ["quadratic", "o(n", "vectorize", "vectorise", "unnecessary copy", "in a loop",
              "slow", "performance", "n+1", "cache this", "use a set"],
    "reuse": ["already have", "already exists", "use the existing", "can you reuse",
              "duplicate of", "duplicat", "there is a helper", "same as the"],
}
LABEL_QUERIES = {"smell": ["Performance"]}     # repo labels that signal the same thing
HINTS = ["tests", "smell", "reuse", "negative"]

COLUMNS = ["repo", "number", "url", "hint", "merged_at", "title", "author", "matched",
           "snippet", "comment_url", "path", "hunk", "snapshot_sha", "additions", "deletions",
           "changed_files", "py_source_files", "has_tests", "review_comments", "comments",
           "discussion", "base_sha", "head_sha"]


# ---------------------------------------------------------------------------
# Pure helpers (these are unit-tested)
# ---------------------------------------------------------------------------

def find_reviewer_comment(comments: list[dict], phrases: Iterable[str],
                          author: str) -> Optional[dict]:
    """First comment by someone else (not a bot) containing a phrase.

    Returns the phrase, a snippet, the link, and (for inline comments) the file,
    the diff hunk and the commit the comment was made on."""
    for c in comments:
        user = c.get("user") or {}
        if user.get("login") == author or user.get("type") == "Bot" or user.get("login", "").endswith("[bot]"):
            continue
        body = c.get("body") or ""
        low = body.lower()
        for ph in phrases:
            at = low.find(ph.lower())
            if at >= 0:
                start = max(0, at - 120)
                snippet = re.sub(r"\s+", " ", body[start:at + len(ph) + 220]).strip()
                return {"phrase": ph, "snippet": snippet, "url": c.get("html_url", ""),
                        "path": c.get("path", ""), "hunk": (c.get("diff_hunk") or "")[-600:],
                        "commit_id": c.get("original_commit_id") or "",
                        "created_at": c.get("created_at", "")}
    return None


def source_stats(files: list[dict]) -> tuple[int, bool]:
    """(# non-test .py files that changed, whether the PR also changes a test file)."""
    names = [f.get("filename", "") for f in files]
    sources = [n for n in names if n.endswith(".py") and not is_test_file(n)
               and not n.startswith(("doc/", "docs/", "examples/", "benchmarks/"))]
    return len(sources), any(is_test_file(n) for n in names)


def size_ok(pr: dict, min_add: int, max_add: int, max_files: int) -> bool:
    return (min_add <= pr.get("additions", 0) <= max_add
            and 0 < pr.get("changed_files", 0) <= max_files)


def search_queries(repo: str, hint: str, since: str, until: str, state: str = "merged") -> list[str]:
    if state == "merged":
        base = f"repo:{repo} is:pr is:merged merged:{since}..{until}"
    else:                                   # open PRs: no merge date, so use the creation date
        base = f"repo:{repo} is:pr is:open created:{since}..{until}"
    if hint == "negative":
        return [f"{base} comments:0"]
    qs = [f'{base} "{p}" in:comments' for p in PHRASES[hint]]
    qs += [f"{base} label:{lab}" for lab in LABEL_QUERIES.get(hint, [])]
    return qs


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def _row(repo: str, pr: dict, hint: str, files: list[dict], match: Optional[dict],
         snapshot_sha: str) -> dict:
    n_src, has_tests = source_stats(files)
    match = match or {}
    head = (pr.get("head") or {}).get("sha", "")
    return {
        "repo": repo, "number": pr["number"], "url": f"https://github.com/{repo}/pull/{pr['number']}",
        "hint": hint, "merged_at": (pr.get("merged_at") or "")[:10], "title": pr.get("title", ""),
        "author": (pr.get("user") or {}).get("login", ""), "matched": match.get("phrase", ""),
        "snippet": match.get("snippet", ""), "comment_url": match.get("url", ""),
        "path": match.get("path", ""), "hunk": match.get("hunk", ""),
        "snapshot_sha": snapshot_sha or head, "additions": pr.get("additions", 0), "deletions": pr.get("deletions", 0),
        "changed_files": pr.get("changed_files", 0), "py_source_files": n_src,
        "has_tests": has_tests, "review_comments": pr.get("review_comments", 0),
        "comments": pr.get("comments", 0),
        "discussion": pr.get("comments", 0) + pr.get("review_comments", 0),
        "base_sha": (pr.get("base") or {}).get("sha", ""), "head_sha": head,
    }


def collect(gh: GitHub, repos: list[str], *, since: str, until: str, hints: list[str],
            per_hint: int = 15, min_add: int = 15, max_add: int = 400, max_files: int = 30,
            max_discussion: int = 25, state: str = "merged", seed: int = 7,
            log=print) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for repo in repos:
        for hint in hints:
            seen: set[int] = set()
            found: list[dict] = []
            numbers: list[int] = []
            for q in search_queries(repo, hint, since, until, state):
                page = rng.randint(1, 5) if hint == "negative" else 1
                try:
                    items = gh.search_prs(q, page=page)
                except Exception as exc:  # noqa: BLE001 - one bad query must not stop the run
                    log(f"  search failed ({exc}): {q}")
                    continue
                for it in items:
                    if it["number"] not in seen:
                        seen.add(it["number"])
                        numbers.append(it["number"])
            rng.shuffle(numbers)
            for number in numbers:
                if len(found) >= per_hint:
                    break
                try:
                    pr = gh.pr(repo, number)
                    if state == "merged" and not pr.get("merged_at"):
                        continue
                    if not size_ok(pr, min_add, max_add, max_files):
                        continue
                    if pr.get("comments", 0) + pr.get("review_comments", 0) > max_discussion:
                        continue                       # a long argument, not a clear review comment
                    if hint == "negative" and (pr.get("review_comments", 0) or pr.get("comments", 0) > 3):
                        continue
                    files = gh.pr_files(repo, number)
                    n_src, _ = source_stats(files)
                    if n_src == 0:
                        continue
                    match, snapshot = None, ""
                    if hint != "negative":
                        author = (pr.get("user") or {}).get("login", "")
                        match = find_reviewer_comment(gh.pr_comments(repo, number), PHRASES[hint], author)
                        if match is None:
                            continue                   # no reviewer sentence to quote: not usable
                        snapshot = snapshot_sha_for(match, gh.pr_commits(repo, number))
                        if not snapshot:
                            continue                   # cannot tell which version the reviewer saw
                    found.append(_row(repo, pr, hint, files, match, snapshot))
                except Exception as exc:  # noqa: BLE001
                    log(f"  skipped {repo}#{number}: {exc}")
            log(f"{repo:35s} {hint:9s} {len(found)} candidate(s)")
            rows += found
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["repo"], r["hint"], r["number"])))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Find candidate PRs to label.")
    ap.add_argument("--repos", nargs="+", required=True, help="owner/repo ...")
    ap.add_argument("--since", default="2019-01-01")
    ap.add_argument("--until", default="2022-10-31", help="before ChatGPT by default")
    ap.add_argument("--hints", nargs="+", choices=HINTS, default=HINTS)
    ap.add_argument("--per-hint", type=int, default=15, help="max candidates per repo and hint")
    ap.add_argument("--state", choices=["merged", "open"], default="merged",
                    help="merged PRs (default) or open ones; --since/--until then apply to creation date")
    ap.add_argument("--max-discussion", type=int, default=25,
                    help="skip PRs with more than this many comments in total")
    ap.add_argument("--out", default="evaluation/candidates.csv")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    gh = GitHub()
    if not gh.token:
        print("warning: no GITHUB_TOKEN set; the search API will stop you after a few requests",
              file=sys.stderr)
    rows = collect(gh, args.repos, since=args.since, until=args.until, hints=args.hints,
                   per_hint=args.per_hint, max_discussion=args.max_discussion,
                   state=args.state, seed=args.seed)
    write_csv(rows, args.out)
    print(f"wrote {len(rows)} candidate(s) to {args.out}. Open each PR and read the comment before labelling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
