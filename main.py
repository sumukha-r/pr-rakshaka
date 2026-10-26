"""
main.py — run all three checks on a PR and write one Markdown review.

    python main.py <PR URL>
    python main.py <PR URL> --provider groq --out review.md
    python main.py <PR URL> --checks tests,smells        # skip the reuse check
    python main.py <PR URL> --json                        # machine-readable output
    python main.py <PR URL> --index                       # index the repo first if needed

Pipeline:  fetch PR -> check 1, 2, 3 (each isolated) -> evidence filter over
everything -> Markdown report.

Each check already drops findings whose quote is not in the diff. The filter is
applied once more here, on the combined list, so the "evidence only" guarantee
is enforced in one visible place and does not depend on every check remembering
to do it. Streamlit (app.py) calls `run_review()` and renders the same Review.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Callable, Optional, Union

from src.ingestion.pr_parser import PullRequest, fetch_pr_from_url
from src.llm import client
from src.llm.aggregator import Finding, keep_evidenced
from src.llm.prompts import is_test_file
from src.report.generator import CheckResult, Review, render_markdown

log = logging.getLogger("pr-rakshaka")

# name -> (heading, icon). Order here is the order in the report.
CHECK_INFO = {
    "missing_tests": ("Missing tests", "🧪"),
    "reuse": ("Possible reuse", "♻️"),
    "smell": ("Hot-path smells", "🔥"),
}
ALIASES = {"tests": "missing_tests", "missing_tests": "missing_tests", "reuse": "reuse",
           "smells": "smell", "smell": "smell", "hotpath": "smell"}

# A runner takes (pr, report_dict) and returns that check's findings.
Runner = Callable[[PullRequest, dict], list[Finding]]


def default_runners() -> dict[str, Runner]:
    """Imported lazily so `--help` and the unit tests do not load model code."""
    from src.checks.hot_path_smells import check_hot_path_smells
    from src.checks.missing_tests import check_missing_tests
    from src.checks.reuse_detector import check_reuse
    return {
        "missing_tests": lambda pr, rep: check_missing_tests(pr, report=rep),
        "reuse": lambda pr, rep: check_reuse(pr, report=rep),
        "smell": lambda pr, rep: check_hot_path_smells(pr, report=rep),
    }


# ---------------------------------------------------------------------------
# Index handling (Check 2 needs the repo in ChromaDB)
# ---------------------------------------------------------------------------

def repo_is_indexed(repo: str) -> Optional[bool]:
    """True/False, or None when the vector store cannot be opened at all."""
    try:
        import chromadb
        from src.rag.indexer import CHROMA_PATH, _collection_name
        store = chromadb.PersistentClient(path=CHROMA_PATH)
        names = [getattr(c, "name", c) for c in store.list_collections()]
        if _collection_name(repo) not in names:
            return False
        return store.get_collection(_collection_name(repo)).count() > 0
    except Exception as exc:  # noqa: BLE001 - chromadb missing, corrupt store, ...
        log.debug("cannot inspect the index: %s", exc)
        return None


def _index(repo: str, token: Optional[str]) -> None:
    from src.rag.indexer import index_repo
    index_repo(repo, token=token)


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

def parse_checks(text: Optional[str]) -> list[str]:
    """'tests,smells' -> ['missing_tests', 'smell'] in report order. None -> all."""
    if not text:
        return list(CHECK_INFO)
    wanted = set()
    for part in text.split(","):
        key = ALIASES.get(part.strip().lower())
        if not key:
            raise ValueError(f"unknown check {part.strip()!r}; choose from "
                             f"{', '.join(sorted(set(ALIASES)))}")
        wanted.add(key)
    return [c for c in CHECK_INFO if c in wanted]


def _python_sources(pr: PullRequest) -> list:
    return [f for f in pr.files if f.filename.endswith(".py") and f.patch
            and f.status not in ("removed", "deleted") and not is_test_file(f.filename)]


def run_review(pr_or_url: Union[str, PullRequest], *, checks: Optional[list[str]] = None,
               provider: Optional[str] = None, token: Optional[str] = None,
               index_if_missing: bool = False,
               runners: Optional[dict[str, Runner]] = None,
               indexed: Optional[Callable[[str], Optional[bool]]] = None,
               progress: Optional[Callable[[str], None]] = None) -> Review:
    """
    Fetch (if given a URL) and review a PR. Never raises because of one check:
    a check that fails is reported as failed and the others still run.
    `runners` and `indexed` exist so tests can run the pipeline without a model.
    `progress`, if given, is called with a short status line before each step
    (the web UI shows these while it waits).
    """
    say = progress or (lambda message: None)
    token = token or os.getenv("GITHUB_TOKEN")
    if isinstance(pr_or_url, PullRequest):
        pr = pr_or_url
    else:
        say("Fetching the pull request from GitHub...")
        pr = fetch_pr_from_url(pr_or_url, token=token)
    say(f"Fetched #{pr.number}: {len(pr.files)} file(s) changed")
    checks = checks or list(CHECK_INFO)
    runners = runners or default_runners()
    is_indexed = indexed or repo_is_indexed

    client.reset_stats()
    results: list[CheckResult] = []
    no_python = not _python_sources(pr)

    for name in checks:
        title, icon = CHECK_INFO[name]
        res = CheckResult(name=name, title=title, icon=icon)
        results.append(res)

        if no_python:
            res.status, res.note = "skipped", "this PR changes no Python source files"
            continue
        if name == "reuse":
            state = is_indexed(pr.repo)
            if state is not True and index_if_missing:
                log.info("indexing %s (this can take several minutes)...", pr.repo)
                try:
                    _index(pr.repo, token)
                    state = is_indexed(pr.repo)
                except Exception as exc:  # noqa: BLE001
                    res.status, res.note = "failed", f"indexing {pr.repo} failed: {exc}"
                    continue
            if state is not True:
                why = ("the vector store could not be opened" if state is None
                       else f"{pr.repo} is not indexed")
                res.status = "skipped"
                res.note = f"{why}. Run: python src/rag/indexer.py {pr.repo}   (or pass --index)"
                continue

        say(f"Running: {title}...")
        rep: dict = {}
        start = time.monotonic()
        try:
            if provider:
                with client.use_provider(provider):
                    res.findings = list(runners[name](pr, rep))
            else:
                res.findings = list(runners[name](pr, rep))
        except client.LLMError as exc:
            res.status, res.note = "failed", f"model error: {exc}"
        except Exception as exc:  # noqa: BLE001 - one broken check must not hide the others
            log.exception("check %s crashed", name)
            res.status, res.note = "failed", f"{type(exc).__name__}: {exc}"
        res.seconds = time.monotonic() - start
        res.metrics = {k: v for k, v in rep.items() if not k.startswith("_")}
        log.info("%s: %s, %d finding(s), %.1fs", name, res.status, len(res.findings), res.seconds)

    # One last, central evidence pass over everything that is about to be shown.
    patches = {f.filename: f.patch for f in pr.files if f.patch}
    dropped_total = 0
    for res in results:
        kept, dropped = keep_evidenced(res.findings, patches)
        res.findings, dropped_total = kept, dropped_total + len(dropped)
    if dropped_total:
        log.warning("final evidence filter dropped %d finding(s) a check let through",
                    dropped_total)

    return Review(pr=pr, results=results, provider=provider or client.PROVIDER,
                  llm_stats=client.stats(), dropped_by_final_filter=dropped_total)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python main.py",
                                 description="Review a GitHub PR with pr-rakshaka.")
    ap.add_argument("url", help="GitHub PR URL, e.g. https://github.com/owner/repo/pull/123")
    ap.add_argument("--checks", help="comma list of: tests, reuse, smells (default: all)")
    ap.add_argument("--provider", choices=client.PROVIDERS, help="force an LLM backend")
    ap.add_argument("--out", help="write the report to this file instead of stdout")
    ap.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    ap.add_argument("--index", action="store_true",
                    help="index the repo first if it is not indexed (slow on big repos)")
    ap.add_argument("-v", "--verbose", action="store_true", help="show debug logging")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="[%(name)s] %(message)s", stream=sys.stderr)
    try:
        checks = parse_checks(args.checks)
        review = run_review(args.url, checks=checks, provider=args.provider,
                            index_if_missing=args.index)
    except ValueError as exc:                    # bad --checks or unparseable URL
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - e.g. GitHub 404 / rate limit
        print(f"error: could not review the PR: {exc}", file=sys.stderr)
        return 1

    text = (json.dumps(review.to_dict(), indent=2) if args.json else render_markdown(review))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
