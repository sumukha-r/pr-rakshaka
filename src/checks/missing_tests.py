"""
missing_tests.py — Check 1: did this PR add logic that needs a test?

The LLM makes the judgement. Python only does the plumbing around it:
pick which files are worth a call (token budget), build the prompt, parse the
reply, and drop any finding whose quoted line is not really in the diff.

Usage:
    python -m src.checks.missing_tests https://github.com/owner/repo/pull/123
"""

import logging
import os

from src.ingestion.pr_parser import PRFile, PullRequest
from src.llm.aggregator import Finding, added_lines, keep_evidenced
from src.llm.client import chat_json
from src.llm.prompts import (
    MISSING_TESTS_SYSTEM,
    build_missing_tests_prompt,
    is_test_file,
)

log = logging.getLogger(__name__)

CHECK_NAME = "missing_tests"
MAX_SOURCE_FILES = 5   # one LLM call per file; protects the per-minute token limit
MAX_TEST_FILES = 3     # test diffs shown as context for each call


def _has_code_changes(f: PRFile) -> bool:
    """True if the file adds at least one line that is not blank or a comment."""
    return any(
        ln.strip() and not ln.strip().startswith("#")
        for ln in added_lines(f.patch or "")
    )


def _candidate_files(pr: PullRequest) -> list[PRFile]:
    files = [
        f for f in pr.files
        if f.filename.endswith(".py")
        and f.patch
        and f.status not in ("removed", "deleted")
        and not is_test_file(f.filename)
        and _has_code_changes(f)
    ]
    return sorted(files, key=lambda f: f.additions, reverse=True)


def _review_file(pr: PullRequest, f: PRFile,
                 test_files: list[tuple[str, str]]) -> list[Finding]:
    prompt = build_missing_tests_prompt(pr.title, f.filename, f.patch, test_files)
    log.info("missing_tests: sending %s (~%d tokens)", f.filename,
             (len(prompt) + len(MISSING_TESTS_SYSTEM)) // 4)

    reply = chat_json(MISSING_TESTS_SYSTEM, prompt)

    raw = reply.get("findings", [])
    if not isinstance(raw, list):
        log.warning("missing_tests: 'findings' was not a list for %s", f.filename)
        return []

    findings = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        evidence, message = item.get("evidence"), item.get("message")
        if isinstance(evidence, str) and isinstance(message, str):
            findings.append(Finding(CHECK_NAME, f.filename, evidence, message))
    return findings


def check_missing_tests(pr: PullRequest, *,
                        max_files: int = MAX_SOURCE_FILES) -> list[Finding]:
    """Run Check 1 over a PR. Returns only evidence-backed findings."""
    test_files = [
        (f.filename, f.patch)
        for f in pr.files
        if f.patch and is_test_file(f.filename) and f.status not in ("removed", "deleted")
    ][:MAX_TEST_FILES]

    candidates = _candidate_files(pr)
    if len(candidates) > max_files:
        log.info("missing_tests: %d candidate files, reviewing the %d largest",
                 len(candidates), max_files)
        candidates = candidates[:max_files]

    patches = {f.filename: f.patch for f in candidates}
    findings: list[Finding] = []
    for f in candidates:
        findings.extend(_review_file(pr, f, test_files))

    kept, dropped = keep_evidenced(findings, patches)
    if dropped:
        log.info("missing_tests: dropped %d finding(s) with no matching diff line",
                 len(dropped))
        for d in dropped:
            log.debug("dropped: %s | evidence=%r", d.filename, d.evidence)
    return kept


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from src.ingestion.pr_parser import fetch_pr_from_url

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.checks.missing_tests <PR URL>")

    pr = fetch_pr_from_url(sys.argv[1], token=os.getenv("GITHUB_TOKEN"))
    print(f"PR #{pr.number}: {pr.title} ({len(pr.files)} files)")

    results = check_missing_tests(pr)
    if not results:
        print("No missing-test findings.")
    for r in results:
        print(f"\n{r.filename}\n  evidence: {r.evidence.strip()}\n  {r.message}")