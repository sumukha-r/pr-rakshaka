"""
missing_tests.py — Check 1: did this PR add logic that needs a test?

The LLM makes the judgement. Python only does the cheap plumbing around it:
pick which files are worth a call (token budget), build the prompt, parse the
reply, and drop any finding whose quoted line is not really in the diff.

Usage:
    python -m src.checks.missing_tests <PR URL>                    # active backend
    python -m src.checks.missing_tests <PR URL> --provider groq    # force one
    python -m src.checks.missing_tests <PR URL> --compare          # local vs Groq
"""

import logging
import os
from typing import Optional

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
                 test_files: list[tuple[str, str]]) -> tuple[list[Finding], str]:
    """Returns (findings, the model's analysis text)."""
    prompt = build_missing_tests_prompt(pr.title, f.filename, f.patch, test_files)
    log.info("missing_tests: sending %s (~%d tokens)", f.filename,
             (len(prompt) + len(MISSING_TESTS_SYSTEM)) // 4)

    reply = chat_json(MISSING_TESTS_SYSTEM, prompt)
    analysis = reply.get("analysis")
    analysis = analysis if isinstance(analysis, str) else ""

    raw = reply.get("findings", [])
    if not isinstance(raw, list):
        log.warning("missing_tests: 'findings' was not a list for %s", f.filename)
        return [], analysis

    findings = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        evidence, message = item.get("evidence"), item.get("message")
        if isinstance(evidence, str) and isinstance(message, str):
            findings.append(Finding(CHECK_NAME, f.filename, evidence, message))
    return findings, analysis


def check_missing_tests(pr: PullRequest, *, max_files: int = MAX_SOURCE_FILES,
                        report: Optional[dict] = None) -> list[Finding]:
    """
    Run Check 1 over a PR. Returns only evidence-backed findings.
    If `report` is given it is filled with {"files_sent", "raw_findings",
    "dropped"}, which is how backends are compared.
    """
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
    analyses: list[tuple[str, str]] = []
    for f in candidates:
        file_findings, analysis = _review_file(pr, f, test_files)
        findings.extend(file_findings)
        analyses.append((f.filename, analysis))

    kept, dropped = keep_evidenced(findings, patches)
    if report is not None:
        report.update(files_sent=len(candidates), raw_findings=len(findings),
                      dropped=len(dropped))
        # keys starting with "_" hold details, not table columns
        report["_analyses"] = analyses
        report["_dropped"] = dropped
    if dropped:
        log.info("missing_tests: dropped %d finding(s) with no matching diff line",
                 len(dropped))
        for d in dropped:
            log.debug("dropped: %s | evidence=%r", d.filename, d.evidence)
    return kept


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _format(findings: list[Finding]) -> str:
    if not findings:
        return "No missing-test findings."
    return "\n".join(
        f"{f.filename}\n  evidence: {f.evidence.strip()}\n  {f.message}\n"
        for f in findings
    )


if __name__ == "__main__":
    import argparse

    from src.ingestion.pr_parser import fetch_pr_from_url
    from src.llm.client import PROVIDERS, use_provider
    from src.llm.compare import print_comparison, run_with_each

    ap = argparse.ArgumentParser(prog="python -m src.checks.missing_tests")
    ap.add_argument("url", help="GitHub PR URL")
    ap.add_argument("--provider", choices=PROVIDERS, help="force a backend")
    ap.add_argument("--compare", action="store_true",
                    help="run the PR through local and Groq and compare")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    pr = fetch_pr_from_url(args.url, token=os.getenv("GITHUB_TOKEN"))
    print(f"PR #{pr.number}: {pr.title} ({len(pr.files)} files)")

    if args.compare:
        outcomes = run_with_each(lambda rep: check_missing_tests(pr, report=rep))
        print_comparison(outcomes, show=_format)
    elif args.provider:
        with use_provider(args.provider):
            print(_format(check_missing_tests(pr)))
    else:
        print(_format(check_missing_tests(pr)))