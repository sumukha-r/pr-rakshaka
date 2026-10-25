"""Check 3: LLM judgement of hot-path performance smells in changed Python code.

Static analysis supplies optional context only. Findings are returned only when
their evidence is present in added diff lines.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Callable, Optional

from src.checks.smell_catalog import (
    SMELL_KEYS,
    catalogue_text,
    complexity_signals,
    touched_complexity_signals,
)
from src.ingestion.diff_defs import added_definitions, added_line_numbers
from src.ingestion.pr_parser import PRFile, PullRequest, fetch_file_text, fetch_pr_from_url
from src.llm.aggregator import Finding, keep_evidenced
from src.llm.client import PROVIDERS, chat_json, use_provider
from src.llm.prompts import HOT_PATH_SYSTEM, build_hot_path_prompt, is_test_file

log = logging.getLogger(__name__)
CHECK_NAME = "smell"
MAX_FILES = 5
MAX_PATCH_CHARS = 7000
SEVERITIES = ("low", "medium", "high")

# fetch_source(repo, path, ref) -> whole file text at the PR head, or None
SourceFetcher = Callable[[str, str, str], Optional[str]]


def _default_fetch(repo: str, path: str, ref: str) -> Optional[str]:
    return fetch_file_text(repo, path, ref, token=os.getenv("GITHUB_TOKEN"))


def _has_added_code(patch: str) -> bool:
    return any(line.strip() and not line.lstrip().startswith("#")
               for line in _added_source_lines(patch))


def _added_source_lines(patch: str) -> list[str]:
    return [line[1:] for line in patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
            and line[1:].strip()]


def _candidate_files(pr: PullRequest) -> list[PRFile]:
    """Changed production Python files with meaningful additions, largest first."""
    files = [
        f for f in pr.files
        if f.filename.endswith(".py") and f.patch
        and f.status not in ("removed", "deleted")
        and not is_test_file(f.filename) and _has_added_code(f.patch)
    ]
    return sorted(files, key=lambda f: (-f.additions, f.filename))[:MAX_FILES]


def _signals(f: PRFile, full_source: Optional[str] = None) -> list[str]:
    """
    Radon signals. With the whole file (read at the PR head commit) every
    function the diff touches is scored, edited ones included. Without it, only
    complete newly-added definitions can be scored, because a diff alone holds
    fragments that do not parse.
    """
    if full_source is not None:
        return touched_complexity_signals(
            full_source, added_line_numbers(f.patch or ""), filename=f.filename)
    signals: list[str] = []
    for definition in added_definitions(f.filename, f.patch or ""):
        signals.extend(complexity_signals(
            definition.source, filename=f"{f.filename}:{definition.start_line}"
        ))
    return signals


def _finding_from(item: dict, filename: str) -> Optional[Finding]:
    evidence, message = item.get("evidence"), item.get("message")
    if not isinstance(evidence, str) or not isinstance(message, str):
        return None
    smell = item.get("smell")
    smell = smell.strip().lower() if isinstance(smell, str) else ""
    severity = item.get("severity")
    severity = severity.strip().lower() if isinstance(severity, str) else ""
    return Finding(CHECK_NAME, filename, evidence, message,
                   kind=smell if smell in SMELL_KEYS else "other",
                   severity=severity if severity in SEVERITIES else "")


def check_hot_path_smells(
    pr: PullRequest, *, max_files: int = MAX_FILES,
    report: Optional[dict] = None,
    fetch_source: Optional[SourceFetcher] = None,
) -> list[Finding]:
    """Review up to `max_files` changed production Python files.

    `report` is optional instrumentation used by the provider comparison runner.
    `fetch_source` reads a whole file at the PR head so radon can score edited
    functions; the default calls GitHub, and it is skipped when the PR carries
    no head commit. A failed fetch only costs the radon hint.
    """
    fetch_source = fetch_source or _default_fetch
    files = _candidate_files(pr)[:max(0, max_files)]
    patches = {f.filename: f.patch for f in pr.files if f.patch}
    findings: list[Finding] = []
    analyses: list[tuple[str, str]] = []
    calls = 0
    fetched = 0
    radon_signal_count = 0
    radon_details: list[tuple[str, str]] = []
    for f in files:
        full = fetch_source(pr.repo, f.filename, pr.head_sha) if pr.head_sha else None
        fetched += full is not None
        signals = _signals(f, full)
        radon_signal_count += len(signals)
        radon_details.extend((f.filename, signal) for signal in signals)
        prompt = build_hot_path_prompt(pr.title, f.filename, f.patch or "", signals, catalogue=catalogue_text())
        try:
            reply = chat_json(HOT_PATH_SYSTEM, prompt)
            calls += 1
        except Exception:
            log.exception("hot-path review failed for %s", f.filename)
            raise
        analysis = reply.get("analysis", "")
        analyses.append((f.filename, analysis if isinstance(analysis, str) else ""))
        raw = reply.get("findings", [])
        if not isinstance(raw, list):
            log.warning("smell: 'findings' was not a list for %s", f.filename)
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            finding = _finding_from(item, f.filename)
            if finding:
                findings.append(finding)

    kept, dropped = keep_evidenced(findings, patches)
    if report is not None:
        report.update(files_considered=len(files), llm_calls=calls,
                      raw_findings=len(findings), dropped=len(dropped),
                      kept_findings=len(kept), radon_signals=radon_signal_count,
                      files_fetched=fetched)
        report["_analyses"] = analyses
        report["_dropped"] = dropped
        report["_radon_details"] = radon_details
    for item in dropped:
        log.debug("smell: dropped %s | evidence=%r", item.filename, item.evidence)
    return kept


def _format(findings: list[Finding]) -> str:
    if not findings:
        return "No hot-path smell findings."
    return "\n".join(
        f"{f.filename}  [{f.kind or 'smell'}"
        f"{', ' + f.severity if f.severity else ''}]\n"
        f"  evidence: {f.evidence.strip()}\n  {f.message}\n"
        for f in findings
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.checks.hot_path_smells")
    parser.add_argument("url", help="GitHub PR URL")
    parser.add_argument("--provider", choices=PROVIDERS, help="force a backend")
    parser.add_argument("--compare", action="store_true",
                        help="run through local and Groq and compare")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    pr = fetch_pr_from_url(args.url, token=os.getenv("GITHUB_TOKEN"))
    print(f"PR #{pr.number}: {pr.title} ({len(pr.files)} files)")
    if args.compare:
        from src.llm.compare import print_comparison, run_with_each
        print_comparison(run_with_each(
            lambda rep: check_hot_path_smells(pr, report=rep)), show=_format)
    elif args.provider:
        with use_provider(args.provider):
            print(_format(check_hot_path_smells(pr)))
    else:
        print(_format(check_hot_path_smells(pr)))


if __name__ == "__main__":
    main()
