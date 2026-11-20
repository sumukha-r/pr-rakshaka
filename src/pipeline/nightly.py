"""
nightly.py — the steps of the nightly maintenance job, as plain Python functions (M6).

Airflow (airflow/dags/pr_rakshaka_nightly.py) only decides WHEN and IN WHAT ORDER these run, retries
them and shows the result. The work itself lives here so it can be tested without Airflow:

    watched_repos()         repos whose default-branch index is kept fresh for live PR reviews
    refresh_index(repo)     re-run the indexer for one repo (skips files it already has)
    validate_labels()       every evaluation label must be well formed; a broken one fails the job
    run_unit_tests()        the offline unit tests must still pass
    write_report(...)       one markdown file summarising the night's run

It deliberately does NOT run the evaluation: that needs a hosted model's quota, and the sealed test
sets must only ever be run once, by hand, with --final.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "evaluation" / "nightly_report.md"


def repo_of(label_id: str) -> str:
    """'pandas-dev__pandas__45928' -> 'pandas-dev/pandas'."""
    return label_id.rsplit("__", 1)[0].replace("__", "/", 1)


def watched_repos(labels: Optional[dict] = None) -> list[str]:
    """PRR_WATCHED_REPOS (comma separated) if set, else every repo that has an evaluation label."""
    env = os.getenv("PRR_WATCHED_REPOS", "").strip()
    if env:
        return [r.strip() for r in env.split(",") if r.strip()]
    if labels is None:
        from evaluation import labels as lab
        labels = lab.load_all()
    return sorted({repo_of(i) for i in labels})


def refresh_index(repo: str, *, index: Optional[Callable] = None) -> dict:
    if index is None:
        from src.rag.indexer import index_repo as index
    return index(repo, token=os.getenv("GITHUB_TOKEN"))


def validate_labels(labels: Optional[dict] = None) -> dict:
    """Counts per expectation; raises ValueError listing every broken label."""
    from evaluation import labels as lab
    labels = lab.load_all() if labels is None else labels
    bad = {i: lab.problems(l) for i, l in labels.items() if lab.problems(l)}
    if bad:
        lines = [f"{i}: {'; '.join(p)}" for i, p in sorted(bad.items())]
        raise ValueError(f"{len(bad)} label(s) are broken:\n" + "\n".join(lines))
    counts: dict[str, int] = {}
    for l in labels.values():
        for c in l["checks"].values():
            counts[c["expect"]] = counts.get(c["expect"], 0) + 1
    return {"labels": len(labels), "by_expectation": counts}


def run_unit_tests(*, run: Callable = subprocess.run) -> dict:
    proc = run([sys.executable, "-m", "pytest", "tests/unit", "-q", "-x"], cwd=ROOT,
               capture_output=True, text=True)
    last = (proc.stdout.strip().splitlines() or [""])[-1]
    if proc.returncode != 0:
        raise RuntimeError(f"unit tests failed: {last}")
    return {"summary": last}


def write_report(index_results: list[dict], label_summary: dict, tests: dict,
                 path: Path = REPORT_PATH, now: Optional[datetime] = None) -> Path:
    now = now or datetime.now(timezone.utc)
    lines = [f"# Nightly report — {now:%Y-%m-%d %H:%M} UTC", "",
             f"- Labels: {label_summary['labels']} PRs, by expectation {label_summary['by_expectation']}",
             f"- Unit tests: {tests['summary']}", "", "## Index refresh", "",
             "| repo | new files | new chunks |", "|---|---|---|"]
    for r in sorted(index_results, key=lambda r: r.get("repo", "")):
        lines.append(f"| {r.get('repo')} | {r.get('files_indexed', 0)} | {r.get('chunks_indexed', 0)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
