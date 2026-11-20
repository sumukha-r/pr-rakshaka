"""M6: the nightly job's steps. No Airflow, no network, no model."""
import ast
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.pipeline import nightly


def label(expect="finding", **over):
    base = {"id": "o__r__1", "repo": "o/r", "number": 1, "url": "u", "base_sha": "b", "snapshot_sha": "s",
            "checks": {c: {"expect": "none", "targets": [], "source": {"url": "", "quote": ""}, "note": ""}
                       for c in ("missing_tests", "reuse", "smell")}}
    base["checks"]["missing_tests"] = {"expect": expect, "targets": [{"file": "a.py", "evidence_contains": "x"}],
                                       "source": {"url": "u", "quote": "q"}, "note": ""}
    base.update(over)
    return base


def test_repo_of_turns_a_label_id_into_a_repo_name():
    assert nightly.repo_of("pandas-dev__pandas__45928") == "pandas-dev/pandas"
    assert nightly.repo_of("scikit-learn__scikit-learn__9") == "scikit-learn/scikit-learn"


def test_watched_repos_come_from_labels_unless_overridden(monkeypatch):
    monkeypatch.delenv("PRR_WATCHED_REPOS", raising=False)
    labels = {"b__y__2": {}, "a__x__1": {}, "a__x__3": {}}
    assert nightly.watched_repos(labels) == ["a/x", "b/y"]
    monkeypatch.setenv("PRR_WATCHED_REPOS", " o/one, o/two ,")
    assert nightly.watched_repos(labels) == ["o/one", "o/two"]


def test_refresh_index_passes_the_repo_and_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    seen = []
    out = nightly.refresh_index("o/r", index=lambda repo, token=None: seen.append((repo, token)) or {"repo": repo})
    assert seen == [("o/r", "tok")] and out == {"repo": "o/r"}


def test_validate_labels_counts_and_fails_loudly_on_a_broken_label():
    ok = nightly.validate_labels({"o__r__1": label(), "o__r__2": label("none", id="o__r__2")})
    assert ok["labels"] == 2 and ok["by_expectation"]["finding"] == 1
    with pytest.raises(ValueError, match="o__r__3"):
        nightly.validate_labels({"o__r__3": label(id="o__r__3", url="")})


def test_unit_test_step_raises_when_pytest_fails():
    class P:
        def __init__(self, code, out): self.returncode, self.stdout = code, out
    assert nightly.run_unit_tests(run=lambda *a, **k: P(0, "...\n312 passed in 8s"))["summary"] == "312 passed in 8s"
    with pytest.raises(RuntimeError, match="1 failed"):
        nightly.run_unit_tests(run=lambda *a, **k: P(1, "1 failed, 5 passed"))


def test_report_lists_every_repo_and_the_totals(tmp_path):
    path = nightly.write_report(
        [{"repo": "b/y", "files_indexed": 2, "chunks_indexed": 9}, {"repo": "a/x", "files_indexed": 0, "chunks_indexed": 0}],
        {"labels": 5, "by_expectation": {"finding": 2}}, {"summary": "312 passed"},
        path=tmp_path / "r.md", now=datetime(2026, 11, 21, 3, 0, tzinfo=timezone.utc))
    text = path.read_text()
    assert "2026-11-21 03:00 UTC" in text and "312 passed" in text
    assert text.index("a/x") < text.index("b/y")                       # sorted by repo
    assert "| b/y | 2 | 9 |" in text


def test_the_dag_file_is_valid_python_and_names_its_dag():
    src = (Path(__file__).resolve().parents[2] / "airflow" / "dags" / "pr_rakshaka_nightly.py").read_text()
    ast.parse(src)
    assert 'dag_id="pr_rakshaka_nightly"' in src and "refresh_index.expand" in src
