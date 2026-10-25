"""Orchestrator tests: fake checks, no model, no network, no vector store."""

import json

import pytest

import main
from src.llm import client
from src.llm.aggregator import Finding
from src.llm.client import LLMError
from tests.unit.helpers import hunk, make_file, make_pr

PATCH = hunk("+for i in ids:", "+    db.get(i)")
PR = make_pr(make_file("src/a.py", PATCH))


def finding(check="smell", evidence="db.get(i)", fname="src/a.py"):
    return Finding(check, fname, evidence, "msg")


def runners(**overrides):
    base = {"missing_tests": lambda pr, rep: [],
            "reuse": lambda pr, rep: [],
            "smell": lambda pr, rep: []}
    base.update(overrides)
    return base


def run(pr=PR, **kw):
    kw.setdefault("runners", runners())
    kw.setdefault("indexed", lambda repo: True)
    return main.run_review(pr, **kw)


def by_name(review):
    return {r.name: r for r in review.results}


def test_findings_from_every_check_are_collected_in_report_order():
    rv = run(runners=runners(smell=lambda pr, rep: [finding("smell")],
                             missing_tests=lambda pr, rep: [finding("missing_tests")]))
    assert [r.name for r in rv.results] == ["missing_tests", "reuse", "smell"]
    assert [f.check for f in rv.findings] == ["missing_tests", "smell"]


def test_a_crashing_check_does_not_stop_the_others():
    def boom(pr, rep):
        raise RuntimeError("kaboom")
    rv = run(runners=runners(reuse=boom, smell=lambda pr, rep: [finding()]))
    r = by_name(rv)
    assert r["reuse"].status == "failed" and "kaboom" in r["reuse"].note
    assert r["smell"].status == "ok" and len(r["smell"].findings) == 1


def test_a_model_error_is_reported_as_such():
    def llm_down(pr, rep):
        raise LLMError("Groq quota exhausted")
    r = by_name(run(runners=runners(smell=llm_down)))["smell"]
    assert r.status == "failed" and r.note == "model error: Groq quota exhausted"


def test_final_filter_drops_a_finding_a_check_let_through():
    fake = Finding("smell", "src/a.py", "imaginary_call()", "invented")
    rv = run(runners=runners(smell=lambda pr, rep: [finding(), fake]))
    assert len(by_name(rv)["smell"].findings) == 1
    assert rv.dropped_by_final_filter == 1


def test_finding_for_a_file_not_in_the_pr_is_dropped():
    rv = run(runners=runners(smell=lambda pr, rep: [finding(fname="other.py")]))
    assert rv.findings == [] and rv.dropped_by_final_filter == 1


def test_check_metrics_are_kept_but_private_keys_are_not():
    def smell(pr, rep):
        rep.update(llm_calls=2, dropped=1, _analyses=[("a.py", "long text")])
        return []
    assert by_name(run(runners=runners(smell=smell)))["smell"].metrics == {"llm_calls": 2, "dropped": 1}


def test_reuse_is_skipped_with_instructions_when_repo_not_indexed():
    called = []
    rv = run(runners=runners(reuse=lambda pr, rep: called.append(1) or []),
             indexed=lambda repo: False)
    r = by_name(rv)["reuse"]
    assert r.status == "skipped" and "not indexed" in r.note
    assert "python src/rag/indexer.py demo/demo" in r.note
    assert not called


def test_reuse_is_skipped_when_the_vector_store_cannot_be_opened():
    r = by_name(run(indexed=lambda repo: None))["reuse"]
    assert r.status == "skipped" and "could not be opened" in r.note


def test_index_flag_indexes_then_runs_reuse(monkeypatch):
    state = {"indexed": False}
    monkeypatch.setattr(main, "_index", lambda repo, token: state.update(indexed=True))
    rv = run(runners=runners(reuse=lambda pr, rep: [finding("reuse")]),
             indexed=lambda repo: state["indexed"], index_if_missing=True)
    assert state["indexed"] and by_name(rv)["reuse"].status == "ok"


def test_index_failure_is_reported_not_raised(monkeypatch):
    def fail(repo, token):
        raise RuntimeError("rate limited")
    monkeypatch.setattr(main, "_index", fail)
    r = by_name(run(indexed=lambda repo: False, index_if_missing=True))["reuse"]
    assert r.status == "failed" and "rate limited" in r.note


def test_pr_without_python_source_skips_every_check_without_calling_them():
    pr = make_pr(make_file("README.md", hunk("+text")), make_file("tests/test_a.py", hunk("+assert 1")))
    never = lambda pr, rep: pytest.fail("must not run")   # noqa: E731
    rv = run(pr, runners=runners(missing_tests=never, reuse=never, smell=never))
    assert all(r.status == "skipped" and "no Python source" in r.note for r in rv.results)


def test_only_the_requested_checks_run():
    never = lambda pr, rep: pytest.fail("must not run")   # noqa: E731
    rv = run(checks=["smell"], runners=runners(missing_tests=never, reuse=never))
    assert [r.name for r in rv.results] == ["smell"]


def test_provider_is_forced_for_the_duration_of_the_check():
    seen = []
    run(provider="groq", runners=runners(smell=lambda pr, rep: seen.append(client.active_provider()) or []))
    assert seen == ["groq"]


def test_parse_checks():
    assert main.parse_checks(None) == ["missing_tests", "reuse", "smell"]
    assert main.parse_checks("smells, tests") == ["missing_tests", "smell"]
    with pytest.raises(ValueError, match="unknown check"):
        main.parse_checks("tests,bogus")


# --- CLI ------------------------------------------------------------------

def test_cli_bad_checks_exits_2(capsys):
    assert main.main(["https://github.com/o/r/pull/1", "--checks", "bogus"]) == 2
    assert "unknown check" in capsys.readouterr().err


def test_cli_fetch_failure_exits_1(monkeypatch, capsys):
    monkeypatch.setattr(main, "run_review", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("404")))
    assert main.main(["https://github.com/o/r/pull/1"]) == 1
    assert "could not review the PR: 404" in capsys.readouterr().err


def test_cli_writes_markdown_and_json(monkeypatch, tmp_path, capsys):
    rv = run(runners=runners(smell=lambda pr, rep: [finding()]))
    monkeypatch.setattr(main, "run_review", lambda *a, **k: rv)
    out = tmp_path / "review.md"
    assert main.main(["https://github.com/o/r/pull/1", "--out", str(out)]) == 0
    assert "pr-rakshaka review" in out.read_text(encoding="utf-8")
    assert main.main(["https://github.com/o/r/pull/1", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["checks"][2]["findings"][0]["evidence"] == "db.get(i)"
