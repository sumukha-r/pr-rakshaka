"""M2: the agent versions of missing_tests and reuse, the runner wiring and the new score columns.
The model is a script; nothing touches the network."""
import contextlib
import json
import types

import pytest

from evaluation import run_eval, score
from src.agent import checks
from src.agent.github import GitHubReader
from src.agent.llm import Turn
from src.agent.loop import AgentFailed, AgentResult, Trace
from tests.unit.helpers import hunk, make_file, make_pr

SRC_PATCH = hunk("+def connect(host):", "+    if not host:", "+        raise ValueError('host')", "+    return host")
SRC = make_file("pkg/hooks.py", SRC_PATCH)
TEST = make_file("tests/test_hooks.py", hunk("+def test_connect():", "+    assert connect('h')"))


def result(data, *, stop="answered", steps=2, calls=1, tokens=100):
    return AgentResult(json.dumps(data) if data is not None else "not json", data, stop, steps, calls,
                       tokens // 2, tokens - tokens // 2, 0.1, "fake", "m", Trace())


class FakeRun:
    """Stands in for run_agent: returns scripted results and records what it was given."""
    def __init__(self, *results, sees=()):
        self.results, self.sees, self.calls = list(results), sees, []

    def __call__(self, system, user, tools, *, budget=None, **kw):
        self.calls.append({"system": system, "user": user, "tools": tools, "budget": budget})
        tools.seen_paths.update(self.sees)
        return self.results.pop(0)


@pytest.fixture
def reader():
    return GitHubReader(get=lambda p, q, raw: (404, None))


# --------------------------------------------------------------------------- missing tests

def test_missing_tests_keeps_only_findings_that_survive_the_filters(monkeypatch, reader):
    data = {"analysis": "no test reaches the new raise", "findings": [
        {"evidence": "if not host:", "message": "The new guard is untested."},
        {"evidence": "return something_else()", "message": "quote is not in the diff"},
        {"evidence": "host", "message": "bare name"}]}
    run = FakeRun(result(data))
    monkeypatch.setattr(checks, "run_agent", run)
    rep = {}
    out = checks.agent_missing_tests(make_pr(SRC, TEST), reader, report=rep)
    assert [f.evidence for f in out] == ["if not host:"]
    assert (rep["raw_findings"], rep["dropped"]) == (3, 2)
    assert (rep["agent_runs"], rep["agent_steps"], rep["agent_tool_calls"], rep["agent_tokens"]) == (1, 2, 1, 100)
    assert rep["agent_stops"] == "answered:1"


def test_missing_tests_agent_is_told_to_look_and_cannot_read_comments(monkeypatch, reader):
    run = FakeRun(result({"analysis": "", "findings": []}))
    monkeypatch.setattr(checks, "run_agent", run)
    checks.agent_missing_tests(make_pr(SRC, TEST), reader)
    call = run.calls[0]
    names = [s["function"]["name"] for s in call["tools"].specs()]
    assert "find_tests" in names and "read_file" in names and "pr_comments" not in names
    assert "Call find_tests" in call["system"] and "pkg/hooks.py" in call["user"]


def test_missing_tests_caps_findings_per_file(monkeypatch, reader):
    data = {"analysis": "", "findings": [{"evidence": e, "message": "m"} for e in
                                         ("def connect(host):", "if not host:", "return host")]}
    monkeypatch.setattr(checks, "run_agent", FakeRun(result(data)))
    assert len(checks.agent_missing_tests(make_pr(SRC), reader)) == 2


def test_missing_tests_failure_modes(monkeypatch, reader):
    monkeypatch.setattr(checks, "run_agent", FakeRun(result(None)))
    with pytest.raises(AgentFailed):
        checks.agent_missing_tests(make_pr(SRC), reader)
    two = make_pr(SRC, make_file("pkg/other.py", hunk("+def other(x):", "+    return x + 1", "+# c")))
    monkeypatch.setattr(checks, "run_agent", FakeRun(result(None), result({"analysis": "", "findings": []})))
    rep = {}
    assert checks.agent_missing_tests(two, reader, report=rep) == []      # one bad file does not sink the check
    assert rep["agent_failures"] == 1


def test_traces_are_written_when_a_directory_is_given(monkeypatch, reader, tmp_path):
    monkeypatch.setattr(checks, "run_agent", FakeRun(result({"analysis": "", "findings": []})))
    checks.agent_missing_tests(make_pr(SRC), reader, trace_dir=tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["demo__demo__1__missing_tests__hooks.json"]


# --------------------------------------------------------------------------- reuse

NEW_FN = hunk("+def parse_date(s):", "+    parts = s.split('-')", "+    year = int(parts[0])",
              "+    return year, int(parts[1])")
REUSE_SRC = make_file("pkg/new.py", NEW_FN, status="added")
EXISTING = {"path": "pkg/dates.py", "symbol": "def split_date", "start_line": 3, "end_line": 5,
            "snippet": "def split_date(text):\n    bits = text.split('-')\n    return int(bits[0]), int(bits[1], 10)",
            "similarity": 0.8}


def patch_search(monkeypatch, hits):
    monkeypatch.setattr(checks, "make_similar_search", lambda repo, ref=None: (lambda q, k: list(hits)))
    monkeypatch.setattr(checks, "make_index_search", lambda repo, ref=None: None)


def finding(existing="pkg/dates.py::def split_date", evidence="return year, int(parts[1])"):
    return {"analysis": "same job", "findings": [{"existing": existing, "evidence": evidence,
                                                  "message": "split_date already does this."}]}


def test_reuse_keeps_a_finding_that_cites_code_the_model_was_shown(monkeypatch, reader):
    patch_search(monkeypatch, [EXISTING])
    run = FakeRun(result(finding()))
    monkeypatch.setattr(checks, "run_agent", run)
    rep = {}
    out = checks.agent_reuse(make_pr(REUSE_SRC), reader, report=rep, fetch_source=lambda *a: None)
    assert len(out) == 1 and out[0].kind == "reuse"
    assert out[0].related == "pkg/dates.py:3-5 def split_date"
    assert "pkg/dates.py" in run.calls[0]["user"]                 # the starting point reached the prompt
    assert "find_similar" in [s["function"]["name"] for s in run.calls[0]["tools"].specs()]
    assert rep["agent_defs"] == 1 and rep["agent_tool_calls"] == 1


def test_reuse_drops_a_citation_no_tool_ever_showed(monkeypatch, reader):
    patch_search(monkeypatch, [])
    monkeypatch.setattr(checks, "run_agent", FakeRun(result(finding("pkg/made_up.py::def ghost"))))
    rep = {}
    assert checks.agent_reuse(make_pr(REUSE_SRC), reader, report=rep, fetch_source=lambda *a: None) == []
    assert rep["dropped"] == 1 and "no tool showed" in rep["_dropped"][0].related


def test_reuse_accepts_a_citation_found_through_a_tool_call(monkeypatch, reader):
    patch_search(monkeypatch, [])
    monkeypatch.setattr(checks, "run_agent", FakeRun(result(finding("pkg/dates.py::def split_date")),
                                                     sees=["pkg/dates.py"]))
    assert len(checks.agent_reuse(make_pr(REUSE_SRC), reader, fetch_source=lambda *a: None)) == 1


def test_reuse_drops_code_the_new_function_already_calls(monkeypatch, reader):
    calls_it = hunk("+def parse_date(s):", "+    parts = split_date(s)", "+    year = int(parts[0])",
                    "+    return year, int(parts[1])")
    patch_search(monkeypatch, [EXISTING])
    monkeypatch.setattr(checks, "run_agent", FakeRun(result(finding(evidence="parts = split_date(s)"))))
    rep = {}
    assert checks.agent_reuse(make_pr(make_file("pkg/new.py", calls_it, "added")), reader, report=rep,
                              fetch_source=lambda *a: None) == []
    assert "already calls" in rep["_dropped"][0].related


def test_reuse_reports_exact_copies_without_calling_the_model(monkeypatch, reader):
    twin = dict(EXISTING, snippet="def parse_data(t):\n    bits = t.split('-')\n    yr = int(bits[0])\n    return yr, int(bits[1])")
    patch_search(monkeypatch, [twin])
    run = FakeRun()
    monkeypatch.setattr(checks, "run_agent", run)
    out = checks.agent_reuse(make_pr(REUSE_SRC), reader, fetch_source=lambda *a: None)
    assert out and out[0].kind == "structural_duplicate" and run.calls == []


def test_reuse_reports_the_same_function_added_twice_in_one_pr(monkeypatch, reader):
    patch_search(monkeypatch, [])
    two = make_pr(REUSE_SRC, make_file("pkg/copy.py", hunk(
        "+def read_date(t):", "+    bits = t.split('-')", "+    yr = int(bits[0])", "+    return yr, int(bits[1])"), "added"))
    run = FakeRun(result({"analysis": "", "findings": []}))
    monkeypatch.setattr(checks, "run_agent", run)
    out = checks.agent_reuse(two, reader, fetch_source=lambda *a: None)
    assert [f.kind for f in out] == ["structural_duplicate"] and len(run.calls) == 1


def test_reuse_fails_loudly_when_the_agent_never_answers(monkeypatch, reader):
    patch_search(monkeypatch, [])
    monkeypatch.setattr(checks, "run_agent", FakeRun(result(None)))
    with pytest.raises(AgentFailed):
        checks.agent_reuse(make_pr(REUSE_SRC), reader, fetch_source=lambda *a: None)


# --------------------------------------------------------------------------- wiring

def test_runners_fall_back_to_v1_only_when_asked(monkeypatch, reader):
    import main
    monkeypatch.setattr(main, "default_runners", lambda: {
        "missing_tests": lambda pr, rep: ["v1 finding"], "reuse": lambda pr, rep: [], "smell": lambda pr, rep: []})

    def boom(*a, **k):
        raise AgentFailed("no answer")
    monkeypatch.setattr(checks, "agent_missing_tests", boom)

    rep = {}
    assert checks.agent_runners(reader=reader, fallback=True)["missing_tests"](make_pr(SRC), rep) == ["v1 finding"]
    assert "AgentFailed" in rep["fallback"]
    with pytest.raises(AgentFailed):                          # evaluation mode: no quiet substitution
        checks.agent_runners(reader=reader, fallback=False)["missing_tests"](make_pr(SRC), {})
    assert set(checks.agent_runners(reader=reader)) == {"missing_tests", "reuse", "smell"}


def test_run_one_passes_runners_only_when_given():
    seen = {}

    def review(pr, **kw):
        seen.update(kw)
        return types.SimpleNamespace(to_dict=lambda: {"checks": []})
    label = {"id": "x", "base_sha": "b", "checks": {"missing_tests": {"expect": "finding"},
                                                    "reuse": {"expect": "unclear"}, "smell": {"expect": "unclear"}}}
    kw = dict(review_fn=review, ref_ctx=lambda ref: contextlib.nullcontext(), pr_builder=lambda l: None)
    run_eval.run_one(label, "groq", **kw)
    assert "runners" not in seen
    run_eval.run_one(label, "groq", runners={"missing_tests": 1}, **kw)
    assert seen["runners"] == {"missing_tests": 1}


def test_score_reports_time_tokens_and_tool_calls_per_pr():
    labels = {"a": {"checks": {"missing_tests": {"expect": "none", "targets": []},
                               "reuse": {"expect": "unclear", "targets": []},
                               "smell": {"expect": "unclear", "targets": []}}}}
    run = {"checks": [{"name": "missing_tests", "status": "ok", "findings": [], "seconds": 12.0,
                       "metrics": {"agent_tokens": 3000, "agent_tool_calls": 4}}]}
    m = score.score_run(labels, ["a"], {"a": run}, {})["missing_tests"]
    assert (m["seconds_per_pr"], m["tokens_per_pr"], m["tool_calls_per_pr"]) == (12.0, 3000, 4.0)
    v1 = dict(run, checks=[dict(run["checks"][0], metrics={})])
    assert score.score_run(labels, ["a"], {"a": v1}, {})["missing_tests"]["tokens_per_pr"] is None
    table = score.format_table("groq-agent", score.score_run(labels, ["a"], {"a": run}, {}))
    assert "sec/PR" in table and "3000" in table


def test_agent_prompt_requires_a_test_of_the_new_behaviour_not_just_the_function():
    from src.agent.prompts import MISSING_TESTS_AGENT_SYSTEM as p
    assert "NEW behaviour" in p and "NOT count" in p and 'quote in "analysis" the test name' in p
    assert "even indirectly, covers it" not in p              # v1's lenient rule is overridden
    from src.llm.prompts import MISSING_TESTS_SYSTEM as v1
    assert "even indirectly, covers it" in v1                 # and v1 itself is untouched
    