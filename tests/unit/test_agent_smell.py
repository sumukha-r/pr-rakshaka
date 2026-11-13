"""M3a: the smell agent and its find_callers tool. The model is a script; nothing touches the network."""
import json
import re

import pytest

from src.agent import checks, tools
from src.agent.github import GitHubReader
from src.agent.loop import AgentFailed, AgentResult, Trace
from src.agent.prompts import SMELL_AGENT_SYSTEM
from src.agent.tools import AgentContext, RepoTools
from src.llm.prompts import HOT_PATH_SYSTEM
from tests.unit.helpers import hunk, make_file, make_pr

CALLER = """def sync_all(rows):
    total = 0
    for row in rows:
        total += fetch_price(row.id)
    return total"""

ONCE = """def setup():
    price = fetch_price(1)
    return price"""

DEFINITION = """def fetch_price(item_id):
    return db.query(item_id)"""


def hit(path, symbol, snippet, start=10):
    return {"path": path, "symbol": symbol, "start_line": start, "end_line": start + 5, "snippet": snippet}


def make_tools(hits, pr=None):
    reader = GitHubReader(get=lambda p, q, raw: (404, None))
    ctx = AgentContext(pr or make_pr(), index_search=(lambda q, k: hits) if hits is not None else None)
    return RepoTools(ctx, reader)


# --------------------------------------------------------------------------- loop detection

@pytest.mark.parametrize("src,line,expected", [
    (CALLER, 3, True),
    (ONCE, 1, False),
    ("def f(xs):\n    return [g(x) for x in xs]", 1, True),                      # a comprehension
    ("def f(xs):\n    for x in xs:\n        pass\n    return g(1)", 3, False),  # after the loop, not in it
    ("for x in xs:\n    def inner():\n        return g(1)", 2, False),           # a function boundary stops the climb
    ("def f(xs):\n    while xs:\n        if xs[0]:\n            g(xs)", 3, True),  # nested block inside a loop
])
def test_in_loop(src, line, expected):
    assert tools._in_loop(src.splitlines(), line) is expected


# --------------------------------------------------------------------------- find_callers

def test_find_callers_marks_calls_inside_loops_and_skips_the_definition():
    t = make_tools([hit("pkg/sync.py", "def sync_all", CALLER), hit("pkg/setup.py", "def setup", ONCE),
                    hit("pkg/db.py", "def fetch_price", DEFINITION)])
    out = t.call("find_callers", {"name": "fetch_price"})
    assert "pkg/sync.py" in out and "[inside a loop]" in out and "total += fetch_price(row.id)" in out
    assert "pkg/setup.py" in out
    assert out.count("[inside a loop]") == 1                  # only the loop call is marked
    assert "return db.query" not in out                        # the definition is not a caller
    assert "pkg/sync.py" in t.seen_paths


def test_find_callers_accepts_a_dotted_or_called_name_and_rejects_junk():
    t = make_tools([hit("pkg/sync.py", "def sync_all", CALLER)])
    assert "sync.py" in t.call("find_callers", {"name": "self.client.fetch_price()"})
    assert t.call("find_callers", {"name": "a b; rm -rf"}).startswith("error: not a function name")
    assert "error" in t.call("find_callers", {})


def test_find_callers_also_shows_calls_the_pr_adds_and_works_without_an_index():
    pr = make_pr(make_file("pkg/new.py", hunk("+def run(rows):", "+    for r in rows:", "+        fetch_price(r)")))
    out = make_tools(None, pr).call("find_callers", {"name": "fetch_price"})
    assert "calls added by this PR" in out and "pkg/new.py: fetch_price(r)" in out
    assert "calls to fetch_price() before this PR" not in out


def test_find_callers_says_plainly_when_it_finds_nothing():
    out = make_tools([]).call("find_callers", {"name": "fetch_price"})
    assert "no call to fetch_price()" in out and "do not assume it runs in a loop" in out


def test_find_callers_is_offered_to_the_model():
    assert "find_callers" in [s["function"]["name"] for s in make_tools([]).specs()]


# --------------------------------------------------------------------------- the prompt

def test_smell_agent_prompt_widens_v1s_repetition_rule_and_leaves_v1_alone():
    assert "return no finding" in SMELL_AGENT_SYSTEM
    assert "When the diff does not establish repetition" not in SMELL_AGENT_SYSTEM
    assert "neither the diff nor a caller you looked at" in SMELL_AGENT_SYSTEM
    assert "find_callers" in SMELL_AGENT_SYSTEM and "never from a tool result" in SMELL_AGENT_SYSTEM
    assert "When the diff does not establish repetition" in HOT_PATH_SYSTEM


# --------------------------------------------------------------------------- the check

PATCH = hunk("+def sync(rows):", "+    for row in rows:", "+        price = session.query(Item).get(row.id)",
             "+        total += price")
SRC = make_file("pkg/sync.py", PATCH)


def result(data, stop="answered"):
    return AgentResult(json.dumps(data) if data is not None else "not json", data, stop, 3, 2, 50, 50,
                       0.1, "fake", "m", Trace())


class FakeRun:
    def __init__(self, *results):
        self.results, self.calls = list(results), []

    def __call__(self, system, user, tools, *, budget=None, **kw):
        self.calls.append({"system": system, "user": user, "tools": tools})
        return self.results.pop(0)


@pytest.fixture
def reader():
    return GitHubReader(get=lambda p, q, raw: (404, None))


def test_agent_smell_keeps_only_evidenced_findings_and_reports_usage(monkeypatch, reader):
    data = {"analysis": "query per row", "findings": [
        {"smell": "n_plus_one", "evidence": "price = session.query(Item).get(row.id)",
         "message": "One query per row inside the loop.", "severity": "high"},
        {"smell": "n_plus_one", "evidence": "not a line in the diff", "message": "invented"}]}
    run = FakeRun(result(data))
    monkeypatch.setattr(checks, "run_agent", run)
    rep = {}
    out = checks.agent_smell(make_pr(SRC), reader, report=rep)
    assert [f.evidence for f in out] == ["price = session.query(Item).get(row.id)"]
    assert out[0].check == "smell" and out[0].severity == "high"
    assert (rep["raw_findings"], rep["dropped"], rep["agent_runs"]) == (2, 1, 1)
    call = run.calls[0]
    assert call["system"] == SMELL_AGENT_SYSTEM and "pkg/sync.py" in call["user"]
    assert "find_callers" in [s["function"]["name"] for s in call["tools"].specs()]
    assert "pr_comments" not in [s["function"]["name"] for s in call["tools"].specs()]


def test_agent_smell_fails_loudly_when_no_file_got_an_answer(monkeypatch, reader):
    monkeypatch.setattr(checks, "run_agent", FakeRun(result(None)))
    with pytest.raises(AgentFailed):
        checks.agent_smell(make_pr(SRC), reader)


def test_agent_smell_with_nothing_to_review_asks_the_model_nothing(monkeypatch, reader):
    run = FakeRun()
    monkeypatch.setattr(checks, "run_agent", run)
    assert checks.agent_smell(make_pr(make_file("README.md", hunk("+hello"))), reader) == []
    assert run.calls == []


def test_changed_function_names_come_from_added_defs_and_hunk_headers():
    patch = ("@@ -1,3 +1,5 @@ def old_helper(x):\n+def fetch_price(item_id):\n+    return 1\n"
             "+    async def inner(a):\n+def __init__(self):\n+def fetch_price(again):")
    assert checks._changed_function_names(patch) == ["old_helper", "fetch_price", "inner"]
    assert checks._changed_function_names("+x = 1\n+print(x)") == []
    many = "\n".join(f"+def f{i}(): pass" for i in range(9))
    assert len(checks._changed_function_names(many)) == checks.MAX_CALLER_HINTS


def test_agent_smell_asks_for_caller_tracing_when_a_function_changes(monkeypatch, reader):
    seen = []

    def run(system, user, tools, *, budget=None, min_tool_calls=0, **kw):
        seen.append((user, min_tool_calls))
        return result({"analysis": "x", "findings": []})

    monkeypatch.setattr(checks, "run_agent", run)
    checks.agent_smell(make_pr(SRC), reader)                      # PATCH adds `def sync`
    user, minimum = seen[0]
    assert minimum == 1 and "Functions this diff adds or changes: sync" in user and "find_callers" in user
    plain = make_file("pkg/consts.py", hunk("+LIMIT = 5", "+for i in range(LIMIT):", "+    print(i)"))
    checks.agent_smell(make_pr(plain), reader)
    assert seen[1][1] == 0 and "Functions this diff adds" not in seen[1][0]


def test_runners_use_the_smell_agent_only_when_asked(monkeypatch):
    monkeypatch.setattr("main.default_runners", lambda: {"missing_tests": None, "reuse": None, "smell": "v1-smell"})
    assert checks.agent_runners(reader=object())["smell"] == "v1-smell"
    assert checks.agent_runners(reader=object(), smell_agent=True)["smell"] != "v1-smell"
    