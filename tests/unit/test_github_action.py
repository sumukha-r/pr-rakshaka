"""M7: the opt-in GitHub Action. The comment logic is tested with a fake HTTP layer; the workflow files
are checked for the safety properties that matter (they never run the pull request's code)."""
from pathlib import Path

import pytest

from src.integrations import github_comment as gc

ROOT = Path(__file__).resolve().parents[2]


class Resp:
    def __init__(self, data=None):
        self.data = data if data is not None else {}

    def raise_for_status(self): pass
    def json(self): return self.data


class FakeHttp:
    def __init__(self, comments):
        self.comments, self.calls = comments, []

    def get(self, url, **kw):
        self.calls.append(("get", url, kw.get("params")))
        page = (kw.get("params") or {}).get("page", 1)
        return Resp(self.comments if page == 1 else [])

    def post(self, url, **kw):
        self.calls.append(("post", url, kw["json"]))
        return Resp()

    def patch(self, url, **kw):
        self.calls.append(("patch", url, kw["json"]))
        return Resp()


def bot(i, body): return {"id": i, "body": body, "user": {"type": "Bot"}}
def human(i, body): return {"id": i, "body": body, "user": {"type": "User"}}


def test_first_review_creates_a_comment_that_starts_with_the_marker():
    http = FakeHttp([human(1, "looks good")])
    assert gc.upsert_comment("o/r", 7, "## review", "tok", http=http, api="https://api") == "created"
    verb, url, payload = http.calls[-1]
    assert verb == "post" and url == "https://api/repos/o/r/issues/7/comments"
    assert payload["body"].startswith(gc.MARKER) and "## review" in payload["body"]


def test_a_rerun_edits_the_earlier_bot_comment_instead_of_adding_another():
    http = FakeHttp([human(1, "hi"), bot(42, f"{gc.MARKER}\nold review")])
    assert gc.upsert_comment("o/r", 7, "new", "tok", http=http, api="https://api") == "updated"
    verb, url, payload = http.calls[-1]
    assert verb == "patch" and url == "https://api/repos/o/r/issues/comments/42" and "new" in payload["body"]


def test_a_person_copying_the_marker_cannot_make_the_tool_edit_their_comment():
    http = FakeHttp([human(9, f"{gc.MARKER} pretend")])
    assert gc.upsert_comment("o/r", 7, "x", "tok", http=http, api="https://api") == "created"


def test_the_marker_is_not_added_twice():
    http = FakeHttp([])
    gc.upsert_comment("o/r", 1, f"{gc.MARKER}\nbody", "tok", http=http, api="https://api")
    assert http.calls[-1][2]["body"].count(gc.MARKER) == 1


def test_cli_needs_a_token(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    f = tmp_path / "r.md"
    f.write_text("x")
    assert gc.main(["--repo", "o/r", "--pr", "1", "--file", str(f)]) == 2


# --------------------------------------------------------------------------- the workflow files

yaml = pytest.importorskip("yaml")


def load(path):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_reusable_workflow_never_checks_out_the_pull_request():
    text = (ROOT / ".github/workflows/rakshaka-review.yml").read_text(encoding="utf-8")
    assert "pull_request.head" not in text and "github.head_ref" not in text
    wf = load(".github/workflows/rakshaka-review.yml")
    checkouts = [s for s in wf["jobs"]["review"]["steps"] if str(s.get("uses", "")).startswith("actions/checkout")]
    assert len(checkouts) == 1 and checkouts[0]["with"]["repository"] == "sumukha-r/pr-rakshaka"
    assert wf["permissions"] == {"contents": "read", "pull-requests": "write"}


def test_the_example_caller_is_opt_in_by_label_and_has_minimal_permissions():
    wf = load("examples/github-action-caller.yml")
    triggers = wf.get("on") or wf.get(True)                      # YAML turns the key `on` into True
    assert triggers == {"pull_request_target": {"types": ["labeled"]}}
    assert wf["jobs"]["review"]["if"] == "github.event.label.name == 'rakshaka'"
    assert wf["permissions"] == {"contents": "read", "pull-requests": "write"}


def test_action_requirements_leave_out_the_heavy_packages():
    lines = [l.strip().lower() for l in (ROOT / "requirements-action.txt").read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    names = {l.split(">")[0].split("=")[0].split("<")[0] for l in lines}
    assert {"requests", "groq", "radon"} <= names
    assert not names & {"chromadb", "sentence-transformers", "streamlit", "langgraph"}
