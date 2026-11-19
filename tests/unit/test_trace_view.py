"""M5 (tracing): the trace viewer turns saved traces into one HTML page."""
import json

from src.agent import trace_view as tv

EVENTS = [
    {"kind": "model", "step": 1, "tools": ["find_callers"], "prompt_tokens": 100, "completion_tokens": 20,
     "seconds": 1.5, "text": ""},
    {"kind": "tool", "step": 1, "name": "find_callers", "args": {"name": "fetch"}, "chars": 40,
     "preview": "<untrusted> a <script>alert(1)</script>"},
    {"kind": "nudge", "step": 2},
    {"kind": "budget", "reason": "max_steps"},
    {"kind": "model", "step": 3, "tools": [], "prompt_tokens": 50, "completion_tokens": 10, "seconds": 0.5,
     "text": '{"findings": []}'},
]


def test_summary_counts_turns_tools_tokens_and_budget():
    s = tv.summarise(EVENTS)
    assert s == {"turns": 2, "tool_calls": 1, "tokens": 180, "seconds": 2.0, "stopped_by_budget": True}


def test_render_escapes_tool_output_and_shows_every_kind_of_event():
    page = tv.render([("o__r__7__smell", EVENTS)])
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page
    for needle in ("o__r__7__smell", "find_callers", "final answer", "nudge", "budget", "stopped by budget"):
        assert needle in page


def test_load_traces_reads_a_folder_and_skips_files_that_are_not_traces(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"events": EVENTS}))
    (tmp_path / "b.json").write_text(json.dumps({"something": "else"}))
    names = [n for n, _ in tv.load_traces(tmp_path)]
    assert names == ["a"]


def test_cli_writes_the_page_and_reports_a_missing_path(tmp_path, capsys):
    (tmp_path / "t.json").write_text(json.dumps({"events": EVENTS}))
    out = tmp_path / "out.html"
    assert tv.main([str(tmp_path), "-o", str(out)]) == 0
    assert out.read_text().startswith("<!doctype html>")
    assert tv.main([str(tmp_path / "missing"), "-o", str(out)]) == 1
