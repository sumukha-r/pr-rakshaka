"""compare_runs: the diagnostic that explains why two run sets differ. No model, no network."""
import json

from evaluation import compare_runs as cr
from evaluation import run_eval


def lab(expect, targets=()):
    blank = {"expect": "unclear", "targets": []}
    return {"checks": {"missing_tests": {"expect": expect, "targets": list(targets)},
                       "reuse": dict(blank), "smell": dict(blank)}}


def run(findings=None, status="ok", note=""):
    return {"checks": [{"name": "missing_tests", "status": status, "note": note,
                        "findings": findings or []}]}


def f(ev, fn="a.py"):
    return {"filename": fn, "evidence": ev}


T = {"file": "a.py", "evidence_contains": "x = 1"}
LABELS = {"p1": lab("finding", [T]), "p2": lab("finding", [T]), "p3": lab("finding", [T]),
          "p4": lab("none")}


def test_compare_lists_lost_gained_not_run_and_extra(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "run_dir", lambda split, prov, root=None: tmp_path / prov)
    monkeypatch.setattr(cr, "run_dir", lambda split, prov, root=None: tmp_path / prov)
    tdir = tmp_path / "agent" / "traces"
    tdir.mkdir(parents=True)
    (tdir / "p1__missing_tests__a.json").write_text(json.dumps(
        {"events": [{"kind": "model"}, {"kind": "tool", "name": "find_tests"}]}))
    base = {"p1": run([f("x = 1")]), "p2": run([]), "p3": run([f("x = 1")]), "p4": run([])}
    other = {"p1": run([]), "p2": run([f("x = 1")]),
             "p3": run(status="failed", note="model error: boom"), "p4": run([f("y = 2")])}
    r = cr.compare(LABELS, list(LABELS), base, other, "dev", "agent")
    assert [x["id"] for x in r["lost"]] == ["p1"] and r["lost"][0]["tools"] == ["find_tests"]
    assert [x["id"] for x in r["gained"]] == ["p2"]
    assert r["not_run"][0]["id"] == "p3" and r["not_run"][0]["note"] == "model error: boom" and r["not_run"][0]["base_ran"]
    assert r["extra"][0]["id"] == "p4"
    text = cr.format_report(r, "base", "agent")
    assert "find_tests" in text and "model error: boom" in text and "p2" in text


def test_a_lost_target_with_no_trace_says_none():
    r = cr.compare(LABELS, ["p1"], {"p1": run([f("x = 1")])}, {"p1": run([])}, "dev", "")
    assert r["lost"][0]["tools"] == [] and "NONE" in cr.format_report(r, "b", "o")
