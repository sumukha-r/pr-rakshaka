"""The two time-savers: the model judge for judgements.csv and --assist for labelling.
No network and no model: every chat call is a stub."""
import pytest

from evaluation import auto_judge as aj
from evaluation import label_candidates as lc
from evaluation import score
from evaluation.snapshot import PRFile, PullRequest

PATCH = "@@ -1 +1,3 @@\n+import os\n+def foo(x):\n+    return x * 2"
TEST_PATCH = "@@ -0,0 +1,2 @@\n+def test_foo():\n+    assert foo(1) == 2"


def make_pr(with_tests=True):
    files = [PRFile("pkg/a.py", "modified", 3, 0, PATCH)]
    if with_tests:
        files.append(PRFile("tests/test_a.py", "added", 2, 0, TEST_PATCH))
    return PullRequest(repo="o/r", number=5, title="Double foo", author="", state="closed", merged=True,
                       body="", additions=5, deletions=0, changed_files=len(files), files=files,
                       head_sha="9" * 40)


def label(quote="please add a test"):
    return {"id": "o__r__5", "checks": {"missing_tests": {"expect": "finding", "targets": [],
            "source": {"url": "u", "quote": quote}}, "reuse": {"expect": "none", "targets": [], "source": {}},
            "smell": {"expect": "none", "targets": [], "source": {}}}}


def row(**over):
    r = {"key": "o__r__5|missing_tests|pkg/a.py|def foo(x):", "id": "o__r__5", "check": "missing_tests",
         "file": "pkg/a.py", "evidence": "def foo(x):", "message": "foo is untested",
         "verdict": "", "auto_verdict": "", "auto_reason": ""}
    r.update(over)
    return r


# --------------------------------------------------------------------------- auto judge

def test_the_judge_sees_the_diff_the_tests_and_the_reviewer_but_not_who_wrote_the_finding():
    p = aj.judge_prompt(label(), make_pr(), row())
    assert "def foo(x):" in p and "test_foo" in p and "please add a test" in p and "foo is untested" in p
    assert "CHANGED TESTS: none" in aj.judge_prompt(label(), make_pr(with_tests=False), row())
    assert "remote" not in p and "agent" not in p.lower()


def test_verdict_parsing_accepts_only_the_three_verdicts():
    assert aj.parse_verdict({"verdict": " Valid ", "reason": "a\nb"}) == ("valid", "a b")
    assert aj.parse_verdict({"verdict": "maybe"}) is None and aj.parse_verdict({}) is None


def test_judging_fills_only_empty_rows_and_never_touches_a_human_verdict():
    rows = [row(), row(key="k2", verdict="invalid"), row(key="k3", auto_verdict="valid")]
    calls = []
    out = aj.run_judging(rows, {"o__r__5": label()}, lambda pid: make_pr(),
                         lambda s, u: calls.append(1) or {"verdict": "trivial", "reason": "r"})
    assert out["judged"] == 1 and len(calls) == 1
    assert rows[0]["auto_verdict"] == "trivial" and rows[1]["verdict"] == "invalid" and rows[1]["auto_verdict"] == ""
    assert rows[2]["auto_verdict"] == "valid"


def test_a_broken_model_reply_or_unknown_pr_is_counted_not_fatal():
    def bad(s, u):
        raise RuntimeError("boom")
    out = aj.run_judging([row(), row(key="k2", id="unknown")], {"o__r__5": label()}, lambda pid: make_pr(), bad)
    assert out["judged"] == 0 and out["failed"] == 2


def test_calibration_compares_with_the_humans_and_writes_nothing():
    rows = [row(key="a", verdict="valid"), row(key="b", verdict="invalid"), row(key="c", verdict="trivial"),
            row(key="d")]
    answers = iter(["valid", "valid", "invalid"])
    out = aj.run_judging(rows, {"o__r__5": label()}, lambda pid: make_pr(),
                         lambda s, u: {"verdict": next(answers), "reason": ""}, calibrate=True)
    a = aj.agreement(out["pairs"])
    assert a == {"n": 3, "binary": 2, "exact": 1, "model_valid_human_not": 1, "human_valid_model_not": 0}
    assert all(r["auto_verdict"] == "" for r in rows)


def test_score_uses_the_model_verdict_only_with_auto_and_a_human_verdict_wins():
    rows = {"k1": {"verdict": "", "auto_verdict": "valid"},
            "k2": {"verdict": "invalid", "auto_verdict": "valid"},
            "k3": {"verdict": "", "auto_verdict": ""}}
    assert score.effective_verdicts(rows) == ({"k1": "", "k2": "invalid", "k3": ""}, 0)
    assert score.effective_verdicts(rows, True) == ({"k1": "valid", "k2": "invalid", "k3": ""}, 1)


def test_the_judgements_file_keeps_the_new_columns(tmp_path):
    path = tmp_path / "j.csv"
    score.write_judgements({}, [row(auto_verdict="valid", auto_reason="ok")], path)
    assert score.load_judgements(path)[row()["key"]]["auto_verdict"] == "valid"


# --------------------------------------------------------------------------- assisted labelling

def candidate(**over):
    r = {"repo": "o/r", "number": "5", "url": "https://github.com/o/r/pull/5", "hint": "tests",
         "merged_at": "2021-07-26", "title": "Fix json", "author": "a", "matched": "add a test",
         "snippet": "please add a test for this", "comment_url": "https://x/c", "path": "pkg/a.py",
         "hunk": "@@ -1 +1 @@\n+def foo(x):", "snapshot_sha": "9" * 40, "additions": "47", "deletions": "18",
         "changed_files": "7", "py_source_files": "4", "has_tests": "False", "review_comments": "4",
         "comments": "2", "discussion": "6", "base_sha": "a" * 40, "head_sha": "1" * 40}
    r.update(over)
    return r


def scripted(*answers):
    it = iter(answers)
    return lambda prompt: next(it)


GOOD = {"real_problem": True, "reason": "new branch", "file": "pkg/a.py", "lines": [2, 3],
        "silent_checks": ["smell", "bogus"]}


def test_a_proposal_is_checked_against_the_real_pr():
    pr = make_pr()
    assert lc.clean_proposal(GOOD, pr, "missing_tests") == {
        "real": True, "reason": "new branch", "file": "pkg/a.py", "lines": [2, 3], "silent": ["smell"]}
    assert lc.clean_proposal({**GOOD, "file": "tests/test_a.py"}, pr, "missing_tests") is None   # a test file
    assert lc.clean_proposal({**GOOD, "lines": [9]}, pr, "missing_tests") is None                # no such line
    assert lc.clean_proposal({**GOOD, "lines": []}, pr, "missing_tests") is None
    assert lc.clean_proposal({"real_problem": "yes"}, pr, "missing_tests") is None
    assert lc.clean_proposal({"real_problem": False, "reason": "unrelated"}, pr, "missing_tests")["real"] is False
    # the check being labelled cannot also be proposed as silent
    assert "missing_tests" not in lc.clean_proposal({**GOOD, "silent_checks": ["missing_tests"]}, pr, "missing_tests")["silent"]


def test_enter_accepts_the_proposal_and_writes_a_valid_label():
    out = []
    label_ = lc.decide(candidate(), lambda r: make_pr(), scripted(""), out.append,
                       suggest=lambda r, pr, c: lc.clean_proposal(GOOD, pr, c))
    c = label_["checks"]
    assert c["missing_tests"]["targets"] == [{"file": "pkg/a.py", "evidence_contains": "def foo(x):"},
                                             {"file": "pkg/a.py", "evidence_contains": "return x * 2"}]
    assert c["missing_tests"]["note"] == "label proposed by a model, accepted by a human"
    assert c["smell"]["expect"] == "none" and c["reuse"]["expect"] == "unclear"
    from evaluation import labels as lab
    assert lab.problems({**label_, "id": "o__r__5"}) == []
    assert any("proposal: a real problem" in line for line in out)


def test_m_falls_back_to_the_manual_questions_s_skips_and_q_stops():
    sug = lambda r, pr, c: lc.clean_proposal(GOOD, pr, c)       # noqa: E731
    manual = lc.decide(candidate(), lambda r: make_pr(), scripted("m", "f", "2", "", ""), lambda m: None, suggest=sug)
    assert manual["checks"]["missing_tests"]["targets"] == [{"file": "pkg/a.py", "evidence_contains": "def foo(x):"}]
    assert lc.decide(candidate(), lambda r: make_pr(), scripted("s"), lambda m: None, suggest=sug) is None
    with pytest.raises(EOFError):
        lc.decide(candidate(), lambda r: make_pr(), scripted("q"), lambda m: None, suggest=sug)


def test_a_not_real_proposal_skips_on_enter_and_a_useless_one_means_manual():
    no = lambda r, pr, c: lc.clean_proposal({"real_problem": False, "reason": "unrelated"}, pr, c)   # noqa: E731
    assert lc.decide(candidate(), lambda r: make_pr(), scripted(""), lambda m: None, suggest=no) is None
    none = lambda r, pr, c: None                                                                       # noqa: E731
    label_ = lc.decide(candidate(), lambda r: make_pr(), scripted("f", "2", "", ""), lambda m: None, suggest=none)
    assert label_["checks"]["missing_tests"]["expect"] == "finding"


def test_suggest_label_never_raises():
    def boom(s, u):
        raise RuntimeError("down")
    assert lc.suggest_label(candidate(), make_pr(), "missing_tests", boom) is None
    assert lc.suggest_label(candidate(), make_pr(), "missing_tests", lambda s, u: GOOD)["lines"] == [2, 3]


# --------------------------------------------------------------------------- no-questions modes

def never_ask(prompt):
    raise AssertionError("auto mode must not ask anything")


def test_auto_accepts_a_good_proposal_without_asking_and_marks_the_label():
    sug = lambda r, pr, c: lc.clean_proposal(GOOD, pr, c)       # noqa: E731
    label_ = lc.decide(candidate(), lambda r: make_pr(), never_ask, lambda m: None, suggest=sug, auto=True)
    assert label_["checks"]["missing_tests"]["note"] == lc.AUTO_NOTE
    assert label_["checks"]["missing_tests"]["expect"] == "finding"
    from evaluation import labels as lab
    assert lab.problems({**label_, "id": "o__r__5"}) == []


def test_auto_skips_not_real_or_unusable_proposals_and_needs_a_suggester():
    no = lambda r, pr, c: lc.clean_proposal({"real_problem": False, "reason": "unrelated"}, pr, c)   # noqa: E731
    assert lc.decide(candidate(), lambda r: make_pr(), never_ask, lambda m: None, suggest=no, auto=True) is None
    nothing = lambda r, pr, c: None                                                                   # noqa: E731
    assert lc.decide(candidate(), lambda r: make_pr(), never_ask, lambda m: None, suggest=nothing, auto=True) is None
    with pytest.raises(ValueError):
        lc.decide(candidate(), lambda r: make_pr(), never_ask, lambda m: None, auto=True)


def test_auto_accepts_quiet_prs_without_asking():
    quiet = candidate(hint="negative", has_tests="True")
    label_ = lc.decide(quiet, lambda r: make_pr(), never_ask, lambda m: None, suggest=lambda *a: None, auto=True)
    assert label_["checks"]["smell"]["expect"] == "none" and label_["checks"]["missing_tests"]["expect"] == "none"


def test_auto_judge_pending_drafts_verdicts_and_scoring_then_uses_them():
    existing = {"k1": row(key="k1"), "k2": row(key="k2", verdict="valid")}
    res = score.auto_judge_pending(existing, {"o__r__5": label()}, get_pr=lambda pid: make_pr(),
                                   chat=lambda s, u: {"verdict": "invalid", "reason": "covered"})
    assert res["judged"] == 1
    verdicts, from_model = score.effective_verdicts(existing, use_auto=True)
    assert verdicts == {"k1": "invalid", "k2": "valid"} and from_model == 1
