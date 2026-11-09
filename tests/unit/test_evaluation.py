"""The evaluation tools, offline: a fake GitHub, temp folders, no model."""

import csv
import json

import pytest

from evaluation import collect_candidates as cc
from evaluation import labels as lab
from evaluation import run_eval, score, split
from evaluation.github_api import GitHub


# ---------------------------------------------------------------------------
# collect_candidates
# ---------------------------------------------------------------------------

def comment(login, body, typ="User", url="https://x/c1", **extra):
    return {"user": {"login": login, "type": typ}, "body": body, "html_url": url,
            "created_at": "2020-01-01T12:00:00Z", **extra}


def test_reviewer_comment_skips_the_author_and_bots():
    comments = [comment("author", "I will add a test later"),
                comment("ci[bot]", "please add a test", typ="Bot"),
                comment("reviewer", "Looks good, but please Add a Test for the empty case.", url="https://x/c9")]
    m = cc.find_reviewer_comment(comments, cc.PHRASES["tests"], "author")
    assert m["phrase"] == "add a test" and "empty case" in m["snippet"] and m["url"] == "https://x/c9"


def test_no_matching_reviewer_comment_gives_none():
    assert cc.find_reviewer_comment([comment("r", "LGTM")], cc.PHRASES["tests"], "a") is None
    assert cc.find_reviewer_comment([comment("a", "add a test")], cc.PHRASES["tests"], "a") is None


def test_source_stats_ignores_tests_and_docs():
    files = [{"filename": "pkg/core.py"}, {"filename": "pkg/tests/test_core.py"},
             {"filename": "doc/conf.py"}, {"filename": "README.md"}]
    assert cc.source_stats(files) == (1, True)
    assert cc.source_stats([{"filename": "docs/x.rst"}]) == (0, False)


def test_size_limits():
    pr = {"additions": 50, "changed_files": 3}
    assert cc.size_ok(pr, 15, 400, 30)
    assert not cc.size_ok({"additions": 5, "changed_files": 1}, 15, 400, 30)
    assert not cc.size_ok({"additions": 900, "changed_files": 1}, 15, 400, 30)
    assert not cc.size_ok({"additions": 50, "changed_files": 80}, 15, 400, 30)


def test_search_queries_use_the_date_window_and_labels():
    qs = cc.search_queries("o/r", "smell", "2019-01-01", "2022-10-31")
    assert all("repo:o/r is:pr is:merged merged:2019-01-01..2022-10-31" in q for q in qs)
    assert any("label:Performance" in q for q in qs)
    assert cc.search_queries("o/r", "negative", "a", "b")[0].endswith("comments:0")


class FakeGitHub:
    """Answers the few endpoints collect() uses."""
    def __init__(self):
        self.prs = {
            1: dict(number=1, title="Add thing", merged_at="2020-05-01T00:00:00Z", additions=60,
                    deletions=2, changed_files=2, review_comments=2, comments=1,
                    user={"login": "alice"}, base={"sha": "b" * 40}, head={"sha": "1" * 40}, labels=[]),
            2: dict(number=2, title="Docs only", merged_at="2020-05-02T00:00:00Z", additions=60,
                    deletions=0, changed_files=1, review_comments=1, comments=0,
                    user={"login": "alice"}, base={"sha": "c" * 40}, head={"sha": "2" * 40}, labels=[]),
            3: dict(number=3, title="Quiet change", merged_at="2020-05-03T00:00:00Z", additions=40,
                    deletions=1, changed_files=1, review_comments=0, comments=0,
                    user={"login": "bob"}, base={"sha": "d" * 40}, head={"sha": "3" * 40}, labels=[]),
            4:dict(number=4, title="Long argument", merged_at="2020-05-04T00:00:00Z", additions=60,
                    deletions=0, changed_files=1, review_comments=30, comments=40,
                    user={"login": "alice"}, base={"sha": "e" * 40}, head={"sha": "4" * 40}, labels=[]),
        }
        self.files = {1: [{"filename": "pkg/a.py"}], 2: [{"filename": "docs/a.rst"}],
                      3: [{"filename": "pkg/b.py"}, {"filename": "pkg/tests/test_b.py"}],
                      4: [{"filename": "pkg/c.py"}]}

    def get(self, path, params=None):
        if path == "/search/issues":
            return {"items": [{"number": n} for n in (1, 2, 3, 4)]}
        n = int(path.split("/")[5])
        if path.endswith("/files"):
            return self.files[n]
        if path.endswith("/commits"):
            return [{"sha": "a" * 40, "commit": {"committer": {"date": "2020-01-01T10:00:00Z"}}},
                    {"sha": "f" * 40, "commit": {"committer": {"date": "2020-01-02T10:00:00Z"}}}]
        if path.endswith("/pulls/%d/comments" % n):
            return ([comment("carol", "Please add a test for this.", path="pkg/a.py",
                             diff_hunk="@@ -1 +1 @@\n+def foo(x):", original_commit_id="9" * 40)]
                    if n in (1, 4) else [])
        if path.endswith("/issues/%d/comments" % n):
            return []
        return self.prs[n]


def test_collect_keeps_only_prs_with_a_reviewer_quote_and_python_source():
    gh = GitHub(get=FakeGitHub().get, pause=0)
    rows = cc.collect(gh, ["o/r"], since="2019-01-01", until="2022-10-31", hints=["tests"],
                      log=lambda m: None)
    assert [r["number"] for r in rows] == [1]               # 4 is skipped: a 70-comment argument
    r = rows[0]
    assert r["matched"] == "add a test" and r["base_sha"] == "b" * 40
    assert r["snapshot_sha"] == "9" * 40 and r["path"] == "pkg/a.py" and "def foo" in r["hunk"]
    assert r["head_sha"] == "1" * 40 and r["discussion"] == 3
    assert r["url"] == "https://github.com/o/r/pull/1" and r["py_source_files"] == 1


def test_collect_negatives_need_no_review_comments_and_some_source():
    gh = GitHub(get=FakeGitHub().get, pause=0)
    rows = cc.collect(gh, ["o/r"], since="a", until="b", hints=["negative"], log=lambda m: None)
    assert [r["number"] for r in rows] == [3] and rows[0]["has_tests"] is True
    assert rows[0]["snapshot_sha"] == "3" * 40               # no comment: the PR's own last commit


def test_csv_round_trip(tmp_path):
    gh = GitHub(get=FakeGitHub().get, pause=0)
    rows = cc.collect(gh, ["o/r"], since="a", until="b", hints=["tests", "negative"], log=lambda m: None)
    out = tmp_path / "c.csv"
    cc.write_csv(rows, str(out))
    back = list(csv.DictReader(open(out, encoding="utf-8")))
    assert [(r["hint"], r["number"]) for r in back] == [("negative", "3"), ("tests", "1")]
    assert set(back[0]) == set(cc.COLUMNS)


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------

def make_label(number, repo="o/r", **expects):
    meta = {"number": number, "title": f"PR {number}", "merged_at": "2020-01-01T00:00:00Z",
            "base": {"sha": "a" * 40}, "head": {"sha": "7" * 40}}
    label = lab.stub(meta, repo)
    for check, exp in expects.items():
        c = label["checks"][check]
        c["expect"] = exp
        if exp == "finding":
            c["targets"] = [{"file": "pkg/a.py", "evidence_contains": "def foo"}]
            c["source"] = {"url": "https://x", "quote": "please add a test"}
    return label


def test_stub_starts_unclear_and_valid_otherwise():
    label = make_label(5)
    assert label["id"] == "o__r__5" and label["base_sha"] == "a" * 40
    assert label["snapshot_sha"] == "7" * 40
    assert lab.stub({"number": 1, "base": {}, "head": {"sha": "7" * 40}}, "o/r", "5" * 40)["snapshot_sha"] == "5" * 40
    assert all(c["expect"] == "unclear" for c in label["checks"].values())
    assert lab.problems(label) == []


def test_a_finding_needs_a_target_and_a_reviewer_quote():
    label = make_label(5)
    label["checks"]["smell"]["expect"] = "finding"
    msgs = lab.problems(label)
    assert any("target" in m for m in msgs) and any("quote" in m for m in msgs)
    assert any("one of" in m for m in lab.problems({**label, "checks": {**label["checks"],
               "reuse": {"expect": "maybe"}}}))


# ---------------------------------------------------------------------------
# split + run_eval
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path, monkeypatch):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    for n in range(1, 13):
        repo = "o/r" if n <= 8 else "o/s"
        label = make_label(n, repo, missing_tests="finding" if n % 2 else "none")
        (labels_dir / f"{label['id']}.json").write_text(json.dumps(label))
    monkeypatch.setattr(lab, "LABEL_DIR", labels_dir)
    monkeypatch.setattr(lab, "LABEL_CSV", tmp_path / "no_such_eval_set.csv")
    for mod in (split, run_eval, score):
        monkeypatch.setattr(mod, "SPLIT_FILE", tmp_path / "split.json")
    monkeypatch.setattr(run_eval, "RUN_DIR", tmp_path / "runs")
    monkeypatch.setattr(score, "JUDGEMENTS", tmp_path / "judgements.csv")
    return tmp_path


def test_split_is_repeatable_balanced_and_covers_everything(workspace):
    labels = lab.load_all()
    a, b = split.make_split(labels, 3), split.make_split(labels, 3)
    assert a == b
    assert sorted(a["dev"] + a["test"]) == sorted(labels)
    assert not set(a["dev"]) & set(a["test"])
    assert abs(len(a["dev"]) - len(a["test"])) <= 2
    for repo in ("o__r", "o__s"):                        # both sides see both repos
        assert any(i.startswith(repo) for i in a["dev"]) and any(i.startswith(repo) for i in a["test"])


def test_split_cli_writes_once_and_refuses_to_reroll(workspace, capsys):
    assert split.main(["--seed", "3"]) == 0
    assert split.main(["--seed", "4"]) == 1
    assert "unseen" in capsys.readouterr().err
    data = json.loads((workspace / "split.json").read_text())
    assert data["test_fingerprint"] == split.fingerprint(data["test"])


def test_split_refuses_incomplete_labels(workspace, capsys):
    bad = lab.load_all()["o__r__1"]
    bad["checks"]["smell"]["expect"] = "finding"
    (lab.LABEL_DIR / "o__r__1.json").write_text(json.dumps(bad))
    assert split.main(["--seed", "3"]) == 1
    assert "fix these labels" in capsys.readouterr().err


def fake_run(label, provider, **kw):
    return {"checks": [{"name": c, "status": "ok", "findings": []} for c in lab.scored_checks(label)]}


def test_run_one_pins_the_index_only_when_reuse_is_run():
    seen = []

    class Ctx:
        def __init__(self, ref): self.ref = ref
        def __enter__(self): seen.append(self.ref)
        def __exit__(self, *a): return False

    class R:
        def to_dict(self): return {"checks": []}

    calls = []
    def review(url, **kw):
        calls.append(kw)
        return R()

    with_reuse = make_label(1, reuse="finding", missing_tests="none")
    without = make_label(2, missing_tests="none")
    built = []
    builder = lambda label: built.append(label["snapshot_sha"]) or "PR-OBJECT"
    run_eval.run_one(with_reuse, "groq", review_fn=review, ref_ctx=Ctx, pr_builder=builder)
    run_eval.run_one(without, "groq", review_fn=review, ref_ctx=Ctx, pr_builder=builder)
    assert built == ["7" * 40, "7" * 40]                                # reviewed as of the snapshot
    assert seen == ["a" * 40, None]
    assert sorted(calls[0]["checks"]) == ["missing_tests", "reuse"] and calls[1]["checks"] == ["missing_tests"]
    assert calls[0]["provider"] == "groq" and calls[0]["index_if_missing"] is True


def test_run_one_records_errors_instead_of_raising():
    def boom(url, **kw): raise RuntimeError("rate limited")
    out = run_eval.run_one(make_label(1, missing_tests="none"), "groq", review_fn=boom,
                           ref_ctx=lambda ref: __import__("contextlib").nullcontext(),
                           pr_builder=lambda label: "PR")
    assert "rate limited" in out["error"]


def test_the_test_split_is_protected(workspace, monkeypatch, capsys):
    split.main(["--seed", "3"])
    monkeypatch.setattr(run_eval, "run_one", fake_run)
    base = ["--split", "test", "--provider", "groq"]
    assert run_eval.main(base) == 1                                   # no --final
    assert "--final" in capsys.readouterr().err
    assert run_eval.main(base + ["--final"]) == 0                     # first run is fine
    assert run_eval.main(base + ["--final"]) == 1                     # a second one is not
    assert "once per provider" in capsys.readouterr().err
    assert run_eval.main(["--split", "test", "--provider", "local", "--final"]) == 0  # other backend ok


def test_editing_a_test_label_after_the_split_blocks_the_test_run(workspace, monkeypatch, capsys):
    split.main(["--seed", "3"])
    monkeypatch.setattr(run_eval, "run_one", fake_run)
    test_id = json.loads((workspace / "split.json").read_text())["test"][0]
    path = lab.LABEL_DIR / f"{test_id}.json"
    data = json.loads(path.read_text())
    data["edge_case"] = "edited later"
    path.write_text(json.dumps(data))
    assert run_eval.main(["--split", "test", "--provider", "groq", "--final"]) == 1
    assert "no longer an unseen set" in capsys.readouterr().err


def test_dev_runs_resume_and_skip_finished_prs(workspace, monkeypatch):
    split.main(["--seed", "3"])
    calls = []
    monkeypatch.setattr(run_eval, "run_one", lambda l, p, **kw: calls.append(l["id"]) or fake_run(l, p))
    run_eval.main(["--split", "dev", "--provider", "groq"])
    n = len(calls)
    run_eval.main(["--split", "dev", "--provider", "groq"])
    assert len(calls) == n and n == len(json.loads((workspace / "split.json").read_text())["dev"])


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def finding(file="pkg/a.py", evidence="def foo(x):", message="m"):
    return {"filename": file, "evidence": evidence, "message": message}


def run_of(check, findings, status="ok"):
    return {"checks": [{"name": check, "status": status, "findings": findings}]}


def test_matching_needs_the_same_file_and_the_quoted_text():
    t = {"file": "pkg/a.py", "evidence_contains": "def  FOO"}
    assert score.matches(finding(), t)                                  # whitespace and case ignored
    assert not score.matches(finding(file="pkg/b.py"), t)
    assert not score.matches(finding(evidence="x = 1"), t)
    assert score.matches(finding(evidence="anything"), {"file": "pkg/a.py"})


def labelled():
    return {"o__r__1": make_label(1, missing_tests="finding"),
            "o__r__2": make_label(2, missing_tests="none"),
            "o__r__3": make_label(3, missing_tests="none"),
            "o__r__4": make_label(4, missing_tests="unclear")}


def test_scores_recall_precision_and_silence():
    ids = list(labelled())
    runs = {"o__r__1": run_of("missing_tests", [finding(), finding(evidence="def bar():")]),
            "o__r__2": run_of("missing_tests", [finding(evidence="x = 1")]),
            "o__r__3": run_of("missing_tests", []),
            "o__r__4": run_of("missing_tests", [finding()])}           # unclear: ignored
    judged = {score.finding_key("o__r__1", "missing_tests", finding(evidence="def bar():")): "valid",
              score.finding_key("o__r__2", "missing_tests", finding(evidence="x = 1")): "trivial"}
    m = score.score_run(labelled(), ids, runs, judged)["missing_tests"]
    assert (m["prs"], m["targets"], m["hit"], m["recall"]) == (3, 1, 1, 1.0)
    assert (m["valid"], m["wrong"], m["pending"], m["precision"]) == (2, 1, 0, 0.667)
    assert (m["none_prs"], m["silent"], m["provisional"]) == (2, 1, False)
    assert m["findings_per_pr"] == 1.0


def test_unjudged_findings_make_precision_provisional():
    runs = {"o__r__2": run_of("missing_tests", [finding(evidence="x = 1")])}
    m = score.score_run(labelled(), ["o__r__2"], runs, {})["missing_tests"]
    assert m["pending"] == 1 and m["provisional"] and m["precision"] is None
    assert "provisional" in score.format_table("groq", {"missing_tests": m, "reuse": m, "smell": m})


def test_failed_or_missing_checks_are_counted_not_scored():
    runs = {"o__r__1": run_of("missing_tests", [], status="failed"),
            "o__r__2": {"error": "boom", "checks": []}}
    m = score.score_run(labelled(), ["o__r__1", "o__r__2", "o__r__3"], runs, {})["missing_tests"]
    assert m["not_run"] == 3 and m["prs"] == 0 and m["recall"] is None


def test_the_judging_file_is_blind_and_never_asks_twice(tmp_path):
    labels = labelled()
    groq = {"o__r__2": run_of("missing_tests", [finding(evidence="x = 1", message="from groq")])}
    local = {"o__r__2": run_of("missing_tests", [finding(evidence="x = 1", message="from local")])}
    rows = score.unjudged_rows(labels, ["o__r__2"], groq, {})
    again = score.unjudged_rows(labels, ["o__r__2"], local, {r["key"]: r for r in rows})
    assert len(rows) == 1 and again == []                              # same finding from both models: one row
    path = tmp_path / "j.csv"
    score.write_judgements({}, rows, path)
    text = path.read_text()
    assert "groq" not in text.splitlines()[0] and "provider" not in text
    loaded = score.load_judgements(path)
    assert loaded[rows[0]["key"]]["verdict"] == ""


def test_matched_findings_are_never_sent_for_judging():
    runs = {"o__r__1": run_of("missing_tests", [finding()])}
    assert score.unjudged_rows(labelled(), ["o__r__1"], runs, {}) == []


# ---------------------------------------------------------------------------
# the index at a pinned commit
# ---------------------------------------------------------------------------

def test_index_names_differ_by_commit_and_follow_the_context(monkeypatch):
    import sys
    import types
    for name in ("chromadb", "sentence_transformers"):          # only the naming is tested
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.SentenceTransformer = object
            stub.Collection = object
            monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.delitem(sys.modules, "src.rag.indexer", raising=False)
    from src.rag import indexer
    try:
        assert indexer._collection_name("o/r") == "o__r"
        assert indexer._collection_name("o/r", "abcdef1234567890") == "o__r__at_abcdef1234"
        with indexer.use_index_ref("1234567890abcdef"):
            assert indexer._collection_name("o/r") == "o__r__at_1234567890"
        assert indexer._collection_name("o/r") == "o__r"                # restored afterwards
        assert len(indexer._collection_name("scikit-learn/scikit-learn", "f" * 40)) <= 63
    finally:
        sys.modules.pop("src.rag.indexer", None)


# ---------------------------------------------------------------------------
# snapshots and the quick labeller
# ---------------------------------------------------------------------------

from evaluation import label_candidates as lc  # noqa: E402
from evaluation import snapshot  # noqa: E402

COMMITS = [{"sha": "a" * 40, "commit": {"committer": {"date": "2020-01-01T10:00:00Z"}}},
           {"sha": "b" * 40, "commit": {"committer": {"date": "2020-01-03T10:00:00Z"}}},
           {"sha": "c" * 40, "commit": {"committer": {"date": "2020-01-02T10:00:00Z"}}}]


def test_snapshot_is_the_commit_the_reviewer_saw():
    assert snapshot.snapshot_sha_for({"commit_id": "9" * 40}, COMMITS) == "9" * 40
    # a conversation comment: the newest commit made before it, whatever order the list is in
    assert snapshot.snapshot_sha_for({"created_at": "2020-01-02T12:00:00Z"}, COMMITS) == "c" * 40
    assert snapshot.snapshot_sha_for({"created_at": "2019-12-31T00:00:00Z"}, COMMITS) == ""


def test_build_pr_turns_a_compare_result_into_a_pull_request():
    seen = {}
    def get(path, params=None):
        seen["path"] = path
        return {"files": [{"filename": "pkg/a.py", "status": "added", "additions": 2, "deletions": 0,
                           "patch": "@@ -0,0 +1,2 @@\n+def foo(x):\n+    return x"}]}
    label = make_label(5)
    label["snapshot_sha"] = "9" * 40
    pr = snapshot.build_pr(GitHub(get=get, pause=0), label)
    assert seen["path"] == f"/repos/o/r/compare/{'a' * 40}...{'9' * 40}"
    assert pr.head_sha == "9" * 40 and pr.number == 5 and pr.additions == 2
    assert snapshot.added_lines_of(pr, "pkg/a.py") == ["def foo(x):", "    return x"]
    assert snapshot.added_lines_of(pr, "other.py") == []


def test_open_pr_queries_use_creation_date_and_no_merge_filter():
    q = cc.search_queries("o/r", "tests", "2024-01-01", "2026-10-31", state="open")[0]
    assert "is:open created:2024-01-01..2026-10-31" in q and "is:merged" not in q


def candidate(**over):
    row = {"repo": "o/r", "number": "5", "url": "https://github.com/o/r/pull/5", "hint": "tests",
           "merged_at": "2021-07-26", "title": "Fix json", "author": "a", "matched": "add a test",
           "snippet": "please add a test for this", "comment_url": "https://x/c", "path": "pkg/a.py",
           "hunk": "@@ -1 +1 @@\n+def foo(x):", "snapshot_sha": "9" * 40, "additions": "47",
           "deletions": "18", "changed_files": "7", "py_source_files": "4", "has_tests": "False",
           "review_comments": "4", "comments": "2", "discussion": "6", "base_sha": "a" * 40,
           "head_sha": "1" * 40}
    row.update(over)
    return row


def scripted(*answers):
    it = iter(answers)
    return lambda prompt: next(it)


def fake_pr(_row):
    return snapshot.PullRequest(
        repo="o/r", number=5, title="t", author="", state="closed", merged=True, body="",
        additions=2, deletions=0, changed_files=1, head_sha="9" * 40,
        files=[snapshot.PRFile("pkg/a.py", "modified", 2, 0, "@@ -1 +1,2 @@\n+import os\n+def foo(x):")])


def test_labeller_records_a_finding_with_the_quoted_line_and_reviewer_quote():
    out = []
    ans = scripted("f", "2", "all", "moved_function")           # finding, line 2, others silent, tag
    label = lc.decide(candidate(), fake_pr, ans, out.append)
    c = label["checks"]
    assert c["missing_tests"] == {"expect": "finding", "note": "",
                                  "targets": [{"file": "pkg/a.py", "evidence_contains": "def foo(x):"}],
                                  "source": {"url": "https://x/c", "quote": "please add a test for this"}}
    assert c["smell"]["expect"] == "none" and c["reuse"]["expect"] == "none"
    assert label["edge_case"] == "moved_function" and label["snapshot_sha"] == "9" * 40
    assert lab.problems(label) == []
    assert any("please add a test" in line for line in out)


def test_labeller_other_checks_default_to_unclear_and_skip_writes_nothing():
    label = lc.decide(candidate(), fake_pr, scripted("f", "", "", ""), lambda m: None)
    assert label["checks"]["smell"]["expect"] == "unclear"
    assert label["checks"]["missing_tests"]["targets"] == [{"file": "pkg/a.py"}]      # blank = file only
    assert lc.decide(candidate(), fake_pr, scripted("s"), lambda m: None) is None
    with pytest.raises(EOFError):
        lc.decide(candidate(), fake_pr, scripted("q"), lambda m: None)


def test_labeller_quiet_prs_are_silent_but_check_one_needs_tests_in_the_pr():
    quiet = candidate(hint="negative", has_tests="True", snippet="", path="", hunk="")
    full = lc.decide(quiet, fake_pr, scripted("", ""), lambda m: None)
    assert [full["checks"][c]["expect"] for c in lab.CHECKS] == ["none", "none", "none"]
    no_tests = lc.decide({**quiet, "has_tests": "False"}, fake_pr, scripted("", ""), lambda m: None)
    assert no_tests["checks"]["missing_tests"]["expect"] == "unclear"
    assert no_tests["checks"]["smell"]["expect"] == "none"
    assert lab.problems(full) == []


def test_labeller_picks_the_only_source_file_and_warns_when_tests_are_already_there():
    out = []
    pr = fake_pr(None)
    pr.files.append(snapshot.PRFile("pkg/tests/test_a.py", "modified", 3, 0, "@@ -1 +1 @@\n+x"))
    label = lc.decide(candidate(path=""), lambda r: pr, scripted("f", "2", "", ""), out.append)
    assert label["checks"]["missing_tests"]["targets"][0]["file"] == "pkg/a.py"     # no question asked
    text = "\n".join(out)
    assert "[test] pkg/tests/test_a.py" in text and "tests are already touched" in text


def test_labeller_asks_which_file_when_several_sources_changed():
    pr = fake_pr(None)
    pr.files.append(snapshot.PRFile("pkg/b.py", "modified", 1, 0, "@@ -1 +1 @@\n+y = 1"))
    label = lc.decide(candidate(path=""), lambda r: pr, scripted("f", "2", "1", "", "", ""),
                      lambda m: None)
    assert label["checks"]["missing_tests"]["targets"][0]["file"] == "pkg/b.py"


def test_a_comment_on_a_test_file_does_not_make_the_test_the_target():
    label = lc.decide(candidate(path="pkg/tests/test_a.py"), fake_pr, scripted("f", "2", "", ""),
                      lambda m: None)
    assert label["checks"]["missing_tests"]["targets"][0]["file"] == "pkg/a.py"


def test_only_list_reads_pr_references(tmp_path):
    f = tmp_path / "keep.txt"
    f.write_text("# my picks\nscikit-learn/scikit-learn#16182   tests\n\npsf/requests#5856\n")
    assert lc.read_only_list(str(f)) == {"scikit-learn__scikit-learn__16182", "psf__requests__5856"}


# ---------------------------------------------------------------------------
# the one-file label set (eval_set.csv)
# ---------------------------------------------------------------------------

def test_csv_round_trip_keeps_targets_notes_and_unclear_checks(tmp_path):
    a = make_label(1, missing_tests="finding", smell="none")
    a["checks"]["missing_tests"]["targets"] = [{"file": "pkg/a.py", "evidence_contains": "def foo"},
                                               {"file": "", "evidence_contains": ""}]
    a["checks"]["missing_tests"]["note"] = "reviewer asked twice"
    b = make_label(2, "o/s", reuse="none")
    path = tmp_path / "set.csv"
    lab.write_csv({a["id"]: a, b["id"]: b}, path)
    back = lab.from_csv(path)
    assert back[a["id"]] == a and back[b["id"]] == b
    assert back[a["id"]]["checks"]["reuse"]["expect"] == "unclear"      # no row = unclear
    assert lab.problems(back[a["id"]]) == [] and lab.problems(back[b["id"]]) == []


def test_load_all_reads_the_csv_and_json_wins_for_the_same_pr(workspace, monkeypatch):
    csv_path = workspace / "eval_set.csv"
    only_csv = make_label(50, missing_tests="none")
    dup = make_label(1, missing_tests="none")                           # also a JSON file (finding)
    lab.write_csv({only_csv["id"]: only_csv, dup["id"]: dup}, csv_path)
    monkeypatch.setattr(lab, "LABEL_CSV", csv_path)
    labels = lab.load_all()
    assert "o__r__50" in labels and len(labels) == 13
    assert labels["o__r__1"]["checks"]["missing_tests"]["expect"] == "finding"


def test_blank_target_file_means_any_file_and_blank_evidence_any_line():
    assert score.matches(finding(file="pkg/zzz.py", evidence="x"), {"file": "", "evidence_contains": ""})
    assert not score.matches(finding(file="pkg/zzz.py"), {"file": "pkg/a.py", "evidence_contains": ""})


def test_test_labels_in_the_csv_are_frozen_too(workspace, monkeypatch, capsys):
    labels = lab.load_all()
    lab.write_csv(labels, workspace / "eval_set.csv")
    for f in lab.LABEL_DIR.glob("*.json"):
        f.unlink()
    monkeypatch.setattr(lab, "LABEL_CSV", workspace / "eval_set.csv")
    assert split.main(["--seed", "3"]) == 0
    monkeypatch.setattr(run_eval, "run_one", fake_run)
    rows = list(csv.DictReader(open(workspace / "eval_set.csv", encoding="utf-8")))
    test_ids = json.loads((workspace / "split.json").read_text())["test"]
    for r in rows:                                                     # edit a test PR's label in the CSV
        if r["id"] == test_ids[0]:
            r["expect"] = "none" if r["expect"] == "finding" else "finding"
            break
    with open(workspace / "eval_set.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=lab.CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    assert run_eval.main(["--split", "test", "--provider", "groq", "--final"]) == 1
    assert "no longer an unseen set" in capsys.readouterr().err
    

def test_common_scores_only_the_prs_every_run_completed():
    labels = labelled()
    ids = ["o__r__1", "o__r__2", "o__r__3"]
    base = {i: run_of("missing_tests", []) for i in ids}
    other = {"o__r__1": run_of("missing_tests", [finding()]),
             "o__r__2": run_of("missing_tests", [], status="failed"),
             "o__r__3": run_of("missing_tests", [])}
    only = score.common_ids(labels, ids, {"base": base, "other": other})
    assert only["missing_tests"] == {"o__r__1", "o__r__3"}
    m = score.score_run(labels, ids, base, {}, only)["missing_tests"]
    assert m["prs"] == 2 and m["not_run"] == 0 and m["targets"] == 1


def test_retry_failed_picks_up_errors_and_failed_checks_only(tmp_path):
    ok, bad, err, junk = (tmp_path / n for n in ("ok.json", "bad.json", "err.json", "junk.json"))
    ok.write_text(json.dumps(run_of("missing_tests", [])))
    bad.write_text(json.dumps(run_of("missing_tests", [], status="failed")))
    err.write_text(json.dumps({"error": "boom", "checks": []}))
    junk.write_text("{not json")
    assert [run_eval.needs_retry(p) for p in (ok, bad, err, junk)] == [False, True, True, True]
    