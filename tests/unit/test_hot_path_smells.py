from src.checks import hot_path_smells as check
from src.ingestion.pr_parser import PRFile, PullRequest
from src.llm.aggregator import Finding


def make_pr(files):
    return PullRequest(
        repo="example/project", number=1, title="Improve processing",
        author="author", state="open", merged=False, body="",
        additions=sum(f.additions for f in files), deletions=0,
        changed_files=len(files), files=files,
    )


def file(path, patch, additions=3, status="modified"):
    return PRFile(path, status, additions, 0, patch)


def test_candidate_files_excludes_tests_non_python_and_deletions():
    pr = make_pr([
        file("src/a.py", "@@ -1 +1,2 @@\n+work()\n+save()", 2),
        file("tests/test_a.py", "@@ -1 +1 @@\n+assert True"),
        file("README.md", "@@ -1 +1 @@\n+text"),
        file("src/gone.py", "@@ -1 +0,0 @@\n-old()", 0, "deleted"),
    ])
    assert [f.filename for f in check._candidate_files(pr)] == ["src/a.py"]


def test_finding_requires_exact_added_line(monkeypatch):
    patch = "@@ -1,2 +1,3 @@\n def run(items):\n+    for item in items:\n+        db.query(item)\n     return None"
    pr = make_pr([file("src/jobs.py", patch, 2)])
    monkeypatch.setattr(check, "_signals", lambda f, full=None: [])
    monkeypatch.setattr(check, "chat_json", lambda *a, **k: {
        "analysis": "A database query is performed for every item.",
        "findings": [
            {"evidence": "db.query(item)", "message": "This may create N+1 queries."},
            {"evidence": "imaginary_call()", "message": "Unsupported finding."},
        ],
    })
    report = {}
    findings = check.check_hot_path_smells(pr, report=report)
    assert len(findings) == 1
    assert findings[0].evidence == "db.query(item)"
    assert report["raw_findings"] == 2
    assert report["dropped"] == 1
    assert report["llm_calls"] == 1


def test_no_candidate_means_no_llm_call(monkeypatch):
    pr = make_pr([file("tests/test_only.py", "@@ -1 +1 @@\n+assert 1 == 1")])
    monkeypatch.setattr(check, "chat_json", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("should not call model")))
    assert check.check_hot_path_smells(pr) == []


def test_malformed_findings_is_safely_ignored(monkeypatch):
    pr = make_pr([file("src/a.py", "@@ -1 +1,2 @@\n+def run():\n+    return list(range(10))", 2)])
    monkeypatch.setattr(check, "_signals", lambda f, full=None: [])
    monkeypatch.setattr(check, "chat_json", lambda *a, **k: {
        "analysis": "No issue.", "findings": "not a list"
    })
    assert check.check_hot_path_smells(pr) == []


def test_max_files_caps_llm_calls(monkeypatch):
    files = [
        file(f"src/f{i}.py", f"@@ -1 +1,2 @@\n+def f{i}():\n+    return {i}", 2)
        for i in range(4)
    ]
    pr = make_pr(files)
    calls = []
    monkeypatch.setattr(check, "_signals", lambda f, full=None: [])
    monkeypatch.setattr(check, "chat_json", lambda *a, **k: (
        calls.append(1) or {"analysis": "No performance issue.", "findings": []}
    ))
    assert check.check_hot_path_smells(pr, max_files=2) == []
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Smell key, severity, context lines, whole-file radon
# ---------------------------------------------------------------------------

LOOP_PATCH = (
    "@@ -1,3 +1,4 @@\n"
    " def cards(users):\n"
    "     for user in users:\n"
    "+        photo = db.query_photo(user.id)\n"
    "         use(user)"
)


def _reply(*findings):
    return lambda *a, **k: {"analysis": "x", "findings": list(findings)}


def test_finding_keeps_smell_key_and_severity(monkeypatch):
    pr = make_pr([file("src/c.py", LOOP_PATCH, 1)])
    monkeypatch.setattr(check, "chat_json", _reply(
        {"smell": "N_Plus_One", "evidence": "photo = db.query_photo(user.id)",
         "message": "One query per user.", "severity": "HIGH"}))
    [f] = check.check_hot_path_smells(pr, fetch_source=lambda *a: None)
    assert (f.kind, f.severity) == ("n_plus_one", "high")


def test_unknown_smell_key_is_kept_as_other_and_bad_severity_is_blank(monkeypatch):
    pr = make_pr([file("src/c.py", LOOP_PATCH, 1)])
    monkeypatch.setattr(check, "chat_json", _reply(
        {"smell": "made_up", "evidence": "photo = db.query_photo(user.id)",
         "message": "m", "severity": "catastrophic"}))
    [f] = check.check_hot_path_smells(pr, fetch_source=lambda *a: None)
    assert (f.kind, f.severity) == ("other", "")


def test_missing_smell_and_severity_fields_still_work(monkeypatch):
    pr = make_pr([file("src/c.py", LOOP_PATCH, 1)])
    monkeypatch.setattr(check, "chat_json", _reply(
        {"evidence": "photo = db.query_photo(user.id)", "message": "m"}))
    [f] = check.check_hot_path_smells(pr, fetch_source=lambda *a: None)
    assert (f.kind, f.severity) == ("other", "")


def test_quoting_an_unchanged_loop_line_is_dropped(monkeypatch):
    # the loop header is context: the model may use it, but may not cite it
    pr = make_pr([file("src/c.py", LOOP_PATCH, 1)])
    monkeypatch.setattr(check, "chat_json", _reply(
        {"smell": "n_plus_one", "evidence": "for user in users:", "message": "m"}))
    assert check.check_hot_path_smells(pr, fetch_source=lambda *a: None) == []


def test_prompt_shows_the_unchanged_loop_header_as_context(monkeypatch):
    pr = make_pr([file("src/c.py", LOOP_PATCH, 1)])
    seen = []
    monkeypatch.setattr(check, "chat_json", lambda system, user, **k: (
        seen.append((system, user)) or {"analysis": "", "findings": []}))
    check.check_hot_path_smells(pr, fetch_source=lambda *a: None)
    system, user = seen[0]
    assert "\n     for user in users:\n" in user          # context line, one-space prefix
    assert "UNCHANGED context" in system
    assert '"smell"' in system


def test_prompt_catalogue_lists_every_smell_key(monkeypatch):
    from src.checks.smell_catalog import SMELLS
    pr = make_pr([file("src/c.py", LOOP_PATCH, 1)])
    seen = []
    monkeypatch.setattr(check, "chat_json", lambda system, user, **k: (
        seen.append(user) or {"analysis": "", "findings": []}))
    check.check_hot_path_smells(pr, fetch_source=lambda *a: None)
    assert all(f'"{s.key}"' in seen[0] for s in SMELLS)


_COMPLEX = "def route(x):\n" + "".join(
    f"    if x == {i}:\n        return {i}\n" for i in range(10)) + "    return -1\n"


def test_whole_file_gives_radon_signal_for_an_edited_function(monkeypatch):
    # line 4 of the new file is inside route(); the patch alone could never parse it
    patch = "@@ -3,3 +3,4 @@\n     if x == 1:\n+        log()\n         return 1\n     if x == 2:"
    pr = make_pr([file("src/r.py", patch, 1)])
    pr.head_sha = "abc123"
    seen, fetched = [], []

    def fetch(repo, path, ref):
        fetched.append((repo, path, ref))
        return _COMPLEX

    monkeypatch.setattr(check, "chat_json", lambda system, user, **k: (
        seen.append(user) or {"analysis": "", "findings": []}))
    report = {}
    check.check_hot_path_smells(pr, fetch_source=fetch, report=report)
    assert fetched == [("example/project", "src/r.py", "abc123")]
    assert "route: Radon cyclomatic complexity 11" in seen[0]
    assert report["radon_signals"] == 1 and report["files_fetched"] == 1


def test_function_not_touched_by_the_diff_gets_no_signal():
    from src.checks.smell_catalog import touched_complexity_signals
    assert touched_complexity_signals(_COMPLEX, {1000}, filename="r.py") == []
    assert len(touched_complexity_signals(_COMPLEX, {3}, filename="r.py")) == 1


def test_no_head_commit_means_no_fetch_and_no_error(monkeypatch):
    pr = make_pr([file("src/c.py", LOOP_PATCH, 1)])     # head_sha defaults to ""
    monkeypatch.setattr(check, "chat_json", _reply())
    check.check_hot_path_smells(pr, fetch_source=lambda *a: (_ for _ in ()).throw(
        AssertionError("must not fetch without a head commit")))


def test_failed_fetch_falls_back_to_diff_only_signals(monkeypatch):
    pr = make_pr([file("src/c.py", LOOP_PATCH, 1)])
    pr.head_sha = "abc"
    monkeypatch.setattr(check, "chat_json", _reply())
    report = {}
    check.check_hot_path_smells(pr, fetch_source=lambda *a: None, report=report)
    assert report["files_fetched"] == 0 and report["llm_calls"] == 1


def test_cli_format_has_real_newlines_and_the_smell():
    out = check._format([Finding("smell", "a.py", "x = 1", "msg", kind="n_plus_one",
                                 severity="high")])
    assert "\\n" not in out
    assert out.splitlines()[0] == "a.py  [n_plus_one, high]"
