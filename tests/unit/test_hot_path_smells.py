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
    monkeypatch.setattr(check, "_signals", lambda f: [])
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
    monkeypatch.setattr(check, "_signals", lambda f: [])
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
    monkeypatch.setattr(check, "_signals", lambda f: [])
    monkeypatch.setattr(check, "chat_json", lambda *a, **k: (
        calls.append(1) or {"analysis": "No performance issue.", "findings": []}
    ))
    assert check.check_hot_path_smells(pr, max_files=2) == []
    assert len(calls) == 2
