import json

from src.llm.aggregator import Finding
from src.report.generator import CheckResult, Review, render_markdown
from tests.unit.helpers import hunk, make_file, make_pr

PR = make_pr(make_file("src/a.py", hunk("+x = 1")), title="Speed up   loading")


def result(name, *findings, status="ok", note="", **metrics):
    title, icon = {"missing_tests": ("Missing tests", "🧪"), "reuse": ("Possible reuse", "♻️"),
                   "smell": ("Hot-path smells", "🔥")}[name]
    return CheckResult(name, title, icon, status=status, note=note,
                       findings=list(findings), seconds=1.0, metrics=metrics)


def review(*results, **kw):
    return Review(pr=PR, results=list(results), provider="groq",
                  llm_stats={"groq": {"model": "gpt-oss-20b", "calls": 3,
                                      "prompt_tokens": 100, "completion_tokens": 20}}, **kw)


def test_header_has_pr_number_title_and_count():
    md = render_markdown(review(result("smell", Finding("smell", "src/a.py", "x = 1", "msg", kind="n_plus_one"))))
    assert md.startswith("## 🛡️ pr-rakshaka review: #1 Speed up loading")
    assert "**1 finding(s)**" in md


def test_finding_shows_file_message_and_quoted_evidence():
    md = render_markdown(review(result("missing_tests",
                                       Finding("missing_tests", "src/a.py", "return a / b", "Zero division is untested."))))
    assert "### 🧪 Missing tests (1)" in md
    assert "**`src/a.py`**" in md and "Zero division is untested." in md
    assert "  ```\n  return a / b\n  ```" in md


def test_reuse_finding_names_the_existing_function():
    f = Finding("reuse", "src/n.py", "def fetch_json(l):", "Same logic.",
                related="src/http.py:10-14 def get_json (similarity 0.97)", kind="structural_duplicate")
    md = render_markdown(review(result("reuse", f)))
    assert "same logic, names changed" in md
    assert "  \n  Existing: `src/http.py:10-14 def get_json (similarity 0.97)`" in md   # hard line break


def test_smells_are_ordered_by_severity_and_tagged():
    lo = Finding("smell", "a.py", "aaaaaa()", "low one", kind="repeated_work", severity="low")
    hi = Finding("smell", "z.py", "zzzzzz()", "high one", kind="n_plus_one", severity="high")
    md = render_markdown(review(result("smell", lo, hi)))
    assert md.index("high one") < md.index("low one")
    assert "N+1 I/O or database operations · high" in md


def test_evidence_with_backticks_gets_a_longer_fence():
    f = Finding("smell", "a.py", 'msg = f"""```x```"""', "m")
    md = render_markdown(review(result("smell", f)))
    assert "````" in md          # four backticks outside the three inside the code


def test_no_findings_says_so_and_names_the_silent_checks():
    md = render_markdown(review(result("missing_tests"), result("smell")))
    assert "**no findings**" in md
    assert "No findings from: Missing tests, Hot-path smells." in md
    assert "###" not in md


def test_skipped_and_failed_checks_are_listed_with_the_reason():
    md = render_markdown(review(
        result("reuse", status="skipped", note="owner/repo is not indexed. Run: python src/rag/indexer.py owner/repo"),
        result("smell", status="failed", note="model error: boom")))
    assert "### Checks that did not run" in md
    assert "**Possible reuse**: skipped, owner/repo is not indexed" in md
    assert "**Hot-path smells**: failed, model error: boom" in md
    assert "No findings from" not in md          # a failed check is not a clean check


def test_footer_reports_calls_tokens_and_dropped():
    md = render_markdown(review(result("smell", dropped=2), dropped_by_final_filter=1))
    assert "groq `gpt-oss-20b`: 3 call(s), 120 tokens" in md
    assert "3 unsupported finding(s) dropped" in md


def test_long_reports_are_truncated_under_the_github_limit():
    many = [Finding("smell", "a.py", f"call_{i}()", "m " * 50) for i in range(400)]
    md = render_markdown(review(result("smell", *many)), max_chars=5000)
    assert len(md) < 5200 and "report truncated" in md


def test_to_dict_is_json_serialisable():
    r = review(result("smell", Finding("smell", "a.py", "x = 1", "m", kind="n_plus_one", severity="high")))
    data = json.loads(json.dumps(r.to_dict()))
    assert data["checks"][2 - 2]["findings"][0]["severity"] == "high"
    assert data["pr"]["number"] == 1
