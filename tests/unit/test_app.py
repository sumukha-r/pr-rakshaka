"""The Streamlit page, run headlessly with Streamlit's AppTest. The pipeline is faked:
no model, no GitHub, no browser."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import main  # noqa: E402
from src.llm.aggregator import Finding  # noqa: E402
from src.report.generator import CheckResult, Review  # noqa: E402
from tests.unit.helpers import hunk, make_file, make_pr  # noqa: E402

APP = str(Path(__file__).resolve().parents[2] / "app.py")
URL = "https://github.com/demo/demo/pull/7"

PATCH = hunk(" for user in users:", "+    photo = db.query_photo(user.id)", "     use(user)")


def fake_review(**kw) -> Review:
    pr = make_pr(make_file("src/cards.py", PATCH), make_file("README.md", hunk("+docs")),
                 title="Show photos")
    pr.number, pr.repo = 7, "demo/demo"
    results = [
        CheckResult("missing_tests", "Missing tests", "🧪"),
        CheckResult("reuse", "Possible reuse", "♻️", status="skipped",
                    note="demo/demo is not indexed. Run: python src/rag/indexer.py demo/demo"),
        CheckResult("smell", "Hot-path smells", "🔥", seconds=1.5, metrics={"llm_calls": 1},
                    findings=[Finding("smell", "src/cards.py", "photo = db.query_photo(user.id)",
                                      "One query per user; batch them.", kind="n_plus_one",
                                      severity="high")]),
    ]
    return Review(pr=pr, results=results, provider="groq",
                  llm_stats={"groq": {"model": "gpt-oss-20b", "calls": 1,
                                      "prompt_tokens": 10, "completion_tokens": 5}}, **kw)


@pytest.fixture
def pipeline(monkeypatch):
    calls = []

    def fake(url, **kwargs):
        calls.append((url, kwargs))
        if kwargs.get("progress"):
            kwargs["progress"]("Fetching the pull request from GitHub...")
        return fake_review()

    monkeypatch.setattr(main, "run_review", fake)
    return calls


def open_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=20)
    at.run()
    assert not at.exception
    return at


def submit(at: AppTest, url: str = URL) -> AppTest:
    at.text_input[0].set_value(url)
    at.button[0].click()
    at.run()
    return at


def text_of(at: AppTest) -> str:
    parts = [e.value for e in at.markdown] + [e.value for e in at.caption] \
        + [e.value for e in at.code] + [e.value for e in at.subheader]
    return "\n".join(str(p) for p in parts)


def test_first_load_is_empty_and_calm(pipeline):
    at = open_app()
    assert at.title[0].value.endswith("pr-rakshaka")
    assert not at.metric and not at.error
    assert pipeline == []


def test_a_valid_url_runs_the_pipeline_with_the_chosen_settings(pipeline):
    at = open_app()
    submit(at)
    assert not at.exception and not at.error
    [(url, kw)] = pipeline
    assert url == URL
    assert kw["checks"] == ["missing_tests", "reuse", "smell"]
    assert kw["provider"] is None               # "auto" means: do not force a backend
    assert kw["token"] is None and kw["index_if_missing"] is False


def test_the_provider_and_check_choices_are_passed_through(pipeline):
    at = open_app()
    at.selectbox[0].set_value("groq")
    at.multiselect[0].set_value(["smell", "missing_tests"])
    submit(at)
    [(_, kw)] = pipeline
    assert kw["provider"] == "groq" and kw["checks"] == ["missing_tests", "smell"]


def test_results_show_counts_the_finding_and_its_evidence(pipeline):
    at = submit(open_app())
    values = {m.label: m.value for m in at.metric}
    assert values == {"🧪 Missing tests": "0", "♻️ Possible reuse": "skipped", "🔥 Hot-path smells": "1"}
    text = text_of(at)
    assert "One query per user; batch them." in text
    assert "photo = db.query_photo(user.id)" in text
    assert "N+1 I/O or database operations" in text and "high" in text


def test_a_skipped_check_explains_itself_and_a_clean_one_says_so(pipeline):
    at = submit(open_app())
    assert any("is not indexed" in i.value for i in at.info)
    assert any("Nothing concrete to report" in s.value for s in at.success)


def test_the_pr_comment_tab_holds_the_same_markdown_the_cli_prints(pipeline):
    at = submit(open_app())
    assert any(c.language == "markdown" and "pr-rakshaka review: #7 Show photos" in c.value
               for c in at.code)


def test_the_diff_tab_lists_files_with_findings_first_and_expanded(pipeline):
    at = submit(open_app())
    labels = [e.label for e in at.expander]
    assert labels[0].startswith("src/cards.py") and "1 finding(s)" in labels[0]
    assert labels[1].startswith("README.md") and "finding" not in labels[1]


def test_a_bad_url_is_rejected_before_any_work(pipeline):
    at = submit(open_app(), "not a url")
    assert any("does not look like a pull request URL" in e.value for e in at.error)
    assert pipeline == [] and not at.metric


def test_no_checks_selected_is_an_error_not_a_crash(pipeline):
    at = open_app()
    at.multiselect[0].set_value([])
    submit(at)
    assert any("at least one check" in e.value for e in at.error)
    assert pipeline == []


def test_a_failing_pipeline_shows_the_message_instead_of_crashing(monkeypatch):
    def boom(url, **kw):
        raise RuntimeError("404 Not Found")
    monkeypatch.setattr(main, "run_review", boom)
    at = submit(open_app())
    assert not at.exception
    assert any("Could not review the pull request: 404 Not Found" in e.value for e in at.error)
    assert not at.metric


def test_a_second_run_replaces_the_first_result(pipeline):
    at = submit(open_app())
    assert at.metric
    submit(at, "not a url")
    assert at.error and not at.metric
    