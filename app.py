"""
app.py — the pr-rakshaka web UI.

    streamlit run app.py

Paste a GitHub PR URL, pick the checks and the model backend, and read the
findings. All the work is done by `main.run_review()`, the same function the
command line uses, so the page and `python main.py` always agree.

"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque

import streamlit as st

import main as pipeline
from src.ingestion.pr_parser import PullRequest, parse_pr_url
from src.llm.client import PROVIDERS
from src.report.generator import (
    CheckResult,
    Review,
    finding_tags,
    ordered_findings,
    render_markdown,
)

st.set_page_config(page_title="pr-rakshaka", page_icon="🛡️", layout="wide")

CHECK_LABELS = {"missing_tests": "Missing tests", "reuse": "Possible reuse",
                "smell": "Hot-path smells"}
STATUS_ICON = {"ok": "✅", "skipped": "⏭️", "failed": "⚠️"}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _load_streamlit_secrets() -> None:
    """Copy keys from Streamlit's secrets file into the environment, if there is one."""
    for key in ("GROQ_API_KEY", "GITHUB_TOKEN"):
        if os.getenv(key):
            continue
        try:
            value = st.secrets.get(key)
        except Exception:  # noqa: BLE001 - no secrets file is the normal local case
            value = None
        if value:
            os.environ[key] = str(value)


_load_streamlit_secrets()


# ---------------------------------------------------------------------------
# Demo limits. Off on your own machine; set DEMO_MODE=1 on a public Space.
# ---------------------------------------------------------------------------

DEMO_MODE = os.getenv("DEMO_MODE") == "1"
MAX_PER_VISIT = int(os.getenv("MAX_REVIEWS_PER_VISIT", "3"))
MAX_PER_HOUR = int(os.getenv("MAX_REVIEWS_PER_HOUR", "20"))


@st.cache_resource
def _hour_log():
    """Start times of recent reviews, shared by every visitor of this running app."""
    return deque(), threading.Lock()


def _demo_limit_message() -> str | None:
    """None if this review may run (and it is counted), else why it may not."""
    if not DEMO_MODE:
        return None
    if st.session_state.get("reviews_used", 0) >= MAX_PER_VISIT:
        return (f"This demo allows {MAX_PER_VISIT} reviews per visit. "
                "Clone the repository to run it without limits.")
    log, lock = _hour_log()
    now = time.time()
    with lock:
        while log and now - log[0] > 3600:
            log.popleft()
        if len(log) >= MAX_PER_HOUR:
            return ("The demo has reached its hourly limit to protect the free model quota. "
                    "Please try again later.")
        log.append(now)
    st.session_state["reviews_used"] = st.session_state.get("reviews_used", 0) + 1
    return None


# ---------------------------------------------------------------------------
# Pieces of the page
# ---------------------------------------------------------------------------

def _sidebar() -> dict:
    with st.sidebar:
        st.header("Settings")
        provider = st.selectbox(
            "Model backend", [p for p in PROVIDERS],
            help="auto: your local Ollama if it is running, otherwise Groq. "
                 "A hosted app has no local model, so use groq there.")
        picked = st.multiselect(
            "Checks to run", list(CHECK_LABELS), default=list(CHECK_LABELS),
            format_func=CHECK_LABELS.get)
        index_first = st.checkbox(
            "Index the repo first if needed", value=False,
            help="The reuse check needs the repository indexed once. Indexing a big "
                 "repository takes minutes, so leave this off on a hosted app.")
        token = st.text_input(
            "GitHub token (optional)", type="password",
            help="Raises the GitHub rate limit and allows private repositories. "
                 "Used for this run only; not stored.")
        st.divider()
        st.caption(
            "Every finding quotes a line the pull request adds, or it is dropped "
            "before you see it. A check with nothing concrete to say stays silent.")
    return {"provider": provider, "checks": picked, "index": index_first, "token": token}


def _summary(review: Review) -> None:
    cols = st.columns(len(review.results) or 1)
    for col, r in zip(cols, review.results):
        with col:
            if r.status == "ok":
                st.metric(f"{r.icon} {r.title}", len(r.findings))
            else:
                st.metric(f"{r.icon} {r.title}", r.status)


def _finding_card(r: CheckResult, f) -> None:
    with st.container(border=True):
        tags = finding_tags(r.name, f)
        st.markdown(f"**`{f.filename}`**" + (f"  ·  {'  ·  '.join(tags)}" if tags else ""))
        st.write(f.message)
        st.code(f.evidence.strip("\n"), language="python")
        if r.name == "reuse" and f.related:
            st.caption(f"Existing: `{f.related}`")


def _findings_tab(review: Review) -> None:
    for r in review.results:
        st.subheader(f"{r.icon} {r.title}  ({len(r.findings)})"
                     if r.status == "ok" else f"{r.icon} {r.title}")
        if r.status != "ok":
            (st.warning if r.status == "failed" else st.info)(
                f"This check {r.status}. {r.note}".strip())
        elif not r.findings:
            st.success("Nothing concrete to report.")
        for f in ordered_findings(r):
            _finding_card(r, f)


def _diff_tab(review: Review) -> None:
    by_file: dict[str, int] = {}
    for f in review.findings:
        by_file[f.filename] = by_file.get(f.filename, 0) + 1
    shown = [f for f in review.pr.files if f.patch]
    if not shown:
        st.info("This pull request has no text diffs to show.")
        return
    st.caption("Files with findings are expanded.")
    for pf in sorted(shown, key=lambda p: (-by_file.get(p.filename, 0), p.filename)):
        n = by_file.get(pf.filename, 0)
        label = f"{pf.filename}  (+{pf.additions} / -{pf.deletions})"
        if n:
            label += f"  ·  {n} finding(s)"
        with st.expander(label, expanded=bool(n)):
            st.code(pf.patch, language="diff")


def _markdown_tab(review: Review) -> None:
    text = render_markdown(review)
    st.caption("This is the review comment as it would be posted on the pull request.")
    st.code(text, language="markdown")
    left, right, _ = st.columns([1, 1, 4])
    name = f"pr-rakshaka-{review.pr.repo.replace('/', '_')}-{review.pr.number}"
    left.download_button("Download .md", text, file_name=f"{name}.md", mime="text/markdown")
    right.download_button("Download .json", json.dumps(review.to_dict(), indent=2),
                          file_name=f"{name}.json", mime="application/json")


def _details_tab(review: Review) -> None:
    for r in review.results:
        st.markdown(f"{STATUS_ICON.get(r.status, '')} **{r.title}**: {r.status}, "
                    f"{len(r.findings)} finding(s), {r.seconds:.1f}s")
        if r.note:
            st.caption(r.note)
        if r.metrics:
            st.json(r.metrics, expanded=False)
    st.divider()
    st.markdown("**Model usage**")
    if review.llm_stats:
        st.json(review.llm_stats, expanded=False)
    else:
        st.caption("No model calls were made.")
    st.caption(f"Findings dropped by the final evidence filter: {review.dropped_by_final_filter}")


def _show(review: Review) -> None:
    pr: PullRequest = review.pr
    st.markdown(f"### [{pr.repo}#{pr.number}](https://github.com/{pr.repo}/pull/{pr.number}): {pr.title}")
    st.caption(f"{len(pr.files)} file(s) changed  ·  +{pr.additions} / -{pr.deletions}"
               + (f"  ·  by {pr.author}" if pr.author else ""))
    _summary(review)
    t1, t2, t3, t4 = st.tabs(["Findings", "Diff", "PR comment", "Run details"])
    with t1:
        _findings_tab(review)
    with t2:
        _diff_tab(review)
    with t3:
        _markdown_tab(review)
    with t4:
        _details_tab(review)


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

st.title("🛡️ pr-rakshaka")
st.write("Paste a GitHub pull request. It reads the diff and runs three checks: "
         "missing tests, possible reuse of existing code, and hot-path performance smells.")

settings = _sidebar()

with st.form("analyse"):
    url = st.text_input("Pull request URL", placeholder="https://github.com/owner/repo/pull/123")
    go = st.form_submit_button("Analyse", type="primary")

if go:
    st.session_state.pop("review", None)
    st.session_state.pop("error", None)
    try:
        parse_pr_url(url.strip())
    except ValueError:
        st.session_state["error"] = ("That does not look like a pull request URL. "
                                     "Use https://github.com/owner/repo/pull/123")
    else:
        limit = _demo_limit_message() if settings["checks"] else None
        if not settings["checks"]:
            st.session_state["error"] = "Pick at least one check in the sidebar."
        elif limit:
            st.session_state["error"] = limit
        else:
            with st.status("Reviewing...", expanded=True) as status:
                try:
                    review = pipeline.run_review(
                        url.strip(),
                        checks=[c for c in CHECK_LABELS if c in settings["checks"]],
                        provider=("groq" if DEMO_MODE else
                                  None if settings["provider"] == "auto" else settings["provider"]),
                        token=settings["token"] or None,
                        index_if_missing=settings["index"] and not DEMO_MODE,
                        progress=lambda message: status.write(message),
                    )
                    st.session_state["review"] = review
                    status.update(label="Done", state="complete", expanded=False)
                except Exception as exc:  # noqa: BLE001 - show it, do not crash the page
                    st.session_state["error"] = f"Could not review the pull request: {exc}"
                    status.update(label="Failed", state="error", expanded=False)

if st.session_state.get("error"):
    st.error(st.session_state["error"])
elif st.session_state.get("review"):
    _show(st.session_state["review"])
else:
    st.caption("Results appear here. Public repositories work without a GitHub token.")
    