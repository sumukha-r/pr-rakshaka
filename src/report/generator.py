"""
generator.py — turn verified findings into a Markdown review comment.

Deterministic on purpose: no LLM pass. Every sentence in the report is either a
finding that already passed the evidence filter, or a fact about the run. A
model-written summary could include claims the evidence filter never verified,
which would break this project's evidence-only rule.

    from src.report.generator import CheckResult, Review, render_markdown
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from src.checks.smell_catalog import SMELLS
from src.ingestion.pr_parser import PullRequest
from src.llm.aggregator import Finding

MAX_COMMENT_CHARS = 60_000     # GitHub rejects comments over 65,536 characters
_SMELL_TITLES = {s.key: s.title for s in SMELLS}
_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "": 3}


@dataclass
class CheckResult:
    name: str                  # "missing_tests" | "reuse" | "smell"
    title: str                 # heading shown to the reader
    icon: str
    status: str = "ok"         # "ok" | "skipped" | "failed"
    note: str = ""             # why it was skipped or failed
    findings: list[Finding] = field(default_factory=list)
    seconds: float = 0.0
    metrics: dict = field(default_factory=dict)   # counts from the check's report dict


@dataclass
class Review:
    pr: PullRequest
    results: list[CheckResult]
    provider: str = ""                 # what was asked for: auto | local | groq
    llm_stats: dict = field(default_factory=dict)   # client.stats(): per-backend calls/tokens
    dropped_by_final_filter: int = 0

    @property
    def findings(self) -> list[Finding]:
        return [f for r in self.results for f in r.findings]

    def to_dict(self) -> dict:
        """JSON-friendly form (for --json and the Streamlit app)."""
        return {
            "pr": {"repo": self.pr.repo, "number": self.pr.number, "title": self.pr.title,
                   "files": len(self.pr.files)},
            "provider": self.provider,
            "llm_stats": self.llm_stats,
            "dropped_by_final_filter": self.dropped_by_final_filter,
            "checks": [
                {"name": r.name, "status": r.status, "note": r.note,
                 "seconds": round(r.seconds, 1), "metrics": r.metrics,
                 "findings": [asdict(f) for f in r.findings]}
                for r in self.results
            ],
        }


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _fence(code: str) -> str:
    """A code block whose fence is longer than any backtick run inside the code."""
    longest = max((len(m) for m in re.findall(r"`+", code)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}\n{code.strip(chr(10))}\n{ticks}"


def _indent(block: str, spaces: int = 2) -> str:
    pad = " " * spaces
    return "\n".join(pad + ln if ln else ln for ln in block.splitlines())


def _heading(r: CheckResult) -> str:
    return f"### {r.icon} {r.title} ({len(r.findings)})"


def _ordered(r: CheckResult) -> list[Finding]:
    if r.name == "smell":
        return sorted(r.findings, key=lambda f: (_SEVERITY_ORDER.get(f.severity, 3), f.filename))
    return sorted(r.findings, key=lambda f: f.filename)


def _render_finding(r: CheckResult, f: Finding) -> str:
    message = _one_line(f.message)
    if r.name == "reuse":
        label = {"structural_duplicate": "same logic, names changed",
                 "duplicate": "likely duplicate", "reuse": "possible reuse"}.get(f.kind, f.kind)
        head = f"- **`{f.filename}`** · {label}\n  {message}"
        if f.related:
            head += f"  \n  Existing: `{_one_line(f.related)}`"   # two spaces = line break
    elif r.name == "smell":
        smell = _SMELL_TITLES.get(f.kind, "")
        tags = [t for t in (smell, f.severity) if t]
        head = f"- **`{f.filename}`**" + (f" · {' · '.join(tags)}" if tags else "") + f"\n  {message}"
    else:
        head = f"- **`{f.filename}`**\n  {message}"
    return head + "\n" + _indent(_fence(f.evidence))


def _footer(review: Review) -> str:
    parts = []
    for backend, s in sorted(review.llm_stats.items()):
        parts.append(f"{backend} `{s.get('model', '?')}`: {s.get('calls', 0)} call(s), "
                     f"{s.get('prompt_tokens', 0) + s.get('completion_tokens', 0)} tokens")
    dropped = sum(r.metrics.get("dropped", 0) for r in review.results) \
        + review.dropped_by_final_filter
    secs = sum(r.seconds for r in review.results)
    line = " · ".join(parts) if parts else "no LLM calls"
    return (f"<sub>pr-rakshaka · {line} · {dropped} unsupported finding(s) dropped by the "
            f"evidence filter · {secs:.0f}s</sub>")


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def render_markdown(review: Review, *, max_chars: int = MAX_COMMENT_CHARS) -> str:
    pr = review.pr
    total = len(review.findings)
    out = [f"## 🛡️ pr-rakshaka review: #{pr.number} {_one_line(pr.title)}", ""]
    out.append(f"`{pr.repo}` · {len(pr.files)} file(s) · "
               + (f"**{total} finding(s)**" if total else "**no findings**"))
    out += ["", "> Every finding quotes a line this PR adds. A check with nothing "
                "concrete to say stays silent.", ""]

    for r in review.results:
        if r.status != "ok" or not r.findings:
            continue
        out += [_heading(r), ""]
        out += [_render_finding(r, f) + "\n" for f in _ordered(r)]

    notes = [r for r in review.results if r.status != "ok"]
    if notes:
        out += ["### Checks that did not run", ""]
        out += [f"- **{r.title}**: {r.status}" + (f", {_one_line(r.note)}" if r.note else "")
                for r in notes]
        out.append("")

    clean = [r.title for r in review.results if r.status == "ok" and not r.findings]
    if clean:
        out += [f"No findings from: {', '.join(clean)}.", ""]

    out += ["---", _footer(review)]
    text = "\n".join(out).rstrip() + "\n"
    if len(text) > max_chars:
        cut = text[:max_chars].rsplit("\n", 1)[0]
        text = cut + "\n\n_(report truncated to fit a GitHub comment)_\n"
    return text
