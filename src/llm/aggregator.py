"""
aggregator.py — the evidence-only filter.

Core principle of pr-rakshaka: every finding must quote real code from the
diff, or it never reaches the user. LLMs sometimes invent code; this module
is the guard that drops those findings.
"""

import re
from dataclasses import dataclass


@dataclass
class Finding:
    check: str        # "missing_tests" | "reuse" | "smell"
    filename: str     # file in the PR the finding is about
    evidence: str     # the exact diff code that justifies it
    message: str      # what the reviewer should know
    related: str = ""  # reuse: the existing function this points at ("path::symbol")
    kind: str = ""     # reuse: "structural_duplicate" | "duplicate" | "reuse"


def added_lines(patch: str) -> list[str]:
    """The '+' lines of a unified diff, without the '+' prefix or file headers."""
    return [
        ln[1:] for ln in patch.splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
    ]


def _added_runs(patch: str) -> list[list[str]]:
    """Added lines grouped into runs of consecutive '+' lines."""
    runs: list[list[str]] = []
    cur: list[str] = []
    for ln in patch.splitlines():
        if ln.startswith("+") and not ln.startswith("+++"):
            cur.append(ln[1:])
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def evidence_in_diff(evidence: str, patch: str, *, min_len: int = 6,
                     max_lines: int = 5) -> bool:
    """
    True if `evidence` appears verbatim in the code the diff ADDS, ignoring
    whitespace differences and stray leading '+' markers.

    The quote may be part of one added line, or span up to `max_lines`
    consecutive added lines (models often join two lines into one). It may not
    bridge context or removed lines, and very short quotes are rejected
    because they match almost anything.
    """
    parts = [ln[1:] if ln.startswith("+") else ln for ln in evidence.splitlines()]
    ev = _norm(" ".join(parts))
    if len(ev) < min_len:
        return False
    for run in _added_runs(patch):
        for start in range(len(run)):
            window = ""
            for line in run[start:start + max_lines]:
                window = _norm(f"{window} {line}")
                if ev in window:
                    return True
    return False


def keep_evidenced(
    findings: list[Finding], patches: dict[str, str]
) -> tuple[list[Finding], list[Finding]]:
    """
    Split findings into (kept, dropped). `patches` maps filename -> raw patch.
    A finding is kept only if its evidence appears in its own file's patch.
    """
    kept: list[Finding] = []
    dropped: list[Finding] = []
    for f in findings:
        patch = patches.get(f.filename, "")
        (kept if evidence_in_diff(f.evidence, patch) else dropped).append(f)
    return kept, dropped