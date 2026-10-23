"""
aggregator.py — the evidence-only filter.

Core principle of pr-rakshaka: every finding must quote a real line from the
diff, or it never reaches the user. LLMs sometimes invent code; this module
is the guard that drops those findings.

For now it holds the Finding type and the evidence check. Merging findings
from the three checks into one report comes later.
"""

import re
from dataclasses import dataclass


@dataclass
class Finding:
    check: str        # "missing_tests" | "reuse" | "smell"
    filename: str     # file in the PR the finding is about
    evidence: str     # the exact diff line that justifies it
    message: str      # what the reviewer should know


def added_lines(patch: str) -> list[str]:
    """The '+' lines of a unified diff, without the '+' prefix or file headers."""
    return [
        ln[1:] for ln in patch.splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
    ]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def evidence_in_diff(evidence: str, patch: str, *, min_len: int = 6) -> bool:
    """
    True if `evidence` is (part of) a single added line in `patch`, ignoring
    whitespace differences and a stray leading '+'. Very short quotes are
    rejected because they match almost anything.
    """
    ev = _norm(evidence.lstrip("+"))
    if len(ev) < min_len:
        return False
    return any(ev in _norm(line) for line in added_lines(patch))


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