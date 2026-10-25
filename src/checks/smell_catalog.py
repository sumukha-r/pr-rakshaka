"""Catalogue and optional static-analysis signals for Check 3.

Signals are context for the LLM, never findings on their own.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Smell:
    key: str
    title: str
    description: str
    evidence_hint: str


SMELLS: tuple[Smell, ...] = (
    Smell("repeated_work", "Repeated expensive work",
          "Repeated parsing, compilation, serialization, regex construction, or other expensive work inside a loop when the result can be computed once.",
          "Quote an added line that performs the repeated work."),
    Smell("quadratic_scan", "Repeated full scans / avoidable quadratic work",
          "Nested iteration or repeated linear searches that plausibly scale poorly with input size, where a set/dict/index or one-pass approach preserves behaviour.",
          "Quote an added loop or repeated-search line."),
    Smell("n_plus_one", "N+1 I/O or database operations",
          "A database, network, filesystem, or other blocking I/O operation executed once per item when batching or prefetching is practical.",
          "Quote the added I/O operation inside the iteration."),
    Smell("async_blocking", "Blocking operation in async code",
          "Synchronous blocking I/O or sleep inside an async function that can stall the event loop; do not flag CPU work or intentional sync boundaries without context.",
          "Quote the blocking call added to an async function."),
    Smell("large_materialization", "Unnecessary materialization",
          "Building a large list/dict or copying an entire iterable when streaming or iterating lazily would materially reduce memory use and preserve required semantics.",
          "Quote the added materialization or copy."),
    Smell("repeated_computation", "Repeated invariant computation",
          "A value independent of the loop item is recalculated in the loop, with meaningful cost and a safe hoist opportunity.",
          "Quote the added invariant computation."),
)

SMELL_KEYS: frozenset[str] = frozenset(s.key for s in SMELLS)
COMPLEXITY_THRESHOLD = 8

def catalogue_text() -> str:
    """One line per smell. The key is what the model puts in a finding's "smell" field."""
    return "\n".join(
        f'- "{s.key}" ({s.title}): {s.description} Evidence: {s.evidence_hint}'
        for s in SMELLS
    )


def complexity_signals(source: str, *, filename: str = "<unknown>") -> list[str]:
    """Return radon complexity signals for parseable functions in source.

    Radon is optional at runtime; absence or parse failures yield no signals.
    A complexity score only prioritises review and must not itself trigger a finding.
    """
    try:
        from radon.complexity import cc_visit
        blocks = cc_visit(source)
    except (ImportError, SyntaxError, ValueError):
        return []
    except Exception:  # defensive: static analysis must never break a review
        return []

    signals = []
    for block in blocks:
        complexity = getattr(block, "complexity", None)
        name = getattr(block, "name", "<function>")
        lineno = getattr(block, "lineno", None)
        if isinstance(complexity, int) and complexity >= COMPLEXITY_THRESHOLD:
            signals.append(
                f"{filename}:{lineno or '?'} {name}: Radon cyclomatic complexity "
                f"{complexity} (review signal only; not automatically a finding)"
            )
    return signals


def touched_complexity_signals(source: str, added_lines: set[int], *,
                               filename: str = "<unknown>") -> list[str]:
    """
    Radon signals for functions of a WHOLE file that the diff touches, whether
    the PR added them or only edited inside them. `added_lines` are line numbers
    in this version of the file (see diff_defs.added_line_numbers).

    This is what lets an edited function get a complexity hint: a diff alone
    only contains fragments, but the whole file parses.
    """
    try:
        from radon.complexity import cc_visit
        blocks = cc_visit(source)
    except Exception:  # ImportError, SyntaxError, anything: a hint must never break a review
        return []

    functions = {}      # keyed by line: radon reports a method on its own and inside its class
    for block in blocks:
        methods = getattr(block, "methods", None)   # a Class block: score its methods
        for fn in (methods if methods is not None else [block]):
            functions.setdefault(getattr(fn, "lineno", None), fn)

    signals = []
    for fn in functions.values():
        complexity = getattr(fn, "complexity", None)
        lineno, endline = getattr(fn, "lineno", None), getattr(fn, "endline", None)
        if not (isinstance(complexity, int) and isinstance(lineno, int)
                and isinstance(endline, int)):
            continue
        if complexity < COMPLEXITY_THRESHOLD:
            continue
        if not any(lineno <= n <= endline for n in added_lines):
            continue
        name = getattr(fn, "fullname", None) or getattr(fn, "name", "<function>")
        signals.append(
            f"{filename}:{lineno} {name}: Radon cyclomatic complexity {complexity} "
            "(this PR changes lines inside it; review signal only, "
            "not automatically a finding)"
        )
    return signals
