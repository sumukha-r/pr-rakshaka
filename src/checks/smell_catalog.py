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


def catalogue_text() -> str:
    return "\n".join(
        f"- {s.title}: {s.description} Evidence: {s.evidence_hint}"
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
        if isinstance(complexity, int) and complexity >= 8:
            signals.append(
                f"{filename}:{lineno or '?'} {name}: Radon cyclomatic complexity "
                f"{complexity} (review signal only; not automatically a finding)"
            )
    return signals
