"""
compare.py — run the same job through several backends and compare.

    from src.llm.compare import run_with_each, print_comparison

    outcomes = run_with_each(lambda report: check_missing_tests(pr, report=report))
    print_comparison(outcomes, show=format_findings)

Each backend is forced strictly (no 'auto' fallback), so the numbers really
belong to the backend named. `fn` receives a dict it may fill with extra
metrics (e.g. report["dropped"]); those appear in the table.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from src.llm import client


@dataclass
class Outcome:
    provider: str
    result: Any = None
    error: str = ""
    seconds: float = 0.0
    model: str = ""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    report: dict = field(default_factory=dict)


def run_with_each(fn: Callable[[dict], Any],
                  providers: Iterable[str] = ("local", "groq")) -> list[Outcome]:
    outcomes = []
    for name in providers:
        client.reset_stats()
        out = Outcome(provider=name)
        start = time.monotonic()
        try:
            with client.use_provider(name):
                out.result = fn(out.report)
        except client.LLMError as exc:
            out.error = str(exc)
        out.seconds = time.monotonic() - start
        s = client.stats().get(name, {})
        out.model = s.get("model", "")
        out.calls = s.get("calls", 0)
        out.prompt_tokens = s.get("prompt_tokens", 0)
        out.completion_tokens = s.get("completion_tokens", 0)
        outcomes.append(out)
    return outcomes


def print_comparison(outcomes: list[Outcome],
                     show: Callable[[Any], str] = str) -> None:
    extra_keys = sorted({k for o in outcomes for k in o.report})
    cols = ["backend", "model", "time(s)", "calls", "prompt tok", "output tok"] + extra_keys
    rows = [
        [o.provider, o.model or "-", f"{o.seconds:.1f}", str(o.calls),
         str(o.prompt_tokens), str(o.completion_tokens)]
        + [str(o.report.get(k, "-")) for k in extra_keys]
        if not o.error else [o.provider, "FAILED"] + [""] * (len(cols) - 2)
        for o in outcomes
    ]
    widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths))  # noqa: E731
    print("\n" + line(cols))
    print(line(["-" * w for w in widths]))
    for r in rows:
        print(line(r))

    for o in outcomes:
        print(f"\n=== {o.provider} ===")
        print(f"error: {o.error}" if o.error else show(o.result))