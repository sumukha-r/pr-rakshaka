"""
compare_check1.py — run Check 1 (missing tests) on four small hand-written
PRs with known answers, through each backend, and show where they differ.

These are sanity fixtures to compare models quickly (no GitHub, no token). They
are not the evaluation dataset; that comes from real PRs.

    python -m tests.integration.compare_check1                    # local vs Groq
    python -m tests.integration.compare_check1 --provider local   # one backend
    python -m tests.integration.compare_check1 --only tested      # one case

Run from the repo root. Each case has an expected answer:
    flag  = the model should report at least one finding
    clean = the model should report nothing
"""

import argparse
import logging
import time

from src.checks.missing_tests import check_missing_tests
from src.ingestion.pr_parser import PRFile, PullRequest
from src.llm.client import PROVIDERS, LLMError, use_provider
from src.llm.compare import print_comparison, run_with_each

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DISCOUNT_SRC = '''@@ -1,4 +1,12 @@
 """Pricing helpers."""

+def apply_discount(price, percent, member=False):
+    if percent < 0 or percent > 100:
+        raise ValueError("percent must be between 0 and 100")
+    discount = price * percent / 100
+    if member:
+        discount += price * 0.05
+    return round(price - discount, 2)
+
 def tax(price):
     return price * 1.1
'''

_DISCOUNT_TESTS = '''@@ -1,4 +1,16 @@
 from pricing import tax
+from pricing import apply_discount
+import pytest

 def test_tax():
     assert tax(100) > 100
+
+def test_discount_basic():
+    assert apply_discount(100, 10) == 90.0
+
+def test_discount_member():
+    assert apply_discount(100, 10, member=True) == 85.0
+
+def test_discount_invalid_percent():
+    with pytest.raises(ValueError):
+        apply_discount(100, 120)
'''

_RENAME_SRC = '''@@ -1,3 +1,3 @@
-def tax(p):
-    r = p * 1.1
-    return r
+def tax(price: float) -> float:
+    total = price * 1.1
+    return total
'''

_LOAD_SRC = '''@@ -10,3 +10,7 @@ def load(path):
     with open(path) as fh:
         data = fh.read()
-    return json.loads(data)
+    try:
+        return json.loads(data)
+    except json.JSONDecodeError:
+        logger.warning("bad json in %s", path)
+        return {}
'''

_UNRELATED_TESTS = '''@@ -1,2 +1,5 @@
 from parser import load_config

+def test_load_config_defaults():
+    assert load_config({})["debug"] is False
+
'''

CASES = [
    {
        "name": "untested_logic",
        "expect": "flag",
        "why": "new function with branches and an error path, no tests at all",
        "title": "Add apply_discount()",
        "files": [("src/pricing.py", _DISCOUNT_SRC)],
    },
    {
        "name": "tested",
        "expect": "clean",
        "why": "same function, but the PR adds tests for every branch",
        "title": "Add apply_discount() with tests",
        "files": [("src/pricing.py", _DISCOUNT_SRC),
                  ("tests/test_pricing.py", _DISCOUNT_TESTS)],
    },
    {
        "name": "refactor_only",
        "expect": "clean",
        "why": "renamed variables and added type hints, behaviour unchanged",
        "title": "Rename variables and add type hints in tax()",
        "files": [("src/pricing.py", _RENAME_SRC)],
    },
    {
        "name": "wrong_tests",
        "expect": "flag",
        "why": "new except branch; the only changed test covers a different function",
        "title": "Make load() tolerate invalid JSON",
        "files": [("src/parser.py", _LOAD_SRC),
                  ("tests/test_parser.py", _UNRELATED_TESTS)],
    },
]


def _build_pr(case: dict) -> PullRequest:
    files = [
        PRFile(filename=name, status="modified",
               additions=sum(1 for ln in patch.splitlines() if ln.startswith("+")),
               deletions=sum(1 for ln in patch.splitlines() if ln.startswith("-")),
               patch=patch)
        for name, patch in case["files"]
    ]
    return PullRequest(
        repo="demo/demo", number=1, title=case["title"], author="demo",
        state="open", merged=False, body="", additions=0, deletions=0,
        changed_files=len(files), files=files,
    )


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def _make_runner(cases: list[dict]):
    def run_all(report: dict) -> list[dict]:
        rows, correct, raw, dropped = [], 0, 0, 0
        for case in cases:
            row = {"case": case, "findings": [], "error": "", "seconds": 0.0}
            rep: dict = {}
            start = time.monotonic()
            try:
                row["findings"] = check_missing_tests(_build_pr(case), report=rep)
            except LLMError as exc:
                row["error"] = str(exc)
            row["seconds"] = time.monotonic() - start
            row["ok"] = (not row["error"]) and (
                bool(row["findings"]) == (case["expect"] == "flag"))
            correct += row["ok"]
            raw += rep.get("raw_findings", 0)
            dropped += rep.get("dropped", 0)
            rows.append(row)
        report.update(correct=f"{correct}/{len(cases)}",
                      raw_findings=raw, dropped=dropped)
        return rows
    return run_all


def _show(rows: list[dict] | None) -> str:
    out = []
    for r in rows or []:
        c = r["case"]
        got = "ERROR" if r["error"] else ("flagged" if r["findings"] else "clean")
        verdict = "ok" if r["ok"] else "WRONG"
        out.append(f"[{verdict}] {c['name']}: expected {c['expect']}, got {got} "
                   f"({r['seconds']:.1f}s)  - {c['why']}")
        if r["error"]:
            out.append(f"      error: {r['error']}")
        for f in r["findings"]:
            out.append(f"      evidence: {f.evidence.strip()}")
            out.append(f"      says:     {f.message}")
    return "\n".join(out)


def _print_matrix(outcomes, cases) -> None:
    names = [o.provider for o in outcomes]
    print("\nCase-by-case (the diff between backends):")
    header = ["case", "expected"] + names
    table = []
    for i, c in enumerate(cases):
        row = [c["name"], c["expect"]]
        for o in outcomes:
            if o.error or o.result is None:
                row.append("FAILED")
                continue
            r = o.result[i]
            got = "error" if r["error"] else ("flagged" if r["findings"] else "clean")
            row.append(f"{'ok' if r['ok'] else 'WRONG'} ({got})")
        table.append(row)
    widths = [max(len(h), *(len(r[j]) for r in table)) for j, h in enumerate(header)]
    fmt = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths))  # noqa: E731
    print(fmt(header))
    print(fmt(["-" * w for w in widths]))
    for r in table:
        print(fmt(r))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tests.integration.compare_check1")
    ap.add_argument("--provider", choices=[p for p in PROVIDERS if p != "auto"],
                    help="run one backend only (default: local and groq)")
    ap.add_argument("--only", choices=[c["name"] for c in CASES],
                    help="run a single case")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="[%(name)s] %(message)s")
    cases = [c for c in CASES if not args.only or c["name"] == args.only]
    providers = [args.provider] if args.provider else ["local", "groq"]

    outcomes = run_with_each(_make_runner(cases), providers)
    print_comparison(outcomes, show=_show)
    if len(outcomes) > 1:
        _print_matrix(outcomes, cases)
    print("\nNote: the first local call includes loading the model into GPU "
          "memory, so its time is inflated.")


if __name__ == "__main__":
    main()