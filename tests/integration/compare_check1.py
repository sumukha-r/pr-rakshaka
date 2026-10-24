"""
compare_check1.py — run Check 1 (missing tests) on ten hand-written PRs with
known answers, through each backend, and show where they differ.

These are sanity fixtures to compare models quickly (no GitHub, no token). They
are not the evaluation dataset; that comes from real PRs.

    python -m tests.integration.compare_check1                    # local vs Groq
    python -m tests.integration.compare_check1 --provider local   # one backend
    python -m tests.integration.compare_check1 --only tested      # one case
    python -m tests.integration.compare_check1 --level hard       # one difficulty

Run from the repo root. Each case has an expected answer:
    flag  = the model should report at least one finding
    clean = the model should report nothing
"""

import argparse
import difflib
import logging
import time

from src.checks.missing_tests import check_missing_tests
from src.ingestion.pr_parser import PRFile, PullRequest
from src.llm.client import PROVIDERS, LLMError
from src.llm.compare import print_comparison, run_with_each

# ---------------------------------------------------------------------------
# Fixtures, cases 1-4: hand-written patches
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

# ---------------------------------------------------------------------------
# Fixtures, cases 5-10: written as old/new file versions; the diff is generated
# ---------------------------------------------------------------------------


def _diff(old: str, new: str) -> str:
    """Unified diff the way GitHub's `patch` field looks (no ---/+++ header)."""
    lines = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(), lineterm="", n=3))
    return "\n".join(lines[2:]) + "\n"


def _file(path: str, old: str, new: str) -> tuple[str, str, str]:
    return (path, _diff(old, new), "added" if not old else "modified")


# 5: one-line bug fix, no test
_PAGE_OLD = '''def page_bounds(page, per_page, total):
    """Return (start, end) slice indexes for a 1-based page number."""
    start = (page - 1) * per_page
    end = start + per_page - 1
    return start, min(end, total)
'''
_PAGE_NEW = _PAGE_OLD.replace("start + per_page - 1", "start + per_page")

# 6: new class, the only changed test is for an unrelated module
_CACHE_NEW = '''from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._data = OrderedDict()

    def get(self, key, default=None):
        if key not in self._data:
            return default
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key, value):
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)
'''
_STRINGS_TEST_OLD = '''from strings import title_case


def test_title_case():
    assert title_case("a b") == "A B"
'''
_STRINGS_TEST_NEW = '''from strings import title_case


def test_title_case_empty():
    assert title_case("") == ""


def test_title_case():
    assert title_case("a b") == "A B"
'''

# 7: a rename (refactor) mixed with one genuinely new branch; tests only follow the rename
_REPORT_OLD = '''def _fmt(v):
    return "" if v is None else str(v)


def render(rows):
    lines = []
    for row in rows:
        lines.append(",".join(_fmt(v) for v in row))
    return "\\n".join(lines)
'''
_REPORT_NEW = '''def _format_value(value):
    return "" if value is None else str(value)


def render(rows, strict=False):
    lines = []
    for row in rows:
        if strict and len(row) != len(rows[0]):
            raise ValueError("ragged row")
        lines.append(",".join(_format_value(v) for v in row))
    return "\\n".join(lines)
'''
_REPORT_TEST_OLD = '''from report import _fmt


def test_none_is_empty():
    assert _fmt(None) == ""
'''
_REPORT_TEST_NEW = _REPORT_TEST_OLD.replace("_fmt", "_format_value")

# 8: new function fully covered by a parametrized test
_TEXT_OLD = '''def shout(text):
    return text.upper()
'''
_TEXT_NEW = '''import re


def slugify(text, max_len=None):
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if max_len is not None:
        slug = slug[:max_len].rstrip("-")
    return slug


def shout(text):
    return text.upper()
'''
_TEXT_TEST_OLD = '''from text import shout


def test_shout():
    assert shout("a") == "A"
'''
_TEXT_TEST_NEW = '''import pytest

from text import shout, slugify


def test_shout():
    assert shout("a") == "A"


@pytest.mark.parametrize(
    "text, max_len, expected",
    [
        ("Hello, World!", None, "hello-world"),
        ("  spaced  out  ", None, "spaced-out"),
        ("Hello, World!", 7, "hello-w"),
        ("Hello, World!", 6, "hello"),
        ("", None, ""),
    ],
)
def test_slugify(text, max_len, expected):
    assert slugify(text, max_len) == expected
'''

# 9: loop rewritten as a comprehension; behaviour identical, no test change
_STATS_OLD = '''def mean_positive(values):
    total = 0
    count = 0
    for v in values:
        if v > 0:
            total += v
            count += 1
    return total / count if count else 0.0
'''
_STATS_NEW = '''def mean_positive(values):
    positives = [v for v in values if v > 0]
    return sum(positives) / len(positives) if positives else 0.0
'''

# 10: only logging and unused constants
_HTTP_OLD = '''import requests


def fetch(url):
    resp = requests.get(url, timeout=10)
    return resp.text
'''
_HTTP_NEW = '''import requests
import logging

logger = logging.getLogger(__name__)
RETRY_LIMIT = 3
DEFAULT_TIMEOUT = 30


def fetch(url):
    logger.debug("fetching %s", url)
    resp = requests.get(url, timeout=10)
    logger.debug("got status %s", resp.status_code)
    return resp.text
'''

CASES = [
    {
        "name": "untested_logic", "level": "easy", "expect": "flag",
        "why": "new function with branches and an error path, no tests at all",
        "title": "Add apply_discount()",
        "files": [("src/pricing.py", _DISCOUNT_SRC)],
    },
    {
        "name": "tested", "level": "easy", "expect": "clean",
        "why": "same function, but the PR adds tests for every branch",
        "title": "Add apply_discount() with tests",
        "files": [("src/pricing.py", _DISCOUNT_SRC),
                  ("tests/test_pricing.py", _DISCOUNT_TESTS)],
    },
    {
        "name": "refactor_only", "level": "medium", "expect": "clean",
        "why": "renamed variables and added type hints, behaviour unchanged",
        "title": "Rename variables and add type hints in tax()",
        "files": [("src/pricing.py", _RENAME_SRC)],
    },
    {
        "name": "wrong_tests", "level": "hard", "expect": "flag",
        "why": "new except branch; the only changed test covers a different function",
        "title": "Make load() tolerate invalid JSON",
        "files": [("src/parser.py", _LOAD_SRC),
                  ("tests/test_parser.py", _UNRELATED_TESTS)],
    },
    {
        "name": "off_by_one_fix", "level": "hard", "expect": "flag",
        "why": "one-line change that alters the returned slice end, no test added",
        "title": "Fix page_bounds dropping the last item of each page",
        "files": [_file("src/pagination.py", _PAGE_OLD, _PAGE_NEW)],
    },
    {
        "name": "new_class_wrong_tests", "level": "medium", "expect": "flag",
        "why": "new class with eviction and an error path; the changed test is for "
               "strings.py",
        "title": "Add an LRU cache",
        "files": [_file("src/cache.py", "", _CACHE_NEW),
                  _file("tests/test_strings.py", _STRINGS_TEST_OLD, _STRINGS_TEST_NEW)],
    },
    {
        "name": "refactor_plus_new_branch", "level": "hard", "expect": "flag",
        "why": "rename plus a new `strict` branch that raises; the test change only "
               "follows the rename",
        "title": "Rename _fmt and add strict mode to render()",
        "files": [_file("src/report.py", _REPORT_OLD, _REPORT_NEW),
                  _file("tests/test_report.py", _REPORT_TEST_OLD, _REPORT_TEST_NEW)],
    },
    {
        "name": "parametrized_tests", "level": "hard", "expect": "clean",
        "why": "new slugify() with a branch; the added parametrized test exercises "
               "both branches and the empty string",
        "title": "Add slugify()",
        "files": [_file("src/text.py", _TEXT_OLD, _TEXT_NEW),
                  _file("tests/test_text.py", _TEXT_TEST_OLD, _TEXT_TEST_NEW)],
    },
    {
        "name": "equivalent_rewrite", "level": "hard", "expect": "clean",
        "why": "loop replaced by a comprehension; same inputs give the same outputs",
        "title": "Simplify mean_positive()",
        "files": [_file("src/stats.py", _STATS_OLD, _STATS_NEW)],
    },
    {
        "name": "logging_and_constants", "level": "medium", "expect": "clean",
        "why": "only debug logging and constants nothing uses; behaviour unchanged",
        "title": "Add debug logging to fetch()",
        "files": [_file("src/http_client.py", _HTTP_OLD, _HTTP_NEW)],
    },
]

LEVELS = ("easy", "medium", "hard")


def _build_pr(case: dict) -> PullRequest:
    files = []
    for name, patch, *rest in case["files"]:
        lines = patch.splitlines()
        files.append(PRFile(
            filename=name, status=rest[0] if rest else "modified",
            additions=sum(1 for ln in lines if ln.startswith("+")),
            deletions=sum(1 for ln in lines if ln.startswith("-")),
            patch=patch,
        ))
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
            row["analyses"] = rep.get("_analyses", [])
            row["dropped"] = rep.get("_dropped", [])
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


def _show(rows: list[dict] | None, verbose: bool = False) -> str:
    out = []
    for r in rows or []:
        c = r["case"]
        got = "ERROR" if r["error"] else ("flagged" if r["findings"] else "clean")
        verdict = "ok" if r["ok"] else "WRONG"
        out.append(f"[{verdict}] {c['name']} ({c['level']}): expected {c['expect']}, "
                   f"got {got} ({r['seconds']:.1f}s)  - {c['why']}")
        if r["error"]:
            out.append(f"      error: {r['error']}")
        for f in r["findings"]:
            out.append(f"      evidence: {f.evidence.strip()}")
            out.append(f"      says:     {f.message}")
        # findings the evidence filter removed: the model quoted a line that is not in the diff
        for d in r.get("dropped", []):
            out.append(f"      DROPPED (quote not in diff): {d.evidence.strip()!r}")
            out.append(f"                  it said: {d.message}")
        if verbose:
            for name, analysis in r.get("analyses", []):
                out.append(f"      analysis: {analysis or '(none returned)'}")
    return "\n".join(out)


def _print_matrix(outcomes, cases) -> None:
    names = [o.provider for o in outcomes]
    print("\nCase-by-case (the diff between backends):")
    header = ["case", "level", "expected"] + names
    table = []
    for i, c in enumerate(cases):
        row = [c["name"], c["level"], c["expect"]]
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

    # score by difficulty, so "3/4 overall" doesn't hide where a model fails
    print("\nCorrect by difficulty:")
    for lvl in LEVELS:
        idx = [i for i, c in enumerate(cases) if c["level"] == lvl]
        if not idx:
            continue
        parts = []
        for o in outcomes:
            if o.error or o.result is None:
                parts.append(f"{o.provider}: FAILED")
            else:
                parts.append(f"{o.provider}: {sum(o.result[i]['ok'] for i in idx)}/{len(idx)}")
        print(f"  {lvl:<7} " + "   ".join(parts))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tests.integration.compare_check1")
    ap.add_argument("--provider", choices=[p for p in PROVIDERS if p != "auto"],
                    help="run one backend only (default: local and groq)")
    ap.add_argument("--only", choices=[c["name"] for c in CASES],
                    help="run a single case")
    ap.add_argument("--level", choices=LEVELS, help="run only one difficulty")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="also print each model's analysis text")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="[%(name)s] %(message)s")
    cases = [c for c in CASES
             if (not args.only or c["name"] == args.only)
             and (not args.level or c["level"] == args.level)]
    providers = [args.provider] if args.provider else ["local", "groq"]

    outcomes = run_with_each(_make_runner(cases), providers)
    print_comparison(outcomes, show=lambda rows: _show(rows, verbose=args.verbose))
    if len(outcomes) > 1:
        _print_matrix(outcomes, cases)
    print("\nNote: the first local call includes loading the model into GPU "
          "memory, so its time is inflated. Groq's free tier has a per-minute "
          "token cap, so a full run may pause while the client waits it out.")


if __name__ == "__main__":
    main()