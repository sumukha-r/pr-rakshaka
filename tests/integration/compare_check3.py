"""compare_check3.py — compare Check 3 on hand-written sanity PRs.

These fixtures test whether the hot-path check distinguishes plausible
performance problems from ordinary or justified code. They are quick prompt /
provider sanity checks, not the curated real-PR evaluation dataset.

    python -m tests.integration.compare_check3                    # local vs Groq
    python -m tests.integration.compare_check3 --provider local   # one backend
    python -m tests.integration.compare_check3 --only n_plus_one  # one case
    python -m tests.integration.compare_check3 --level hard       # one level
    python -m tests.integration.compare_check3 -v                 # include analysis

Run from the repository root. "flag" means at least one evidence-backed
finding is expected; "clean" means no finding is expected.
"""
from __future__ import annotations

import argparse
import logging
import time

from src.checks.hot_path_smells import check_hot_path_smells
from src.ingestion.pr_parser import PRFile, PullRequest
from src.llm.client import PROVIDERS, LLMError
from src.llm.compare import print_comparison, run_with_each

LEVELS = ("easy", "medium", "hard")


def _patch(*lines: str) -> str:
    """Create a GitHub-style all-added hunk from source lines."""
    return f"@@ -0,0 +1,{len(lines)} @@\n" + "\n".join("+" + line for line in lines)


# Keep the fixtures intentionally small and readable. They are smoke tests,
# not a substitute for reviewed real-PR examples.
CASES = [
    {
        "name": "n_plus_one", "level": "easy", "expect": "flag",
        "why": "one database operation is issued for every item",
        "title": "Load users", "file": "src/users.py",
        "lines": [
            "def load_users(user_ids):",
            "    users = []",
            "    for user_id in user_ids:",
            "        users.append(db.query_user(user_id))",
            "    return users",
        ],
    },
    {
        "name": "blocking_async", "level": "medium", "expect": "flag",
        "why": "time.sleep blocks the event loop inside an async function",
        "title": "Poll service status", "file": "src/polling.py",
        "lines": [
            "async def poll_status():",
            "    time.sleep(2)",
            "    return await fetch_status()",
        ],
    },
    {
        "name": "invariant_regex", "level": "medium", "expect": "flag",
        "why": "the same regular expression is compiled for each item",
        "title": "Match values", "file": "src/matching.py",
        "lines": [
            "def match_all(pattern, values):",
            "    results = []",
            "    for value in values:",
            "        results.append(re.compile(pattern).search(value))",
            "    return results",
        ],
    },
    {
        "name": "quadratic_search", "level": "medium", "expect": "flag",
        "why": "membership in a list is repeated for every item in another list",
        "title": "Find common values", "file": "src/collections.py",
        "lines": [
            "def common(left, right):",
            "    return [item for item in left if item in right]",
        ],
    },
    {
        "name": "repeated_json_parse", "level": "hard", "expect": "flag",
        "why": "the same JSON document is parsed repeatedly inside a loop",
        "title": "Build event records", "file": "src/events.py",
        "lines": [
            "def build_records(payload, event_ids):",
            "    records = []",
            "    for event_id in event_ids:",
            "        data = json.loads(payload)",
            "        records.append(data[event_id])",
            "    return records",
        ],
    },
    {
        "name": "large_materialization", "level": "hard", "expect": "flag",
        "why": "a large intermediate list is built only to be iterated once",
        "title": "Process records", "file": "src/records.py",
        "lines": [
            "def process_records(records):",
            "    prepared = list(transform(record) for record in records)",
            "    for record in prepared:",
            "        write_record(record)",
        ],
    },
    {
        "name": "ordinary_loop", "level": "easy", "expect": "clean",
        "why": "a simple linear transformation is not itself a performance smell",
        "title": "Double values", "file": "src/math_utils.py",
        "lines": [
            "def double(values):",
            "    for value in values:",
            "        yield value * 2",
        ],
    },
    {
        "name": "cold_small_constant", "level": "easy", "expect": "clean",
        "why": "a tiny fixed tuple conversion is not a meaningful hot-path issue",
        "title": "Return labels", "file": "src/labels.py",
        "lines": [
            "def labels():",
            '    return list(("one", "two", "three"))',
        ],
    },
    {
        "name": "set_membership", "level": "medium", "expect": "clean",
        "why": "the lookup collection is already a set, so membership is efficient",
        "title": "Filter allowed values", "file": "src/filters.py",
        "lines": [
            "def allowed_values(values, allowed):",
            "    allowed_set = set(allowed)",
            "    return [value for value in values if value in allowed_set]",
        ],
    },
    {
        "name": "compiled_regex", "level": "medium", "expect": "clean",
        "why": "the regular expression is compiled once before iteration",
        "title": "Match values efficiently", "file": "src/matching.py",
        "lines": [
            "def match_all(pattern, values):",
            "    compiled = re.compile(pattern)",
            "    return [compiled.search(value) for value in values]",
        ],
    },
    {
        "name": "async_sleep", "level": "hard", "expect": "clean",
        "why": "asyncio.sleep yields control instead of blocking the event loop",
        "title": "Poll service status asynchronously", "file": "src/polling.py",
        "lines": [
            "async def poll_status():",
            "    await asyncio.sleep(2)",
            "    return await fetch_status()",
        ],
    },
    {
        "name": "required_materialization", "level": "hard", "expect": "clean",
        "why": "the API contract explicitly requires returning a list",
        "title": "Return sorted results", "file": "src/results.py",
        "lines": [
            "def sorted_results(records):",
            "    return list(sorted(records, key=lambda record: record.created_at))",
        ],
    },
]


def _build_pr(case: dict) -> PullRequest:
    patch = _patch(*case["lines"])
    additions = len(case["lines"])
    f = PRFile(filename=case["file"], status="added", additions=additions,
               deletions=0, patch=patch)
    return PullRequest(
        repo="demo/demo", number=1, title=case["title"], author="demo",
        state="open", merged=False, body="", additions=additions,
        deletions=0, changed_files=1, files=[f],
    )


def _make_runner(cases: list[dict]):
    """Run all selected fixtures once per backend, aggregating check metrics."""
    def run_all(report: dict) -> list[dict]:
        rows = []
        correct = raw = dropped = calls = files_considered = radon_signals = 0
        for case in cases:
            row = {"case": case, "findings": [], "error": "", "seconds": 0.0}
            rep: dict = {}
            start = time.monotonic()
            try:
                row["findings"] = check_hot_path_smells(_build_pr(case), report=rep)
            except LLMError as exc:
                row["error"] = str(exc)
            row["seconds"] = time.monotonic() - start
            row["analyses"] = rep.get("_analyses", [])
            row["dropped_findings"] = rep.get("_dropped", [])
            row["radon_details"] = rep.get("_radon_details", [])
            row["report"] = rep
            row["ok"] = (not row["error"]) and (
                bool(row["findings"]) == (case["expect"] == "flag"))
            correct += int(row["ok"])
            raw += rep.get("raw_findings", 0)
            dropped += rep.get("dropped", 0)
            calls += rep.get("llm_calls", 0)
            files_considered += rep.get("files_considered", 0)
            radon_signals += rep.get("radon_signals", 0)
            rows.append(row)
        report.update(correct=f"{correct}/{len(cases)}", files_considered=files_considered,
                      llm_calls=calls, raw_findings=raw, dropped=dropped,
                      radon_signals=radon_signals)
        return rows
    return run_all


def _show(rows: list[dict] | None, verbose: bool = False) -> str:
    out = []
    for row in rows or []:
        case = row["case"]
        got = "ERROR" if row["error"] else ("flagged" if row["findings"] else "clean")
        verdict = "ok" if row.get("ok") else "WRONG"
        out.append(f"[{verdict}] {case['name']} ({case['level']}): expected {case['expect']}, "
                   f"got {got} ({row['seconds']:.1f}s) — {case['why']}")
        if row["error"]:
            out.append(f"      error: {row['error']}")
        rep = row.get("report", {})
        out.append("      metrics: "
                    f"files={rep.get('files_considered', 0)}, "
                    f"LLM calls={rep.get('llm_calls', 0)}, "
                    f"Radon signals={rep.get('radon_signals', 0)}, "
                    f"raw findings={rep.get('raw_findings', 0)}, "
                    f"dropped={rep.get('dropped', 0)}")
        for finding in row["findings"]:
            out.append(f"      evidence: {finding.evidence.strip()}")
            out.append(f"      says:     {finding.message}")
        for finding in row.get("dropped_findings", []):
            out.append(f"      DROPPED (quote not in diff): {finding.evidence.strip()!r}")
            out.append(f"                  it said: {finding.message}")
        if verbose:
            for filename, signal in row.get("radon_details", []):
                out.append(f"      Radon context [{filename}]: {signal}")
            for filename, analysis in row.get("analyses", []):
                out.append(f"      analysis [{filename}]: {analysis or '(none returned)'}")
    return "\n".join(out)


def _print_matrix(outcomes, cases: list[dict]) -> None:
    print("\nCase-by-case (the diff between backends):")
    header = ["case", "level", "expected"] + [o.provider for o in outcomes]
    table = []
    for i, case in enumerate(cases):
        row = [case["name"], case["level"], case["expect"]]
        for outcome in outcomes:
            if outcome.error or outcome.result is None:
                row.append("FAILED")
                continue
            result = outcome.result[i]
            got = "ERROR" if result["error"] else ("flagged" if result["findings"] else "clean")
            row.append(f"{'ok' if result.get('ok') else 'WRONG'} ({got})")
        table.append(row)
    widths = [max(len(head), *(len(row[i]) for row in table)) for i, head in enumerate(header)]
    fmt = lambda cells: "  ".join(cell.ljust(width) for cell, width in zip(cells, widths))
    print(fmt(header))
    print(fmt(["-" * width for width in widths]))
    for row in table:
        print(fmt(row))

    print("\nCorrect by difficulty:")
    for level in LEVELS:
        subset = [i for i, case in enumerate(cases) if case["level"] == level]
        if not subset:
            continue
        scores = []
        for outcome in outcomes:
            if outcome.error or outcome.result is None:
                scores.append(f"{outcome.provider}: FAILED")
            else:
                score = sum(bool(outcome.result[i].get("ok")) for i in subset)
                scores.append(f"{outcome.provider}: {score}/{len(subset)}")
        print(f"  {level:<7} " + "   ".join(scores))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tests.integration.compare_check3")
    parser.add_argument("--provider", choices=[p for p in PROVIDERS if p != "auto"],
                        help="run one backend only (default: local and groq)")
    parser.add_argument("--only", choices=[case["name"] for case in CASES],
                        help="run one fixture")
    parser.add_argument("--level", choices=LEVELS, help="run only one difficulty")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="include model analysis and Radon context")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="[%(name)s] %(message)s")
    cases = [case for case in CASES
             if (not args.only or case["name"] == args.only)
             and (not args.level or case["level"] == args.level)]
    providers = [args.provider] if args.provider else ["local", "groq"]
    outcomes = run_with_each(_make_runner(cases), providers)
    print_comparison(outcomes, show=lambda rows: _show(rows, verbose=args.verbose))
    if len(outcomes) > 1:
        _print_matrix(outcomes, cases)
    print("\nNote: these hand-written fixtures are initial sanity checks, not a curated "
          "benchmark. A future versioned real-PR dataset should be evaluated separately.")
    print("The first local call may include model-loading time; provider token limits can add pauses.")


if __name__ == "__main__":
    main()
