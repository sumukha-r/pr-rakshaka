"""
compare_check2.py — run Check 2 (reuse) on eight hand-written cases with known
answers, through each backend, and show where they differ.

Retrieval is replaced by fixed candidates, so this tests only the judgement
(and the exact-match shortcut), not the embeddings. It needs no index, no
GitHub and no token. Not the evaluation dataset; that comes from real PRs.

    python -m tests.integration.compare_check2                    # local vs Groq
    python -m tests.integration.compare_check2 --provider local
    python -m tests.integration.compare_check2 --only same_job_different_impl
    python -m tests.integration.compare_check2 --level hard -v

Run from the repo root. Each case has an expected answer:
    flag  = at least one finding (and, where stated, pointing at the right candidate)
    clean = no findings
"""

import argparse
import logging
import time
from dataclasses import dataclass

from src.checks.reuse_detector import check_reuse
from src.ingestion.pr_parser import PRFile, PullRequest
from src.llm.client import PROVIDERS, LLMError
from src.llm.compare import print_comparison, run_with_each
from tests.integration.compare_check1 import LEVELS, _diff, _print_matrix


@dataclass
class Chunk:
    """Stands in for src.rag.retriever.RetrievedChunk."""
    filepath: str
    symbol: str
    source: str
    start_line: int
    end_line: int
    similarity: float


def _chunk(path: str, symbol: str, source: str, similarity: float) -> Chunk:
    return Chunk(path, symbol, source, 10, 10 + len(source.splitlines()) - 1, similarity)


# --- existing code the "search" returns -------------------------------------

_GET_JSON = '''def get_json(url, timeout=10):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data
'''
_FLATTEN_LIST = '''def flatten_list(nested):
    for item in nested:
        if isinstance(item, (list, tuple)):
            yield from flatten_list(item)
        else:
            yield item
'''
_NORMALIZE_USERNAME = '''def normalize_username(name):
    name = name.strip().lower()
    name = name.replace(" ", "_")
    if len(name) > 32:
        name = name[:32]
    return name
'''
_LOAD_USER = '''def load_user(uid):
    row = db.query("SELECT * FROM users WHERE id = ?", uid)
    if row is None:
        raise KeyError(uid)
    return User(**row)
'''
_BATCH = '''def batch(seq, size):
    for start in range(0, len(seq), size):
        yield seq[start:start + size]
'''
_CUMULATIVE_MEAN = '''def cumulative_mean(values):
    total = 0.0
    means = []
    for i, v in enumerate(values, start=1):
        total += v
        means.append(total / i)
    return means
'''
_PARSE_INT = '''def parse_int(text, default=0):
    try:
        return int(text.strip())
    except ValueError:
        return default
'''
_STR_TO_BOOL = '''def str_to_bool(value):
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"not a boolean: {value!r}")
'''
_PARSE_FLOAT = '''def parse_float(text, default=0.0):
    try:
        return float(text.strip())
    except ValueError:
        return default
'''
_READ_TEXT = '''def read_text(path, encoding="utf-8"):
    with open(path, encoding=encoding) as fh:
        return fh.read()
'''

# --- new code in the PR ------------------------------------------------------

_FETCH_JSON = '''def fetch_json(link, wait=10):
    reply = requests.get(link, timeout=wait)
    reply.raise_for_status()
    payload = reply.json()
    return payload
'''
_FLATTEN = '''def flatten(items):
    stack = list(reversed(items))
    result = []
    while stack:
        current = stack.pop()
        if isinstance(current, (list, tuple)):
            stack.extend(reversed(current))
        else:
            result.append(current)
    return result
'''
_NORMALIZE_EMAIL = '''def normalize_email(address):
    address = address.strip().lower()
    if "@" not in address:
        raise ValueError("not an email address")
    local, _, domain = address.partition("@")
    return f"{local}@{domain}"
'''
_GET_USER_NAME = '''def get_user_name(uid):
    user = load_user(uid)
    return user.name
'''
_CHUNKED = '''def chunked(items, n):
    out = []
    for i in range(0, len(items), n):
        out.append(items[i:i + n])
    return out
'''
_ROLLING_MEAN = '''def rolling_mean(values, window):
    means = []
    for i in range(len(values) - window + 1):
        chunk = values[i:i + window]
        means.append(sum(chunk) / window)
    return means
'''
_PARSE_BOOL = '''def parse_bool(s):
    s = s.lower().strip()
    truthy = {"true", "yes", "1", "on"}
    falsy = {"false", "no", "0", "off"}
    if s in truthy:
        return True
    if s in falsy:
        return False
    raise ValueError("invalid boolean: " + s)
'''
_CHECKSUM = '''def compute_checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()
'''

CASES = [
    {
        "name": "exact_copy_renamed", "level": "easy", "expect": "flag",
        "why": "same code as get_json with every name changed; reported without an LLM call",
        "title": "Add fetch_json()", "file": "src/net.py", "new": _FETCH_JSON,
        "candidates": [_chunk("src/http_utils.py", "def get_json", _GET_JSON, 0.97)],
    },
    {
        "name": "same_job_different_impl", "level": "hard", "expect": "flag",
        "why": "iterative flatten does what the recursive flatten_list generator does",
        "title": "Add flatten()", "file": "src/lists.py", "new": _FLATTEN,
        "candidates": [_chunk("src/iterutils.py", "def flatten_list", _FLATTEN_LIST, 0.84)],
    },
    {
        "name": "same_shape_different_job", "level": "hard", "expect": "clean",
        "why": "both lowercase and strip a string, but one validates an email and the "
               "other builds a username slug",
        "title": "Add normalize_email()", "file": "src/mail.py", "new": _NORMALIZE_EMAIL,
        "candidates": [_chunk("src/users.py", "def normalize_username",
                              _NORMALIZE_USERNAME, 0.81)],
    },
    {
        "name": "delegates_to_existing", "level": "medium", "expect": "clean",
        "why": "the new function calls load_user(); it reuses it, it does not copy it",
        "title": "Add get_user_name()", "file": "src/profile.py", "new": _GET_USER_NAME,
        "candidates": [_chunk("src/users.py", "def load_user", _LOAD_USER, 0.78)],
    },
    {
        "name": "reimplements_repo_helper", "level": "medium", "expect": "flag",
        "why": "chunked() splits a sequence into chunks, exactly what batch() already does",
        "title": "Add chunked()", "file": "src/export.py", "new": _CHUNKED,
        "candidates": [_chunk("src/iterutils.py", "def batch", _BATCH, 0.86)],
    },
    {
        "name": "similar_math_different_output", "level": "hard", "expect": "clean",
        "why": "rolling_mean uses a fixed window, cumulative_mean grows from the start: "
               "different results",
        "title": "Add rolling_mean()", "file": "src/stats.py", "new": _ROLLING_MEAN,
        "candidates": [_chunk("src/metrics.py", "def cumulative_mean",
                              _CUMULATIVE_MEAN, 0.88)],
    },
    {
        "name": "right_candidate_of_three", "level": "hard", "expect": "flag",
        "expect_related": "str_to_bool",
        "why": "parse_bool does what str_to_bool does; the other two candidates are "
               "parsers for other types, so the finding must point at candidate 2",
        "title": "Add parse_bool()", "file": "src/settings.py", "new": _PARSE_BOOL,
        "candidates": [_chunk("src/parse.py", "def parse_int", _PARSE_INT, 0.83),
                       _chunk("src/parse.py", "def str_to_bool", _STR_TO_BOOL, 0.80),
                       _chunk("src/parse.py", "def parse_float", _PARSE_FLOAT, 0.79)],
    },
    {
        "name": "no_equivalent", "level": "easy", "expect": "clean",
        "why": "hashing a file has nothing in common with reading its text",
        "title": "Add compute_checksum()", "file": "src/files.py", "new": _CHECKSUM,
        "candidates": [_chunk("src/io_utils.py", "def read_text", _READ_TEXT, 0.76)],
    },
]


def _build_pr(case: dict) -> PullRequest:
    patch = _diff("", case["new"])
    f = PRFile(filename=case["file"], status="added",
               additions=len(case["new"].splitlines()), deletions=0, patch=patch)
    return PullRequest(repo="demo/demo", number=1, title=case["title"], author="demo",
                       state="open", merged=False, body="", additions=f.additions,
                       deletions=0, changed_files=1, files=[f])


def _make_runner(cases: list[dict]):
    def run_all(report: dict) -> list[dict]:
        rows, correct = [], 0
        totals = {"exact": 0, "raw_findings": 0, "dropped": 0}
        for case in cases:
            row = {"case": case, "findings": [], "error": "", "seconds": 0.0}
            rep: dict = {}
            start = time.monotonic()
            try:
                row["findings"] = check_reuse(
                    _build_pr(case), report=rep,
                    retrieve=lambda repo, source, filename, top_k, c=case: list(c["candidates"]))
            except LLMError as exc:
                row["error"] = str(exc)
            row["seconds"] = time.monotonic() - start
            row["analyses"] = rep.get("_analyses", [])
            row["dropped"] = rep.get("_dropped", [])
            flagged = bool(row["findings"])
            right_target = (not case.get("expect_related")) or any(
                case["expect_related"] in f.related for f in row["findings"])
            row["ok"] = (not row["error"]) and flagged == (case["expect"] == "flag") \
                and (not flagged or right_target)
            row["wrong_target"] = flagged and case["expect"] == "flag" and not right_target
            correct += row["ok"]
            for k in totals:
                totals[k] += rep.get(k, 0)
            rows.append(row)
        report.update(correct=f"{correct}/{len(cases)}", **totals)
        return rows
    return run_all


def _show(rows: list[dict] | None, verbose: bool = False) -> str:
    out = []
    for r in rows or []:
        c = r["case"]
        got = "ERROR" if r["error"] else ("flagged" if r["findings"] else "clean")
        if r.get("wrong_target"):
            got += " (WRONG candidate)"
        out.append(f"[{'ok' if r['ok'] else 'WRONG'}] {c['name']} ({c['level']}): expected "
                   f"{c['expect']}, got {got} ({r['seconds']:.1f}s)  - {c['why']}")
        if r["error"]:
            out.append(f"      error: {r['error']}")
        for f in r["findings"]:
            out.append(f"      [{f.kind}] new:      {f.evidence.strip()}")
            out.append(f"                existing: {f.related}")
            out.append(f"                says:     {f.message}")
        for d in r.get("dropped", []):
            out.append(f"      DROPPED ({d.related}): quote {d.evidence.strip()!r}")
            out.append(f"                it said: {d.message}")
        if verbose:
            for name, analysis in r.get("analyses", []):
                out.append(f"      analysis: {analysis or '(none returned)'}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tests.integration.compare_check2")
    ap.add_argument("--provider", choices=[p for p in PROVIDERS if p != "auto"],
                    help="run one backend only (default: local and groq)")
    ap.add_argument("--only", choices=[c["name"] for c in CASES], help="run a single case")
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
    print("\nNote: 'exact' cases are decided in Python and make no LLM call. The first "
          "local call includes loading the model into GPU memory.")


if __name__ == "__main__":
    main()