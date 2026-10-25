"""
reuse_detector.py — Check 2: does newly added code re-implement something that
already exists in the repository?

For each new function or class in the PR diff:
  1. RAG: fetch the most similar existing chunks from the indexed repo
     (the repo must be indexed first: python src/rag/indexer.py owner/repo)
  2. Drop the function's own earlier version (same file, same name) and anything
     below the similarity floor
  3. Same code apart from names? Report it directly: that is certain, so it
     needs no LLM call
  4. Otherwise the LLM reads the new code next to the candidates and says which
     candidate, if any, does the same job
  5. Two-sided evidence: the quoted line must be in the diff AND the candidate
     number must be one we actually showed. Otherwise the finding is dropped.

Usage:
    python -m src.checks.reuse_detector <PR URL>
    python -m src.checks.reuse_detector <PR URL> --provider groq
    python -m src.checks.reuse_detector <PR URL> --compare        # local vs Groq
"""

import logging
import os
from typing import Callable, Optional

from src.ingestion.diff_defs import AddedDef, added_definitions
from src.ingestion.pr_parser import PullRequest
from src.llm.aggregator import Finding, keep_evidenced
from src.llm.client import chat_json
from src.llm.prompts import REUSE_SYSTEM, build_reuse_prompt, is_test_file
from src.rag.fingerprint import structurally_identical

log = logging.getLogger(__name__)

CHECK_NAME = "reuse"
TOP_K = 3                    # candidates shown to the LLM per new definition
MIN_SIMILARITY = 0.75        # below this a candidate is not worth an LLM call
DUPLICATE_SIMILARITY = 0.92  # at or above: reported as "duplicate", else "reuse"
MAX_DEFS = 5                 # LLM calls per PR; protects the per-minute token limit

# retrieve(repo, source, filename, top_k) -> list of chunks with .filepath, .symbol,
# .source, .start_line, .end_line, .similarity (src.rag.retriever.RetrievedChunk)
Retriever = Callable[[str, str, str, int], list]


def _default_retrieve(repo: str, source: str, filename: str, top_k: int) -> list:
    from src.rag.retriever import retrieve_similar   # lazy: loads chromadb and the embedding model
    return retrieve_similar(repo, source, top_k=top_k)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_name(symbol: str) -> str:
    """'def Session.get' -> 'get'"""
    return symbol.split(" ", 1)[-1].split(".")[-1]


def _is_self(chunk, d: AddedDef) -> bool:
    """The function's own earlier (or, for a merged PR, current) version in the index."""
    return chunk.filepath == d.filename and _last_name(chunk.symbol) == d.name.split(".")[-1]


def _related(c) -> str:
    return f"{c.filepath}:{c.start_line}-{c.end_line} {c.symbol} (similarity {c.similarity:.2f})"


def _kind(c) -> str:
    return "duplicate" if c.similarity >= DUPLICATE_SIMILARITY else "reuse"


def _as_int(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _collect_defs(pr: PullRequest) -> list[AddedDef]:
    defs: list[AddedDef] = []
    for f in pr.files:
        if (f.filename.endswith(".py") and f.patch
                and f.status not in ("removed", "deleted")
                and not is_test_file(f.filename)):
            defs.extend(added_definitions(f.filename, f.patch))
    return defs


def _review_def(pr: PullRequest, d: AddedDef, cands: list
                ) -> tuple[list[Finding], list[Finding], str]:
    """Ask the LLM about one new definition. Returns (findings, invalid, analysis)."""
    prompt = build_reuse_prompt(pr.title, d, cands)
    log.info("reuse: sending %s in %s (~%d tokens)", d.name, d.filename,
             (len(prompt) + len(REUSE_SYSTEM)) // 4)

    reply = chat_json(REUSE_SYSTEM, prompt)
    analysis = reply.get("analysis")
    analysis = analysis if isinstance(analysis, str) else ""

    raw = reply.get("findings", [])
    if not isinstance(raw, list):
        log.warning("reuse: 'findings' was not a list for %s", d.name)
        return [], [], analysis

    findings: list[Finding] = []
    invalid: list[Finding] = []
    used: set[int] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        evidence, message = item.get("evidence"), item.get("message")
        if not (isinstance(evidence, str) and isinstance(message, str)):
            continue
        n = _as_int(item.get("candidate"))
        if n is None or not 1 <= n <= len(cands):
            # points at code we never showed: cannot be trusted
            invalid.append(Finding(CHECK_NAME, d.filename, evidence, message,
                                   related=f"(no such candidate: {item.get('candidate')!r})"))
            continue
        if n in used:
            continue
        used.add(n)
        c = cands[n - 1]
        findings.append(Finding(CHECK_NAME, d.filename, evidence, message,
                                related=_related(c), kind=_kind(c)))
    return findings, invalid, analysis


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

def check_reuse(pr: PullRequest, *, retrieve: Optional[Retriever] = None,
                max_defs: int = MAX_DEFS, report: Optional[dict] = None) -> list[Finding]:
    """
    Run Check 2 over a PR. Returns only evidence-backed findings.
    If `report` is given it is filled with counts (defs_found, with_candidates,
    exact, llm_calls, raw_findings, dropped) so backends can be compared.
    """
    retrieve = retrieve or _default_retrieve
    patches = {f.filename: f.patch for f in pr.files if f.patch}

    defs = _collect_defs(pr)
    findings: list[Finding] = []
    to_review: list[tuple[float, AddedDef, list]] = []
    exact = 0
    within_pr = 0
    seen: list[AddedDef] = []

    for d in defs:
        # Two copies added in the same PR: neither is in the index yet, so the
        # search cannot see them. Compare new definitions with each other
        # (exact matches only), and report the second copy.
        first = next((s for s in seen if structurally_identical(d.source, s.source)), None)
        seen.append(d)
        if first:
            within_pr += 1
            findings.append(Finding(
                CHECK_NAME, d.filename, d.first_line,
                f"`{d.name}` has exactly the same logic as `{first.name}`, which this PR "
                f"also adds in {first.filename}; only names, type hints or comments "
                "differ. Keep one and reuse it.",
                related=f"{first.filename}:{first.start_line}-{first.end_line} "
                        f"{first.kind} {first.name} (added in this PR)",
                kind="structural_duplicate"))
            continue

        hits = [c for c in retrieve(pr.repo, d.source, d.filename, TOP_K + 1)
                if not _is_self(c, d)]
        cands = sorted((c for c in hits if c.similarity >= MIN_SIMILARITY),
                       key=lambda c: -c.similarity)[:TOP_K]
        if not cands:
            continue

        twin = next((c for c in cands if structurally_identical(d.source, c.source)), None)
        if twin:
            exact += 1
            findings.append(Finding(
                CHECK_NAME, d.filename, d.first_line,
                f"`{d.name}` has exactly the same logic as existing "
                f"`{twin.symbol.split(' ', 1)[-1]}` in {twin.filepath}; only names, "
                "type hints or comments differ. Reuse it instead of copying it.",
                related=_related(twin), kind="structural_duplicate"))
            continue
        to_review.append((cands[0].similarity, d, cands))

    to_review.sort(key=lambda t: -t[0])
    if len(to_review) > max_defs:
        log.info("reuse: %d new definitions have candidates, reviewing the %d most similar",
                 len(to_review), max_defs)
    reviewed = to_review[:max_defs]

    llm_findings: list[Finding] = []
    invalid: list[Finding] = []
    analyses: list[tuple[str, str]] = []
    for _, d, cands in reviewed:
        fs, bad, analysis = _review_def(pr, d, cands)
        llm_findings.extend(fs)
        invalid.extend(bad)
        analyses.append((f"{d.name} ({d.filename})", analysis))

    kept, dropped = keep_evidenced(findings + llm_findings, patches)
    dropped = dropped + invalid

    if report is not None:
        report.update(defs_found=len(defs), with_candidates=exact + len(to_review),
                      exact=exact, within_pr=within_pr, llm_calls=len(reviewed),
                      raw_findings=len(llm_findings) + len(invalid), dropped=len(dropped))
        report["_analyses"] = analyses
        report["_dropped"] = dropped
    for d in dropped:
        log.debug("reuse: dropped %s | evidence=%r | %s", d.filename, d.evidence, d.related)
    return kept


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _format(findings: list[Finding]) -> str:
    if not findings:
        return "No reuse findings."
    return "\n".join(
        f"{f.filename}  [{f.kind}]\n  new code: {f.evidence.strip()}\n"
        f"  existing: {f.related}\n  {f.message}\n"
        for f in findings
    )


def _warn_if_not_indexed(repo: str) -> None:
    try:
        import chromadb
        from src.rag.indexer import CHROMA_PATH, _collection_name
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        names = [getattr(c, "name", c) for c in client.list_collections()]
    except Exception:  # noqa: BLE001 - a hint only; never block the run
        return
    if _collection_name(repo) not in names:
        print(f"WARNING: {repo} is not indexed, so nothing can be matched. "
              f"Run: python src/rag/indexer.py {repo}")


if __name__ == "__main__":
    import argparse

    from src.ingestion.pr_parser import fetch_pr_from_url
    from src.llm.client import PROVIDERS, use_provider
    from src.llm.compare import print_comparison, run_with_each

    ap = argparse.ArgumentParser(prog="python -m src.checks.reuse_detector")
    ap.add_argument("url", help="GitHub PR URL")
    ap.add_argument("--provider", choices=PROVIDERS, help="force a backend")
    ap.add_argument("--compare", action="store_true",
                    help="run the PR through local and Groq and compare")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    pr = fetch_pr_from_url(args.url, token=os.getenv("GITHUB_TOKEN"))
    print(f"PR #{pr.number}: {pr.title} ({len(pr.files)} files)")
    _warn_if_not_indexed(pr.repo)

    if args.compare:
        print_comparison(run_with_each(lambda rep: check_reuse(pr, report=rep)), show=_format)
    elif args.provider:
        with use_provider(args.provider):
            print(_format(check_reuse(pr)))
    else:
        print(_format(check_reuse(pr)))