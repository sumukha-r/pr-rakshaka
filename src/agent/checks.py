"""
checks.py — the missing-tests and reuse checks, rebuilt as agents.

Same contract as v1 so they drop into main.run_review through `runners`:
a runner takes (pr, report) and returns evidence-backed Findings. What changes is
how the model decides. v1 shows it a fixed prompt; here it may first look: find the
tests that cover a file, read them, search for existing code. Everything it says
still has to survive the same filters as v1 (the quote must be a line the PR adds,
the substance gate, one-per-file), plus one new rule for reuse: it may only cite an
existing function that a tool or the starting points actually showed it.

    runners = agent_runners()                      # missing_tests + reuse as agents, smell as v1
    run_review(pr, runners=runners, index_if_missing=True)

`fallback=True` runs the v1 check when the agent fails (right for live use). For an
evaluation leave it off, so a failed agent shows up as 'not run' instead of quietly
being scored as v1.
"""

from __future__ import annotations

import logging
import re
import types
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

from src.agent.cache import Cache
from src.agent.github import GitHubReader, ReaderError
from src.agent.loop import AgentFailed, AgentResult, Budget, run_agent, with_fallback
from src.agent.prompts import (MISSING_TESTS_AGENT_SYSTEM, REUSE_AGENT_SYSTEM, SMELL_AGENT_SYSTEM,
                               build_agent_reuse_prompt)
from src.agent.tools import AgentContext, RepoTools, make_index_search, make_similar_search
from src.checks.hot_path_smells import MAX_FILES as MAX_SMELL_FILES
from src.checks.hot_path_smells import _candidate_files as _smell_files
from src.checks.hot_path_smells import _finding_from as _smell_finding
from src.checks.hot_path_smells import _signals as _smell_signals
from src.checks.missing_tests import MAX_FINDINGS_PER_FILE, MAX_SOURCE_FILES, MAX_TEST_FILES, _candidate_files
from src.checks.reuse_detector import _already_calls, _is_self, _last_name
from src.ingestion.diff_defs import AddedDef
from src.ingestion.pr_parser import PullRequest
from src.llm.aggregator import Finding, is_substantive, keep_evidenced, limit_per_file
from src.checks.smell_catalog import catalogue_text
from src.llm.prompts import build_hot_path_prompt, build_missing_tests_prompt, is_test_file
from src.rag.fingerprint import structurally_identical

log = logging.getLogger(__name__)

CACHE_DIR = "data/agent_cache"
MAX_AGENT_DEFS = 4                 # reuse agent runs per PR
MIN_STARTING_SIMILARITY = 0.6
DEFAULT_BUDGET = Budget(max_steps=6, max_tool_calls=10, max_total_tokens=40_000, max_seconds=180)


class _Usage:
    """Totals across the agent runs of one check, for the report and the comparison with v1."""
    def __init__(self):
        self.steps = self.tool_calls = self.tokens = self.runs = self.failures = 0
        self.stops: Counter = Counter()

    def add(self, r: AgentResult) -> None:
        self.runs += 1
        self.steps += r.steps
        self.tool_calls += r.tool_calls
        self.tokens += r.prompt_tokens + r.completion_tokens
        self.stops[r.stop] += 1

    def into(self, report: Optional[dict]) -> None:
        if report is not None:
            report.update(agent_runs=self.runs, agent_steps=self.steps, agent_tool_calls=self.tool_calls,
                          agent_tokens=self.tokens, agent_failures=self.failures,
                          agent_stops=",".join(f"{k}:{v}" for k, v in sorted(self.stops.items())))


def _save_trace(trace_dir: Optional[Path], pr: PullRequest, name: str, r: AgentResult) -> None:
    if trace_dir:
        r.trace.write(str(Path(trace_dir) / f"{pr.repo.replace('/', '__')}__{pr.number}__{name}.json"))


def _agent_call(system: str, user: str, tools: RepoTools, budget: Budget, usage: _Usage,
                pr: PullRequest, name: str, trace_dir: Optional[Path], min_tool_calls: int = 0) -> Optional[dict]:
    """One agent run. Returns the parsed answer, or None (counted as a failure) if it had none."""
    r = run_agent(system, user, tools, budget=budget, min_tool_calls=min_tool_calls)
    usage.add(r)
    _save_trace(trace_dir, pr, name, r)
    if not r.ok:
        usage.failures += 1
        log.info("%s: no valid JSON answer (%s)", name, r.stop)
        return None
    return r.data


def _analysis(data: dict) -> str:
    a = data.get("analysis")
    return a if isinstance(a, str) else ""


# ---------------------------------------------------------------------------
# Missing tests
# ---------------------------------------------------------------------------

def agent_missing_tests(pr: PullRequest, reader: GitHubReader, *, budget: Optional[Budget] = None,
                        trace_dir: Optional[Path] = None, report: Optional[dict] = None) -> list[Finding]:
    budget = budget or DEFAULT_BUDGET
    tools = RepoTools(AgentContext(pr, allow_comments=False), reader)
    test_files = [(f.filename, f.patch) for f in pr.files
                  if f.patch and is_test_file(f.filename) and f.status not in ("removed", "deleted")][:MAX_TEST_FILES]
    candidates = _candidate_files(pr)[:MAX_SOURCE_FILES]
    usage, findings, analyses = _Usage(), [], []
    for f in candidates:
        prompt = build_missing_tests_prompt(pr.title, f.filename, f.patch, test_files)
        data = _agent_call(MISSING_TESTS_AGENT_SYSTEM, prompt, tools, budget, usage, pr,
                           f"missing_tests__{Path(f.filename).stem}", trace_dir, min_tool_calls=1)
        if data is None:
            continue
        analyses.append((f.filename, _analysis(data)))
        raw = data.get("findings", [])
        for item in raw if isinstance(raw, list) else []:
            if isinstance(item, dict) and isinstance(item.get("evidence"), str) and isinstance(item.get("message"), str):
                findings.append(Finding("missing_tests", f.filename, item["evidence"], item["message"]))
    if candidates and usage.failures == len(candidates):
        raise AgentFailed("the agent gave no valid answer for any file")

    patches = {f.filename: f.patch for f in candidates}
    kept, dropped = keep_evidenced(findings, patches)
    weak = [f for f in kept if not is_substantive(f)]
    kept = limit_per_file([f for f in kept if is_substantive(f)], MAX_FINDINGS_PER_FILE)
    dropped = dropped + weak
    if report is not None:
        report.update(files_sent=len(candidates), raw_findings=len(findings), dropped=len(dropped))
        usage.into(report)
        report["_analyses"], report["_dropped"] = analyses, dropped
    return kept


# ---------------------------------------------------------------------------
# Reuse
# ---------------------------------------------------------------------------

def _parse_existing(item: dict) -> tuple[str, str]:
    """'path::def name' -> (path, symbol); a bare path gives an empty symbol."""
    ex = item.get("existing")
    if not isinstance(ex, str) or not ex.strip():
        return "", ""
    path, _, symbol = ex.strip().partition("::")
    return path.strip(), symbol.strip()


def _review_def(pr: PullRequest, d: AddedDef, tools: RepoTools, start: list[dict], budget: Budget,
                usage: _Usage, trace_dir: Optional[Path]) -> tuple[list[Finding], list[Finding], str]:
    for c in start:                       # what the model was shown counts as seen
        tools.seen_paths.add(c["path"])
        tools.seen_chunks[f"{c['path']}::{c['symbol']}"] = c
    data = _agent_call(REUSE_AGENT_SYSTEM, build_agent_reuse_prompt(pr.title, d, start), tools, budget,
                       usage, pr, f"reuse__{d.name.replace('.', '_')}", trace_dir)
    if data is None:
        return [], [], ""
    raw = data.get("findings", [])
    findings, invalid, used = [], [], set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        evidence, message = item.get("evidence"), item.get("message")
        if not (isinstance(evidence, str) and isinstance(message, str)):
            continue
        path, symbol = _parse_existing(item)
        if not path or path not in tools.seen_paths:
            invalid.append(Finding("reuse", d.filename, evidence, message,
                                   related=f"(cited code no tool showed: {item.get('existing')!r})"))
            continue
        cand = types.SimpleNamespace(filepath=path, symbol=symbol or "def ?")
        if symbol and _is_self(cand, d):
            invalid.append(Finding("reuse", d.filename, evidence, message, related="(cites the new code itself)"))
            continue
        if symbol and _already_calls(d, cand):
            invalid.append(Finding("reuse", d.filename, evidence, message,
                                   related=f"(the new code already calls {_last_name(symbol)})"))
            continue
        if (path, symbol) in used:
            continue
        used.add((path, symbol))
        hit = tools.seen_chunks.get(f"{path}::{symbol}")
        related = (f"{path}:{hit['start_line']}-{hit['end_line']} {symbol}" if hit else f"{path} {symbol}".strip())
        findings.append(Finding("reuse", d.filename, evidence, message, related=related, kind="reuse"))
    return findings, invalid, _analysis(data)


def agent_reuse(pr: PullRequest, reader: GitHubReader, *, budget: Optional[Budget] = None,
                trace_dir: Optional[Path] = None, report: Optional[dict] = None,
                fetch_source: Optional[Callable[[str, str, str], Optional[str]]] = None) -> list[Finding]:
    from src.checks.reuse_detector import _collect_defs
    budget = budget or DEFAULT_BUDGET
    similar = make_similar_search(pr.repo, None)
    ctx = AgentContext(pr, allow_comments=False, index_search=make_index_search(pr.repo, None),
                       similar_search=similar)
    tools = RepoTools(ctx, reader)

    def fetch(repo: str, path: str, ref: str) -> Optional[str]:
        try:
            return reader.file_text(repo, path, ref)
        except ReaderError:
            return None                    # those blocks are simply not reviewed, as in v1

    defs, fetched = _collect_defs(pr, fetch_source or fetch)
    patches = {f.filename: f.patch for f in pr.files if f.patch}
    findings: list[Finding] = []
    todo: list[tuple[AddedDef, list[dict]]] = []
    seen: list[AddedDef] = []
    exact = within_pr = 0
    for d in defs:
        first = next((s for s in seen if structurally_identical(d.source, s.source)), None)
        seen.append(d)
        if first:                          # the same code added twice in this PR: certain, no model needed
            within_pr += 1
            findings.append(Finding("reuse", d.filename, d.first_line,
                f"`{d.name}` has exactly the same logic as `{first.name}`, which this PR also adds in "
                f"{first.filename}; only names, type hints or comments differ. Keep one and reuse it.",
                related=f"{first.filename}:{first.start_line}-{first.end_line} {first.kind} {first.name} "
                        "(added in this PR)", kind="structural_duplicate"))
            continue
        start = [c for c in (similar(d.source, 4) if similar else [])
                 if c["similarity"] >= MIN_STARTING_SIMILARITY
                 and not _is_self(types.SimpleNamespace(filepath=c["path"], symbol=c["symbol"]), d)][:3]
        twin = next((c for c in start if structurally_identical(d.source, c["snippet"])), None)
        if twin:                           # same code apart from names: certain, no model needed
            exact += 1
            findings.append(Finding("reuse", d.filename, d.first_line,
                f"`{d.name}` has exactly the same logic as existing `{twin['symbol'].split(' ', 1)[-1]}` in "
                f"{twin['path']}; only names, type hints or comments differ. Reuse it instead of copying it.",
                related=f"{twin['path']}:{twin['start_line']}-{twin['end_line']} {twin['symbol']}",
                kind="structural_duplicate"))
            continue
        if sum(1 for ln in d.source.splitlines() if ln.strip()) >= 3:
            todo.append((d, start))

    todo.sort(key=lambda t: -len(t[0].source))
    usage, llm_findings, invalid, analyses = _Usage(), [], [], []
    for d, start in todo[:MAX_AGENT_DEFS]:
        fs, bad, analysis = _review_def(pr, d, tools, start, budget, usage, trace_dir)
        llm_findings += fs
        invalid += bad
        analyses.append((f"{d.name} ({d.filename})", analysis))
    if todo and usage.runs and usage.failures == usage.runs and not findings:
        raise AgentFailed("the agent gave no valid answer for any new definition")

    kept, dropped = keep_evidenced(findings + llm_findings, patches)
    dropped = dropped + invalid
    if report is not None:
        report.update(defs_found=len(defs), blocks_found=sum(d.kind == "block" for d in defs),
                      files_fetched=fetched, exact=exact, within_pr=within_pr,
                      agent_defs=min(len(todo), MAX_AGENT_DEFS),
                      raw_findings=len(llm_findings) + len(invalid), dropped=len(dropped))
        usage.into(report)
        report["_analyses"], report["_dropped"] = analyses, dropped
    return kept


# ---------------------------------------------------------------------------
# Hot-path smell
# ---------------------------------------------------------------------------

_DEF_RE = re.compile(r"^\+\s*(?:async\s+)?def\s+([A-Za-z_]\w*)", re.M)
_HUNK_DEF_RE = re.compile(r"^@@[^@]*@@\s*(?:async\s+)?def\s+([A-Za-z_]\w*)", re.M)
MAX_CALLER_HINTS = 3


def _changed_function_names(patch: str) -> list[str]:
    """Functions the diff adds or edits (defined on a `+` line, or named in a hunk header), in order."""
    names: list[str] = []
    for n in _HUNK_DEF_RE.findall(patch) + _DEF_RE.findall(patch):
        if n not in names and not n.startswith("__"):
            names.append(n)
    return names[:MAX_CALLER_HINTS]


def agent_smell(pr: PullRequest, reader: GitHubReader, *, budget: Optional[Budget] = None,
                trace_dir: Optional[Path] = None, report: Optional[dict] = None) -> list[Finding]:
    """v1's smell check, plus tools to trace callers (is this code really on a hot path?)."""
    budget = budget or DEFAULT_BUDGET
    ctx = AgentContext(pr, allow_comments=False, index_search=make_index_search(pr.repo, None))
    tools = RepoTools(ctx, reader)
    files = _smell_files(pr)[:MAX_SMELL_FILES]
    usage, findings, analyses = _Usage(), [], []
    for f in files:
        try:
            full = reader.file_text(pr.repo, f.filename, pr.head_sha) if pr.head_sha else None
        except ReaderError:
            full = None                                     # only the radon hint is lost
        prompt = build_hot_path_prompt(pr.title, f.filename, f.patch or "", _smell_signals(f, full),
                                       catalogue=catalogue_text())
        changed = _changed_function_names(f.patch or "")
        if changed:      # the model must look at who calls what changed, not just guess from the diff
            prompt += ("\n\nFunctions this diff adds or changes: " + ", ".join(changed) +
                       ". Their callers are not in the diff: check them with find_callers before you "
                       "answer.")
        data = _agent_call(SMELL_AGENT_SYSTEM, prompt, tools, budget, usage, pr,
                           f"smell__{Path(f.filename).stem}", trace_dir,
                           min_tool_calls=1 if changed else 0)
        if data is None:
            continue
        analyses.append((f.filename, _analysis(data)))
        raw = data.get("findings", [])
        for item in raw if isinstance(raw, list) else []:
            finding = _smell_finding(item, f.filename) if isinstance(item, dict) else None
            if finding:
                findings.append(finding)
    if files and usage.failures == len(files):
        raise AgentFailed("the agent gave no valid answer for any file")

    patches = {f.filename: f.patch for f in pr.files if f.patch}
    kept, dropped = keep_evidenced(findings, patches)
    weak = [f for f in kept if not is_substantive(f)]
    kept = [f for f in kept if is_substantive(f)]
    dropped = dropped + weak
    if report is not None:
        report.update(files_considered=len(files), raw_findings=len(findings), dropped=len(dropped),
                      kept_findings=len(kept))
        usage.into(report)
        report["_analyses"], report["_dropped"] = analyses, dropped
    return kept


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def agent_runners(*, reader: Optional[GitHubReader] = None, budget: Optional[Budget] = None,
                  trace_dir: Optional[Path] = None, fallback: bool = False,
                  smell_agent: bool = False) -> dict:
    """Runners for main.run_review: missing_tests and reuse as agents; smell is v1 unless
    `smell_agent` (milestone 3: the smell check with caller tracing)."""
    from main import default_runners
    base = default_runners()
    reader = reader or GitHubReader(cache=Cache(CACHE_DIR))
    if trace_dir:
        Path(trace_dir).mkdir(parents=True, exist_ok=True)

    def guard(name: str, run: Callable[[PullRequest, dict], list]) -> Callable[[PullRequest, dict], list]:
        if not fallback:
            return run

        def guarded(pr: PullRequest, rep: dict) -> list:
            value, why = with_fallback(lambda: run(pr, rep), lambda: base[name](pr, rep))
            if why:
                rep["fallback"] = why
            return value
        return guarded

    return {
        "missing_tests": guard("missing_tests", lambda pr, rep: agent_missing_tests(
            pr, reader, budget=budget, trace_dir=trace_dir, report=rep)),
        "reuse": guard("reuse", lambda pr, rep: agent_reuse(
            pr, reader, budget=budget, trace_dir=trace_dir, report=rep)),
        "smell": guard("smell", lambda pr, rep: agent_smell(
            pr, reader, budget=budget, trace_dir=trace_dir, report=rep)) if smell_agent else base["smell"],
    }
    