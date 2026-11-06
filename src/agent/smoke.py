"""
smoke.py — check that tool calling works end to end on a real PR and a real model.

    python -m src.agent.smoke https://github.com/owner/repo/pull/123 --provider groq
    python -m src.agent.smoke <PR URL> --provider local --trace trace.json

The task is deliberately small: for each changed source file, find the tests that
cover it. It exercises find_tests and read_file, prints every tool call, and shows
the tokens and time used. Run it once per backend before building on the loop:
whether a given local model supports tool calling at all is worth knowing early.
"""

from __future__ import annotations

import argparse
import json
import os

from src.agent.cache import Cache
from src.agent.github import GitHubReader
from src.agent.loop import Budget, run_agent
from src.agent.tools import AgentContext, RepoTools, make_index_search
from src.ingestion.pr_parser import fetch_pr_from_url
from src.llm.client import LLMError
from src.llm.prompts import is_test_file

SYSTEM = (
    "You are a code reviewer's assistant. For each changed Python source file in the pull request, "
    "find the test files that exercise it, using the tools. Confirm a test really calls the changed "
    "code with read_file(pattern=...) before listing it.\n"
    'Final answer, one JSON object: {"files": [{"source": "<path>", "tests": ["<path>"], '
    '"evidence": "<one line from a test that calls the code, or empty>"}]}'
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pr_url")
    ap.add_argument("--provider", choices=["auto", "local", "groq"])
    ap.add_argument("--model")
    ap.add_argument("--trace", help="write the full trace to this JSON file")
    args = ap.parse_args(argv)

    pr = fetch_pr_from_url(args.pr_url, os.getenv("GITHUB_TOKEN"))
    reader = GitHubReader(cache=Cache("data/agent_cache"))
    ctx = AgentContext(pr, allow_comments=False,
                       index_search=make_index_search(pr.repo, None))
    sources = [f.filename for f in pr.files if f.filename.endswith(".py") and not is_test_file(f.filename)]
    user = f"Pull request: {pr.title}\nChanged source files:\n" + "\n".join(f"- {s}" for s in sources[:5])
    try:
        res = run_agent(SYSTEM, user, RepoTools(ctx, reader), provider=args.provider,
                        model=args.model, budget=Budget(max_steps=6))
    except LLMError as exc:
        print(f"model error: {exc}")
        return 1
    for ev in res.trace.events:
        if ev["kind"] == "tool":
            print(f"  step {ev['step']}: {ev['name']}({json.dumps(ev['args'])})  -> {ev['chars']} chars")
    print(f"\nstopped: {res.stop} after {res.steps} step(s), {res.tool_calls} tool call(s), "
          f"{res.prompt_tokens + res.completion_tokens} tokens, {res.seconds}s ({res.provider}/{res.model})")
    print(json.dumps(res.data, indent=2) if res.data else f"no valid JSON answer. Raw reply:\n{res.text}")
    if args.trace:
        res.trace.write(args.trace)
        print(f"trace written to {args.trace}")
    return 0 if res.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
