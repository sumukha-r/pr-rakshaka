"""
run_eval.py — run the pipeline over a split and save every review.

    python -m evaluation.run_eval --split dev  --provider groq
    python -m evaluation.run_eval --split dev  --provider local
    python -m evaluation.run_eval --split test --provider groq --final     # once
    python -m evaluation.run_eval --split dev  --provider groq --mode agent   # v2 agent checks

The test split is protected: it needs --final, it checks the test labels were
not edited since the split was made, and it will not run a second time for the
same provider.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

from evaluation.labels import load_all, scored_checks
from evaluation.split import SPLIT_FILE, fingerprint, fingerprint_key, split_names

RUN_DIR = Path(__file__).parent / "runs"


def run_dir(split: str, provider: str, root: Optional[Path] = None) -> Path:
    return Path(root or RUN_DIR) / split / provider


def _snapshot_pr(label: dict):
    from evaluation.github_api import GitHub
    from evaluation.snapshot import build_pr
    return build_pr(GitHub(), label)


def run_one(label: dict, provider: str, *, review_fn: Optional[Callable] = None,
            ref_ctx=None, pr_builder: Optional[Callable] = None,
            runners: Optional[dict] = None) -> dict:
    """Review one labelled PR as of its snapshot commit. Returns the Review as a dict,
    or {"error": ...}."""
    if review_fn is None:
        from main import run_review as review_fn
    if ref_ctx is None:
        from src.rag.indexer import use_index_ref as ref_ctx
    checks = scored_checks(label)
    try:
        pr = (pr_builder or _snapshot_pr)(label)
        with ref_ctx(label["base_sha"] if "reuse" in checks else None):
            extra = {"runners": runners} if runners else {}
            review = review_fn(pr, checks=checks, provider=provider,
                               index_if_missing=True, **extra)
        return review.to_dict()
    except Exception as exc:  # noqa: BLE001 - record it, keep going with the other PRs
        return {"error": f"{type(exc).__name__}: {exc}", "checks": []}


def needs_retry(path: Path) -> bool:
    """True if a saved review is an error, unreadable, or has a check that failed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    return "error" in data or any(c.get("status") == "failed" for c in data.get("checks", []))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the pipeline over the dev or test split.")
    ap.add_argument("--split", choices=split_names(), required=True)
    ap.add_argument("--provider", choices=["groq", "local", "remote"], required=True)
    ap.add_argument("--mode", choices=["pipeline", "agent", "agent-m3"], default="pipeline",
                    help="pipeline = v1 checks; agent = v2 agent checks (saved under <provider>-agent); "
                         "agent-m3 = the same plus the smell agent with caller tracing "
                         "(saved under <provider>-agent-m3)")
    ap.add_argument("--final", action="store_true", help="required for the test split")
    ap.add_argument("--force", action="store_true", help="re-run PRs that already have output")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-run only PRs whose saved output has an error or a failed check "
                         "(finish a run after a rate limit without re-spending tokens on the rest)")
    args = ap.parse_args(argv)

    if not SPLIT_FILE.exists():
        print("no split yet: python -m evaluation.split --seed N", file=sys.stderr)
        return 1
    split = json.loads(SPLIT_FILE.read_text())
    ids = split[args.split]
    out = run_dir(args.split, args.provider + {"agent": "-agent", "agent-m3": "-agent-m3"}.get(args.mode, ""))

    if args.split.startswith("test"):
        if not args.final:
            print(f"the {args.split} split needs --final: run it once, after your last prompt change",
                  file=sys.stderr)
            return 1
        if fingerprint(ids) != split[fingerprint_key(args.split)]:
            print("test labels changed since the split was made; this is no longer an unseen set",
                  file=sys.stderr)
            return 1
        if out.exists() and any(out.glob("*.json")) and not args.force:
            print(f"{out} already has results; the test set is run once per provider",
                  file=sys.stderr)
            return 1

    labels = load_all()
    out.mkdir(parents=True, exist_ok=True)
    runners = None
    if args.mode in ("agent", "agent-m3"):
        from src.agent.checks import agent_runners
        # No fallback to v1 here: a failed agent must show up as "not run", not be scored as v1.
        runners = agent_runners(trace_dir=out / "traces", fallback=False, smell_agent=args.mode == "agent-m3")
    for n, lid in enumerate(ids, 1):
        path = out / f"{lid}.json"
        if path.exists() and not args.force and not (args.retry_failed and needs_retry(path)):
            continue
        print(f"[{n}/{len(ids)}] {lid}", flush=True)
        path.write_text(json.dumps(run_one(labels[lid], args.provider, runners=runners), indent=2), encoding="utf-8")
    print(f"done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
