"""
run_eval.py — run the pipeline over a split and save every review.

    python -m evaluation.run_eval --split dev  --provider groq
    python -m evaluation.run_eval --split dev  --provider local
    python -m evaluation.run_eval --split test --provider groq --final     # once

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

from evaluation.labels import load_all, scored_checks
from evaluation.split import SPLIT_FILE, fingerprint

RUN_DIR = Path(__file__).parent / "runs"


def run_dir(split: str, provider: str, root: Optional[Path] = None) -> Path:
    return Path(root or RUN_DIR) / split / provider


def run_one(label: dict, provider: str, *, review_fn: Optional[Callable] = None,
            ref_ctx=None) -> dict:
    """Review one labelled PR. Returns the Review as a dict, or {"error": ...}."""
    if review_fn is None:
        from main import run_review as review_fn
    if ref_ctx is None:
        from src.rag.indexer import use_index_ref as ref_ctx
    checks = scored_checks(label)
    try:
        with ref_ctx(label["base_sha"] if "reuse" in checks else None):
            review = review_fn(label["url"], checks=checks, provider=provider,
                               index_if_missing=True)
        return review.to_dict()
    except Exception as exc:  # noqa: BLE001 - record it, keep going with the other PRs
        return {"error": f"{type(exc).__name__}: {exc}", "checks": []}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the pipeline over the dev or test split.")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--provider", choices=["groq", "local"], required=True)
    ap.add_argument("--final", action="store_true", help="required for the test split")
    ap.add_argument("--force", action="store_true", help="re-run PRs that already have output")
    args = ap.parse_args(argv)

    if not SPLIT_FILE.exists():
        print("no split yet: python -m evaluation.split --seed N", file=sys.stderr)
        return 1
    split = json.loads(SPLIT_FILE.read_text())
    ids = split[args.split]
    out = run_dir(args.split, args.provider)

    if args.split == "test":
        if not args.final:
            print("the test split needs --final: run it once, after your last prompt change",
                  file=sys.stderr)
            return 1
        if fingerprint(ids) != split["test_fingerprint"]:
            print("test labels changed since the split was made; this is no longer an unseen set",
                  file=sys.stderr)
            return 1
        if out.exists() and any(out.glob("*.json")) and not args.force:
            print(f"{out} already has results; the test set is run once per provider",
                  file=sys.stderr)
            return 1

    labels = load_all()
    out.mkdir(parents=True, exist_ok=True)
    for n, lid in enumerate(ids, 1):
        path = out / f"{lid}.json"
        if path.exists() and not args.force:
            continue
        print(f"[{n}/{len(ids)}] {lid}", flush=True)
        path.write_text(json.dumps(run_one(labels[lid], args.provider), indent=2), encoding="utf-8")
    print(f"done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
