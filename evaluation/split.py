"""
split.py — divide the labelled PRs into dev and test, once.

    python -m evaluation.split --seed <seed_number>

The script refuses to overwrite an existing split (so you cannot re-roll it
after seeing results) and records a fingerprint of the test labels. run_eval
checks that fingerprint, which proves the test labels were not edited after the
split was made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from typing import Optional

from evaluation.labels import CHECKS, load_all, problems, scored_checks

SPLIT_FILE = Path(__file__).parent / "split.json"


def stratum(label: dict) -> str:
    pos = [c for c in CHECKS if label["checks"][c]["expect"] == "finding"]
    return f"{label['repo']}|{'+'.join(pos) or 'negative'}"


def make_split(labels: dict[str, dict], seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for lid, lab in sorted(labels.items()):
        if scored_checks(lab):                       # PRs with nothing to score are left out
            groups.setdefault(stratum(lab), []).append(lid)
    sides: dict[str, list[str]] = {"dev": [], "test": []}
    flip = rng.random() < 0.5
    for key in sorted(groups):
        ids = groups[key]
        rng.shuffle(ids)
        for i, lid in enumerate(ids):
            sides["dev" if (i % 2 == 0) == flip else "test"].append(lid)
        if len(ids) % 2:
            flip = not flip                           # odd groups: alternate who gets the extra one
    return {k: sorted(v) for k, v in sides.items()}


def fingerprint(ids: list[str], labels: Optional[dict] = None) -> str:
    """A hash of what the test labels say, so any edit after the split is noticed."""
    labels = load_all() if labels is None else labels
    h = hashlib.sha256()
    for lid in sorted(ids):
        h.update(lid.encode())
        h.update(json.dumps(labels[lid], sort_keys=True).encode())
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Split labelled PRs into dev and test.")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--force", action="store_true", help="overwrite an existing split (don't)")
    args = ap.parse_args(argv)
    if SPLIT_FILE.exists() and not args.force:
        print(f"{SPLIT_FILE} exists. The split is made once so the test set stays unseen.",
              file=sys.stderr)
        return 1
    labels = load_all()
    bad = [(lid, p) for lid, lab in labels.items() for p in problems(lab)]
    if bad:
        print("fix these labels first (python -m evaluation.labels validate):", file=sys.stderr)
        for lid, p in bad[:10]:
            print(f"  {lid}: {p}", file=sys.stderr)
        return 1
    sides = make_split(labels, args.seed)
    SPLIT_FILE.write_text(json.dumps({"seed": args.seed, **sides,
                                      "test_fingerprint": fingerprint(sides["test"])}, indent=2) + "\n")
    print(f"dev={len(sides['dev'])}  test={len(sides['test'])}. Commit and tag this now.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
