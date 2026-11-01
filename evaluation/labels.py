"""
labels.py — the label files (evaluation/labels/<owner>__<repo>__<number>.json).

    python -m evaluation.labels stub https://github.com/owner/repo/pull/123
    python -m evaluation.labels validate

per check:
  finding       the tool should report something here (needs targets + a reviewer quote)
  none          the tool should stay silent
  unclear       you are not sure: left out of scoring
  out_of_scope  the problem is in code the PR does not add, so the tool cannot be
                blamed for missing it: left out of scoring
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

LABEL_DIR = Path(__file__).parent / "labels"
CHECKS = ("missing_tests", "reuse", "smell")
EXPECTS = ("finding", "none", "unclear", "out_of_scope")
SCORED = ("finding", "none")


def label_id(repo: str, number: int) -> str:
    return f"{repo.replace('/', '__')}__{number}"


def blank_check() -> dict:
    return {"expect": "unclear", "targets": [], "source": {"url": "", "quote": ""}, "note": ""}


def stub(pr_meta: dict, repo: str, snapshot_sha: Optional[str] = None) -> dict:
    """A label template from GitHub's /pulls/N JSON.

    `snapshot_sha` is the commit the PR is reviewed at (see snapshot.py). By default
    it is the PR's last commit; for a positive, use the commit the reviewer saw."""
    number = pr_meta["number"]
    return {
        "id": label_id(repo, number), "repo": repo, "number": number,
        "url": f"https://github.com/{repo}/pull/{number}", "title": pr_meta.get("title", ""),
        "merged_at": (pr_meta.get("merged_at") or "")[:10],
        "base_sha": (pr_meta.get("base") or {}).get("sha", ""),
        "snapshot_sha": snapshot_sha or (pr_meta.get("head") or {}).get("sha", ""),
        "edge_case": "",
        "checks": {c: blank_check() for c in CHECKS},
    }


def problems(label: dict) -> list[str]:
    """Everything wrong with one label, as readable sentences (empty list = fine)."""
    out = []
    for key in ("id", "repo", "number", "url", "base_sha", "snapshot_sha"):
        if not label.get(key):
            out.append(f"missing '{key}'")
    checks = label.get("checks", {})
    for name in CHECKS:
        c = checks.get(name)
        if c is None:
            out.append(f"{name}: missing")
            continue
        if c.get("expect") not in EXPECTS:
            out.append(f"{name}: expect must be one of {', '.join(EXPECTS)}")
        if c.get("expect") == "finding":
            if not c.get("targets") or not all(t.get("file") for t in c["targets"]):
                out.append(f"{name}: a 'finding' needs at least one target with a file")
            if not (c.get("source", {}).get("quote") or "").strip():
                out.append(f"{name}: a 'finding' needs the reviewer's quote in source.quote")
    return out


def load_all(directory: Optional[Path] = None) -> dict[str, dict]:
    labels = {}
    for path in sorted(Path(directory or LABEL_DIR).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        labels[data.get("id") or path.stem] = data
    return labels


def scored_checks(label: dict) -> list[str]:
    """The checks of this PR that count: labelled 'finding' or 'none'."""
    return [c for c in CHECKS if label["checks"][c]["expect"] in SCORED]


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "stub":
        from evaluation.github_api import GitHub
        from src.ingestion.pr_parser import parse_pr_url
        repo, number = parse_pr_url(argv[1])
        data = stub(GitHub().pr(repo, number), repo)
        LABEL_DIR.mkdir(parents=True, exist_ok=True)
        path = LABEL_DIR / f"{data['id']}.json"
        if path.exists():
            print(f"{path} already exists; not overwriting it", file=sys.stderr)
            return 1
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}: now read the PR and edit the file")
        return 0
    if argv == ["validate"]:
        labels = load_all()
        bad = 0
        for lid, lab in labels.items():
            for msg in problems(lab):
                bad += 1
                print(f"{lid}: {msg}")
        counts = {c: {e: sum(1 for l in labels.values() if l['checks'][c]['expect'] == e)
                      for e in EXPECTS} for c in CHECKS}
        print(f"{len(labels)} label file(s), {bad} problem(s)")
        for c, row in counts.items():
            print(f"  {c:14s} " + "  ".join(f"{e}={n}" for e, n in row.items()))
        return 1 if bad else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
