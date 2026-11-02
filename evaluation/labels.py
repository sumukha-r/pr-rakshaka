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

import csv
import json
import sys
from pathlib import Path
from typing import Optional

LABEL_DIR = Path(__file__).parent / "labels"
LABEL_CSV = Path(__file__).parent / "eval_set.csv"      # the whole labelled set in one file
CSV_COLUMNS = ["id", "repo", "number", "url", "title", "merged_at", "base_sha", "snapshot_sha",
               "edge_case", "check", "expect", "target_file", "evidence_contains",
               "source_url", "source_quote", "note"]
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
            if not c.get("targets"):
                out.append(f"{name}: a 'finding' needs at least one target")
            if not (c.get("source", {}).get("quote") or "").strip():
                out.append(f"{name}: a 'finding' needs the reviewer's quote in source.quote")
    return out


def from_csv(path: Path) -> dict[str, dict]:
    """Labels from the one-file format: one row per (PR, check).

    A check with no row is 'unclear'. Several rows for the same PR and check, all
    'finding', are several targets. A blank target_file means "any file in the PR";
    a blank evidence_contains means "any line in that file"."""
    labels: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lid = row["id"]
            lab = labels.setdefault(lid, {
                "id": lid, "repo": row["repo"], "number": int(row["number"]), "url": row["url"],
                "title": row["title"], "merged_at": row["merged_at"], "base_sha": row["base_sha"],
                "snapshot_sha": row["snapshot_sha"], "edge_case": row["edge_case"],
                "checks": {c: blank_check() for c in CHECKS}})
            check = row["check"]
            if check not in CHECKS:
                continue
            c = lab["checks"][check]
            c["expect"] = row["expect"]
            c["note"] = row["note"]
            if row["expect"] == "finding":
                c["targets"].append({"file": row["target_file"],
                                     "evidence_contains": row["evidence_contains"]})
                c["source"] = {"url": row["source_url"], "quote": row["source_quote"]}
    return labels


def to_csv_rows(label: dict) -> list[dict]:
    """The rows for one label (checks that are 'unclear' are left out)."""
    rows = []
    base = {k: label.get(k, "") for k in ("id", "repo", "number", "url", "title", "merged_at",
                                          "base_sha", "snapshot_sha", "edge_case")}
    for check in CHECKS:
        c = label["checks"][check]
        if c["expect"] == "unclear":
            continue
        targets = c["targets"] if c["expect"] == "finding" else [{}]
        for t in targets:
            rows.append({**base, "check": check, "expect": c["expect"],
                         "target_file": t.get("file", ""), "evidence_contains": t.get("evidence_contains", ""),
                         "source_url": c.get("source", {}).get("url", ""),
                         "source_quote": c.get("source", {}).get("quote", ""), "note": c.get("note", "")})
    return rows


def write_csv(labels: dict[str, dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for lid in sorted(labels):
            w.writerows(to_csv_rows(labels[lid]))


def load_all(directory: Optional[Path] = None) -> dict[str, dict]:
    """Every label: the CSV set, plus any single JSON files (a JSON file wins for the same PR)."""
    labels: dict[str, dict] = {}
    if Path(LABEL_CSV).exists():
        labels.update(from_csv(Path(LABEL_CSV)))
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
