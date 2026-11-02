"""
label_candidates.py — turn candidates.csv into label files, quickly, in the terminal.

    python -m evaluation.label_candidates evaluation/candidates.csv --limit 25
    python -m evaluation.label_candidates evaluation/candidates.csv --hint smell

For each candidate it shows the reviewer's comment and the code it was about, and
asks one question. You answer in a keystroke; a label file is written. Rows that
already have a label are skipped.

"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable, Optional

from evaluation import labels as label_store
from evaluation.labels import CHECKS, blank_check, label_id
from evaluation.snapshot import added_lines_of
from src.llm.prompts import is_test_file

HINT_TO_CHECK = {"tests": "missing_tests", "smell": "smell", "reuse": "reuse"}


def load_candidates(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def base_label(row: dict) -> dict:
    n = int(row["number"])
    return {"id": label_id(row["repo"], n), "repo": row["repo"], "number": n, "url": row["url"],
            "title": row["title"], "merged_at": row["merged_at"], "base_sha": row["base_sha"],
            "snapshot_sha": row["snapshot_sha"], "edge_case": "",
            "checks": {c: blank_check() for c in CHECKS}}


def show(row: dict, say: Callable[[str], None]) -> None:
    say("\n" + "=" * 78)
    say(f"{row['repo']}#{row['number']}  [{row['hint']}]  {row['merged_at']}  +{row['additions']}/-{row['deletions']}"
        f"  in {row['changed_files']} file(s), {row['discussion']} comments")
    say(row["title"])
    if row["hint"] != "negative":
        say(f"\nreviewer: {row['comment_url']}")
        say(f"  \"{row['snippet']}\"")
        if row["path"]:
            say(f"\nabout {row['path']}:")
            for ln in row["hunk"].splitlines()[-10:]:
                say("    " + ln)


def decide(row: dict, get_pr: Callable[[dict], object], ask: Callable[[str], str],
           say: Callable[[str], None]) -> Optional[dict]:
    """Ask about one candidate. Returns a label, or None to skip. Raises EOFError to stop."""
    show(row, say)
    label = base_label(row)

    if row["hint"] == "negative":
        silent = ["smell", "reuse"] + (["missing_tests"] if row["has_tests"] in ("True", "TRUE", "true", True) else [])
        say(f"\nA quiet PR. Silent checks would be: {', '.join(silent)} "
            f"({'tests are in the PR' if 'missing_tests' in silent else 'no tests in the PR, so Check 1 stays unclear'})")
        answer = ask("Enter = accept, s = skip, q = quit: ").strip().lower()
        if answer == "q":
            raise EOFError
        if answer == "s":
            return None
        for c in silent:
            label["checks"][c]["expect"] = "none"
        label["edge_case"] = ask("edge-case tag (optional): ").strip()
        return label

    check = HINT_TO_CHECK[row["hint"]]
    answer = ask("\nIs this a real problem the tool could catch in the code the PR adds?  "
                 "f = yes, s = skip, q = quit: ").strip().lower()
    if answer == "q":
        raise EOFError
    if answer != "f":
        return None

    pr = get_pr(row)
    say("\nfiles changed at the commit the reviewer saw:")
    for f in pr.files:
        say(f"  {'[test]' if is_test_file(f.filename) else '      '} {f.filename}  +{f.additions}/-{f.deletions}")
    if check == "missing_tests" and any(is_test_file(f.filename) for f in pr.files):
        say("  note: tests are already touched here. Is the reviewer asking for more, or for a new one?")
    sources = [f.filename for f in pr.files if f.filename.endswith(".py") and not is_test_file(f.filename)]
    filename = row["path"] if row["path"] in sources else ""
    if not filename and len(sources) == 1:
        filename = sources[0]
    if not filename:
        for i, name in enumerate(sources, 1):
            say(f"  {i}. {name}")
        pick = ask("number of the source file the tool should quote from (or type a path): ").strip()
        filename = sources[int(pick) - 1] if pick.isdigit() and 1 <= int(pick) <= len(sources) else pick
    added = added_lines_of(pr, filename)
    say(f"\nlines the PR adds to {filename}:")
    for i, ln in enumerate(added[:40], 1):
        say(f"  {i:3d}  {ln}")
    picks = ask("number(s) of the line(s) the tool should quote, e.g. 3 or 3,9 (blank = just the file): ")
    targets = []
    for token in [t for t in picks.replace(" ", "").split(",") if t.isdigit()]:
        i = int(token)
        if 1 <= i <= len(added):
            targets.append({"file": filename, "evidence_contains": added[i - 1].strip()})
    label["checks"][check] = {"expect": "finding", "targets": targets or [{"file": filename}],
                              "source": {"url": row["comment_url"], "quote": row["snippet"]}, "note": ""}

    others = [c for c in CHECKS if c != check]
    silent = ask(f"checks that should stay silent on this PR (of {', '.join(others)}; blank = none, "
                 f"'all' = both): ").strip().lower()
    for c in others:
        if silent == "all" or c in silent or (c == "missing_tests" and "tests" in silent):
            label["checks"][c]["expect"] = "none"
    label["edge_case"] = ask("edge-case tag (optional): ").strip()
    return label


def read_only_list(path: str) -> set[str]:
    """Label ids from a text file with one PR per line: 'owner/repo#123' (or an id)."""
    ids = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("  ")[0].strip()
        if not line or line.startswith("#") and "/" not in line:
            continue
        if "#" in line:
            repo, number = line.rsplit("#", 1)
            ids.add(label_id(repo.strip(), int(number)))
        else:
            ids.add(line)
    return ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Label candidate PRs one keystroke at a time.")
    ap.add_argument("candidates")
    ap.add_argument("--only", help="a text file of PRs to go through (owner/repo#123, one per line)")
    ap.add_argument("--hint", choices=list(HINT_TO_CHECK) + ["negative"])
    ap.add_argument("--limit", type=int, default=25, help="stop after this many labels this session")
    args = ap.parse_args(argv)

    from evaluation.github_api import GitHub
    from evaluation.snapshot import build_pr
    gh = GitHub()
    rows = [r for r in load_candidates(args.candidates) if not args.hint or r["hint"] == args.hint]
    if args.only:
        wanted = read_only_list(args.only)
        rows = [r for r in rows if label_id(r["repo"], int(r["number"])) in wanted]
    label_store.LABEL_DIR.mkdir(parents=True, exist_ok=True)
    skipped_file = Path(__file__).parent / "skipped.txt"
    skipped = set(skipped_file.read_text().split()) if skipped_file.exists() else set()
    have = set(label_store.load_all())
    todo = [r for r in rows
            if label_id(r["repo"], int(r["number"])) not in skipped
            and label_id(r["repo"], int(r["number"])) not in have]
    print(f"{len(todo)} unlabelled candidate(s)")
    written = 0
    try:
        for row in todo:
            if written >= args.limit:
                break
            label = decide(row, lambda r: build_pr(gh, base_label(r)), input, print)
            if label is None:
                with open(skipped_file, "a", encoding="utf-8") as fh:
                    fh.write(label_id(row["repo"], int(row["number"])) + "\n")
                continue
            (label_store.LABEL_DIR / f"{label['id']}.json").write_text(
                json.dumps(label, indent=2) + "\n", encoding="utf-8")
            written += 1
            print(f"saved {label['id']}  ({written} this session)")
    except (EOFError, KeyboardInterrupt):
        print("\nstopped")
    print(f"{written} label(s) written. Check them with: python -m evaluation.labels validate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
