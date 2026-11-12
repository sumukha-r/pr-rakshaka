"""
label_candidates.py — turn candidates.csv into label files, quickly, in the terminal.

    python -m evaluation.label_candidates evaluation/candidates.csv --limit 25
    python -m evaluation.label_candidates evaluation/candidates.csv --hint smell
    python -m evaluation.label_candidates evaluation/candidates.csv --assist --provider remote

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
from evaluation.snapshot import added_lines_of, reachable
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


SUGGEST_SYSTEM = """\
You help a person label pull requests for an evaluation of an automatic code reviewer. You are given \
a human reviewer's comment and the lines the pull request ADDS. Decide whether the comment points at a \
real problem, of the kind the check looks for, in code the PR adds. If so, choose the source file and the \
numbered added line(s) that best implement the problematic behaviour (an executable line: a def, an if, a \
return, a call; never a comment or a bare name). Also say which of the OTHER checks have nothing to find \
in this PR (they should stay silent); leave a check out when you are unsure.
Reply as JSON: {"real_problem": true|false, "reason": "one sentence", "file": "path", \
"lines": [numbers], "silent_checks": ["smell", "reuse", "missing_tests"]}."""


def suggest_prompt(row: dict, pr, check: str) -> str:
    from evaluation.auto_judge import CHECK_DEFINITIONS
    parts = [f"CHECK: {check}: {CHECK_DEFINITIONS[check]}", f"PULL REQUEST: {row['title']}",
             f"REVIEWER SAID: \"{row['snippet']}\"", f"IT WAS ABOUT: {row['path'] or '(not a specific file)'}"]
    if row.get("hunk"):
        parts.append("THE CODE IT WAS ABOUT:\n" + "\n".join(row["hunk"].splitlines()[-10:]))
    sources = [f.filename for f in pr.files if f.filename.endswith(".py") and not is_test_file(f.filename)][:4]
    for name in sources:
        added = added_lines_of(pr, name)[:60]
        parts.append(f"LINES ADDED TO {name}:\n" + "\n".join(f"{i:3d}  {ln}" for i, ln in enumerate(added, 1)))
    tests = [f.filename for f in pr.files if is_test_file(f.filename)]
    parts.append("TEST FILES CHANGED: " + (", ".join(tests) if tests else "none"))
    return "\n\n".join(parts)


def clean_proposal(raw: dict, pr, check: str) -> Optional[dict]:
    """The model's suggestion, checked against the real PR. None if anything does not hold up."""
    if not isinstance(raw, dict) or not isinstance(raw.get("real_problem"), bool):
        return None
    out = {"real": raw["real_problem"], "reason": " ".join(str(raw.get("reason", "")).split())[:200],
           "file": "", "lines": [], "silent": []}
    if not out["real"]:
        return out
    sources = [f.filename for f in pr.files if f.filename.endswith(".py") and not is_test_file(f.filename)]
    if raw.get("file") not in sources:
        return None
    added = added_lines_of(pr, raw["file"])
    lines = raw.get("lines")
    if not isinstance(lines, list) or not lines or not all(isinstance(i, int) and 1 <= i <= len(added) for i in lines):
        return None
    out["file"], out["lines"] = raw["file"], sorted(set(lines))
    others = [c for c in CHECKS if c != check]
    silent = raw.get("silent_checks") if isinstance(raw.get("silent_checks"), list) else []
    out["silent"] = [c for c in others if c in silent]
    return out


def suggest_label(row: dict, pr, check: str, chat: Callable[[str, str], dict]) -> Optional[dict]:
    try:
        return clean_proposal(chat(SUGGEST_SYSTEM, suggest_prompt(row, pr, check)), pr, check)
    except Exception as exc:  # noqa: BLE001 - any failure just means "label it by hand"
        print(f"  (no suggestion: {exc})", file=sys.stderr)
        return None


AUTO_NOTE = "label proposed by a model, auto-accepted (not reviewed by a human)"


def _accept(label: dict, row: dict, check: str, prop: dict, pr, note: str = "label proposed by a model, accepted by a human") -> dict:
    added = added_lines_of(pr, prop["file"])
    targets = [{"file": prop["file"], "evidence_contains": added[i - 1].strip()} for i in prop["lines"]]
    label["checks"][check] = {"expect": "finding", "targets": targets,
                              "source": {"url": row["comment_url"], "quote": row["snippet"]},
                              "note": note}
    for c in prop["silent"]:
        label["checks"][c]["expect"] = "none"
    return label


def decide(row: dict, get_pr: Callable[[dict], object], ask: Callable[[str], str],
           say: Callable[[str], None], suggest: Optional[Callable] = None, auto: bool = False) -> Optional[dict]:
    """Ask about one candidate. Returns a label, or None to skip. Raises EOFError to stop.

    With `suggest` (--assist) a model proposes the whole label and one keystroke accepts it.
    With `auto` (--auto, needs suggest) nothing is asked: a proposal that checks out against the real PR
    is accepted, and anything else (no proposal, "not a real problem") is skipped. Quiet PRs are accepted."""
    show(row, say)
    label = base_label(row)

    if auto and row["hint"] != "negative":
        if suggest is None:
            raise ValueError("auto labelling needs a suggestion function (--assist)")
        check = HINT_TO_CHECK[row["hint"]]
        pr = get_pr(row)
        prop = suggest(row, pr, check)
        if prop is None or not prop["real"]:
            say("auto: skipped (" + (prop["reason"] if prop else "no usable proposal") + ")")
            return None
        say(f"auto: accepted ({prop['reason']}); {prop['file']} line(s) {prop['lines']}")
        return _accept(label, row, check, prop, pr, AUTO_NOTE)

    if row["hint"] == "negative":
        silent = ["smell", "reuse"] + (["missing_tests"] if row["has_tests"] in ("True", "TRUE", "true", True) else [])
        say(f"\nA quiet PR. Silent checks would be: {', '.join(silent)} "
            f"({'tests are in the PR' if 'missing_tests' in silent else 'no tests in the PR, so Check 1 stays unclear'})")
        answer = "" if auto else ask("Enter = accept, s = skip, q = quit: ").strip().lower()
        if answer == "q":
            raise EOFError
        if answer == "s":
            return None
        for c in silent:
            label["checks"][c]["expect"] = "none"
        label["edge_case"] = "" if auto else ask("edge-case tag (optional): ").strip()
        return label

    check = HINT_TO_CHECK[row["hint"]]
    pr = None
    if suggest is not None:
        pr = get_pr(row)
        prop = suggest(row, pr, check)
        if prop is not None:
            if not prop["real"]:
                say(f"\nproposal: NOT a real problem for this check ({prop['reason']})")
                answer = ask("Enter = skip, m = decide manually, q = quit: ").strip().lower()
                if answer == "q":
                    raise EOFError
                if answer != "m":
                    return None
            else:
                say(f"\nproposal: a real problem ({prop['reason']})")
                say(f"  file: {prop['file']}")
                added = added_lines_of(pr, prop["file"])
                for i in prop["lines"]:
                    say(f"  line {i}: {added[i - 1]}")
                say(f"  other checks that should stay silent: {', '.join(prop['silent']) or 'none'}")
                answer = ask("Enter = accept, m = decide manually, s = skip, q = quit: ").strip().lower()
                if answer == "q":
                    raise EOFError
                if answer == "s":
                    return None
                if answer != "m":
                    return _accept(label, row, check, prop, pr)
    answer = ask("\nIs this a real problem the tool could catch in the code the PR adds?  "
                 "f = yes, s = skip, q = quit: ").strip().lower()
    if answer == "q":
        raise EOFError
    if answer != "f":
        return None

    pr = pr or get_pr(row)
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
    ap.add_argument("--assist", action="store_true",
                    help="a model proposes each label; Enter accepts it (m = decide by hand)")
    ap.add_argument("--auto", action="store_true",
                    help="with --assist: no questions at all. Good proposals are accepted, the rest skipped; "
                         "accepted labels carry a note so you can spot-check them")
    ap.add_argument("--provider", help="provider for --assist (groq, local or remote)")
    ap.add_argument("--model", help="model for --assist")
    args = ap.parse_args(argv)
    if args.auto and not args.assist:
        ap.error("--auto needs --assist (a model has to propose the labels)")

    from evaluation.github_api import GitHub
    from evaluation.snapshot import build_pr
    gh = GitHub()
    unreachable_file = Path(__file__).parent / "unreachable.txt"
    rows = [r for r in load_candidates(args.candidates) if not args.hint or r["hint"] == args.hint]
    if args.only:
        wanted = read_only_list(args.only)
        rows = [r for r in rows if label_id(r["repo"], int(r["number"])) in wanted]
    suggest = None
    if args.assist:
        from src.llm.client import chat_json
        suggest = lambda r, pr, check: suggest_label(   # noqa: E731
            r, pr, check, lambda system, user: chat_json(system, user, provider=args.provider, model=args.model))
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
            ok, why = reachable(gh, base_label(row))
            if not ok:
                with open(unreachable_file, "a", encoding="utf-8") as fh:
                    fh.write(f"{label_id(row['repo'], int(row['number']))}  {why}\n")
                with open(skipped_file, "a", encoding="utf-8") as fh:
                    fh.write(label_id(row["repo"], int(row["number"])) + "\n")
                print(f"skipped {row['repo']}#{row['number']}: {why}")
                continue
            label = decide(row, lambda r: build_pr(gh, base_label(r)), input, print, suggest, auto=args.auto)
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
