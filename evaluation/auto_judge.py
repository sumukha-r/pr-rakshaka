"""
auto_judge.py — let a model draft the verdicts in judgements.csv.

    python -m evaluation.score --split dev --providers remote remote-agent   # adds new rows
    python -m evaluation.auto_judge --calibrate          # first: does it agree with YOU?
    python -m evaluation.auto_judge                      # then: judge the empty rows
    python -m evaluation.score --split dev --providers remote remote-agent --auto

Why it is built this way:
  * Your own verdicts (the `verdict` column) are never touched. The model writes to
    `auto_verdict` / `auto_reason`, and `score --auto` uses it only where you left `verdict` blank.
  * The judge is blind, like you: the rows do not say which system wrote a finding.
  * A model judge is a noisy judge. `--calibrate` runs it on the rows YOU already judged and
    reports how often it agrees, so you know how far to trust it before you use it.
  * It sees the diff only (the file, the changed tests, the reviewers' comments for that PR),
    so on "missing_tests" it cannot know about tests that live outside the diff.
    Use a different model from the one being evaluated when you can.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Optional

from evaluation import score
from evaluation.labels import load_all
from src.llm.prompts import is_test_file

CHECK_DEFINITIONS = {
    "missing_tests": "the PR adds or changes behaviour that could break, and no changed test plausibly exercises it",
    "reuse": "code the PR adds re-implements something that already exists in the codebase, so the author should reuse it",
    "smell": "code the PR adds has a performance problem in a hot path (work repeated in a loop, "
             "needless copying, quadratic scans)",
}

JUDGE_SYSTEM = """\
You are a senior maintainer deciding whether a finding from an automatic code reviewer is worth \
reporting on a pull request. You see the pull request's diff for the file, any changed tests, and \
what human reviewers said about the PR. Judge ONLY the finding you are given.

Verdicts:
- "valid": a real, specific problem of the kind the check looks for, in code this PR adds, that a \
reasonable maintainer would want fixed or flagged.
- "invalid": the claim is wrong or unsupported: the behaviour is unchanged, a changed test already \
covers it, the code does not do what the finding says, or the problem is not in the added code.
- "trivial": technically true but not worth a reviewer's time: constants, a template field, logging, \
a rename, style, a micro-optimisation with no real effect.

When unsure between valid and invalid, choose invalid. Reply as JSON: \
{"verdict": "valid" | "invalid" | "trivial", "reason": "one sentence"}."""


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n... (cut)"


def judge_prompt(label: dict, pr, row: dict) -> str:
    """Everything the judge sees for one row. Pure: no network, no model."""
    check = row["check"]
    patch = next((f.patch for f in pr.files if f.filename == row["file"] and f.patch), "") or "(diff not available)"
    tests = [f for f in pr.files if is_test_file(f.filename) and f.patch][:2]
    reviewer = (label.get("checks", {}).get(check, {}).get("source", {}) or {}).get("quote", "")
    parts = [f"CHECK: {check}: {CHECK_DEFINITIONS.get(check, check)}",
             f"PULL REQUEST: {pr.title}",
             f"FILE: {row['file']}", "DIFF OF THE FILE:\n" + _cut(patch, 6000)]
    if tests:
        parts.append("CHANGED TESTS:\n" + "\n\n".join(f"{t.filename}\n{_cut(t.patch, 2500)}" for t in tests))
    elif check == "missing_tests":
        parts.append("CHANGED TESTS: none in this pull request")
    if reviewer:
        parts.append(f"WHAT A HUMAN REVIEWER SAID ABOUT THIS PR: \"{reviewer}\"")
    parts.append(f"FINDING TO JUDGE:\nquoted line: {row['evidence']}\nclaim: {row['message']}")
    return "\n\n".join(parts)


def parse_verdict(reply: dict) -> Optional[tuple[str, str]]:
    verdict = str((reply or {}).get("verdict", "")).strip().lower()
    if verdict not in score.VERDICTS:
        return None
    return verdict, " ".join(str(reply.get("reason", "")).split())[:300]


def judge_row(row: dict, label: dict, pr, chat: Callable[[str, str], dict]) -> Optional[tuple[str, str]]:
    """(verdict, reason) for one row, or None if the model gave no usable verdict."""
    try:
        return parse_verdict(chat(JUDGE_SYSTEM, judge_prompt(label, pr, row)))
    except Exception as exc:  # noqa: BLE001 - one bad row must not stop the batch
        print(f"  could not judge {row['key'][:60]}: {exc}", file=sys.stderr)
        return None


def agreement(pairs: list[tuple[str, str]]) -> dict:
    """pairs = [(human, model)]. Precision only cares valid vs not valid, so that is the headline."""
    n = len(pairs)
    binary = sum(1 for h, m in pairs if (h == "valid") == (m == "valid"))
    exact = sum(1 for h, m in pairs if h == m)
    return {"n": n, "binary": binary, "exact": exact,
            "model_valid_human_not": sum(1 for h, m in pairs if m == "valid" and h != "valid"),
            "human_valid_model_not": sum(1 for h, m in pairs if h == "valid" and m != "valid")}


def run_judging(rows: list[dict], labels: dict, get_pr: Callable[[str], object],
                chat: Callable[[str, str], dict], *, calibrate: bool = False,
                limit: Optional[int] = None, save: Optional[Callable[[], None]] = None) -> dict:
    """Judge rows. Normal mode fills auto_verdict on rows with no verdict yet; calibrate mode judges the
    rows a human already judged and writes nothing. Returns counts (and the pairs when calibrating)."""
    if calibrate:
        todo = [r for r in rows if (r.get("verdict") or "").strip().lower() in score.VERDICTS]
    else:
        todo = [r for r in rows if not (r.get("verdict") or "").strip() and not (r.get("auto_verdict") or "").strip()]
    if limit:
        todo = todo[:limit]
    prs: dict[str, object] = {}
    done, failed, pairs = 0, 0, []
    for i, row in enumerate(todo, 1):
        label = labels.get(row["id"])
        if label is None:
            failed += 1
            continue
        try:
            if row["id"] not in prs:
                prs[row["id"]] = get_pr(row["id"])
        except Exception as exc:  # noqa: BLE001
            print(f"  could not fetch {row['id']}: {exc}", file=sys.stderr)
            failed += 1
            continue
        result = judge_row(row, label, prs[row["id"]], chat)
        if result is None:
            failed += 1
            continue
        if calibrate:
            pairs.append((row["verdict"].strip().lower(), result[0]))
        else:
            row["auto_verdict"], row["auto_reason"] = result
            if save and i % 5 == 0:
                save()
        done += 1
    if save and not calibrate:
        save()
    return {"judged": done, "failed": failed, "pairs": pairs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Draft verdicts for evaluation/judgements.csv with a model.")
    ap.add_argument("--calibrate", action="store_true",
                    help="judge the rows you already judged and report agreement; writes nothing")
    ap.add_argument("--provider", help="groq, local or remote (ideally not the model being evaluated)")
    ap.add_argument("--model")
    ap.add_argument("--limit", type=int, help="judge at most this many rows")
    args = ap.parse_args(argv)

    from evaluation.github_api import GitHub
    from evaluation.snapshot import build_pr
    from src.llm.client import chat_json

    existing = score.load_judgements()
    if not existing:
        print("judgements.csv is empty or missing: run evaluation.score first")
        return 1
    labels = load_all()
    gh = GitHub()
    result = run_judging(
        list(existing.values()), labels, lambda pid: build_pr(gh, labels[pid]),
        lambda system, user: chat_json(system, user, provider=args.provider, model=args.model),
        calibrate=args.calibrate, limit=args.limit,
        save=lambda: score.write_judgements(existing, []))
    if args.calibrate:
        a = agreement(result["pairs"])
        if not a["n"]:
            print("no rows with your verdicts to calibrate on")
            return 1
        print(f"calibration on {a['n']} row(s) you judged:")
        print(f"  agrees on valid vs not valid: {a['binary']}/{a['n']} ({a['binary'] / a['n']:.0%})  <- what precision uses")
        print(f"  agrees exactly (3 classes):   {a['exact']}/{a['n']} ({a['exact'] / a['n']:.0%})")
        print(f"  model said valid, you did not: {a['model_valid_human_not']}   "
              f"you said valid, model did not: {a['human_valid_model_not']}")
        print("  rule of thumb: under about 80% agreement, judge by hand or change --provider/--model")
    else:
        print(f"auto-judged {result['judged']} row(s), {result['failed']} failed. "
              f"Use them with: python -m evaluation.score ... --auto")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
