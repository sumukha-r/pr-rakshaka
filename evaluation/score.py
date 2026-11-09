"""
score.py — compare saved reviews with the labels.

    python -m evaluation.score --split dev --providers groq local
    python -m evaluation.score --split test --providers groq local --json

How a finding is judged:
  * It matches a labelled target (same file, and its quoted line contains the
    target's `evidence_contains`): counted as correct.
  * Anything else (a finding on a PR labelled 'none', or an extra finding on a
    PR labelled 'finding') goes into evaluation/judgements.csv with an empty
    `verdict`. You fill it in: valid, invalid or trivial. The rows are shuffled
    and do not say which model wrote them, so the judging is blind.
    'valid' counts as correct; 'invalid' and 'trivial' count as wrong.
    Until every row is judged, precision is marked provisional.

Reported per check: recall (labelled targets found), precision, how many 'none'
PRs stayed silent, findings per PR, and how many checks did not run (failed or
were skipped), because a check that errors out is also a result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Optional

from evaluation.labels import CHECKS, load_all
from evaluation.run_eval import run_dir
from evaluation.split import SPLIT_FILE

JUDGEMENTS = Path(__file__).parent / "judgements.csv"
VERDICTS = {"valid": True, "invalid": False, "trivial": False}
J_COLUMNS = ["key", "id", "check", "file", "evidence", "message", "verdict"]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def finding_key(pr_id: str, check: str, f: dict) -> str:
    return f"{pr_id}|{check}|{f.get('filename', '')}|{norm(f.get('evidence', ''))[:80]}"


def matches(finding: dict, target: dict) -> bool:
    if target.get("file") and finding.get("filename") != target["file"]:
        return False                  # a blank target file means any file in the PR
    want = norm(target.get("evidence_contains", ""))
    return not want or want in norm(finding.get("evidence", ""))


def _check_result(run: dict, name: str) -> Optional[dict]:
    """A check that was skipped on purpose (no Python in the PR) is a result with
    no findings, not a failure; only 'failed' or a missing run counts as not run."""
    for c in run.get("checks", []):
        if c.get("name") == name:
            if c.get("status") == "skipped" and "no Python source" in (c.get("note") or ""):
                return dict(c, status="ok", findings=[])
            return c
    return None


def common_ids(labels: dict, ids: list[str], all_runs: dict[str, dict]) -> dict[str, set]:
    """Per check, the PRs on which EVERY run set produced a result, so tables compare like with like."""
    out = {}
    for check in CHECKS:
        ok = set()
        for pid in ids:
            if labels[pid]["checks"][check]["expect"] not in ("finding", "none"):
                continue
            if all((r := _check_result(runs.get(pid, {}), check)) and r.get("status") == "ok"
                   for runs in all_runs.values()):
                ok.add(pid)
        out[check] = ok
    return out


def score_run(labels: dict[str, dict], ids: list[str], runs: dict[str, dict],
              judgements: dict[str, str], only: Optional[dict[str, set]] = None) -> dict:
    """Metrics per check for one provider. `runs` maps PR id -> saved review dict.
    `only` (from common_ids) restricts each check to those PRs."""
    out = {}
    for check in CHECKS:
        m = dict(prs=0, not_run=0, targets=0, hit=0, findings=0, valid=0, wrong=0, pending=0,
                 none_prs=0, silent=0, noisy_valid=0, seconds=0.0, tokens=0, token_prs=0,
                 tool_calls=0)
        for pid in ids:
            exp = labels[pid]["checks"][check]["expect"]
            if exp not in ("finding", "none"):
                continue
            if only is not None and pid not in only[check]:
                continue
            res = _check_result(runs.get(pid, {}), check)
            if res is None or res.get("status") != "ok":
                m["not_run"] += 1
                continue
            m["prs"] += 1
            m["seconds"] += float(res.get("seconds") or 0)
            met = res.get("metrics") or {}
            if "agent_tokens" in met:                  # only the agent reports these
                m["token_prs"] += 1
                m["tokens"] += int(met.get("agent_tokens") or 0)
                m["tool_calls"] += int(met.get("agent_tool_calls") or 0)
            found = res.get("findings", [])
            m["findings"] += len(found)
            targets = labels[pid]["checks"][check]["targets"] if exp == "finding" else []
            m["targets"] += len(targets)
            m["hit"] += sum(1 for t in targets if any(matches(f, t) for f in found))
            extras = []
            for f in found:
                if any(matches(f, t) for t in targets):
                    m["valid"] += 1
                else:
                    extras.append(f)
            for f in extras:
                verdict = judgements.get(finding_key(pid, check, f), "").strip().lower()
                if verdict in VERDICTS:
                    m["valid" if VERDICTS[verdict] else "wrong"] += 1
                else:
                    m["pending"] += 1
            if exp == "none":
                m["none_prs"] += 1
                m["silent"] += 0 if found else 1
                m["noisy_valid"] += 1 if found and all(
                    judgements.get(finding_key(pid, check, f), "").strip().lower() == "valid"
                    for f in found) else 0
        judged = m["valid"] + m["wrong"]
        m["recall"] = round(m["hit"] / m["targets"], 3) if m["targets"] else None
        m["precision"] = round(m["valid"] / judged, 3) if judged else None
        m["provisional"] = m["pending"] > 0
        m["findings_per_pr"] = round(m["findings"] / m["prs"], 2) if m["prs"] else None
        m["seconds_per_pr"] = round(m["seconds"] / m["prs"], 1) if m["prs"] else None
        m["tokens_per_pr"] = round(m["tokens"] / m["token_prs"]) if m["token_prs"] else None
        m["tool_calls_per_pr"] = round(m["tool_calls"] / m["token_prs"], 1) if m["token_prs"] else None
        out[check] = m
    return out


def load_judgements(path: Path = JUDGEMENTS) -> dict[str, dict]:
    if not Path(path).exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["key"]: r for r in csv.DictReader(fh)}


def unjudged_rows(labels, ids, runs, existing) -> list[dict]:
    """Findings that matched no target and are not in the judgements file yet."""
    rows = {}
    for pid in ids:
        for check in CHECKS:
            exp = labels[pid]["checks"][check]["expect"]
            res = _check_result(runs.get(pid, {}), check)
            if exp not in ("finding", "none") or not res or res.get("status") != "ok":
                continue
            targets = labels[pid]["checks"][check]["targets"] if exp == "finding" else []
            for f in res.get("findings", []):
                if any(matches(f, t) for t in targets):
                    continue
                key = finding_key(pid, check, f)
                if key not in existing and key not in rows:
                    rows[key] = {"key": key, "id": pid, "check": check, "file": f.get("filename", ""),
                                 "evidence": f.get("evidence", ""), "message": f.get("message", ""),
                                 "verdict": ""}
    return list(rows.values())


def write_judgements(existing: dict[str, dict], new_rows: list[dict], path: Path = JUDGEMENTS) -> int:
    allrows = list(existing.values()) + new_rows
    allrows.sort(key=lambda r: hashlib.sha256(r["key"].encode()).hexdigest())   # shuffled, stable
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=J_COLUMNS)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in J_COLUMNS} for r in allrows)
    return len(new_rows)


def load_runs(split: str, provider: str) -> dict[str, dict]:
    d = run_dir(split, provider)
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))}


def format_table(provider: str, result: dict) -> str:
    def pct(x):
        return "n/a" if x is None else f"{x:.0%}"
    lines = [f"\n{provider}",
             f"{'check':14s} {'PRs':>4s} {'recall':>14s} {'precision':>15s} {'silent on none':>15s} {'per PR':>7s} {'not run':>8s} {'sec/PR':>7s} {'tok/PR':>7s} {'calls':>6s}"]
    for check, m in result.items():
        rec = f"{pct(m['recall'])} ({m['hit']}/{m['targets']})"
        prec = f"{pct(m['precision'])} ({m['valid']}/{m['valid'] + m['wrong']})" + ("*" if m["provisional"] else "")
        sil = f"{m['silent']}/{m['none_prs']}"
        lines.append(f"{check:14s} {m['prs']:>4d} {rec:>14s} {prec:>15s} {sil:>15s} "
                     f"{(m['findings_per_pr'] if m['findings_per_pr'] is not None else 'n/a')!s:>7} {m['not_run']:>8d} "
                     f"{(m.get('seconds_per_pr') if m.get('seconds_per_pr') is not None else 'n/a')!s:>7} "
                     f"{(m.get('tokens_per_pr') if m.get('tokens_per_pr') is not None else 'n/a')!s:>7} "
                     f"{(m.get('tool_calls_per_pr') if m.get('tool_calls_per_pr') is not None else 'n/a')!s:>6}")
    if any(m["provisional"] for m in result.values()):
        pend = sum(m["pending"] for m in result.values())
        lines.append(f"* provisional: {pend} finding(s) still need a verdict in evaluation/judgements.csv")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score saved reviews against the labels.")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--providers", nargs="+", default=["groq", "local"])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--common", action="store_true",
                    help="score only the PRs every listed provider ran, so the tables are comparable")
    args = ap.parse_args(argv)

    if not SPLIT_FILE.exists():
        print("no split yet", file=sys.stderr)
        return 1
    ids = json.loads(SPLIT_FILE.read_text())[args.split]
    labels = load_all()
    existing = load_judgements()
    all_runs = {p: load_runs(args.split, p) for p in args.providers}

    new_rows: dict[str, dict] = {}
    for runs in all_runs.values():
        for row in unjudged_rows(labels, ids, runs, {**existing, **new_rows}):
            new_rows[row["key"]] = row
    if new_rows:
        write_judgements(existing, list(new_rows.values()))
        print(f"{len(new_rows)} new finding(s) added to {JUDGEMENTS.name}: fill in the verdict column "
              f"(valid / invalid / trivial) and run this again")
    verdicts = {k: r.get("verdict", "") for k, r in existing.items()}

    only = common_ids(labels, ids, all_runs) if args.common else None
    if only is not None:
        print("common PRs per check: " + ", ".join(f"{c}={len(v)}" for c, v in only.items()))
    results = {p: score_run(labels, ids, runs, verdicts, only) for p, runs in all_runs.items()}
    print(json.dumps(results, indent=2) if args.json else
          "\n".join(format_table(p, r) for p, r in results.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
