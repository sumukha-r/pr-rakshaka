"""
compare_runs.py — why do two run sets differ?  (a diagnostic, not a score)

    python -m evaluation.compare_runs --split dev --base groq --other groq-agent

It answers the questions the score table cannot:
  * not run:   which PRs failed or were skipped in `other` but not in `base`, and why
               (the note saved with the check, e.g. "model error: ...");
  * lost:      labelled targets `base` found and `other` missed, with the tool calls the
               agent made on that PR (from its traces), to see whether it looked at all;
  * gained:    labelled targets `other` found and `base` missed;
  * extra:     findings `other` made that match no label (to be judged), per PR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from evaluation.labels import CHECKS, load_all
from evaluation.run_eval import run_dir
from evaluation.score import _check_result, load_runs, matches
from evaluation.split import SPLIT_FILE, split_names


def trace_calls(split: str, provider: str, pid: str, check: str) -> list[str]:
    """Tool names the agent called on this PR+check, from its saved traces ([] if none)."""
    d = run_dir(split, provider) / "traces"
    names: list[str] = []
    for p in sorted(d.glob(f"{pid}__{check}__*.json")) if d.exists() else []:
        try:
            events = json.loads(p.read_text(encoding="utf-8")).get("events", [])
        except (OSError, ValueError):
            continue
        names += [e.get("name", "?") for e in events if e.get("kind") == "tool"]
    return names


def trace_finals(split: str, provider: str, pid: str, check: str, limit: int = 700) -> list[str]:
    """The agent's last reply (first `limit` chars, one per file it reviewed) for this PR+check."""
    d = run_dir(split, provider) / "traces"
    out: list[str] = []
    for p in sorted(d.glob(f"{pid}__{check}__*.json")) if d.exists() else []:
        try:
            events = json.loads(p.read_text(encoding="utf-8")).get("events", [])
        except (OSError, ValueError):
            continue
        texts = [e.get("text", "") for e in events if e.get("kind") in ("model", "repair") and e.get("text")]
        if texts:
            out.append(" ".join(texts[-1].split())[:limit])
    return out


def _found(res: Optional[dict], targets: list[dict]) -> list[bool]:
    found = (res or {}).get("findings", [])
    return [any(matches(f, t) for f in found) for t in targets]


def compare(labels: dict, ids: list[str], base_runs: dict, other_runs: dict,
            split: str = "dev", other_name: str = "") -> dict:
    out = {"not_run": [], "lost": [], "gained": [], "extra": []}
    for pid in ids:
        for check in CHECKS:
            exp = labels[pid]["checks"][check]["expect"]
            if exp not in ("finding", "none"):
                continue
            b, o = _check_result(base_runs.get(pid, {}), check), _check_result(other_runs.get(pid, {}), check)
            b_ok, o_ok = bool(b) and b.get("status") == "ok", bool(o) and o.get("status") == "ok"
            if not o_ok:
                out["not_run"].append({"id": pid, "check": check, "base_ran": b_ok,
                                       "note": (o or {}).get("note") or "no saved review"})
                continue
            targets = labels[pid]["checks"][check]["targets"] if exp == "finding" else []
            bf, of = _found(b, targets) if b_ok else [], _found(o, targets)
            for i, t in enumerate(targets):
                if b_ok and bf[i] and not of[i]:
                    out["lost"].append({"id": pid, "check": check, "target": t.get("evidence_contains") or t.get("file", ""),
                                        "tools": trace_calls(split, other_name, pid, check) if other_name else [],
                                        "final": trace_finals(split, other_name, pid, check) if other_name else [],
                                        "raw": (o.get("metrics") or {}).get("raw_findings"),
                                        "dropped": (o.get("metrics") or {}).get("dropped")})
                elif of[i] and not (b_ok and bf[i]):
                    out["gained"].append({"id": pid, "check": check, "target": t.get("evidence_contains") or t.get("file", "")})
            extras = [f for f in o.get("findings", []) if not any(matches(f, t) for t in targets)]
            if extras:
                out["extra"].append({"id": pid, "check": check, "expect": exp,
                                     "findings": [(f.get("filename", ""), f.get("evidence", "")[:80]) for f in extras]})
    return out


def format_report(r: dict, base: str, other: str) -> str:
    L = [f"{other} compared with {base}", ""]
    L.append(f"NOT RUN in {other} ({len(r['not_run'])}):")
    L += [f"  {x['id']} {x['check']}  [{base} {'ran' if x['base_ran'] else 'also failed'}]  {x['note'][:110]}" for x in r["not_run"]] or ["  none"]
    L.append(f"\nLOST: {base} hit, {other} missed ({len(r['lost'])}):")
    for x in r["lost"]:
        L.append(f"  {x['id']} {x['check']}  target={x['target'][:50]!r}  tool calls: {', '.join(x['tools']) or 'NONE'}"
                 f"  (raw findings {x.get('raw')}, dropped by filters {x.get('dropped')})")
        L += [f"      agent said: {t}" for t in x.get("final", [])]
    if not r["lost"]:
        L.append("  none")
    L.append(f"\nGAINED: {other} hit, {base} missed ({len(r['gained'])}):")
    L += [f"  {x['id']} {x['check']}  target={x['target'][:50]!r}" for x in r["gained"]] or ["  none"]
    L.append(f"\nEXTRA findings in {other} matching no label ({len(r['extra'])} PR+check):")
    L += [f"  {x['id']} {x['check']} (label: {x['expect']})  " + "; ".join(f"{fn}: {ev}" for fn, ev in x["findings"]) for x in r["extra"]] or ["  none"]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Explain the difference between two run sets.")
    ap.add_argument("--split", choices=split_names(), default="dev")
    ap.add_argument("--base", default="groq")
    ap.add_argument("--other", default="groq-agent")
    args = ap.parse_args(argv)
    if not SPLIT_FILE.exists():
        print("no split yet", file=sys.stderr)
        return 1
    ids = json.loads(SPLIT_FILE.read_text())[args.split]
    r = compare(load_all(), ids, load_runs(args.split, args.base), load_runs(args.split, args.other),
                args.split, args.other)
    print(format_report(r, args.base, args.other))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
