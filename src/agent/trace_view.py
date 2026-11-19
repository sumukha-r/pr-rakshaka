"""
trace_view.py — turn saved agent traces into one self-contained HTML page (M5, tracing).

    python -m src.agent.trace_view evaluation/results/dev2/remote-agent-m3/traces -o traces.html
    python -m src.agent.trace_view one_trace.json -o one.html

Every agent run already saves a JSON trace (model turns, tool calls, nudges, budget stops).
This renders them as a timeline you can open in a browser or screenshot for the README:
no account, no server, no extra dependency. Each run is a collapsible card; the header shows
steps, tool calls, tokens and seconds, then every event in order.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Optional

CSS = """
:root{--bg:#fff;--fg:#1b1f24;--muted:#5b6570;--card:#f6f8fa;--line:#d8dee4;--model:#0b5fff;--tool:#1a7f4b;--warn:#b35900}
@media (prefers-color-scheme:dark){:root{--bg:#0f1318;--fg:#e6e9ee;--muted:#9aa5b1;--card:#171c23;--line:#2b333d;--model:#6ea1ff;--tool:#4cc38a;--warn:#f0a050}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:920px;margin:0 auto}h1{font-size:1.3rem;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 18px}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:10px 0;padding:0 14px}
summary{cursor:pointer;padding:12px 0;font-weight:600}summary span{font-weight:400;color:var(--muted);margin-left:8px;font-size:.9em}
.ev{border-left:3px solid var(--line);margin:8px 0;padding:2px 0 2px 12px}
.ev.model{border-color:var(--model)}.ev.tool{border-color:var(--tool)}.ev.warn{border-color:var(--warn)}
.k{font-size:.8rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
pre{white-space:pre-wrap;word-break:break-word;margin:4px 0;font:13px/1.45 ui-monospace,monospace}
"""


def load_traces(path: Path) -> list[tuple[str, list[dict]]]:
    """(name, events) for one trace file, or for every *.json in a folder."""
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    out = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        events = data.get("events") if isinstance(data, dict) else None
        if isinstance(events, list):
            out.append((f.stem, events))
    return out


def summarise(events: list[dict]) -> dict:
    model = [e for e in events if e.get("kind") == "model"]
    return {"turns": len(model),
            "tool_calls": sum(1 for e in events if e.get("kind") == "tool"),
            "tokens": sum(int(e.get("prompt_tokens") or 0) + int(e.get("completion_tokens") or 0) for e in model),
            "seconds": round(sum(float(e.get("seconds") or 0) for e in model), 1),
            "stopped_by_budget": any(e.get("kind") == "budget" for e in events)}


def _event_html(e: dict) -> str:
    kind, esc = e.get("kind", "?"), html.escape
    if kind == "model":
        tools = ", ".join(e.get("tools") or []) or "final answer"
        body = (f"<div class='k'>model · step {e.get('step')} · asked for: {esc(tools)} · "
                f"{e.get('prompt_tokens', 0)}+{e.get('completion_tokens', 0)} tokens · {e.get('seconds', 0)}s</div>")
        if e.get("text"):
            body += f"<pre>{esc(str(e['text']))}</pre>"
        return f"<div class='ev model'>{body}</div>"
    if kind == "tool":
        args = esc(json.dumps(e.get("args", {}), ensure_ascii=False))
        return (f"<div class='ev tool'><div class='k'>tool · step {e.get('step')} · {esc(str(e.get('name')))}"
                f"({args}) · {e.get('chars', 0)} chars returned</div><pre>{esc(str(e.get('preview', '')))}</pre></div>")
    detail = {k: v for k, v in e.items() if k != "kind"}
    return (f"<div class='ev warn'><div class='k'>{esc(kind)}</div>"
            f"<pre>{esc(json.dumps(detail, ensure_ascii=False))}</pre></div>")


def render(traces: list[tuple[str, list[dict]]], title: str = "Agent traces") -> str:
    cards = []
    for name, events in traces:
        s = summarise(events)
        flag = " · stopped by budget" if s["stopped_by_budget"] else ""
        head = (f"{html.escape(name)}<span>{s['turns']} turns · {s['tool_calls']} tool calls · "
                f"{s['tokens']} tokens · {s['seconds']}s{flag}</span>")
        cards.append(f"<details><summary>{head}</summary>{''.join(_event_html(e) for e in events)}</details>")
    body = "".join(cards) or "<p class='sub'>No traces found.</p>"
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body><main><h1>{html.escape(title)}</h1>"
            f"<p class='sub'>{len(traces)} run(s). Click a run to see each model turn and tool call.</p>"
            f"{body}</main></body></html>")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Render agent trace JSON files as one HTML page.")
    ap.add_argument("path", help="a trace .json file or a folder of them")
    ap.add_argument("-o", "--out", default="traces.html")
    args = ap.parse_args(argv)
    src = Path(args.path)
    if not src.exists():
        print(f"not found: {src}", file=sys.stderr)
        return 1
    traces = load_traces(src)
    Path(args.out).write_text(render(traces), encoding="utf-8")
    print(f"wrote {args.out} ({len(traces)} run(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
