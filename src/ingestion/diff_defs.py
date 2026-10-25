"""
diff_defs.py — find the functions and classes a PR ADDS, from its diff alone.

A diff is a set of fragments, not whole files. A definition can only be
recovered when all of its lines were added, so a whole new function, a whole
new class, or whole new methods inside an existing class. A hunk that edits the
middle of an existing function does not parse and is skipped on purpose: the
reuse check is about new code, not edited code.

Each result carries its line range in the NEW file, so a finding can say where.
"""

import ast
import re
import textwrap
from dataclasses import dataclass

MIN_LINES = 3   # same floor as the indexer, so we only look for what could be indexed

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_DEF_LINE = re.compile(r"\s*(async\s+def|def|class)\b")


@dataclass
class AddedDef:
    filename: str
    name: str          # "load", "LRUCache" or "LRUCache.get" (class name only when it is in the run)
    kind: str          # "def" | "class"
    source: str        # dedented source text
    start_line: int    # in the new file
    end_line: int
    first_line: str    # the `def ...` / `class ...` line, stripped (used as evidence)


def _runs(patch: str) -> list[tuple[int, list[str]]]:
    """(first new-file line number, lines) for each run of consecutive added lines."""
    runs: list[tuple[int, list[str]]] = []
    cur: list[str] = []
    start = 0
    new_line = None

    def flush():
        nonlocal cur
        if cur:
            runs.append((start, cur))
            cur = []

    for ln in patch.splitlines():
        m = _HUNK.match(ln)
        if m:
            flush()
            new_line = int(m.group(1))
            continue
        if new_line is None:
            continue
        if ln.startswith("+") and not ln.startswith("+++"):
            if not cur:
                start = new_line
            cur.append(ln[1:])
            new_line += 1
        elif ln.startswith("-"):
            flush()                     # a removed line breaks the run
        elif ln.startswith("\\"):       # "\ No newline at end of file"
            continue
        else:                           # context line
            flush()
            new_line += 1
    flush()
    return runs


def _walk_defs(body, prefix: str = ""):
    """Functions, classes and methods; not functions nested inside functions."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield prefix + node.name, "def", node
        elif isinstance(node, ast.ClassDef):
            yield prefix + node.name, "class", node
            yield from _walk_defs(node.body, prefix + node.name + ".")


def added_definitions(filename: str, patch: str) -> list[AddedDef]:
    out: list[AddedDef] = []
    for start, lines in _runs(patch):
        text = textwrap.dedent("\n".join(lines))
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue                    # a fragment of a larger definition
        text_lines = text.splitlines()
        for qualname, kind, node in _walk_defs(tree.body):
            if kind == "def" and node.name.startswith("__") and node.name.endswith("__"):
                continue                # __init__, __repr__ ...: match everywhere, mean nothing
            first = min([node.lineno] + [d.lineno for d in node.decorator_list])
            seg = text_lines[first - 1:node.end_lineno]
            if sum(1 for ln in seg if ln.strip()) < MIN_LINES:
                continue
            head = next((ln.strip() for ln in seg if _DEF_LINE.match(ln)), seg[0].strip())
            out.append(AddedDef(
                filename=filename, name=qualname, kind=kind,
                source=textwrap.dedent("\n".join(seg)),   # a method loses its class indentation
                start_line=start + first - 1,
                end_line=start + node.end_lineno - 1,
                first_line=head,
            ))
    return out