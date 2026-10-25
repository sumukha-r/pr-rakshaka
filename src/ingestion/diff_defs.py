"""
diff_defs.py — find the functions and classes a PR ADDS, from its diff alone.

A diff is a set of fragments, not whole files. A definition can only be
recovered when all of its lines were added, so a whole new function, a whole
new class, or whole new methods inside an existing class. A hunk that edits the
middle of an existing function does not parse, so `added_definitions` skips it.
`edited_blocks` covers that case: given the whole file at the PR head it returns
each block of added lines together with the function it sits in.

Each result carries its line range in the NEW file, so a finding can say where.
"""

import ast
import re
import textwrap
from dataclasses import dataclass

MIN_LINES = 3   # same floor as the indexer, so we only look for what could be indexed
MIN_BLOCK_LINES = 4   # an added block shorter than this is too small to be "reinvented code"

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_DEF_LINE = re.compile(r"\s*(async\s+def|def|class)\b")


@dataclass
class AddedDef:
    filename: str
    name: str          # "load", "LRUCache" or "LRUCache.get" (class name only when it is in the run)
    kind: str          # "def" | "class" | "block" (lines added inside an existing function)
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


def added_line_numbers(patch: str) -> set[int]:
    """Line numbers, in the NEW file, of every line the diff adds."""
    out: set[int] = set()
    for start, lines in _runs(patch):
        out.update(range(start, start + len(lines)))
    return out


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

def _first_line(node) -> int:
    return min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])


def edited_blocks(filename: str, patch: str, full_source: str, *,
                  min_lines: int = MIN_BLOCK_LINES) -> list[AddedDef]:
    """
    Blocks of lines a PR adds INSIDE a function that already existed.

    A diff alone cannot say which function an added run belongs to, and a run
    from the middle of a function does not parse. With the whole file (read at
    the PR head) both are solved: parse the file, find the function that
    encloses each added run, and return the run as one `AddedDef(kind="block")`
    named after that function.

    Skipped on purpose: runs shorter than `min_lines` real lines (too small to
    be reinvented code), runs outside any function, runs that span two
    functions, and functions the PR adds whole (`added_definitions` has those).
    """
    try:
        tree = ast.parse(full_source)
    except SyntaxError:
        return []
    added = added_line_numbers(patch)
    funcs = [(q, n) for q, kind, n in _walk_defs(tree.body) if kind == "def"]

    out: list[AddedDef] = []
    for start, lines in _runs(patch):
        end = start + len(lines) - 1
        real = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        if len(real) < min_lines:
            continue
        inside = [(q, n) for q, n in funcs if _first_line(n) <= start and end <= n.end_lineno]
        if not inside:
            continue
        name, node = min(inside, key=lambda t: t[1].end_lineno - _first_line(t[1]))
        if all(i in added for i in range(_first_line(node), node.end_lineno + 1)):
            continue                      # a whole new function: handled elsewhere
        out.append(AddedDef(
            filename=filename, name=name, kind="block",
            source=textwrap.dedent("\n".join(lines)),
            start_line=start, end_line=end, first_line=real[0].strip(),
        ))
    return out
