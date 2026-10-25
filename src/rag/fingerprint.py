"""
fingerprint.py — structural fingerprint of one Python function or class.

Two definitions get the same fingerprint when they are the same code apart from
names: a renamed function, renamed parameters and local variables, different
type hints, different docstrings or comments.

It is exact, not statistical: a match means the logic is identical. It cannot
see through two different implementations of the same job (a loop versus a
comprehension); that is what the embeddings and the LLM are for.

What it ignores:  the definition's own name, parameter and local-variable names
                  (renamed by order of first use, so `a + a` still differs from
                  `a + b`), type annotations, docstrings, comments, formatting.
What it keeps:    called function names, attribute and keyword names, constants,
                  decorators, and the control flow.
"""

import ast
import hashlib
import textwrap
from dataclasses import dataclass
from typing import Optional

# Below this many statements a match is boilerplate (a getter, a one-line
# wrapper), not duplication worth reporting.
MIN_STATEMENTS = 3

_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True)
class Fingerprint:
    digest: str
    statements: int


def _strip_docstrings(root: ast.AST) -> None:
    for node in ast.walk(root):
        if isinstance(node, _DEFS):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]


def _local_names(root: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(root):
        if isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, _DEFS):
            names.add(node.name)
    return names


class _Renamer(ast.NodeTransformer):
    """Rename local names to v0, v1, ... in order of first appearance."""

    def __init__(self, local: set[str]):
        self.local = local
        self.mapping: dict[str, str] = {}

    def _new(self, name: str) -> str:
        if name not in self.local:
            return name
        return self.mapping.setdefault(name, f"v{len(self.mapping)}")

    def visit_Name(self, node: ast.Name):
        node.id = self._new(node.id)
        return node

    def visit_arg(self, node: ast.arg):
        node.arg = self._new(node.arg)
        node.annotation = None
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if node.name:
            node.name = self._new(node.name)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        node.name = self._new(node.name)
        node.returns = None
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef):
        node.name = self._new(node.name)
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign):
        node.annotation = ast.Constant(value=None)
        self.generic_visit(node)
        return node


def fingerprint(source: str) -> Optional[Fingerprint]:
    """
    Fingerprint of `source`, which must be exactly one function or class.
    None if it does not parse (for example a chunk that was truncated).
    """
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], _DEFS):
        return None
    root = tree.body[0]
    _strip_docstrings(root)
    _Renamer(_local_names(root)).visit(root)
    statements = sum(isinstance(n, ast.stmt) for n in ast.walk(root)) - 1
    dump = ast.dump(root, annotate_fields=False, include_attributes=False)
    return Fingerprint(hashlib.sha1(dump.encode()).hexdigest(), statements)


def structurally_identical(a: str, b: str) -> bool:
    """True if a and b are the same code apart from names, and big enough to matter."""
    fa, fb = fingerprint(a), fingerprint(b)
    return bool(fa and fb and fa.digest == fb.digest and fa.statements >= MIN_STATEMENTS)