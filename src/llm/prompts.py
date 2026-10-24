"""
prompts.py — prompt text and prompt-building helpers shared by the checks.

Kept separate from the checks so prompts can be tuned without touching logic,
and so the token-budget helpers (trim_patch) live in one place.

"""

import re

# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------

_TEST_PATH = re.compile(
    r"(^|/)(tests?|testing)/"        # tests/ or test/ directory
    r"|(^|/)test_[^/]*\.py$"         # test_foo.py
    r"|_tests?\.py$"                 # foo_test.py
    r"|(^|/)conftest\.py$"
)


def is_test_file(path: str) -> bool:
    return bool(_TEST_PATH.search(path))


def trim_patch(patch: str, max_chars: int = 6000) -> str:
    """
    Shrink a unified diff to at most ~max_chars, keeping whole hunks where
    possible. Always says so when it truncates, so the model knows it is not
    seeing everything.
    """
    if len(patch) <= max_chars:
        return patch

    hunks = re.split(r"(?m)^(?=@@ )", patch)
    kept: list[str] = []
    used = 0
    for hunk in hunks:
        if not hunk:
            continue
        if used + len(hunk) > max_chars:
            if not kept:  # even the first hunk is too big: cut on a line boundary
                cut = hunk[:max_chars]
                cut = cut[:cut.rfind("\n")] if "\n" in cut else cut
                kept.append(cut + "\n")
            break
        kept.append(hunk)
        used += len(hunk)
    return "".join(kept).rstrip("\n") + "\n... (diff truncated; later changes not shown)"


# ---------------------------------------------------------------------------
# Check 1 — missing tests
# ---------------------------------------------------------------------------

MISSING_TESTS_SYSTEM = """\
You are a careful code reviewer checking exactly ONE thing: whether a pull \
request adds or changes behaviour in a source file without including a test \
for it.

You are shown the diff of one source file and the diffs of any test files \
changed in the same pull request.

Flag a finding ONLY when the diff adds or changes logic that could break, \
such as: a new function or method, a new branch or condition, new error \
handling, or a changed calculation or return value; AND none of the changed \
test files plausibly exercises it.

Do NOT flag: comments, docstrings, type hints, renames, formatting, imports, \
logging, constants, or configuration. Do NOT flag logic that a changed test \
file clearly covers. When unsure, return no findings. Silence is better than \
a guess.

Before flagging, compare the removed lines ('-') with the added lines ('+'). \
If the added lines compute the same thing as the removed ones and only rename \
variables, add type hints, reformat, or reorder code, this is a refactor and \
the behaviour is unchanged: return no findings. Only flag behaviour that is \
new or different from what the removed lines did.

Every finding MUST include an "evidence" field: ONE line copied exactly from \
the added lines ('+' lines) of the source diff, without the leading '+'. If \
you cannot quote such a line, you have no finding. Never invent or paraphrase \
code.

Think before you decide. Start your JSON object with an "analysis" field of \
2-4 sentences covering: (1) what the removed lines did versus what the added \
lines do, and whether the behaviour is new or different; (2) which changed \
test lines, if any, actually call the new or changed behaviour. Only after \
that, fill "findings". If your analysis says the behaviour is unchanged, or \
that a changed test exercises it, "findings" must be empty.

Keep "message" to one or two sentences saying what behaviour is untested.\
"""


def build_missing_tests_prompt(
    pr_title: str,
    filename: str,
    patch: str,
    test_files: list[tuple[str, str]],
    *,
    max_patch_chars: int = 6000,
    max_test_chars: int = 1500,
) -> str:
    """User message for Check 1. `test_files` is [(filename, patch), ...]."""
    if test_files:
        tests_block = "\n\n".join(
            f"### {name}\n```diff\n{trim_patch(tp, max_test_chars)}\n```"
            for name, tp in test_files
        )
    else:
        tests_block = "(none: this pull request changes no test files)"

    return (
        f"Pull request title: {pr_title}\n\n"
        f"## Source file changed: {filename}\n"
        f"```diff\n{trim_patch(patch, max_patch_chars)}\n```\n\n"
        f"## Test files changed in this pull request\n{tests_block}\n\n"
        "Return a JSON object of this shape:\n"
        '{"analysis": "<2-4 sentences>", '
        '"findings": [{"evidence": "<exact added line>", '
        '"message": "<what is untested>"}]}\n'
        'Write "analysis" first. Use "findings": [] if there is nothing to flag.'
    )