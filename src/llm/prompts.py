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


# ---------------------------------------------------------------------------
# Check 2 — reuse / duplication
# ---------------------------------------------------------------------------

def clip_source(text: str, max_chars: int) -> str:
    """Cut source text on a line boundary, saying so when it is cut."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    cut = cut[:cut.rfind("\n")] if "\n" in cut else cut
    return cut + "\n    # ... (truncated)"


REUSE_SYSTEM = """\
You are a careful code reviewer checking exactly ONE thing: whether a NEW \
function in a pull request reinvents behaviour that already exists in the \
codebase.

You are shown the new function and up to three existing functions from the \
same repository, found by a similarity search. The similarity scores are \
hints from an embedding model, not proof. Names and wording can differ, so \
compare what the code DOES: its inputs, outputs, side effects and \
edge-case handling.

Verdicts:
- "duplicate": an existing function already does what the new one does, so \
the new one could be replaced by a call to it.
- "reuse": the new function re-implements a substantial part of an existing \
one, or could be written by calling or extending it.
- "different": similar-looking but for a different purpose, or the overlap \
is small or incidental (looping, validating arguments, building a dict). \
When unsure, choose "different". Silence is better than a guess.

Think before you decide. Start your JSON object with an "analysis" field of \
2-4 sentences: what the new function does, what each existing function does, \
and where they differ. Only then give the verdict.

If the verdict is "duplicate" or "reuse", also give: "candidate" (the number \
of the existing function, 1-3); "evidence" (ONE line copied exactly from the \
new function, such as its def line or the line that re-implements the \
behaviour; never invent code); and "message" (one or two sentences naming the \
existing function and what overlaps). If the verdict is "different", use 0 \
for "candidate" and empty strings for "evidence" and "message".\
"""


def build_reuse_prompt(
    pr_title: str,
    filename: str,
    new_source: str,
    candidates: list[dict],
    *,
    max_new_chars: int = 3000,
    max_candidate_chars: int = 1800,
) -> str:
    """
    User message for Check 2. Each candidate dict has: filepath, symbol,
    start_line, end_line, similarity, source.
    """
    blocks = []
    for n, c in enumerate(candidates, 1):
        blocks.append(
            f"### Candidate {n}\n"
            f"File: {c['filepath']} (lines {c['start_line']}-{c['end_line']})\n"
            f"Symbol: {c['symbol']}\n"
            f"Embedding similarity: {c['similarity']:.2f} (a hint only)\n"
            f"```python\n{clip_source(c['source'], max_candidate_chars)}\n```"
        )
    return (
        f"Pull request title: {pr_title}\n\n"
        f"## New function (added in {filename})\n"
        f"```python\n{clip_source(new_source, max_new_chars)}\n```\n\n"
        "## Existing functions in the repository\n" + "\n\n".join(blocks) + "\n\n"
        "Return a JSON object of this shape:\n"
        '{"analysis": "<2-4 sentences>", "verdict": "duplicate|reuse|different", '
        '"candidate": <1-3, or 0>, "evidence": "<exact line from the new function>", '
        '"message": "<what overlaps>"}\n'
        'Write "analysis" first.'
    )


# ---------------------------------------------------------------------------
# Check 2 — possible reuse / duplication
# ---------------------------------------------------------------------------

REUSE_SYSTEM = """\
You are a careful code reviewer checking exactly ONE thing: whether code that a \
pull request ADDS re-implements something that already exists in the codebase, \
so the author should reuse the existing code instead.

You are shown one new function or class from the pull request, and up to three \
existing code chunks from the same repository, numbered as candidates, that a \
search found similar. The search is only a hint: most candidates are NOT real \
duplicates.

A candidate counts ONLY if it does the same job as the new code, so that the new \
code could simply call it, wrap it, or be dropped in its favour. The same job \
done a different way (a loop versus a comprehension, a list versus a generator) \
still counts. Do NOT count a candidate that:
- has a similar shape but a different purpose, input, output, or semantics;
- is already called by the new code (the new code does reuse it);
- is boilerplate that is naturally repeated, such as a constructor, a getter, \
or a logging wrapper.
When unsure, return no findings. Silence is better than a guess.

Think before you decide. Start your JSON object with an "analysis" field of 2-4 \
sentences: what the new code does, what each candidate does, and whether any \
candidate does the same job. Only after that, fill "findings", at most one per \
candidate. If your analysis says no candidate does the same job, "findings" \
must be empty.

Each finding needs the candidate number, an "evidence" line copied exactly from \
the NEW code (one line, as shown), and a "message" of one or two sentences \
naming what to reuse and why it does the same job. Never invent code.\
"""


def _cut(text: str, limit: int) -> str:
    """Shorten on a line boundary so the model never sees half a statement."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return cut[:cut.rfind("\n")] + "\n    # ... (truncated)" if "\n" in cut else cut


def build_reuse_prompt(
    pr_title: str,
    new_def,
    candidates,
    *,
    show_similarity: bool = False,
    max_new_chars: int = 3000,
    max_cand_chars: int = 1800,
) -> str:
    """
    User message for Check 2. `new_def` needs .filename/.start_line/.end_line/.source;
    each candidate needs .filepath/.symbol/.start_line/.end_line/.source/.similarity.

    Similarity scores are left out by default: a weak model anchors on a number
    ("0.93, so it must be a duplicate") instead of reading the code.
    """
    blocks = []
    for i, c in enumerate(candidates, 1):
        sim = f" (similarity {c.similarity:.2f})" if show_similarity else ""
        blocks.append(
            f"### Candidate {i}{sim}\n"
            f"File: {c.filepath} (lines {c.start_line}-{c.end_line})  Symbol: {c.symbol}\n"
            f"```python\n{_cut(c.source, max_cand_chars)}\n```"
        )
    return (
        f"Pull request title: {pr_title}\n\n"
        f"## New code added in {new_def.filename} "
        f"(lines {new_def.start_line}-{new_def.end_line})\n"
        f"```python\n{_cut(new_def.source, max_new_chars)}\n```\n\n"
        "## Existing code in the repository\n" + "\n\n".join(blocks) + "\n\n"
        "Return a JSON object of this shape:\n"
        '{"analysis": "<2-4 sentences>", "findings": [{"candidate": <number>, '
        '"evidence": "<exact line from the NEW code>", '
        '"message": "<what to reuse and why>"}]}\n'
        'Write "analysis" first. Use "findings": [] if no candidate does the same job.'
    )