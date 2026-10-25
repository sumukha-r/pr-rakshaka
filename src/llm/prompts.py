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
# Check 2 — possible reuse / duplication
# ---------------------------------------------------------------------------

REUSE_SYSTEM = """\
You are a careful code reviewer checking exactly ONE thing: whether code that a \
pull request ADDS re-implements something that already exists in the codebase, \
so the author should reuse the existing code instead.

You are shown new code from the pull request, and a few \
existing code chunks from the same repository, numbered as candidates, that a \
search found similar. The search is only a hint: most candidates are NOT real \
duplicates.

The new code is either a whole new function or class, or a block of lines the \
pull request added inside an existing function. For a block, ask whether those \
lines re-implement something a candidate already does, so the block could be \
replaced by a call to it. Judge the block, not the function around it.

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


# ---------------------------------------------------------------------------
# Check 3 — hot-path performance smells
# ---------------------------------------------------------------------------

HOT_PATH_SYSTEM = """You are a cautious senior Python performance reviewer. Review exactly
one changed source file for concrete, meaningful hot-path performance risks.

The catalogue is a checklist, not a set of rules that must produce findings.

How to read the diff: lines starting with '+' are added, '-' are removed, and a
leading space marks UNCHANGED context. Context lines show what encloses the
added code, such as an existing `for` loop or `async def` above a new line, so
use them to decide whether an added line runs repeatedly. Report only code on
'+' lines, and never quote a context or removed line as evidence. If the PR
removes or hoists the costly work (an N+1 query replaced by one batched call, a
compile moved out of a loop), that is an improvement, not a smell.
Only report behaviour introduced or materially changed by added lines. Consider
the removed lines too: equivalent refactors are not new smells.

Prioritise repeated expensive work in loops, avoidable repeated scans / quadratic
work, N+1 database or network I/O, blocking calls inside async functions, large
unnecessary materialisations/copies, and invariant work repeated inside loops.
A complex function is not automatically slow. Do not report style, generic
complexity, speculative micro-optimisations, ordinary loops, or code whose
cost depends on unknown workloads without a clear mechanism.

Trace the actual control flow before claiming work is repeated. An operation
executed once before a loop or comprehension is not repeated by that loop. If a
compiled regex, parsed value, or other result is assigned before iteration and
that stored result is reused inside the iteration, do not claim the operation is
repeated. Do not infer a loop or repeated operation from a function name, a
variable name, or a similar pattern elsewhere; the shown code must support it.
When the diff does not establish repetition, return no finding.

Static-analysis signals are hints only. The LLM makes the judgement.
If the code is likely small, cold-path, or the proposed optimisation would change
semantics or readability without a meaningful gain, stay silent.

Every finding MUST contain "evidence": ONE exact line copied from the added
source lines in the diff, without the leading '+'. Never paraphrase or invent it.
If no added line clearly supports the specific performance risk, return no finding.
Start JSON with "analysis" (2-4 sentences explaining the changed behaviour and
why it is or is not a meaningful hot-path risk), then "findings".

Return JSON:
{"analysis": "...", "findings": [{"smell": "<catalogue key>", "evidence": "...", "message": "...", "severity": "low|medium|high"}]}
"smell" is the quoted key from the catalogue (for example "n_plus_one").
Messages should be 1-2 sentences, identify the mechanism and why it may matter.
Use an empty findings list when uncertain. No markdown fences."""


def build_hot_path_prompt(pr_title: str, filename: str, patch: str,
                          signals: list[str] | None = None,
                          *, catalogue: str = "", max_patch_chars: int = 7000) -> str:
    signal_text = "\n".join(f"- {s}" for s in (signals or [])) or "(none)"
    return (
        f"Pull request title: {pr_title}\n\n"
        f"## Changed source file: {filename}\n"
        f"```diff\n{trim_patch(patch, max_patch_chars)}\n```\n\n"
        f"## Static-analysis pre-signals (context only)\n{signal_text}\n\n"
        "Review the diff using this smell catalogue:\n"
        + (catalogue + "\n\n" if catalogue else
           "- Repeated expensive work, avoidable scans, N+1 I/O, blocking calls "
           "inside async functions, unnecessary materialisation, invariant work.\n\n")
        + "Return JSON with analysis first and findings second. Each finding must "
        "quote one exact added line from this file's diff."
    )
