"""
prompts.py — prompts for the agent checks.

The v1 prompts in src/llm/prompts.py are left alone (they are the baseline). The
missing-tests agent keeps every rule of the v1 prompt and adds the instruction to
look before flagging. The reuse agent has its own prompt: v1 shows three numbered
candidates, the agent is told they are only a starting point.
"""

from src.llm.prompts import MISSING_TESTS_SYSTEM, _cut

# The v1 prompt says "a test that calls the changed function, even indirectly, covers it". With tools
# that sentence made the agent declare almost anything covered (dev round 1), so the agent's prompt
# replaces it with the stricter rule below. v1's own prompt is left as it was.
_V1_COVERAGE_RULE = ("a test that calls the changed function or class, even indirectly, covers it. "
                     "Flag only what no changed test line reaches.")
assert _V1_COVERAGE_RULE in MISSING_TESTS_SYSTEM, "v1 prompt changed: update the agent's override"
_STRICT_COVERAGE_RULE = ("a test covers the change only if it exercises the NEW behaviour, as set out "
                         "in the numbered rules below. Being the same function is not enough.")

MISSING_TESTS_AGENT_SYSTEM = MISSING_TESTS_SYSTEM.replace(_V1_COVERAGE_RULE, _STRICT_COVERAGE_RULE) + """

You also have tools. Use them to look before you flag, because a test for the new \
behaviour may live in a file you were not shown:
1. Call find_tests for the source file.
2. Read the most likely test files with read_file(pattern=<name of the new function, \
class, argument or branch>, ref="head").
3. A test counts ONLY if it would fail, or check a different result, without this \
change: it must exercise the NEW behaviour (the new argument, branch, condition or \
return value). A test that merely calls the same function for its old behaviour does \
NOT count, even if the function is well tested overall.
4. To decide there is NO finding, quote in "analysis" the test name AND the assertion \
line, copied exactly from a file you read, that exercises the new behaviour. "The \
function is tested" is not a quote. If you cannot quote one, flag the new behaviour. \
5. A change that guards against a failure (an overflow cast, a copy, a None check) is \
behaviour: flag it unless a test you quoted triggers that failure case. \
Say in "analysis" which files you looked at.
Use as few tool calls as you can and stop as soon as you can decide. "evidence" must \
still be copied from the source diff you were given, never from a tool result."""

REUSE_AGENT_SYSTEM = """\
You are a careful code reviewer checking exactly ONE thing: whether code that a pull \
request ADDS re-implements something that already exists in the codebase, so the \
author should reuse the existing code instead.

You are shown new code from the pull request and up to three existing code chunks \
that a search found similar. They are only a starting point: the search often misses \
the real match, and most of what it returns is NOT a duplicate. Look further with the \
tools before you decide:
- find_similar(query): describe in plain words what the new code does, or paste its \
core, to find existing code with the same meaning.
- search_code(query): find existing code by a name or phrase.
- read_file(path, ref="base"): read the whole existing function before judging it.
Use as few tool calls as you can and stop as soon as you can decide.

An existing function counts ONLY if it does the same job as the new code, so that the \
new code could simply call it, wrap it, or be dropped in its favour. The same job done \
a different way (a loop versus a comprehension, a list versus a generator) still \
counts. Do NOT count code that:
- has a similar shape but a different purpose, input, output, or semantics;
- is already called by the new code (the new code does reuse it);
- is boilerplate that is naturally repeated, such as a constructor, a getter, or a \
logging wrapper.
When unsure, return no findings. Silence is better than a guess.

Think before you decide. Start your JSON object with an "analysis" field of 2-4 \
sentences: what the new code does, what you looked at, and whether any existing \
function does the same job. If your analysis says none does, "findings" must be empty.

Each finding needs "existing" (the path and symbol exactly as a tool or the starting \
points showed them, written path::symbol, for example pkg/utils.py::def parse_date), \
an "evidence" line copied exactly from the NEW code section (one line, never a line \
of existing code), and a "message" of one or two sentences naming what to reuse and \
why it does the same job. Cite only code you were shown. Never invent code.\
"""


def build_agent_reuse_prompt(pr_title: str, new_def, starting_points: list[dict], *,
                             max_new_chars: int = 3000, max_cand_chars: int = 1500) -> str:
    """User message for the reuse agent. `starting_points` are dicts from the semantic search."""
    if starting_points:
        blocks = "\n\n".join(
            f"### Starting point {i}\nFile: {c['path']} (lines {c['start_line']}-{c['end_line']})  "
            f"Symbol: {c['symbol']}\n```python\n{_cut(c['snippet'], max_cand_chars)}\n```"
            for i, c in enumerate(starting_points, 1))
    else:
        blocks = "(the search found nothing close; use the tools to look)"
    return (
        f"Pull request title: {pr_title}\n\n"
        f"## New code added in {new_def.filename} (lines {new_def.start_line}-{new_def.end_line})\n"
        f"```python\n{_cut(new_def.source, max_new_chars)}\n```\n\n"
        f"## Existing code the search found\n{blocks}\n\n"
        "Return a JSON object of this shape:\n"
        '{"analysis": "<2-4 sentences>", "findings": [{"existing": "<path>::<symbol>", '
        '"evidence": "<exact line from the NEW code>", "message": "<what to reuse and why>"}]}\n'
        'Write "analysis" first. Use "findings": [] if no existing function does the same job.'
    )
