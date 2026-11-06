# v2: the agent

v1 is a fixed pipeline: pick the changed files, build one prompt per file, ask the model once, filter the answer. v2 lets the model **look before it decides**: it can read files, find the tests for a file, and search the codebase for existing code, then answer. Everything it says still has to pass the v1 filters. The v1 pipeline is kept unchanged as the baseline (git tag `v1.0`), and the two are compared on the same PRs.

Status: the loop, the tools and the missing-tests and reuse agents are built and unit-tested with a scripted model. They have not yet been run against a live model; the first real runs and the comparison with v1 are the next milestone. The smell check is still v1.

## How one review runs

```
PR diff ──► for each changed file / new function:
              prompt  ──►  model ──► tool call? ──yes──► run tool ──► result back to the model
                              ▲                                   │
                              └───────────────────────────────────┘
                              no tool call ──► final JSON ──► filters ──► findings
```

The loop (`src/agent/loop.py`) stops when the model answers, or when a budget runs out (steps, tool calls, tokens, seconds). If it runs out, the model is asked once for its best final answer without tools.

## Tools (`src/agent/tools.py`)

| Tool | What it does |
|---|---|
| `read_file` | Read a file at the PR's last commit (`head`) or just before it (`base`); `pattern` returns only matching lines, which is cheaper than a whole file |
| `find_tests` | Test files that probably cover a source file (name and folder match, plus tests the PR changes); a lead, not proof |
| `pr_context` | Title, description, changed files |
| `search_code` | Keyword search over the code index (before the PR) |
| `find_similar` | Semantic search over the same index: "does something like this already exist?" |
| `pr_comments` | Live reviews only; absent in evaluation |

Tools only read. Nothing writes to GitHub and no PR code is ever run.

## Guard rails

- **Evidence-only, as in v1.** A finding must quote a line the PR adds, pass the substance gate, and stay within the per-file cap.
- **A new rule for reuse:** the model may only cite an existing function that a tool or the starting points actually showed it. A citation of anything else is dropped.
- **Pinned reads.** `ref` can only be `head` or `base`. The agent cannot open a branch or a later commit, so in an evaluation it can never read the fix.
- **No comments in evaluation.** A reviewer's comment is the label; the `pr_comments` tool does not exist in that mode.
- **Untrusted data.** PR text and tool output are wrapped as data, closing tags inside them are neutralised, and the system prompt says never to follow instructions found there.
- **Budgets, caching, retries.** Commit-addressed GitHub reads are cached on disk, identical tool calls are not repeated, rate limits and dropped connections are retried, and a malformed tool call is reported back to the model.
- **Fallback.** For live use, a failed agent can fall back to the v1 check. In evaluation the fallback is off, so a failed agent shows up as "not run" instead of being scored as v1.

## What the agent does per check

- **Missing tests** (`agent_missing_tests`): same rules as the v1 prompt, plus: call `find_tests`, read the likely test files for the new function's name, and flag only if no test reaches it.
- **Reuse** (`agent_reuse`): exact copies are reported without a model, as in v1. For other new functions the agent starts from the closest existing code, then uses `find_similar`, `search_code` and `read_file` to look further before deciding. v1 showed three fixed candidates and found 1 of 5 expected duplicates; the point is to test whether looking further helps.

## Running and measuring

```
python -m src.agent.smoke <PR URL> --provider groq        # does tool calling work on this model?
python -m evaluation.run_eval --split dev --provider groq --mode agent
python -m evaluation.score --split dev --providers groq groq-agent
```

Agent runs are saved under `evaluation/runs/<split>/<provider>-agent/`, with a trace of every model turn and tool call in `traces/`. The score table adds seconds, tokens and tool calls per PR, so the cost of looking is visible next to the accuracy.

## Limits

- Not yet validated on live models; whether a given local model supports tool calling at all is unknown until the smoke test.
- Calls are sequential, so an agent review is slower and uses more tokens than v1. Measuring that is part of the comparison.
- The old test split has been used once (v1), so it cannot be a sealed set for v2: it is kept for validation, and a fresh set from other repositories is planned.
