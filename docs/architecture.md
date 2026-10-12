# Architecture — pr-rakshaka

## What it is

pr-rakshaka is an LLM-powered pull request reviewer. It takes a GitHub PR URL, reads the actual code diff, and makes intelligent judgments about three critical review concerns, not with regex rules, but with a language model that understands what the code does and why.

The LLM is the brain of the system, not a formatter bolted on at the end.

---

## How it works end-to-end

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: GitHub PR URL                                           │
│  e.g. https://github.com/org/repo/pull/42                       │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  pr_parser.py — PR Ingestion Layer                              │
│  • Calls GitHub REST API (/pulls/{n} and /pulls/{n}/files)      │
│  • Extracts unified diff patches from each changed file         │
│  • Captures metadata: author, title, additions, deletions       │
│  • Returns structured PullRequest + PRFile dataclasses          │
└───────────────────────┬─────────────────────────────────────────┘
                        │  raw diffs + file list
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  RAG Context Builder — Codebase Memory Layer                    │
│  • ChromaDB vector store, indexed from real open-source PRs     │
│    (scikit-learn, FastAPI, Pandas, Airflow)                     │
│  • sentence-transformers (all-MiniLM-L6-v2) for embedding       │
│  • For each changed file, retrieves top-3 semantically          │
│    similar existing code chunks from the indexed codebase       │
│  • Gives the LLM the answer to "does this already exist?"       │
└───────────────────────┬─────────────────────────────────────────┘
                        │  diff + retrieved similar code chunks
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Three LLM Checks (Groq API — Llama 3.1)                        │
│  Each check is a focused prompt. LLM reads the real diff.       │
│  Evidence must be cited from actual lines or it is dropped.     │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Check 1 — Missing Tests                                  │  │
│  │  "Does this diff introduce logic that needs a test?       │  │
│  │   Was one written? Point to the line or stay silent."     │  │
│  │  Pre-signal: coverage diff via pytest-cov                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Check 2 — Possible Reuse / Duplication                   │  │
│  │  "Here is the diff. Here are the 3 most similar existing  │  │
│  │   functions in the codebase. Is this reinventing          │  │
│  │   something? Name the existing one or stay silent."       │  │
│  │  High similarity (>0.92 cosine): flagged as duplication.  │  │
│  │  Moderate similarity: flagged as possible reuse.          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Check 3 — Hot-path Smells + Complexity                   │  │
│  │  "Does this diff introduce known anti-patterns?           │  │
│  │   N+1 queries, bare excepts, mutable defaults, etc.       │  │
│  │   Quote the exact lines or stay silent."                  │  │
│  │  Pre-signal: radon cyclomatic complexity on changed funcs │  │
│  └───────────────────────────────────────────────────────────┘  │
└───────────────────────┬─────────────────────────────────────────┘
                        │  raw findings (may include hallucinations)
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Evidence Aggregator — Hallucination Filter                     │
│  • Every finding must quote a specific line from the diff       │
│  • No line cited → finding is silently dropped                  │
│  • LLM cannot say "this probably needs a test" —               │
│    it must say "line 47 adds a branch with no test"             │
│  • Survivors are structured findings with file + line evidence  │
└───────────────────────┬─────────────────────────────────────────┘
                        │  verified findings only
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Report Generator                                               │
│  • Final LLM pass synthesises findings into a review comment    │
│  • Tone: constructive, specific, not a wall of warnings         │
│  • Output: structured dict → rendered Markdown                  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  Streamlit Dashboard — app.py                                   │
│  • Paste PR URL, click Analyse                                  │
│  • Shows each finding with evidence snippet and diff context    │
│  • Green / amber / red verdict per check                        │
│  • Raw diff viewer alongside the findings                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module map

```
pr-rakshaka/
│
├── src/
│   ├── ingestion/
│   │   └── pr_parser.py          # GitHub API client; returns PullRequest dataclass
│   │
│   ├── rag/
│   │   ├── indexer.py            # Embeds code chunks into ChromaDB at ingest time
│   │   └── retriever.py          # Given a diff chunk, returns top-k similar code
│   │
│   ├── checks/
│   │   ├── missing_tests.py      # LLM prompt: did new logic get a test?
│   │   ├── reuse_detector.py     # RAG + LLM prompt: does this already exist?
│   │   └── smell_catalog.py      # Smell definitions + LLM prompt: any anti-patterns?
│   │
│   ├── llm/
│   │   ├── client.py             # Groq API wrapper (Llama 3.1 free tier)
│   │   ├── prompts.py            # Prompt templates for each check
│   │   └── aggregator.py         # Evidence validation; drops ungrounded findings
│   │
│   └── report/
│       └── generator.py          # Final synthesis prompt → Markdown review comment
│
├── app.py                        # Streamlit UI
├── main.py                       # CLI entry point; orchestrates the full pipeline
├── data/
│   └── chromadb/                 # Persisted vector store (gitignored)
└── docs/
    └── architecture.md           # This file
```

---

## Key design decisions

**LLM as reviewer, not classifier**
The model reads actual diffs and makes judgments the way a senior engineer would, not by matching patterns. A regex can tell you "no test file was touched." Only an LLM can tell you "this new branch in `_validate_input()` is the kind of logic that breaks silently under edge inputs and needs a test."

**RAG for codebase awareness**
Without RAG, the LLM only sees the diff and has no idea what the rest of the codebase looks like. RAG closes that gap: when a PR adds a `calculate_similarity()` function, the LLM gets told that `utils/math.py` already exports one with the same signature. Now it can say something meaningful.

**Evidence-only principle**
Any finding not grounded in a specific line from the diff is dropped before it reaches the user. This is not optional. Hallucinated findings destroy trust in a code review tool faster than missing a real one.

**Static analysis as pre-filter, not the answer**
radon and pytest-cov are run before the LLM and their signals are passed into the prompt as context, not returned as findings directly. A high cyclomatic complexity score goes into the smell prompt as: *"radon flagged this function at complexity 12 - does the diff justify it?"* The LLM decides. The tool just provides numbers.

---

## Tech stack

| Layer | Tool | Role |
|---|---|---|
| PR ingestion | GitHub REST API + requests | Fetch diffs and metadata |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Encode code chunks for RAG |
| Vector store | ChromaDB | Persist + retrieve similar code |
| LLM | Groq API — Llama 3.1 8B (free) | All three checks + report synthesis |
| Orchestration | LangChain | RAG chain + prompt management |
| Pre-filters | radon (complexity), pytest-cov (coverage) | Signals fed into LLM prompts |
| UI | Streamlit | Dashboard |
| Language | Python 3.11 | Everything |
