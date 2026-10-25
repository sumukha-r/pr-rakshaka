# pr-rakshaka 🛡️

*Rakshaka (रक्षक) - guardian, protector in Sanskrit*

An LLM-powered pull request reviewer that reads your diffs the way a senior engineer would, not with regex rules, but by actually understanding what the code does and whether the change makes sense.

---

## What it does

You paste a GitHub PR URL. pr-rakshaka fetches the diff, reads it, and runs three focused checks:

**Missing Tests**
Did this change introduce logic that can fail silently under edge inputs? Is there any evidence a test was written for it? If not, it tells you exactly which function or branch was left uncovered and why it matters.

**Possible Reuse / Duplication**
Does this new code reinvent something that already exists in the codebase? Using vector search over the indexed repo, the LLM is shown the most similar existing functions. If the PR is duplicating logic, it names the original. If the similarity is high enough, it calls it a duplication outright.

**Hot-path Smells**
Does the diff introduce known anti-patterns - N+1 queries, bare `except:` clauses, mutable default arguments, blocking calls in async functions? The LLM quotes the exact lines, not a vague warning.

If there is no concrete evidence for a finding, nothing is said. The reviewer stays silent rather than guessing.

---

## Why

Code review is the highest-leverage point to catch bad patterns before they compound. Most automated tools are linters, they match rules. pr-rakshaka makes judgments.

## How it works

```
PR URL → fetch diff (GitHub API)
       → retrieve similar code from codebase (ChromaDB + RAG)
       → three LLM prompts, each reading the actual diff
       → evidence aggregator drops any finding without a cited line
       → report synthesised into a structured review comment
       → Streamlit dashboard
```

The LLM (Groq / Llama 3.1, free tier) is the decision-maker throughout. Static analysis tools (radon for complexity, pytest-cov for coverage) run beforehand and feed their signals into the prompts as context, the LLM still makes the call.

Full design: [`docs/architecture.md`](docs/architecture.md)

---

## Setup

```bash
git clone https://github.com/your-username/pr-rakshaka
cd pr-rakshaka
pip install -r requirements.txt

# add your Groq API key (free at console.groq.com)
export GROQ_API_KEY=your_key_here

# optional: GitHub token for private repos or higher rate limits
export GITHUB_TOKEN=your_token_here

streamlit run app.py
```

---

## Stack

- **Groq API** (Llama 3.1 8B, free tier) - LLM reviewer
- **ChromaDB** + **sentence-transformers** - RAG for codebase context
- **LangChain** - RAG chain and prompt orchestration
- **GitHub REST API** - PR diff fetching
- **Streamlit** - dashboard
- **radon**, **pytest-cov** - pre-filter signals

---

## Status

| Module | Status |
|---|---|
| PR ingestion (`pr_parser.py`) | v1 |
| RAG indexer + retriever | v1 |
| Check 1 - Missing Tests | v1 |
| Check 2 - Possible Reuse | v1 |
| Check 3 - Hot-path Smells | v1 |
| Evidence aggregator | ⏳ pending |
| Report generator | ⏳ pending |
| Streamlit UI | ⏳ pending |

---
