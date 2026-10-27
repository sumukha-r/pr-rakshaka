# pr-rakshaka 🛡️

*Rakshaka (रक्षक) - guardian, protector in Sanskrit*

An LLM-powered pull request reviewer that reads your diffs the way a senior engineer would, not with regex rules, but by actually understanding what the code does and whether the change makes sense.

https://github.com/user-attachments/assets/708cb824-3475-4774-a3a3-9a5cb64901a9

<p align="center">
  <img src="https://github.com/user-attachments/assets/9c9650cc-e9a1-4d43-bdd4-90ec6e2cd450" width="32%" />
  <img src="https://github.com/user-attachments/assets/4ce86457-55ac-4d3c-b4d8-dcb39d737c56" width="32%" />
  <img src="https://github.com/user-attachments/assets/2972f035-9a15-46d7-964d-a21f961c2eb4" width="32%" />
</p>

---

Try it yourself: https://pr-rakshaka-bbs7et8y9yu8rck3qfzldq.streamlit.app/

## What it does

You paste a GitHub PR URL. pr-rakshaka fetches the diff, reads it, and runs three focused checks:

**Missing Tests**
Did this change introduce logic that can fail silently under edge inputs? Is there any evidence a test was written for it? If not, it tells you exactly which function or branch was left uncovered and why it matters.

**Possible Reuse / Duplication**
Does this new code reinvent something that already exists in the codebase? Using vector search over the indexed repo, the LLM is shown the most similar existing functions. If the PR is duplicating logic, it names the original. If the similarity is high enough, it calls it a duplication outright.

**Hot-path Smells**
Does the added code introduce a performance risk on a hot path: an N+1 query or request per item, expensive work repeated in a loop, a quadratic scan, a blocking call inside `async def`, an avoidable full copy of a large iterable? The LLM reads unchanged loop and function lines as context, but only quotes lines the PR adds, and says what may cost rather than claiming it is slow.

If there is no concrete evidence for a finding, nothing is said. The reviewer stays silent rather than guessing.

---

## Why

Code review is the highest-leverage point to catch bad patterns before they compound. Most automated tools are linters, they match rules. pr-rakshaka makes judgments.

## How it works

```
PR URL → fetch diff (GitHub API)
       → retrieve similar code from codebase (ChromaDB + RAG)
       → three checks, each prompting the LLM with the actual diff
       → evidence filter drops any finding that does not quote a line the PR adds
       → deterministic Markdown report (no LLM summarising pass)
       → Streamlit app or command line
```

The LLM is the decision-maker throughout. It runs on Groq (`openai/gpt-oss-20b`, free tier) or on a local Ollama model, switchable per run. radon computes complexity for the functions a PR touches and passes it to the smell check as a hint; the LLM still makes the call.

The report is built without an LLM on purpose: a summarising model could add claims nobody checked, which would break the evidence-only rule.

Full design: [`docs/architecture.md`](docs/architecture.md)

---

## Setup

```bash
git clone https://github.com/sumukha-r/pr-rakshaka
cd pr-rakshaka
pip install -r requirements.txt

# add your Groq API key (free at console.groq.com)
export GROQ_API_KEY=your_key_here

# optional: GitHub token for private repos or higher rate limits
export GITHUB_TOKEN=your_token_here

# optional: use a local model through Ollama instead of Groq
export LOCAL_MODEL=qwen2.5-coder:7b

# the reuse check needs the repo indexed once (slow on large repos)
python src/rag/indexer.py owner/repo
```

### Run the web app

```bash
streamlit run app.py
```

Paste a PR URL, choose the model backend and which checks to run, and press Analyse. Results appear in four tabs: Findings, Diff, PR comment (downloadable as Markdown or JSON) and Run details. The sidebar also has an option to index the repo first if it has not been indexed yet.

### Or use the command line

```bash
python main.py https://github.com/owner/repo/pull/123            # prints a Markdown review
python main.py <PR URL> --provider groq --out review.md
python main.py <PR URL> --checks tests,smells                     # skip the reuse check
python main.py <PR URL> --json                                    # machine-readable
python main.py <PR URL> --index                                   # index the repo first if needed
```

The web app and the command line call the same function, so they always give the same review.

### Tests

Run the offline unit tests (no keys, no GPU, no network) with `python -m pytest`. The scripts in `tests/integration/` compare the Groq and local models on hand-written cases and do need a model.

> The reuse check only runs on repositories that have been indexed. On any other repo it is skipped and the report says so, rather than guessing.

---

## Stack

- **Groq API** (`openai/gpt-oss-20b`, free tier) and **Ollama** (local, e.g. Qwen2.5-Coder) - LLM reviewer
- **ChromaDB** + **sentence-transformers** (all-MiniLM-L6-v2) - vector search for codebase context
- **GitHub REST API** - PR diff fetching
- **Streamlit** - web app
- **radon** - complexity signals for the smell check
- **pytest** - unit tests

---

## Status

| Module | Status |
|---|---|
| PR ingestion (`pr_parser.py`) | v1 |
| RAG indexer + retriever | v1 |
| Check 1 - Missing Tests | v1 |
| Check 2 - Possible Reuse | v1 |
| Check 3 - Hot-path Smells | v1 |
| Evidence aggregator (`src/llm/aggregator.py`) | v1 |
| Report generator (`src/report/generator.py`, deterministic Markdown) | v1 |
| Pipeline entry point (`main.py`) | v1 |
| Streamlit UI (`app.py`) | v1 |

## Known limitations

- Python only: PRs that change no Python source files are skipped.
- The reuse check needs the repository indexed first, and the index reflects the code at the time it was built.
- The three checks have been tuned and measured on hand-written fixtures. An evaluation on real PRs with a held-out test set is in progress, so no accuracy numbers are claimed yet.
- Findings are leads for a human reviewer, not verdicts. Each one quotes the line it is based on so it can be checked quickly.

---
