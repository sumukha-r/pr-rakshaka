# Evaluation results (v1)

How well do the three checks work on **real pull requests**, and how was that measured? This page gives the numbers, how they were produced, what was found and fixed along the way, and where they should not be trusted.

Short version: the **missing-tests check** is the one with enough data to say something. The **reuse** and **smell** checks rarely raise a false alarm, but the set has too few positive cases (5 and 6, and only 1 of the 6 smell cases could be scored) to measure how often they catch a real issue. Treat those two as early observations.

## What was measured

- **68 real PRs** from Apache Airflow, scikit-learn and Django, each labelled by check as `finding` (a human reviewer raised that concern), `none` (a quiet PR with nothing to report) or excluded (unclear / out of scope). The labels come from reviewers' own comments, linked in [`eval_set.csv`](eval_set.csv).
- **Split once, in advance:** 34 dev PRs (used to tune prompts) and 34 test PRs, sealed. The split is stored in `split.json` with a fingerprint of the test labels; the runner refuses to score the test set without `--final`, refuses a second run, and refuses if the labels changed.
- **Snapshot, not final diff.** A merged PR's diff usually already contains the fix the reviewer asked for. Each PR is rebuilt at the commit the reviewer actually saw, and the reuse index is built at the PR's *base* commit, so the answer is not already in the index.
- **Two backends:** Groq `openai/gpt-oss-20b` and local Ollama (Qwen). One run each per round.

### Metrics

| Metric | Meaning |
|---|---|
| recall | Of the issues a reviewer raised, how many did a finding match (same file, and the quoted line if the label names one). |
| precision | Of the findings produced, how many were right. A finding that matches a label counts as right; every other finding was judged blind (`valid` / `invalid` / `trivial`). Only `valid` counts as right. |
| silent on none | Of the PRs labelled quiet, how many got no finding at all. This is strict: a finding on a quiet PR counts against it even if it is a fair point. |
| per PR | Findings per scored PR. |
| not run | Checks that could not be scored (see "Data gaps"). |

## Results

### Final dev run (after two tuning rounds)

| Check | Model | PRs | Recall | Precision | Silent on none | Per PR | Not run |
|---|---|---|---|---|---|---|---|
| missing_tests | Groq | 24 | 88% (14/16) | 77% (20/26) | 2/8 | 1.08 | 4 |
| missing_tests | local | 24 | 88% (14/16) | 90% (19/21) | 7/8 | 0.88 | 4 |
| reuse | Groq | 8 | 33% (1/3) | 100% (1/1) | 5/5 | 0.12 | 0 |
| reuse | local | 8 | 33% (1/3) | 100% (1/1) | 5/5 | 0.12 | 0 |
| smell | Groq | 9 | 100% (1/1) | 100% (2/2) | 8/8 | 0.22 | 2 |
| smell | local | 9 | 100% (1/1) | 100% (1/1) | 8/8 | 0.11 | 2 |

These numbers were tuned against, so they are optimistic.

### Sealed test run (run once, no changes afterwards)

| Check | Model | PRs | Recall | Precision* | Silent on none | Per PR | Not run |
|---|---|---|---|---|---|---|---|
| missing_tests | Groq | 20 | 88% (15/17) | 71% (15/21) | 0/3 | 1.05 | 3 |
| missing_tests | local | 20 | 59% (10/17) | 69% (11/16) | 1/3 | 0.80 | 3 |
| reuse | Groq | 11 | 0% (0/2) | n/a (no findings) | 9/9 | 0.0 | 0 |
| reuse | local | 11 | 0% (0/2) | n/a (no findings) | 9/9 | 0.0 | 0 |
| smell | Groq | 9 | n/a (no positives) | n/a (no findings) | 9/9 | 0.0 | 3 |
| smell | local | 9 | n/a (no positives) | 0% (0/5) | 6/9 | 0.56 | 3 |

\* **Precision on the test set is approximate.** Unmatched findings were judged after a quick look, in repositories too large to learn in an afternoon; most unmatched findings were marked `invalid` or `trivial` when the diff alone did not show the gap. That probably *understates* precision. Recall does not depend on judging.

### What the numbers support

- **Missing tests (best-supported).** Groq found 14 of 16 expected issues on dev and 15 of 17 on the sealed test set, so its recall held up on unseen PRs. The local model fell from 14/16 to 10/17, so its dev result did not carry over. Both models over-flag: on the 8 quiet dev PRs Groq stayed silent on 2 and local on 7; the test set has only 3 quiet PRs, so it says little about this. Rough test precision is about 70% for both models.
- **Reuse.** Both models were silent on all 14 quiet PRs across dev and test (no false alarms), but together found 1 of 5 expected duplicates. Why it missed the rest has not been diagnosed: the duplicate may not have been among the three retrieved candidates (a search problem) or the model may have rejected a correct candidate (a prompt problem).
- **Smell.** The label file has 6 positives, but 5 of them could not be rebuilt (see "Data gaps"), so only 1 was scored (in dev) and recall cannot be assessed. What can be said is that Groq raised no finding on 17 quiet PRs, while the local model raised findings on 3 of 9 quiet test PRs, none judged right.

Every comparison here is one run on a small set. Differences of one or two findings are within the run-to-run variation of an LLM and should not be read as a ranking of the models.

## How it was debugged

Real PRs exposed problems the hand-written fixtures never did.

| Problem found | Fix |
|---|---|
| Merged PRs already contain the fix, so "missing tests" looked wrong | Rebuild each PR at the commit the reviewer saw (GitHub compare API) |
| The reuse index held the very function being added | Index the repo at the PR's base commit; collection named per commit |
| Smell check quoted comments, hedged guesses ("could be expensive if large") and praised optimisations as smells | Prompt rules, plus a deterministic quality gate after the evidence filter |
| One gap reported as three findings, each quoting a bare name (`waiters`, `array_properties`) | Prompt rule and gate: evidence must be a code line, at most two findings per file |
| GitHub timeouts while indexing Airflow, and a half-built index treated as complete | Retries with backoff; a marker that exists only while a build is unfinished |
| Scorer counted a PR with no Python files as a failed check | A skip for "no Python" now counts as silence |

Dev round 1 to round 2: smell precision for the local model rose from 25% to 100% (findings per PR 0.44 to 0.11), and missing-tests precision for the local model rose from 79% to 90%. The same changes did not help Groq's missing-tests precision (82% to 77%), which is within run-to-run noise.

Evidence-only filtering: on the final dev run the filters removed 14 of 37 raw missing-tests findings from the local model and 1 of 27 from Groq. The count mixes quotes not found in the diff with findings rejected by the quality gate.

## Known gaps and limitations

- **Small sets.** 5 reuse and 6 smell positives in the label file (1 smell positive scored); 3 quiet missing-tests PRs in the test split. Percentages on these are counts, not rates.
- **Labels come from reviewer comments.** A reviewer's silence is not proof a PR is fine, and a comment is not always a complete list of issues. Silent-on-none is therefore a strict measure.
- **Approximate test precision** (see above), judged by one person with limited context.
- **Data gaps skew toward positives.** For 6 dev PRs and 6 test PRs GitHub returned 404 when rebuilding the snapshot (the commit is gone, probably because the branch was force-pushed or deleted). They are reported as "not run", not scored, and the labels were not edited afterwards. The counts show these were almost certainly all PRs where a reviewer *had* raised an issue: all 11 quiet missing-tests PRs and all 17 quiet smell PRs were scored, while 7 of 40 missing-tests positives and 5 of 6 smell positives were not. PRs that attract review comments are the ones that get rebased, so the scored set has fewer positives than the label file. Recovering the old commits is a v2 item.
- **The quality gate has a hole.** Hedges phrased as "may be repeated work if the result can be cached" pass it; three of the local model's wrong smell findings and Groq's on one Airflow PR had that shape. It was found on the test set, so it was not changed afterwards. Fixing it is a v2 item.
- **Test-infrastructure files** (for example scikit-learn's `estimator_checks.py`) are treated as source code and can draw "not tested" findings.
- **One run per model per split**, with no repeat to measure variance.
- **Python only**, and three repositories.

## Reproducing

See [`README.md`](README.md) in this folder for the commands. The per-PR run outputs are in `runs/`, the blind judging file is `judgements.csv`, the labelled PRs are `eval_set.csv`, and the split is `split.json`.
