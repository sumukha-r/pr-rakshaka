# Evaluation on real pull requests

The fixtures in `tests/integration/` were written by hand and the prompts were tuned against them, so
they no longer say how the tool behaves on code it has never seen. This folder measures that on real,
merged pull requests, with a dev set to tune against and a test set that is run once.

## Steps

1. **Collect candidates** (needs `GITHUB_TOKEN`):
   `python -m evaluation.collect_candidates --repos scikit-learn/scikit-learn psf/requests --per-hint 40`
   writes `evaluation/candidates.csv`: PRs where a reviewer asked for a test, flagged a performance
   problem or pointed at existing code, plus quiet PRs to use as negatives. PRs with long arguments
   (`--max-discussion`, default 25 comments) are skipped. Add `--state open` to take open PRs instead.
   Each row records `snapshot_sha`, the commit the reviewer was looking at.
2. **Label** the candidates, one keystroke each:
   `python -m evaluation.label_candidates evaluation/candidates.csv --limit 25`
   It shows the reviewer's comment and the code, and asks whether the tool could be expected to catch
   it. Skipped PRs are remembered. Labels land in `evaluation/labels/`; for a hand-made one use
   `python -m evaluation.labels stub <PR URL>`. Check with `python -m evaluation.labels validate`.
3. **Split once**: `python -m evaluation.split --seed 11`, then commit `split.json` and tag it.
   From here the test labels must not change.
4. **Tune on dev**: `python -m evaluation.run_eval --split dev --provider groq` (and `local`), then
   `python -m evaluation.score --split dev`. Read the failures, change a prompt, repeat. Stop after
   about three rounds.
5. **Judge extra findings**: the scorer adds every finding that matched no label to
   `judgements.csv`, shuffled and without the model name. Fill the `verdict` column with `valid`,
   `invalid` or `trivial`, then score again.
6. **Test once**: `python -m evaluation.run_eval --split test --provider groq --final`, the same for
   `local`, then `python -m evaluation.score --split test`. Do not change a prompt afterwards. If you
   do, the test set has become a dev set.

## Why the PR is reviewed as of the reviewer's comment

By the time a PR was merged the author had usually fixed what the reviewer asked for, so the merged
diff has the test and the tool would rightly say nothing. Each label stores `snapshot_sha` and the PR
is rebuilt from the diff up to that commit (`snapshot.py`). That is also why it does not matter that
the repository has changed since: nothing is read from today's code, except the reuse index, which is
built at the PR's base commit (next section). Merged, closed and open PRs all work this way.

## Reuse check on old PRs

The reuse check searches an index of the repository. For these runs the index is built from the
repository at the PR's base commit (collection name `owner__repo__at_<sha>`), so a 2020 PR is compared
with 2020 code. The first run of each reuse PR builds that index, which takes a few minutes.
You can also build one yourself: `python src/rag/indexer.py owner/repo <commit sha>`.

## What the numbers mean

* **recall**: labelled targets the tool found.
* **precision**: findings that were right (matched a label, or judged `valid`), out of all judged.
* **silent on none**: PRs labelled `none` where the tool reported nothing.
* **per PR**: findings per PR; a tool that reports five things per PR is not read.
* **not run**: checks that failed or were skipped. A check that errors is also a result.

With about 60 PRs the counts are small. Report them as "7 of 10", not as a precise percentage.
