"""M0: skip PRs whose snapshot commit is gone before anyone labels them."""
import types

import pytest
import requests

from evaluation import snapshot
from evaluation.github_api import GitHub


def http_error(code):
    return requests.HTTPError(response=types.SimpleNamespace(status_code=code))


def gh_raising(exc=None):
    def get(path, params=None):
        if exc:
            raise exc
        return {"files": []}
    return GitHub(token="t", get=get)


LABEL = {"repo": "o/r", "base_sha": "b" * 40, "snapshot_sha": "s" * 40}


def test_a_reachable_pr_passes():
    assert snapshot.reachable(gh_raising(), LABEL) == (True, "")


@pytest.mark.parametrize("code", [404, 422])
def test_a_missing_commit_is_reported_with_a_reason(code):
    ok, why = snapshot.reachable(gh_raising(http_error(code)), LABEL)
    assert not ok and str(code) in why


def test_other_errors_are_not_mistaken_for_a_missing_commit():
    with pytest.raises(requests.HTTPError):
        snapshot.reachable(gh_raising(http_error(500)), LABEL)


def test_a_label_without_commits_is_unreachable():
    ok, why = snapshot.reachable(gh_raising(), {"repo": "o/r", "base_sha": "", "snapshot_sha": "s"})
    assert not ok and "no snapshot" in why


def test_the_rebuilt_pr_records_its_base_commit():
    gh = GitHub(token="t", get=lambda path, params=None: {"files": []})
    pr = snapshot.build_pr(gh, dict(LABEL, number=1, title="t"))
    assert pr.base_sha == "b" * 40 and pr.head_sha == "s" * 40
