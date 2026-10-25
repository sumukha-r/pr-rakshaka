"""The hand-written Check 3 fixtures must be well-formed, or a model comparison
measures a typo instead of the model. Offline: no LLM."""

import ast

import pytest

from src.llm.aggregator import added_lines
from tests.integration import compare_check3 as fx

IDS = [c["name"] for c in fx.CASES]


def test_case_names_are_unique_and_labels_valid():
    assert len(set(IDS)) == len(IDS)
    assert all(c["level"] in fx.LEVELS and c["expect"] in ("flag", "clean") for c in fx.CASES)


@pytest.mark.parametrize("case", fx.CASES, ids=IDS)
def test_new_side_of_every_fixture_is_valid_python(case):
    source = case["new"] if "old" in case else "\n".join(case["lines"])
    ast.parse(source)


@pytest.mark.parametrize("case", [c for c in fx.CASES if "old" in c],
                         ids=[c["name"] for c in fx.CASES if "old" in c])
def test_edit_cases_have_real_context_and_added_lines(case):
    patch = fx._build_pr(case).files[0].patch
    rows = patch.splitlines()[1:]
    assert any(r.startswith("+") for r in rows), "an edit must add something"
    assert any(r.startswith(" ") for r in rows), "an edit case must keep context lines"
    assert added_lines(patch)


def test_edit_cases_cover_both_expectations():
    edits = [c["expect"] for c in fx.CASES if "old" in c]
    assert "flag" in edits and "clean" in edits
