"""Check 2 fixtures must be well-formed, and the block cases must really reach the block path."""

import ast

import pytest

from src.ingestion.diff_defs import edited_blocks
from tests.integration import compare_check2 as fx

BLOCKS = [c for c in fx.CASES if "old_file" in c]


def test_names_unique_and_labels_valid():
    names = [c["name"] for c in fx.CASES]
    assert len(set(names)) == len(names) and len(BLOCKS) == 3
    assert all(c["level"] in fx.LEVELS and c["expect"] in ("flag", "clean") for c in fx.CASES)


@pytest.mark.parametrize("case", BLOCKS, ids=[c["name"] for c in BLOCKS])
def test_block_case_produces_exactly_one_block_from_the_diff(case):
    ast.parse(case["new_file"])
    pr = fx._build_pr(case)
    assert pr.head_sha and pr.files[0].status == "modified"
    blocks = edited_blocks(case["file"], pr.files[0].patch, case["new_file"])
    assert [b.name for b in blocks] == ["export_rows"]
