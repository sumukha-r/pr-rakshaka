"""The evidence filter is the project's core promise; pin its behaviour."""

from src.llm.aggregator import Finding, added_lines, evidence_in_diff, keep_evidenced
from tests.unit.helpers import hunk

PATCH = hunk(" def f(items):", "-    old_call()", "+    for i in items:", "+        db.get(i)",
             "+    return 1", "     tail()")


def test_added_lines_strips_prefix_and_headers():
    assert added_lines("+++ b/x.py\n+a\n-b\n c\n+d") == ["a", "d"]


def test_exact_added_line_is_evidence():
    assert evidence_in_diff("db.get(i)", PATCH)


def test_whitespace_and_leading_plus_are_tolerated():
    assert evidence_in_diff("+        db.get(i)", PATCH)
    assert evidence_in_diff("   db.get(i)   ", PATCH)
    assert not evidence_in_diff("db.get( i )", PATCH)      # spaces inside a token still matter


def test_quote_may_span_consecutive_added_lines():
    assert evidence_in_diff("for i in items:\n    db.get(i)", PATCH)


def test_context_and_removed_lines_are_not_evidence():
    assert not evidence_in_diff("def f(items):", PATCH)      # context
    assert not evidence_in_diff("old_call()", PATCH)         # removed
    assert not evidence_in_diff("tail()", PATCH)             # context after the run


def test_quote_may_not_bridge_a_context_line():
    p = hunk("+a_long_statement()", " middle()", "+another_statement()")
    assert not evidence_in_diff("a_long_statement() another_statement()", p)


def test_too_short_or_invented_quotes_are_rejected():
    assert not evidence_in_diff("i", PATCH)
    assert not evidence_in_diff("imaginary_call()", PATCH)


def test_keep_evidenced_checks_the_findings_own_file():
    good = Finding("smell", "a.py", "db.get(i)", "m")
    wrong_file = Finding("smell", "b.py", "db.get(i)", "m")
    kept, dropped = keep_evidenced([good, wrong_file], {"a.py": PATCH, "b.py": hunk("+other()")})
    assert kept == [good] and dropped == [wrong_file]
