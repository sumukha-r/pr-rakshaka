"""Check 2 end to end with a fake retriever and a fake model: no index, no network, no LLM."""

from dataclasses import dataclass

import pytest

import src.checks.reuse_detector as rd
from tests.unit.helpers import make_diff, make_file, make_pr

OLD = '''\
def export_rows(rows, out):
    header = "id,name"
    out.write(header)
    for row in rows:
        out.write(row.line())
    return out
'''
NEW = OLD.replace('''    for row in rows:
        out.write(row.line())
''', '''    chunks = []
    for i in range(0, len(rows), 100):
        chunks.append(rows[i:i + 100])
    for chunk in chunks:
        out.write("".join(r.line() for r in chunk))
''')

BATCH = "def batch(seq, size):\n    for start in range(0, len(seq), size):\n        yield seq[start:start + size]\n"
GET_JSON = ("def get_json(url, timeout=10):\n    resp = requests.get(url, timeout=timeout)\n"
            "    resp.raise_for_status()\n    data = resp.json()\n    return data\n")
FETCH_JSON = ("def fetch_json(link, wait=10):\n    reply = requests.get(link, timeout=wait)\n"
              "    reply.raise_for_status()\n    payload = reply.json()\n    return payload\n")


@dataclass
class Chunk:
    filepath: str
    symbol: str
    source: str
    similarity: float
    start_line: int = 1
    end_line: int = 5


def edit_pr(head_sha="abc123"):
    pr = make_pr(make_file("src/export.py", make_diff(OLD, NEW)))
    pr.head_sha = head_sha
    return pr


@pytest.fixture
def llm(monkeypatch):
    sent = []

    def install(reply):
        def fake(system, user, **kw):
            sent.append((system, user))
            return reply(user) if callable(reply) else reply
        monkeypatch.setattr(rd, "chat_json", fake)
        return sent
    return install


def finding_reply(evidence="chunks.append(rows[i:i + 100])", candidate=1):
    return {"analysis": "same job", "findings": [
        {"candidate": candidate, "evidence": evidence, "message": "Use batch()."}]}


# --- whole new definitions (unchanged behaviour) -----------------------------

def test_renamed_copy_is_reported_without_an_llm_call(llm):
    sent = llm(lambda u: pytest.fail("no LLM call expected"))
    pr = make_pr(make_file("src/net.py", make_diff("", FETCH_JSON), "added"))
    out = rd.check_reuse(pr, retrieve=lambda *a: [Chunk("src/http.py", "def get_json", GET_JSON, 0.97)])
    assert [(f.kind, f.evidence) for f in out] == [("structural_duplicate", "def fetch_json(link, wait=10):")]
    assert not sent


# --- blocks added inside existing functions -----------------------------------

def test_block_inside_an_edited_function_is_reviewed_and_reported(llm):
    sent = llm(finding_reply())
    seen_queries = []

    def retrieve(repo, source, filename, top_k):
        seen_queries.append(source)
        return [Chunk("src/export.py", "def export_rows", OLD, 0.95),      # its own old version
                Chunk("src/iterutils.py", "def batch", BATCH, 0.84)]

    report = {}
    out = rd.check_reuse(edit_pr(), retrieve=retrieve, report=report,
                         fetch_source=lambda repo, path, ref: NEW)
    assert len(out) == 1
    f = out[0]
    assert f.evidence == "chunks.append(rows[i:i + 100])" and f.kind == "reuse"
    assert "src/iterutils.py" in f.related and "def batch" in f.related
    # the query is the added block, not the whole function
    assert seen_queries[0].startswith("chunks = []") and "header" not in seen_queries[0]
    # the function's own old version was excluded from what the model saw
    _, user = sent[0]
    assert "def batch" in user and "def export_rows" not in user
    assert "src/export.py" in user
    assert report["blocks_found"] == 1 and report["files_fetched"] == 1
    assert report["llm_calls"] == 1


def test_block_prompt_tells_the_model_to_judge_the_block(llm):
    sent = llm({"analysis": "", "findings": []})
    rd.check_reuse(edit_pr(), retrieve=lambda *a: [Chunk("a.py", "def batch", BATCH, 0.8)],
                   fetch_source=lambda *a: NEW)
    assert "block of lines the pull request added inside an existing function" in sent[0][0]


def test_quote_from_an_unchanged_line_of_the_function_is_dropped(llm):
    llm(finding_reply(evidence='header = "id,name"'))        # exists in the file, not added
    report = {}
    out = rd.check_reuse(edit_pr(), retrieve=lambda *a: [Chunk("a.py", "def batch", BATCH, 0.8)],
                         fetch_source=lambda *a: NEW, report=report)
    assert out == [] and report["dropped"] == 1


def test_candidate_number_that_was_never_shown_is_dropped(llm):
    llm(finding_reply(candidate=3))
    report = {}
    out = rd.check_reuse(edit_pr(), retrieve=lambda *a: [Chunk("a.py", "def batch", BATCH, 0.8)],
                         fetch_source=lambda *a: NEW, report=report)
    assert out == [] and report["dropped"] == 1


def test_no_head_commit_means_no_fetch_and_no_blocks(llm):
    llm(lambda u: pytest.fail("nothing to review"))
    report = {}
    out = rd.check_reuse(edit_pr(head_sha=""), retrieve=lambda *a: [],
                         fetch_source=lambda *a: pytest.fail("must not fetch"), report=report)
    assert out == [] and report["blocks_found"] == 0 and report["files_fetched"] == 0


def test_failed_fetch_just_means_no_blocks(llm):
    llm(lambda u: pytest.fail("nothing to review"))
    report = {}
    assert rd.check_reuse(edit_pr(), retrieve=lambda *a: [], fetch_source=lambda *a: None,
                          report=report) == []
    assert report["blocks_found"] == 0 and report["files_fetched"] == 0


def test_a_newly_added_file_is_not_fetched(llm):
    llm({"analysis": "", "findings": []})
    pr = make_pr(make_file("src/new.py", make_diff("", NEW), "added"))
    pr.head_sha = "abc"
    rd.check_reuse(pr, retrieve=lambda *a: [], fetch_source=lambda *a: pytest.fail("no fetch"))


def test_candidate_below_the_similarity_floor_means_no_llm_call(llm):
    sent = llm(lambda u: pytest.fail("no LLM call expected"))
    rd.check_reuse(edit_pr(), retrieve=lambda *a: [Chunk("a.py", "def batch", BATCH, 0.60)],
                   fetch_source=lambda *a: NEW)
    assert not sent


def test_blocks_and_new_functions_share_the_per_pr_call_cap(llm):
    sent = llm({"analysis": "", "findings": []})
    pr = edit_pr()
    pr.files.append(make_file("src/more.py", make_diff("", GET_JSON), "added"))
    rd.check_reuse(pr, retrieve=lambda *a: [Chunk("a.py", "def batch", BATCH, 0.8)],
                   fetch_source=lambda *a: NEW, max_defs=1)
    assert len(sent) == 1


def test_broken_file_at_head_does_not_crash(llm):
    llm(lambda u: pytest.fail("nothing to review"))
    assert rd.check_reuse(edit_pr(), retrieve=lambda *a: [],
                          fetch_source=lambda *a: "def broken(:\n") == []
