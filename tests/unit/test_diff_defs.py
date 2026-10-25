"""diff_defs: what a diff adds. `edited_blocks` needs the whole file; no network here."""

from src.ingestion.diff_defs import added_definitions, added_line_numbers, edited_blocks
from tests.unit.helpers import make_diff

OLD = '''\
import os


def export_rows(rows, out):
    header = "id,name"
    out.write(header)
    for row in rows:
        out.write(row.line())
    return out


class Exporter:
    def run(self, rows):
        count = 0
        for row in rows:
            count += 1
        return count
'''

NEW_FUNC_BLOCK = OLD.replace('''    for row in rows:
        out.write(row.line())
    return out
''', '''    chunks = []
    for i in range(0, len(rows), 100):
        chunks.append(rows[i:i + 100])
    for chunk in chunks:
        out.write("".join(r.line() for r in chunk))
    return out
''')


def blocks(old, new, **kw):
    return edited_blocks("src/x.py", make_diff(old, new), new, **kw)


def test_block_added_inside_an_existing_function():
    [b] = blocks(OLD, NEW_FUNC_BLOCK)
    assert (b.name, b.kind, b.filename) == ("export_rows", "block", "src/x.py")
    assert b.source.startswith("chunks = []")          # dedented
    assert "chunks.append(rows[i:i + 100])" in b.source
    # the reported lines really are those lines of the new file
    assert NEW_FUNC_BLOCK.splitlines()[b.start_line - 1].strip() == b.first_line == "chunks = []"
    assert b.end_line - b.start_line + 1 == len(b.source.splitlines())


def test_the_function_is_not_reported_as_a_new_definition_by_the_diff_alone():
    assert added_definitions("src/x.py", make_diff(OLD, NEW_FUNC_BLOCK)) == []


def test_block_inside_a_method_is_named_with_its_class():
    new = OLD.replace("        count = 0\n", "        count = 0\n        seen = set()\n"
                      "        for row in rows:\n            seen.add(row.id)\n"
                      "        if len(seen) != len(rows):\n            raise ValueError('dup')\n")
    [b] = blocks(OLD, new)
    assert b.name == "Exporter.run"


def test_small_additions_are_ignored():
    new = OLD.replace("    return out\n", "    out.flush()\n    return out\n")
    assert blocks(OLD, new) == []


def test_comments_and_blank_lines_do_not_count_toward_the_minimum():
    new = OLD.replace("    return out\n", "    # a\n    # b\n    # c\n    # d\n\n    return out\n")
    assert blocks(OLD, new) == []


def test_min_lines_is_adjustable():
    new = OLD.replace("    return out\n", "    out.flush()\n    return out\n")
    assert len(blocks(OLD, new, min_lines=1)) == 1


def test_a_whole_new_function_is_left_to_added_definitions():
    new = OLD + '''

def helper(items):
    seen = set()
    for item in items:
        seen.add(item)
    return seen
'''
    patch = make_diff(OLD, new)
    assert edited_blocks("src/x.py", patch, new) == []
    assert [d.name for d in added_definitions("src/x.py", patch)] == ["helper"]


def test_module_level_additions_are_not_blocks():
    new = OLD.replace("import os\n", "import os\nA = 1\nB = 2\nC = 3\nD = 4\nE = 5\n")
    assert blocks(OLD, new) == []


def test_two_separate_edits_in_one_function_are_two_blocks():
    new = OLD.replace('    header = "id,name"\n', '    header = "id,name"\n    a = 1\n    b = 2\n    c = 3\n    d = 4\n') \
             .replace("    return out\n", "    w = 1\n    x = 2\n    y = 3\n    z = 4\n    return out\n")
    found = blocks(OLD, new)
    assert [b.name for b in found] == ["export_rows", "export_rows"]
    assert found[0].end_line < found[1].start_line


def test_unparseable_full_source_gives_no_blocks():
    assert edited_blocks("src/x.py", make_diff(OLD, NEW_FUNC_BLOCK), "def broken(:\n") == []


def test_added_line_numbers_are_new_file_numbers():
    patch = make_diff(OLD, NEW_FUNC_BLOCK)
    lines = NEW_FUNC_BLOCK.splitlines()
    numbers = added_line_numbers(patch)
    assert numbers and all(lines[n - 1].strip() for n in numbers)
    assert lines[min(numbers) - 1].strip() == "chunks = []"
