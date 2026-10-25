from types import ModuleType, SimpleNamespace

from src.checks.smell_catalog import SMELLS, catalogue_text, complexity_signals


def test_catalogue_has_six_unique_concrete_hot_path_categories():
    keys = [smell.key for smell in SMELLS]
    assert len(keys) == 6
    assert len(set(keys)) == len(keys)
    assert {
        "repeated_work",
        "quadratic_scan",
        "n_plus_one",
        "async_blocking",
        "large_materialization",
        "repeated_computation",
    } == set(keys)
    assert all(smell.title and smell.description and smell.evidence_hint for smell in SMELLS)


def test_catalogue_text_uses_real_line_breaks():
    text = catalogue_text()
    assert "\n" in text
    assert "\\n" not in text
    assert text.count("\n") == len(SMELLS) - 1
    assert "N+1 I/O or database operations" in text


def test_complexity_signal_reports_high_complexity(monkeypatch):
    # Stub Radon so this unit test is deterministic even when Radon is not installed.
    radon = ModuleType("radon")
    complexity = ModuleType("radon.complexity")
    complexity.cc_visit = lambda source: [
        SimpleNamespace(complexity=8, name="branchy", lineno=1),
        SimpleNamespace(complexity=7, name="simple", lineno=20),
    ]
    radon.complexity = complexity
    monkeypatch.setitem(__import__("sys").modules, "radon", radon)
    monkeypatch.setitem(__import__("sys").modules, "radon.complexity", complexity)

    result = complexity_signals("def branchy(x):\n    return x", filename="sample.py")

    assert len(result) == 1
    assert "sample.py:1 branchy" in result[0]
    assert "complexity 8" in result[0]
    assert "review signal only" in result[0]


def test_complexity_signal_handles_parse_failure(monkeypatch):
    radon = ModuleType("radon")
    complexity = ModuleType("radon.complexity")

    def fail_to_parse(source):
        raise SyntaxError("invalid Python")

    complexity.cc_visit = fail_to_parse
    radon.complexity = complexity
    monkeypatch.setitem(__import__("sys").modules, "radon", radon)
    monkeypatch.setitem(__import__("sys").modules, "radon.complexity", complexity)
    assert complexity_signals("not valid python", filename="broken.py") == []
