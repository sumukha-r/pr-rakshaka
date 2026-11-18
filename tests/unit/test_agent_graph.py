"""M5: the LangGraph engine must behave exactly like the plain loop. Same scripted model, both engines."""
import pytest

from src.agent import checks
from src.agent.graph import run_agent_graph
from src.agent.llm import MalformedToolCall
from src.agent.loop import Budget, run_agent
from src.llm.client import LLMError
from tests.unit.test_agent import ANSWER, Script, call, make_tools, say

BAD = "not json at all"

# name -> (turns, run_agent kwargs)
SCENARIOS = {
    "answers_without_tools": ([say(ANSWER)], {}),
    "tool_then_answer": ([call("pr_context"), say(ANSWER)], {}),
    "repeated_call_is_memoised": ([call("pr_context", cid="a"), call("pr_context", cid="b"), say(ANSWER)], {}),
    "nudge_when_no_tool_used": ([say(ANSWER), call("pr_context"), say(ANSWER)], {"min_tool_calls": 1}),
    "nudge_only_once": ([say(ANSWER), say(ANSWER)], {"min_tool_calls": 1}),
    "malformed_then_answer": ([MalformedToolCall("bad"), say(ANSWER)], {}),
    "bad_arguments_come_back_as_text": ([call("pr_context", error="not json"), say(ANSWER)], {}),
    "unknown_tool": ([call("no_such_tool"), say(ANSWER)], {}),
    "step_budget_forces_an_answer": (
        [call("pr_context", cid="1"), call("find_tests", {"path": "pkg/hooks/base_aws.py"}, cid="2"),
         say(ANSWER)], {"budget": Budget(max_steps=2)}),
    "tool_call_budget": (
        [call("pr_context", cid="1"), call("find_tests", {"path": "pkg/hooks/base_aws.py"}, cid="2"),
         say(ANSWER)], {"budget": Budget(max_tool_calls=1)}),
    "token_budget": ([call("pr_context", cid="1"), say(ANSWER)], {"budget": Budget(max_total_tokens=15)}),
    "invalid_json_is_repaired_once": ([say(BAD), say(ANSWER)], {}),
    "still_invalid_after_repair": ([say(BAD), say(BAD)], {}),
}


def summary(r, script):
    return {"data": r.data, "text": r.text, "stop": r.stop, "steps": r.steps, "calls": r.tool_calls,
            "tokens": (r.prompt_tokens, r.completion_tokens), "who": (r.provider, r.model),
            "events": r.trace.events,
            "sent": [[m.get("content") for m in msgs] for msgs, _ in script.seen]}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_both_engines_agree(name):
    turns, kw = SCENARIOS[name]
    out = []
    for engine in (run_agent, run_agent_graph):
        script = Script(*turns)
        out.append(summary(engine("sys", "task", make_tools(), llm=script, **kw), script))
    assert out[0] == out[1]


def test_too_many_malformed_calls_fail_the_same_way_in_both():
    for engine in (run_agent, run_agent_graph):
        script = Script(MalformedToolCall("a"), MalformedToolCall("b"), MalformedToolCall("c"))
        with pytest.raises(LLMError, match="malformed"):
            engine("sys", "task", make_tools(), llm=script)


def test_the_graph_has_the_four_nodes_it_documents():
    from src.agent.graph import build_graph
    g = build_graph("s", "u", make_tools(), provider=None, model=None, budget=Budget(), min_tool_calls=0,
                    llm=Script())
    assert {"gate", "model", "tools", "finish"} <= set(g.get_graph().nodes)


# --------------------------------------------------------------------------- choosing the engine

def test_engine_defaults_to_the_plain_loop(monkeypatch):
    monkeypatch.delenv("AGENT_ENGINE", raising=False)
    assert checks._engine() is checks.run_agent


def test_engine_can_be_switched_to_langgraph(monkeypatch):
    monkeypatch.setenv("AGENT_ENGINE", "langgraph")
    assert checks._engine() is run_agent_graph


def test_an_unknown_engine_name_is_an_error(monkeypatch):
    monkeypatch.setenv("AGENT_ENGINE", "crewai")
    with pytest.raises(ValueError, match="AGENT_ENGINE"):
        checks._engine()
