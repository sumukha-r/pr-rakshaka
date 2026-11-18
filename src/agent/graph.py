"""
graph.py — the same agent loop as loop.py, written as a LangGraph state machine (M5).

    run_agent_graph(system, user, tools, ...) -> AgentResult      (same signature as run_agent)

Why port it: to show the loop's logic as an explicit graph, which tracing tools draw for you.
The behaviour is meant to be identical, and tests/unit/test_agent_graph.py runs the same scripted
conversations through both engines and compares them.

    START -> gate -> model -> tools -> gate ... -> finish -> END
              |        |                            ^
              +--------+----------------------------+

  gate    stop conditions: out of steps, tokens or time
  model   one model turn: handles a malformed tool call, the "look first" nudge, or a final answer
  tools   runs the tool calls the model asked for (budget, repeat memo, untrusted wrapping)
  finish  if the budget ran out ask once for a final answer; then the one JSON repair turn

The loop in loop.py stays the default. Select this one with AGENT_ENGINE=langgraph.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agent.llm import MalformedToolCall, Turn, chat_with_tools
from src.agent.loop import (UNTRUSTED_RULE, AgentResult, Budget, Trace, _parse, wrap_untrusted)
from src.agent.tools import RepoTools
from src.llm.client import LLMError


class State(TypedDict):
    messages: list
    events: list            # trace events
    seen: dict              # tool-call memo
    steps: int
    calls: int
    ptok: int
    ctok: int
    malformed: int
    nudged: bool
    stop: str
    text: str
    who: tuple
    t0: float
    phase: str              # which node comes next
    pending: list           # tool calls of the last model turn


def _add(events: list, kind: str, **data: Any) -> list:
    return events + [{"kind": kind, **data}]


def build_graph(system: str, user: str, tools: RepoTools, *, provider, model, budget: Budget,
                min_tool_calls: int, llm: Callable[..., Turn]):
    specs = tools.specs()

    def spent(s: State) -> str:
        if s["ptok"] + s["ctok"] >= budget.max_total_tokens:
            return "token_budget"
        if time.monotonic() - s["t0"] >= budget.max_seconds:
            return "time_budget"
        return ""

    def gate(s: State) -> dict:
        if s["steps"] >= budget.max_steps:
            return {"phase": "finish"}
        why = spent(s)
        return {"phase": "finish", "stop": why} if why else {"phase": "model"}

    def model_node(s: State) -> dict:
        steps = s["steps"] + 1
        try:
            turn = llm(s["messages"], specs, provider=provider, model=model)
        except MalformedToolCall as exc:
            malformed = s["malformed"] + 1
            if malformed > 2:
                raise LLMError("the model kept producing malformed tool calls") from exc
            return {"steps": steps, "malformed": malformed, "phase": "gate",
                    "events": _add(s["events"], "malformed_tool_call", step=steps, detail=str(exc)),
                    "messages": s["messages"] + [{"role": "user", "content": "Your last tool call was "
                        "malformed. Call a tool again with valid JSON arguments, or give your final answer."}]}
        upd: dict = {"steps": steps, "ptok": s["ptok"] + turn.prompt_tokens,
                     "ctok": s["ctok"] + turn.completion_tokens, "who": (turn.provider, turn.model),
                     "events": _add(s["events"], "model", step=steps, tools=[c.name for c in turn.tool_calls],
                                    prompt_tokens=turn.prompt_tokens, completion_tokens=turn.completion_tokens,
                                    seconds=round(turn.seconds, 2), text=turn.content[:1500])}
        if not turn.tool_calls:
            if s["calls"] < min_tool_calls and not s["nudged"] and steps < budget.max_steps:
                upd.update(nudged=True, phase="gate",
                           events=_add(upd["events"], "nudge", step=steps),
                           messages=s["messages"] + [turn.message, {"role": "user", "content":
                               "You have not used any tool yet. Use the tools to look at the repository "
                               "first, then give your final answer."}])
                return upd
            upd.update(text=turn.content, stop="answered", phase="finish")
            return upd
        upd.update(messages=s["messages"] + [turn.message], pending=list(turn.tool_calls), phase="tools")
        return upd

    def tools_node(s: State) -> dict:
        messages, events, seen, calls = list(s["messages"]), s["events"], dict(s["seen"]), s["calls"]
        for call in s["pending"]:
            key = f"{call.name}|{json.dumps(call.arguments, sort_keys=True)}"
            if call.error:
                result = f"error: {call.error}"
            elif calls >= budget.max_tool_calls:
                result = "error: tool-call budget used up. Give your final answer now."
            elif key in seen:
                result = "(you already made this exact call; its result is repeated)\n" + seen[key]
            else:
                calls += 1
                result = tools.call(call.name, call.arguments)
                seen[key] = result
            events = _add(events, "tool", step=s["steps"], name=call.name, args=call.arguments,
                          chars=len(result), preview=result[:300])
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": wrap_untrusted(call.name, result)})
        return {"messages": messages, "events": events, "seen": seen, "calls": calls,
                "pending": [], "phase": "gate"}

    def finish(s: State) -> dict:
        messages, events, ptok, ctok, who = list(s["messages"]), s["events"], s["ptok"], s["ctok"], s["who"]
        text, steps = s["text"], s["steps"]
        if s["stop"] != "answered":
            events = _add(events, "budget", reason=s["stop"])
            messages.append({"role": "user", "content": "Your budget is used up. Give your best final "
                             "answer now as the JSON object, with no tool calls."})
            turn = llm(messages, [], provider=provider, model=model)
            ptok, ctok, who = ptok + turn.prompt_tokens, ctok + turn.completion_tokens, (turn.provider, turn.model)
            text = turn.content
            events = _add(events, "model", step=steps + 1, tools=[], prompt_tokens=turn.prompt_tokens,
                          completion_tokens=turn.completion_tokens, seconds=round(turn.seconds, 2),
                          text=text[:1500])
        if _parse(text) is None and text.strip():            # one repair turn, like v1's retry
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "That was not a valid JSON object. Reply again "
                             "with only the JSON object described in the task, nothing else."})
            try:
                turn = llm(messages, [], provider=provider, model=model)
            except (LLMError, MalformedToolCall):
                turn = None
            if turn is not None:
                ptok, ctok, who = ptok + turn.prompt_tokens, ctok + turn.completion_tokens, (turn.provider, turn.model)
                events = _add(events, "repair", text=turn.content[:1500])
                text = turn.content
        return {"text": text, "events": events, "ptok": ptok, "ctok": ctok, "who": who, "messages": messages}

    g = StateGraph(State)
    g.add_node("gate", gate)
    g.add_node("model", model_node)
    g.add_node("tools", tools_node)
    g.add_node("finish", finish)
    g.add_edge(START, "gate")
    g.add_conditional_edges("gate", lambda s: s["phase"], {"model": "model", "finish": "finish"})
    g.add_conditional_edges("model", lambda s: s["phase"], {"gate": "gate", "tools": "tools", "finish": "finish"})
    g.add_edge("tools", "gate")
    g.add_edge("finish", END)
    return g.compile()


def run_agent_graph(system: str, user: str, tools: RepoTools, *, provider: Optional[str] = None,
                    model: Optional[str] = None, budget: Optional[Budget] = None, min_tool_calls: int = 0,
                    llm: Callable[..., Turn] = chat_with_tools) -> AgentResult:
    budget = budget or Budget()
    graph = build_graph(system, user, tools, provider=provider, model=model, budget=budget,
                        min_tool_calls=min_tool_calls, llm=llm)
    start: State = {
        "messages": [{"role": "system", "content": system.rstrip() + "\n\n" + UNTRUSTED_RULE},
                     {"role": "user", "content": user}],
        "events": [], "seen": {}, "steps": 0, "calls": 0, "ptok": 0, "ctok": 0, "malformed": 0,
        "nudged": False, "stop": "max_steps", "text": "", "who": ("", ""), "t0": time.monotonic(),
        "phase": "gate", "pending": []}
    end = graph.invoke(start, {"recursion_limit": budget.max_steps * 3 + 12})
    trace = Trace(end["events"])
    return AgentResult(end["text"], _parse(end["text"]), end["stop"], end["steps"], end["calls"],
                       end["ptok"], end["ctok"], round(time.monotonic() - end["t0"], 2),
                       end["who"][0], end["who"][1], trace)
