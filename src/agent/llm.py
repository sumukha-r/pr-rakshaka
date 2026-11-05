"""
llm.py — one chat turn with tool calling, for Groq and for a local server.

src/llm/client.py (v1) is left untouched on purpose: it is the baseline the agent
is compared against. This module reuses its configuration, retry constants and
usage statistics, and adds what v1 never needed: sending tool definitions and
reading tool calls back.

Both backends speak the OpenAI function-calling format. Two things go wrong in
practice and are handled here rather than crashing the run:
  * the model emits a malformed tool call (Groq answers 400 'tool_use_failed');
    this raises MalformedToolCall so the loop can ask it to try again;
  * a local model returns arguments that are not valid JSON; the call comes back
    with `error` set and the loop tells the model.
Rate limits and dropped connections are retried with backoff.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import requests
from groq import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from src.llm import client as v1

log = logging.getLogger(__name__)


class MalformedToolCall(RuntimeError):
    """The model tried to call a tool but produced something unusable."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    error: str = ""            # set when the arguments could not be parsed


@dataclass
class Turn:
    content: str
    tool_calls: list[ToolCall]
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    message: dict = field(default_factory=dict)   # the assistant message, ready to append


def _parse_calls(raw_calls) -> list[ToolCall]:
    calls = []
    for i, rc in enumerate(raw_calls or []):
        fn = rc["function"] if isinstance(rc, dict) else rc.function
        get = (lambda k: fn.get(k)) if isinstance(fn, dict) else (lambda k: getattr(fn, k, None))
        cid = (rc.get("id") if isinstance(rc, dict) else getattr(rc, "id", None)) or f"call_{i}"
        args, err = get("arguments"), ""
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args, err = {}, "arguments were not valid JSON"
        if not isinstance(args, dict):
            args, err = {}, err or "arguments were not a JSON object"
        calls.append(ToolCall(cid, get("name") or "", args, err))
    return calls


def _assistant_message(content: str, calls: list[ToolCall]) -> dict:
    msg: dict = {"role": "assistant", "content": content or ""}
    if calls:
        msg["tool_calls"] = [{"id": c.id, "type": "function",
                              "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                             for c in calls]
    return msg


def _groq_turn(model: str, messages, tools, temperature, max_tokens) -> Turn:
    client = v1._get_groq()
    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if tools:
        kwargs.update(tools=tools, tool_choice="auto")
    extras = v1._model_extras(model)
    if extras:
        kwargs["extra_body"] = extras
    start = time.monotonic()
    last: Exception = RuntimeError("no attempt made")
    for attempt in range(1, v1.MAX_ATTEMPTS + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            calls = _parse_calls(getattr(msg, "tool_calls", None))
            usage = getattr(resp, "usage", None)
            content = msg.content or ""
            return Turn(content, calls, "groq", model,
                        int(getattr(usage, "prompt_tokens", 0) or 0),
                        int(getattr(usage, "completion_tokens", 0) or 0),
                        time.monotonic() - start, _assistant_message(content, calls))
        except RateLimitError as exc:
            last = exc
            wait = v1._retry_after_seconds(exc) or v1.BASE_BACKOFF_S * 2 ** (attempt - 1)
            if wait > v1.MAX_WAIT_S or attempt == v1.MAX_ATTEMPTS:
                raise v1.LLMError("Groq rate limit reached; wait a minute and retry.") from exc
            log.warning("rate limited on %s; sleeping %.0fs", model, wait)
            time.sleep(wait)
        except (APIConnectionError, APITimeoutError) as exc:
            last = exc
            if attempt < v1.MAX_ATTEMPTS:
                time.sleep(v1.BASE_BACKOFF_S * 2 ** (attempt - 1))
        except APIStatusError as exc:
            last = exc
            code = exc.status_code
            if code == 400 and "tool_use_failed" in str(exc):
                raise MalformedToolCall(str(exc)[:200]) from exc
            if code >= 500 and attempt < v1.MAX_ATTEMPTS:
                time.sleep(v1.BASE_BACKOFF_S * 2 ** (attempt - 1))
                continue
            raise v1.LLMError(f"Groq API error {code}: {exc}") from exc
    raise v1.LLMError(f"Groq request failed after {v1.MAX_ATTEMPTS} attempts: {last}") from last


class _LocalUnavailable(Exception):
    pass


def _local_turn(model: str, messages, tools, temperature, max_tokens) -> Turn:
    payload = {"model": model, "messages": messages, "temperature": temperature,
               "max_tokens": max_tokens, "stream": False}
    if tools:
        payload["tools"] = tools
    start = time.monotonic()
    try:
        resp = requests.post(f"{v1.LOCAL_BASE_URL}/chat/completions", json=payload,
                             timeout=v1.LOCAL_TIMEOUT_S)
    except requests.ConnectionError as exc:
        raise _LocalUnavailable(f"can't reach the local server at {v1.LOCAL_BASE_URL}") from exc
    except requests.Timeout as exc:
        raise v1.LLMError(f"Local model '{model}' did not answer within {v1.LOCAL_TIMEOUT_S:.0f}s.") from exc
    if resp.status_code == 404:
        raise _LocalUnavailable(f"the local server has no model '{model}'")
    if resp.status_code == 400 and tools and "tool" in resp.text.lower():
        raise v1.LLMError(f"Local model '{model}' does not support tool calling "
                          f"(server said: {resp.text[:200]}). Pick a model that does.")
    if resp.status_code >= 400:
        raise v1.LLMError(f"Local server error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    msg = (data.get("choices") or [{}])[0].get("message", {})
    calls = _parse_calls(msg.get("tool_calls"))
    usage = data.get("usage") or {}
    content = msg.get("content") or ""
    return Turn(content, calls, "local", model, int(usage.get("prompt_tokens") or 0),
                int(usage.get("completion_tokens") or 0), time.monotonic() - start,
                _assistant_message(content, calls))


def chat_with_tools(messages: list[dict], tools: list[dict], *, provider: Optional[str] = None,
                    model: Optional[str] = None, temperature: float = 0.0,
                    max_tokens: int = v1.DEFAULT_MAX_TOKENS) -> Turn:
    """One model turn. `tools` may be empty (a plain answer). Same provider rules as v1."""
    prov = v1.active_provider(provider)
    if prov == "groq":
        turn = _groq_turn(model or v1.DEFAULT_MODEL, messages, tools, temperature, max_tokens)
    elif prov == "local":
        try:
            turn = _local_turn(model or v1.LOCAL_MODEL, messages, tools, temperature, max_tokens)
        except _LocalUnavailable as exc:
            raise v1.LLMError(f"Local backend unavailable: {exc}") from exc
    else:
        try:
            turn = _local_turn(model or v1.LOCAL_MODEL, messages, tools, temperature, max_tokens)
        except _LocalUnavailable as exc:
            v1._warn_once(f"local backend unavailable ({exc}); falling back to Groq")
            turn = _groq_turn(v1.DEFAULT_MODEL, messages, tools, temperature, max_tokens)
    v1._record(v1.CallResult(turn.content, turn.provider, turn.model, turn.prompt_tokens,
                             turn.completion_tokens, turn.seconds))
    return turn
