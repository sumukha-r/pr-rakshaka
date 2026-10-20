"""
client.py — thin Groq wrapper used by every pr-rakshaka check.

Responsibilities (and nothing more):
  * one place that talks to the Groq API
  * retries with backoff on rate limits / network blips / 5xx
  * give up fast on errors retrying cannot fix (bad key, retired model,
    oversized prompt)
  * `chat_json()` — ask for a JSON object and parse it defensively

Usage:
    from src.llm.client import chat_json

    result = chat_json(
        system="You review pull requests. Respond with a JSON object.",
        user=prompt_text,
    )

Env vars:
    GROQ_API_KEY           required (free at console.groq.com)
    GROQ_MODEL             default "openai/gpt-oss-20b"
    GROQ_FALLBACK_MODEL    optional, e.g. "openai/gpt-oss-120b" — tried if the
                           primary model's rate limit is exhausted
    GROQ_REASONING_EFFORT  "low" (default) | "medium" | "high" — only used for
                           the gpt-oss models, which are reasoning models

Groq retires models often (llama-3.1-8b-instant stopped working on free
accounts on 2026-08-16). If 404 "model not found", run:

    python -m src.llm.client --models

to list what your key can use, then set GROQ_MODEL.
"""

import json
import logging
import os
import re
import time
from typing import Optional

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    Groq,
    RateLimitError,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL    = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
FALLBACK_MODEL   = os.getenv("GROQ_FALLBACK_MODEL", "")
REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")

# Reasoning models spend part of the output budget on hidden thinking
DEFAULT_MAX_TOKENS = 2048

# Models that accept reasoning_effort / include_reasoning.
_REASONING_MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b")

MAX_ATTEMPTS   = 4        # total tries per model
BASE_BACKOFF_S = 2.0      # 2s, 4s, 8s ... when the server gives no retry-after
MAX_WAIT_S     = 65.0     # a per-minute limit resets within ~60s
TIMEOUT_S      = 60.0


class LLMError(RuntimeError):
    """Raised when the LLM call cannot be completed. Message is user-presentable."""


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise LLMError(
                "GROQ_API_KEY is not set. Get a free key at console.groq.com "
                "and export it before running."
            )
        # max_retries=0: we do our own retrying below, so the SDK doesn't
        # silently multiply the attempts.
        _client = Groq(api_key=key, timeout=TIMEOUT_S, max_retries=0)
    return _client


# ---------------------------------------------------------------------------
# Core call with retry
# ---------------------------------------------------------------------------

def _retry_after_seconds(exc: Exception) -> Optional[float]:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    raw = headers.get("retry-after")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _model_extras(model: str) -> dict:
    """Extra request fields for models that need them (sent via extra_body so
    this works across Groq SDK versions)."""
    if model in _REASONING_MODELS:
        return {"reasoning_effort": REASONING_EFFORT, "include_reasoning": False}
    return {}


def _call_model(model: str, messages: list[dict], *, temperature: float,
                max_tokens: int, json_mode: bool) -> str:
    """One model, up to MAX_ATTEMPTS tries. Raises RateLimitError if still limited."""
    client = _get_client()
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    extras = _model_extras(model)
    if extras:
        kwargs["extra_body"] = extras

    last_exc: Exception = RuntimeError("no attempt made")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            usage = getattr(resp, "usage", None)
            if usage:
                log.debug("groq %s tokens: prompt=%s completion=%s",
                          model, usage.prompt_tokens, usage.completion_tokens)
            choice = resp.choices[0]
            text = choice.message.content or ""
            if not text.strip():
                log.warning(
                    "empty reply from %s (finish_reason=%s) — a reasoning model "
                    "may have used the whole max_tokens budget",
                    model, getattr(choice, "finish_reason", None),
                )
            return text

        except RateLimitError as exc:
            last_exc = exc
            wait = _retry_after_seconds(exc) or BASE_BACKOFF_S * 2 ** (attempt - 1)
            if wait > MAX_WAIT_S or attempt == MAX_ATTEMPTS:
                raise
            log.warning("rate limited on %s; sleeping %.0fs", model, wait)
            time.sleep(wait)

        except (APIConnectionError, APITimeoutError) as exc:
            last_exc = exc
            if attempt == MAX_ATTEMPTS:
                break
            time.sleep(BASE_BACKOFF_S * 2 ** (attempt - 1))

        except APIStatusError as exc:
            last_exc = exc
            code = exc.status_code
            if code >= 500 and attempt < MAX_ATTEMPTS:
                time.sleep(BASE_BACKOFF_S * 2 ** (attempt - 1))
                continue
            if code == 401:
                raise LLMError("Groq rejected the API key (401). Check GROQ_API_KEY.") from exc
            if code == 404:
                raise LLMError(
                    f"Model '{model}' is not available for this API key (404). "
                    "Groq retires models often. Run `python -m src.llm.client --models` "
                    "to list what your key can use, then set GROQ_MODEL."
                ) from exc
            if code == 413:
                raise LLMError(
                    "Prompt is too large for this model's request/token limit (413). "
                    "Send a smaller diff or fewer retrieved chunks."
                ) from exc
            raise LLMError(f"Groq API error {code}: {exc}") from exc

    raise LLMError(f"Groq request failed after {MAX_ATTEMPTS} attempts: {last_exc}") from last_exc


def chat(system: str, user: str, *, model: Optional[str] = None,
         temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS,
         json_mode: bool = False) -> str:
    """Send one system+user exchange and return the raw text reply."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    primary = model or DEFAULT_MODEL
    try:
        return _call_model(primary, messages, temperature=temperature,
                           max_tokens=max_tokens, json_mode=json_mode)
    except RateLimitError as exc:
        if FALLBACK_MODEL and FALLBACK_MODEL != primary:
            log.warning("%s rate limit exhausted; falling back to %s", primary, FALLBACK_MODEL)
            try:
                return _call_model(FALLBACK_MODEL, messages, temperature=temperature,
                                   max_tokens=max_tokens, json_mode=json_mode)
            except RateLimitError as exc2:
                exc = exc2
        raise LLMError(
            "Groq rate limit reached (the free tier has per-minute token and "
            "daily request caps). Wait a minute and retry."
        ) from exc


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object(text: str) -> dict:
    """
    Parse an LLM reply as a JSON object. Tolerates markdown fences and
    chatter around the object. Raises ValueError if no object can be found.
    """
    cleaned = _FENCE.sub("", text.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in model reply")
        data = json.loads(cleaned[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("model reply was JSON but not an object")
    return data


def chat_json(system: str, user: str, *, model: Optional[str] = None,
              temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    """
    Ask for a JSON object and return it parsed. If the first reply isn't
    valid JSON, retries once with a stricter reminder before giving up.
    """
    system = system.rstrip() + "\n\nRespond with a single valid JSON object and nothing else."
    reply = chat(system, user, model=model, temperature=temperature,
                 max_tokens=max_tokens, json_mode=True)
    try:
        return parse_json_object(reply)
    except ValueError as first_err:
        log.warning("invalid JSON from model (%s); retrying once", first_err)
        retry_user = (
            user
            + "\n\nYour previous reply was not valid JSON. "
              "Reply again with ONLY one JSON object."
        )
        reply = chat(system, retry_user, model=model, temperature=temperature,
                     max_tokens=max_tokens, json_mode=True)
        try:
            return parse_json_object(reply)
        except ValueError as exc:
            raise LLMError(f"Model did not return valid JSON twice: {exc}") from exc


# ---------------------------------------------------------------------------
# CLI:  python -m src.llm.client            smoke test
#       python -m src.llm.client --models   list models this key can use
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    try:
        if "--models" in sys.argv:
            for model_id in sorted(m.id for m in _get_client().models.list().data):
                print(model_id)
        else:
            print(f"[client] using model: {DEFAULT_MODEL}")
            print(chat_json(
                system="You are a connectivity test.",
                user='Return {"ok": true, "model_says": "<one short sentence>"}',
            ))
    except LLMError as err:
        print(f"[client] {err}")
        sys.exit(1)