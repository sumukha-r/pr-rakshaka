"""
client.py — LLM access for pr-rakshaka: a local model and/or Groq, one API.

Every check calls `chat_json(system, user)`. Which backend answers is decided
by configuration, never by the check:

    LLM_PROVIDER=auto   (default) local first; if the local server can't be
                        reached or lacks the model, fall back to Groq
    LLM_PROVIDER=local  local only (errors if unavailable)
    LLM_PROVIDER=groq   Groq only

Switch for one block of code without touching env vars:

    from src.llm.client import use_provider
    with use_provider("groq"):
        run_the_check()

Env vars:
    LLM_PROVIDER           auto | local | groq            (default auto)

    LOCAL_BASE_URL         OpenAI-compatible server       (default http://localhost:11434/v1, Ollama)
    LOCAL_MODEL            model name on that server      (default qwen2.5-coder:7b)
    LOCAL_TIMEOUT_S        per-request timeout            (default 300; 7B models are slow on CPU)

    GROQ_API_KEY           required for Groq (free at console.groq.com)
    GROQ_MODEL             default "openai/gpt-oss-20b"
    GROQ_FALLBACK_MODEL    optional, e.g. "openai/gpt-oss-120b", used if the primary is rate limited
    GROQ_REASONING_EFFORT  low (default) | medium | high  (gpt-oss models only)

CLI:
    python -m src.llm.client                      smoke test with the active provider
    python -m src.llm.client --provider local     ... force a backend
    python -m src.llm.client --models             list models each backend offers
"""

import contextvars
import json
import logging
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import requests
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

PROVIDERS = ("auto", "local", "groq")
PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()

LOCAL_BASE_URL  = os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1").rstrip("/")
LOCAL_MODEL     = os.getenv("LOCAL_MODEL", "qwen2.5-coder:7b")
LOCAL_TIMEOUT_S = float(os.getenv("LOCAL_TIMEOUT_S", "300"))

DEFAULT_MODEL    = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
FALLBACK_MODEL   = os.getenv("GROQ_FALLBACK_MODEL", "")
REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")

# Reasoning models spend part of the output budget on hidden thinking, so the
# default is generous; an empty reply usually means this was too small.
DEFAULT_MAX_TOKENS = 2048

_REASONING_MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b")

MAX_ATTEMPTS   = 4        # total tries per Groq model
BASE_BACKOFF_S = 2.0      # 2s, 4s, 8s ... when the server gives no retry-after
MAX_WAIT_S     = 65.0     # per-minute limits reset within ~60s; longer = daily quota
TIMEOUT_S      = 60.0     # Groq request timeout


class LLMError(RuntimeError):
    """Raised when the LLM call cannot be completed. Message is user-presentable."""


class _LocalUnavailable(Exception):
    """Local server unreachable or model missing: the one case 'auto' falls back on."""


@dataclass
class CallResult:
    text: str
    provider: str            # "local" | "groq"
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------

_provider_override: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "llm_provider_override", default=None
)


def _check_provider(name: str) -> str:
    name = name.lower()
    if name not in PROVIDERS:
        raise LLMError(f"Unknown provider '{name}'. Use one of: {', '.join(PROVIDERS)}.")
    return name


@contextmanager
def use_provider(name: str):
    """Force a backend for everything inside the `with` block."""
    token = _provider_override.set(_check_provider(name))
    try:
        yield
    finally:
        _provider_override.reset(token)


def active_provider(explicit: Optional[str] = None) -> str:
    return _check_provider(explicit or _provider_override.get() or PROVIDER)


# ---------------------------------------------------------------------------
# Usage stats (handy for comparing backends and for feeling the Groq limits)
# ---------------------------------------------------------------------------

_stats: dict[str, dict] = {}
_last_call: Optional[CallResult] = None
_warned: set[str] = set()


def _record(r: CallResult) -> None:
    global _last_call
    _last_call = r
    s = _stats.setdefault(r.provider, {"calls": 0, "prompt_tokens": 0,
                                       "completion_tokens": 0, "seconds": 0.0,
                                       "model": r.model})
    s["calls"] += 1
    s["prompt_tokens"] += r.prompt_tokens
    s["completion_tokens"] += r.completion_tokens
    s["seconds"] += r.seconds
    s["model"] = r.model


def stats() -> dict[str, dict]:
    """Per-provider totals since the last reset_stats()."""
    return {k: dict(v) for k, v in _stats.items()}


def reset_stats() -> None:
    _stats.clear()


def last_call() -> Optional[CallResult]:
    return _last_call


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        log.warning(msg)


# ---------------------------------------------------------------------------
# Local backend (any OpenAI-compatible server: Ollama, LM Studio, llama.cpp)
# ---------------------------------------------------------------------------

_local_json_ok = True   # flips to False if the server rejects response_format


def _local_call(model: str, messages: list[dict], *, temperature: float,
                max_tokens: int, json_mode: bool) -> CallResult:
    global _local_json_ok
    url = f"{LOCAL_BASE_URL}/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": temperature,
               "max_tokens": max_tokens, "stream": False}
    use_json = json_mode and _local_json_ok
    if use_json:
        payload["response_format"] = {"type": "json_object"}

    start = time.monotonic()
    for attempt in (1, 2):
        try:
            resp = requests.post(url, json=payload, timeout=LOCAL_TIMEOUT_S)
        except requests.ConnectionError as exc:
            raise _LocalUnavailable(
                f"can't reach the local server at {LOCAL_BASE_URL} (is Ollama / "
                f"LM Studio running?)"
            ) from exc
        except requests.Timeout as exc:
            raise LLMError(
                f"Local model '{model}' did not answer within {LOCAL_TIMEOUT_S:.0f}s. "
                "Raise LOCAL_TIMEOUT_S, shorten the prompt, or check the model is "
                "running on the GPU."
            ) from exc

        if resp.status_code == 404:
            raise _LocalUnavailable(
                f"the local server has no model '{model}'. Run "
                f"`python -m src.llm.client --models` to see what it has "
                f"(Ollama: `ollama list` / `ollama pull`), then set LOCAL_MODEL."
            )
        if resp.status_code == 400 and use_json and attempt == 1:
            # Some servers don't support response_format; retry plain and remember.
            _warn_once("local server rejected JSON mode; continuing without it "
                       "(relying on prompt + tolerant parsing)")
            _local_json_ok = False
            payload.pop("response_format", None)
            use_json = False
            continue
        if resp.status_code >= 400:
            raise LLMError(f"Local server error {resp.status_code}: {resp.text[:300]}")
        break

    data = resp.json()
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    usage = data.get("usage") or {}
    pt, ct = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)

    est = sum(len(m["content"]) for m in messages) // 4
    if pt and est > 500 and pt < 0.6 * est:
        _warn_once(
            f"local server reported only {pt} prompt tokens for a ~{est}-token prompt: "
            "the context window is probably too small and the prompt was cut. "
            "Raise it (Ollama: Modelfile `PARAMETER num_ctx 8192`)."
        )
    if not text.strip():
        log.warning("empty reply from local model %s", model)
    return CallResult(text, "local", model, pt, ct, time.monotonic() - start)


# ---------------------------------------------------------------------------
# Groq backend
# ---------------------------------------------------------------------------

_groq_client: Optional[Groq] = None


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise LLMError(
                "GROQ_API_KEY is not set. Get a free key at console.groq.com "
                "and export it before running."
            )
        # max_retries=0: we do our own retrying, so the SDK doesn't multiply attempts.
        _groq_client = Groq(api_key=key, timeout=TIMEOUT_S, max_retries=0)
    return _groq_client


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


def _groq_call(model: str, messages: list[dict], *, temperature: float,
               max_tokens: int, json_mode: bool) -> CallResult:
    """One Groq model, up to MAX_ATTEMPTS tries. Raises RateLimitError if still limited."""
    client = _get_groq()
    kwargs = dict(model=model, messages=messages, temperature=temperature,
                  max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    extras = _model_extras(model)
    if extras:
        kwargs["extra_body"] = extras

    start = time.monotonic()
    last_exc: Exception = RuntimeError("no attempt made")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            usage = getattr(resp, "usage", None)
            pt = int(getattr(usage, "prompt_tokens", 0) or 0)
            ct = int(getattr(usage, "completion_tokens", 0) or 0)
            choice = resp.choices[0]
            text = choice.message.content or ""
            if not text.strip():
                log.warning(
                    "empty reply from %s (finish_reason=%s): a reasoning model "
                    "may have used the whole max_tokens budget",
                    model, getattr(choice, "finish_reason", None),
                )
            return CallResult(text, "groq", model, pt, ct, time.monotonic() - start)

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


def _groq_with_fallback(model: Optional[str], messages: list[dict], *,
                        temperature: float, max_tokens: int, json_mode: bool) -> CallResult:
    primary = model or DEFAULT_MODEL
    try:
        return _groq_call(primary, messages, temperature=temperature,
                          max_tokens=max_tokens, json_mode=json_mode)
    except RateLimitError as exc:
        if FALLBACK_MODEL and FALLBACK_MODEL != primary:
            log.warning("%s rate limit exhausted; falling back to %s", primary, FALLBACK_MODEL)
            try:
                return _groq_call(FALLBACK_MODEL, messages, temperature=temperature,
                                  max_tokens=max_tokens, json_mode=json_mode)
            except RateLimitError as exc2:
                exc = exc2
        raise LLMError(
            "Groq rate limit reached (the free tier has per-minute token and "
            "daily request caps). Wait a minute and retry."
        ) from exc


# ---------------------------------------------------------------------------
# Public chat API
# ---------------------------------------------------------------------------

def chat_detailed(system: str, user: str, *, model: Optional[str] = None,
                  temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS,
                  json_mode: bool = False, provider: Optional[str] = None) -> CallResult:
    """
    One system+user exchange. `model` overrides the model of the first backend
    tried; a fallback backend always uses its own default model.
    """
    messages = [{"role": "system", "content": system},
                {"role": "user",   "content": user}]
    kw = dict(temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)
    prov = active_provider(provider)

    if prov == "groq":
        result = _groq_with_fallback(model, messages, **kw)
    elif prov == "local":
        try:
            result = _local_call(model or LOCAL_MODEL, messages, **kw)
        except _LocalUnavailable as exc:
            raise LLMError(f"Local backend unavailable: {exc}") from exc
    else:  # auto: local first, Groq if local isn't there
        try:
            result = _local_call(model or LOCAL_MODEL, messages, **kw)
        except _LocalUnavailable as exc:
            _warn_once(f"local backend unavailable ({exc}); falling back to Groq")
            result = _groq_with_fallback(None, messages, **kw)

    _record(result)
    return result


def chat(system: str, user: str, **kwargs) -> str:
    """Like chat_detailed, returning just the reply text."""
    return chat_detailed(system, user, **kwargs).text


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
              temperature: float = 0.0, max_tokens: int = DEFAULT_MAX_TOKENS,
              provider: Optional[str] = None) -> dict:
    """
    Ask for a JSON object and return it parsed. If the first reply isn't
    valid JSON, retries once with a stricter reminder before giving up.
    """
    system = system.rstrip() + "\n\nRespond with a single valid JSON object and nothing else."
    kw = dict(model=model, temperature=temperature, max_tokens=max_tokens,
              json_mode=True, provider=provider)
    reply = chat(system, user, **kw)
    try:
        return parse_json_object(reply)
    except ValueError as first_err:
        log.warning("invalid JSON from model (%s); retrying once", first_err)
        retry_user = (user + "\n\nYour previous reply was not valid JSON. "
                             "Reply again with ONLY one JSON object.")
        reply = chat(system, retry_user, **kw)
        try:
            return parse_json_object(reply)
        except ValueError as exc:
            raise LLMError(f"Model did not return valid JSON twice: {exc}") from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _list_models(which: str) -> None:
    if which in ("local", "auto"):
        print(f"# local ({LOCAL_BASE_URL})")
        try:
            data = requests.get(f"{LOCAL_BASE_URL}/models", timeout=5).json()
            for m in sorted(x["id"] for x in data.get("data", [])):
                print(m)
        except Exception as exc:  # noqa: BLE001 - CLI diagnostic
            print(f"(unreachable: {exc.__class__.__name__})")
    if which in ("groq", "auto"):
        print("# groq")
        try:
            for m in sorted(x.id for x in _get_groq().models.list().data):
                print(m)
        except LLMError as exc:
            print(f"({exc})")


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="python -m src.llm.client")
    ap.add_argument("--provider", choices=PROVIDERS, help="force a backend")
    ap.add_argument("--models", action="store_true", help="list models the backend offers")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[client] %(message)s")

    try:
        if args.models:
            _list_models(active_provider(args.provider))
        else:
            out = chat_json(
                system="You are a connectivity test.",
                user='Return {"ok": true, "model_says": "<one short sentence>"}',
                provider=args.provider,
            )
            r = last_call()
            print(f"[client] answered by {r.provider} / {r.model} in {r.seconds:.1f}s "
                  f"(prompt={r.prompt_tokens} completion={r.completion_tokens} tokens)")
            print(out)
    except LLMError as err:
        print(f"[client] {err}")
        sys.exit(1)