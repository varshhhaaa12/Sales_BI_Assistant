"""Provider-agnostic LLM client, configured entirely by environment variables.

It speaks the OpenAI-compatible Chat Completions protocol (POST {base_url}/chat/completions),
which Gemini, OpenAI, Anthropic, Groq, OpenRouter, Ollama and most others expose. Changing
provider is a configuration change, not a code change:

    LLM_PROVIDER   optional preset: gemini | openai | anthropic | groq | openrouter | ollama
    LLM_MODEL      model name, e.g. gemini-3.5-flash            (required)
    LLM_API_KEY    API key (falls back to the preset's own variable, e.g. GEMINI_API_KEY)
    LLM_BASE_URL   endpoint base URL (overrides the preset; use for any other compatible API)
    LLM_TIMEOUT    seconds, default 30
    LLM_JSON_MODE  none (default) | json_object | json_schema - native JSON mode, if the
                   provider supports it; the schema is always in the prompt anyway

Structured output works the same way everywhere: the JSON schema goes in the prompt, the
reply is validated with Pydantic, and an invalid reply gets one retry with the error shown
to the model.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# preset -> (base URL, environment variable holding its key)
PRESETS: dict[str, tuple[str, str | None]] = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),   # local, no key
}
JSON_MODES = ("none", "json_object", "json_schema")


class LLMError(Exception):
    """Configuration, transport or output problem. Callers fall back to the rules."""


@dataclass
class LLMConfig:
    provider: str
    model: str
    base_url: str
    api_key: str | None
    timeout: float = 30.0
    json_mode: str = "none"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "LLMConfig | None":
        """Build the config from env vars. Returns None when no LLM is configured."""
        env = dict(os.environ) if env is None else env
        if env.get("BI_USE_LLM", "1") == "0":
            return None
        provider = env.get("LLM_PROVIDER", "").strip().lower()
        model = env.get("LLM_MODEL", "").strip()
        if not provider and not env.get("LLM_BASE_URL"):
            return None                                   # nothing configured: rules only
        if provider and provider not in PRESETS and not env.get("LLM_BASE_URL"):
            raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use one of {sorted(PRESETS)} "
                           "or set LLM_BASE_URL for any OpenAI-compatible endpoint.")
        if not model:
            raise LLMError("LLM_MODEL must be set, e.g. LLM_MODEL=gemini-3.5-flash")

        preset_url, key_var = PRESETS.get(provider, ("", None))
        base_url = (env.get("LLM_BASE_URL") or preset_url).rstrip("/")
        api_key = env.get("LLM_API_KEY") or (env.get(key_var) if key_var else None)
        if provider == "gemini" and not api_key:
            api_key = env.get("GOOGLE_API_KEY")
        if not api_key and provider != "ollama" and "localhost" not in base_url:
            raise LLMError(f"No API key: set LLM_API_KEY{f' or {key_var}' if key_var else ''}.")

        json_mode = env.get("LLM_JSON_MODE", "none").strip().lower()
        if json_mode not in JSON_MODES:
            raise LLMError(f"LLM_JSON_MODE must be one of {JSON_MODES}")
        return cls(provider or "custom", model, base_url, api_key,
                   float(env.get("LLM_TIMEOUT", "30")), json_mode)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> str:
    """Models sometimes wrap JSON in ``` fences or add a sentence around it."""
    text = _FENCE_RE.sub("", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise LLMError("reply contains no JSON object")
    return text[start:end + 1]


class LLMClient:
    def __init__(self, config: LLMConfig, http: httpx.Client | None = None):
        self.config = config
        headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
        self.http = http or httpx.Client(timeout=config.timeout, headers=headers)
        if http is not None and config.api_key:
            self.http.headers.update(headers)

    @property
    def label(self) -> str:
        return f"{self.config.provider}/{self.config.model}"

    def chat(self, messages: list[dict], response_format: dict | None = None) -> str:
        body: dict = {"model": self.config.model, "messages": messages}
        if response_format:
            body["response_format"] = response_format
        try:
            r = self.http.post(f"{self.config.base_url}/chat/completions", json=body)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"{self.label} returned HTTP {exc.response.status_code}: "
                           f"{exc.response.text[:300]}") from exc
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(f"{self.label} request failed: {exc!r}") from exc
        if not content:
            raise LLMError(f"{self.label} returned an empty reply")
        return content

    def complete_text(self, system: str, user: str) -> str:
        return self.chat([{"role": "system", "content": system},
                          {"role": "user", "content": user}]).strip()

    def complete_json(self, system: str, user: str, schema: type[T], retries: int = 1) -> T:
        """Ask for JSON matching ``schema``; validate it and retry once with the error."""
        json_schema = schema.model_json_schema()
        system = (f"{system}\n\nReply with ONLY a JSON object (no prose, no code fences) that "
                  f"matches this JSON schema:\n{json.dumps(json_schema)}")
        response_format = {
            "none": None,
            "json_object": {"type": "json_object"},
            "json_schema": {"type": "json_schema",
                            "json_schema": {"name": schema.__name__, "schema": json_schema}},
        }[self.config.json_mode]

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(retries + 1):
            reply = self.chat(messages, response_format)
            try:
                return schema.model_validate_json(extract_json(reply))
            except (LLMError, ValidationError) as exc:
                if attempt == retries:
                    raise LLMError(f"{self.label} gave invalid JSON after {retries + 1} attempts: {exc}") from exc
                messages += [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": f"That reply was not valid: {exc}. "
                                                "Reply again with only the corrected JSON object."},
                ]
        raise AssertionError("unreachable")
