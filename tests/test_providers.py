"""Generic LLM client: env configuration and the HTTP protocol, against a fake server."""

import json

import httpx
import pytest

from app.interpreter import DataContext, Interpreter, LLMInterpreter
from app.llm import LLMClient, LLMConfig, LLMError, extract_json

CTX = DataContext(["2026-03", "2026-04"], {"product": ["Monitor"], "region": ["North"],
                                           "category": ["Electronics"], "sales_channel": ["Online"]})


# --- Configuration from environment variables ---------------------------------------------


@pytest.mark.parametrize("env, provider, base_url, key", [
    ({"LLM_PROVIDER": "gemini", "LLM_MODEL": "gemini-2.5-flash", "GEMINI_API_KEY": "g"},
     "gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "g"),
    ({"LLM_PROVIDER": "openai", "LLM_MODEL": "m", "LLM_API_KEY": "k"}, "openai", "https://api.openai.com/v1", "k"),
    ({"LLM_PROVIDER": "anthropic", "LLM_MODEL": "m", "ANTHROPIC_API_KEY": "a"},
     "anthropic", "https://api.anthropic.com/v1", "a"),
    ({"LLM_PROVIDER": "ollama", "LLM_MODEL": "llama3.2"}, "ollama", "http://localhost:11434/v1", None),
    ({"LLM_BASE_URL": "https://my-proxy.example/v1/", "LLM_MODEL": "m", "LLM_API_KEY": "k"},
     "custom", "https://my-proxy.example/v1", "k"),
])
def test_config_from_env(env, provider, base_url, key):
    cfg = LLMConfig.from_env(env)
    assert (cfg.provider, cfg.base_url, cfg.api_key) == (provider, base_url, key)


@pytest.mark.parametrize("env", [{}, {"LLM_PROVIDER": "gemini", "LLM_MODEL": "m", "GEMINI_API_KEY": "g", "BI_USE_LLM": "0"}])
def test_no_llm_configured(env):
    assert LLMConfig.from_env(env) is None


@pytest.mark.parametrize("env, message", [
    ({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "g"}, "LLM_MODEL"),
    ({"LLM_PROVIDER": "gemini", "LLM_MODEL": "m"}, "API key"),
    ({"LLM_PROVIDER": "nope", "LLM_MODEL": "m"}, "Unknown LLM_PROVIDER"),
    ({"LLM_PROVIDER": "openai", "LLM_MODEL": "m", "LLM_API_KEY": "k", "LLM_JSON_MODE": "xml"}, "LLM_JSON_MODE"),
])
def test_bad_config_is_reported(env, message):
    with pytest.raises(LLMError, match=message):
        LLMConfig.from_env(env)


def test_misconfiguration_falls_back_to_rules(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")          # no model, no key
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert Interpreter().mode == "rules"


# --- Protocol, against a fake OpenAI-compatible server --------------------------------------


def fake_server(*replies: str, status: int = 200):
    """Returns (client, captured_requests). Each call returns the next reply."""
    captured, queue = [], list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if status != 200:
            return httpx.Response(status, json={"error": "boom"})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": queue.pop(0)}}]})

    cfg = LLMConfig("gemini", "gemini-test", "https://llm.example/v1", "secret")
    client = LLMClient(cfg, http=httpx.Client(transport=httpx.MockTransport(handler)))
    return client, captured


def test_request_shape_and_plan():
    client, captured = fake_server('{"intent": "custom_query", "metric": "units", '
                                   '"filters": [{"dimension": "product", "value": "Monitor"}], "months": ["2026-03"]}')
    intent = LLMInterpreter(client).interpret("Monitor units in March", CTX)
    assert (intent.spec.metric, intent.spec.filters, intent.spec.months) == ("units", {"product": ["Monitor"]}, ["2026-03"])

    req = captured[0]
    assert str(req.url) == "https://llm.example/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer secret"
    body = json.loads(req.content)
    assert body["model"] == "gemini-test"
    assert '"LLMFilter"' in body["messages"][0]["content"]   # schema included in the prompt


def test_fenced_json_is_accepted():
    client, _ = fake_server('Sure!\n```json\n{"intent": "monthly_trend"}\n```')
    assert LLMInterpreter(client).interpret("trend?", CTX).name == "monthly_trend"


def test_invalid_reply_is_retried_with_the_error():
    client, captured = fake_server('{"intent": "drop_tables"}', '{"intent": "total_revenue"}')
    assert LLMInterpreter(client).interpret("total?", CTX).name == "total_revenue"
    retry = json.loads(captured[1].content)["messages"]
    assert retry[-1]["role"] == "user" and "not valid" in retry[-1]["content"]


@pytest.mark.parametrize("server", [
    lambda: fake_server('{"intent": "nonsense"}', "still not json"),   # invalid twice
    lambda: fake_server(status=500),                                     # provider error
])
def test_llm_failures_fall_back_to_rules(server):
    interp = Interpreter(use_llm=False)
    interp.llm = LLMInterpreter(server()[0])
    intent, used = interp.interpret("What is the total revenue?", CTX)
    assert (intent.name, used) == ("total_revenue", "rules")


def test_json_mode_is_sent_when_configured():
    client, captured = fake_server('{"intent": "monthly_trend"}')
    client.config.json_mode = "json_object"
    LLMInterpreter(client).interpret("trend?", CTX)
    assert json.loads(captured[0].content)["response_format"] == {"type": "json_object"}


def test_extract_json_rejects_prose():
    with pytest.raises(LLMError):
        extract_json("I cannot help with that.")
