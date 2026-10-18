"""Characterization tests for the OpenAI-compatible tool-calling loops.

A fake HTTP layer replaces ``requests.post`` so each provider's
``generate_with_tools`` can be driven turn by turn without a server.
"""

import json
from copy import deepcopy

import pytest
import requests

from tt_bench.llm import client as llm_client
from tt_bench.llm.client import LLMConfig, create_llm_client


class FakeResponse:
    def __init__(self, data=None, status_code=200, text=""):
        self._data = data
        self.status_code = status_code
        self.text = text or json.dumps(data)

    def json(self):
        return self._data


class FakeHTTP:
    """Queue of replies; records every payload that was posted."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.payloads = []

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.payloads.append(deepcopy(json))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeExecutor:
    def __init__(self, solved=False):
        self.calls = []
        self.solved = solved

    def execute(self, name, args):
        self.calls.append((name, args))
        if not isinstance(args, dict):
            raise TypeError("arguments must be a mapping")
        return {"success": True, "placed": args.get("component_type")}

    def is_solution_found(self):
        return self.solved


def chat(message, prompt_tokens=10, completion_tokens=5, finish_reason="stop"):
    return FakeResponse(
        {
            "model": "served-model",
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }
    )


TOOL_CALL = {
    "id": "c1",
    "type": "function",
    "function": {
        "name": "place_component",
        "arguments": json.dumps({"component_type": "ramp_left", "x": 1, "y": 2}),
    },
}
TOOL_MSG = {"role": "assistant", "content": "thinking", "tool_calls": [TOOL_CALL]}
FINAL_TEXT = '{"final_solution": []}'
FINAL_MSG = {"role": "assistant", "content": FINAL_TEXT}

PROVIDERS = {
    "openai": "gpt-4o",
    "deepseek": "deepseek-chat",
    "lmstudio": "local-model",
    "vllm": "local-model",
    "cloud": "cloud-model",
}
# Providers whose final answer is JSON-parsed; the rest return {"content": raw}.
JSON_FINAL = {"openai", "cloud"}


def make(provider, model=None, **cfg):
    config = LLMConfig(
        provider=provider,
        model=model or PROVIDERS[provider],
        api_key="key",
        temperature=0.3,
        seed=7,
        max_tokens=1000,
        **cfg,
    )
    return create_llm_client(config)


def run(monkeypatch, provider, replies, executor=None, model=None, **kwargs):
    http = FakeHTTP(replies)
    monkeypatch.setattr(llm_client.requests, "post", http)
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    c = make(provider, model)
    out = c.generate_with_tools(
        prompt="solve",
        tools=[{"type": "function", "function": {"name": "place_component"}}],
        tool_executor=executor or FakeExecutor(),
        system_prompt="sys",
        max_turns=5,
        max_tokens=1000,
        **kwargs,
    )
    return out, http


@pytest.mark.parametrize("provider", list(PROVIDERS))
def test_two_turn_tool_call_then_final_answer(monkeypatch, provider):
    executor = FakeExecutor()
    (final, error, calls, results, usage, logprobs), http = run(
        monkeypatch,
        provider,
        [chat(deepcopy(TOOL_MSG), 10, 5), chat(deepcopy(FINAL_MSG), 20, 7)],
        executor,
    )

    assert error == ""
    if provider in JSON_FINAL:
        assert final == {"final_solution": []}
    else:
        assert final == {"content": FINAL_TEXT}

    # Tool call recorded and executed.
    assert executor.calls == [("place_component", {"component_type": "ramp_left", "x": 1, "y": 2})]
    assert len(calls) == 1
    tc = calls[0]
    assert (tc.name, tc.tool_call_id, tc.turn_index) == ("place_component", "c1", 0)
    assert tc.arguments == {"component_type": "ramp_left", "x": 1, "y": 2}
    assert tc.assistant_text == ("thinking" if provider == "openai" else "")
    assert len(results) == 1
    assert results[0].result == {"success": True, "placed": "ramp_left"}
    assert results[0].error is None
    assert (results[0].tool_call_id, results[0].turn_index) == ("c1", 0)

    # Usage accumulated across both turns; one logprob slot per API call.
    assert usage == {"prompt_tokens": 30, "completion_tokens": 12}
    assert logprobs == [None, None]

    # Transcript sent on turn 2.
    assert len(http.payloads) == 2
    msgs = http.payloads[1]["messages"]
    tool_reply = {
        "role": "tool",
        "tool_call_id": "c1",
        "content": json.dumps({"success": True, "placed": "ramp_left"}),
    }
    if provider == "openai":
        assistant = {"role": "assistant", "tool_calls": [TOOL_CALL]}
    else:
        assistant = TOOL_MSG
    assert msgs == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "solve"},
        assistant,
        tool_reply,
    ]


EXPECTED_PAYLOAD = {
    "openai": {"model": "gpt-4o", "max_completion_tokens": 1000, "temperature": 0.3},
    # Fix b: DeepSeek (non-v4) and Cloud used to send no temperature/seed.
    "deepseek": {"model": "deepseek-chat", "max_tokens": 1000, "temperature": 0.3},
    "lmstudio": {"model": "local-model", "max_tokens": 1000, "temperature": 0.3, "seed": 7},
    "vllm": {
        "model": "local-model",
        "tool_choice": "auto",
        "max_tokens": 1000,
        "temperature": 0.3,
        "seed": 7,
    },
    "cloud": {"model": "cloud-model", "max_tokens": 1000, "temperature": 0.3, "seed": 7},
}


@pytest.mark.parametrize("provider", list(PROVIDERS))
def test_payload(monkeypatch, provider):
    _, http = run(monkeypatch, provider, [chat(deepcopy(FINAL_MSG))])
    payload = http.payloads[0]
    assert payload.pop("messages") == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "solve"},
    ]
    assert payload.pop("tools")
    assert payload == EXPECTED_PAYLOAD[provider]


def test_deepseek_v4_payload_disables_thinking_and_skips_temperature(monkeypatch):
    _, http = run(monkeypatch, "deepseek", [chat(deepcopy(FINAL_MSG))], model="deepseek-v4-flash")
    payload = http.payloads[0]
    assert payload["thinking"] == {"type": "disabled"}
    assert "temperature" not in payload


def test_openai_gpt5_payload_uses_reasoning_effort(monkeypatch):
    _, http = run(monkeypatch, "openai", [chat(deepcopy(FINAL_MSG))], model="gpt-5.4")
    payload = http.payloads[0]
    assert payload["reasoning_effort"] == "low"
    assert "temperature" not in payload


@pytest.mark.parametrize("provider", [p for p in PROVIDERS if p != "openai"])
def test_early_stop_when_solution_found(monkeypatch, provider):
    (final, error, calls, results, usage, _), http = run(
        monkeypatch, provider, [chat(deepcopy(TOOL_MSG))], FakeExecutor(solved=True)
    )
    assert final == {"content": "", "solution_found": True}
    assert error == ""
    assert len(calls) == 1 and len(http.payloads) == 1
    assert usage == {"prompt_tokens": 10, "completion_tokens": 5}


def test_openai_does_not_stop_early(monkeypatch):
    (final, _, _, _, _, _), http = run(
        monkeypatch,
        "openai",
        [chat(deepcopy(TOOL_MSG)), chat(deepcopy(FINAL_MSG))],
        FakeExecutor(solved=True),
    )
    assert final == {"final_solution": []}
    assert len(http.payloads) == 2


@pytest.mark.parametrize("provider", list(PROVIDERS))
def test_max_turns_keeps_usage(monkeypatch, provider):
    replies = [chat(deepcopy(TOOL_MSG)) for _ in range(5)]
    (final, error, calls, _, usage, _), _ = run(monkeypatch, provider, replies)
    assert final is None
    assert "Max turns" in error
    assert len(calls) == 5
    assert usage == {"prompt_tokens": 50, "completion_tokens": 25}


def test_vllm_recovers_tool_call_from_content(monkeypatch):
    content = '<tool_call>{"name": "place_component", "arguments": {"component_type": "bit", "x": 0, "y": 0}}</tool_call>'
    executor = FakeExecutor()
    (final, _, calls, _, _, _), _ = run(
        monkeypatch,
        "vllm",
        [chat({"role": "assistant", "content": content, "tool_calls": []}), chat(deepcopy(FINAL_MSG))],
        executor,
    )
    assert executor.calls == [("place_component", {"component_type": "bit", "x": 0, "y": 0})]
    assert calls[0].tool_call_id == "call_0"
    assert final == {"content": FINAL_TEXT}


@pytest.mark.parametrize("provider", ["vllm", "cloud"])
def test_tool_choice_required_fallback(monkeypatch, provider):
    reject = FakeResponse(status_code=400, text="needs --enable-auto-tool-choice")
    (final, error, _, _, _, _), http = run(
        monkeypatch,
        provider,
        [reject, chat(deepcopy(TOOL_MSG)), chat(deepcopy(FINAL_MSG))],
    )
    assert error == ""
    assert http.payloads[1]["tool_choice"] == "required"
    # The fallback sticks for later turns.
    assert http.payloads[2]["tool_choice"] == "required"


@pytest.mark.parametrize("provider", list(PROVIDERS))
def test_network_error_is_retried(monkeypatch, provider):
    (final, error, _, _, _, _), http = run(
        monkeypatch,
        provider,
        [requests.exceptions.ConnectionError("down"), chat(deepcopy(FINAL_MSG))],
    )
    assert error == ""
    assert final is not None
    assert len(http.payloads) == 2


@pytest.mark.parametrize("provider", list(PROVIDERS))
def test_non_200_after_tool_turn(monkeypatch, provider):
    bad = FakeResponse(status_code=400, text="bad request")
    (final, error, calls, _, usage, logprobs), http = run(
        monkeypatch, provider, [chat(deepcopy(TOOL_MSG)), bad]
    )
    assert final is None
    assert error == "API error: bad request"
    assert len(calls) == 1
    # Fix c: the turns already run keep their tokens (was {}).
    assert usage == {"prompt_tokens": 10, "completion_tokens": 5}
    assert logprobs == [None]
    # A 4xx is not retried (Cloud used to re-post it up to four times).
    assert len(http.payloads) == 2


# ── Bug fixes in the shared loop ───────────────────────────────────────────


@pytest.mark.parametrize("provider", list(PROVIDERS))
def test_cut_off_answer_is_not_taken_from_reasoning(monkeypatch, provider):
    """Fix a: an empty, truncated answer is not graded from reasoning_content."""
    cut = chat(
        {"role": "assistant", "content": None, "reasoning_content": '{"final_solution": [1]}'},
        finish_reason="length",
    )
    (final, error, _, _, _, _), _ = run(monkeypatch, provider, [cut])
    assert "final_solution" not in json.dumps(final)
    if provider != "openai":
        assert error.startswith("Response truncated")


def test_vllm_generate_ignores_reasoning_and_sends_seed(monkeypatch):
    """Fixes a and b for VLLMClient.generate."""
    http = FakeHTTP([chat({"role": "assistant", "content": None, "reasoning_content": "monologue"})])
    monkeypatch.setattr(llm_client.requests, "post", http)
    resp = make("vllm").generate("hi")
    assert resp.content == ""
    assert http.payloads[0]["seed"] == 7
    assert http.payloads[0]["temperature"] == 0.3


@pytest.mark.parametrize("provider", list(PROVIDERS))
def test_malformed_tool_arguments_become_a_tool_error(monkeypatch, provider):
    """Fix d: bad JSON arguments are reported to the model, the episode goes on."""
    bad_call = deepcopy(TOOL_CALL)
    bad_call["function"]["arguments"] = '{"component_type": "ramp_left", "x": 1,'
    executor = FakeExecutor()
    (final, error, calls, results, _, _), http = run(
        monkeypatch,
        provider,
        [chat({"role": "assistant", "content": None, "tool_calls": [bad_call]}), chat(deepcopy(FINAL_MSG))],
        executor,
    )
    assert error == ""
    assert final is not None
    assert executor.calls == []
    assert len(calls) == 1 and calls[0].arguments == bad_call["function"]["arguments"]
    assert "not a JSON object" in results[0].error
    tool_msg = http.payloads[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and "not a JSON object" in tool_msg["content"]


def test_cloud_null_final_content_does_not_crash(monkeypatch):
    """Fix d: ``content: null`` on the final turn crashed Cloud with .strip()."""
    (final, error, _, _, _, _), _ = run(
        monkeypatch, "cloud", [chat({"role": "assistant", "content": None})]
    )
    assert final == {"final_answer": ""}
    assert error == ""


def test_cloud_rate_limiter_spaces_concurrent_requests():
    """Fix e: worker threads sharing one client must not fire together."""
    import threading
    import time

    c = make("cloud")
    c._MIN_INTERVAL = 0.05
    stamps = []

    def worker():
        c._wait_for_rate_limit()
        stamps.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stamps.sort()
    assert min(b - a for a, b in zip(stamps, stamps[1:])) >= 0.04


@pytest.mark.parametrize("provider", list(PROVIDERS))
def test_5xx_is_retried(monkeypatch, provider):
    """Fix f: vLLM gave up on the first 5xx; all providers now retry it."""
    (final, error, _, _, _, _), http = run(
        monkeypatch,
        provider,
        [FakeResponse(status_code=503, text="busy"), chat(deepcopy(FINAL_MSG))],
    )
    assert error == ""
    assert final is not None
    assert len(http.payloads) == 2


def test_5xx_retries_are_bounded(monkeypatch):
    busy = FakeResponse(status_code=503, text="busy")
    (final, error, _, _, _, _), http = run(monkeypatch, "vllm", [busy] * 5)
    assert final is None
    assert error == "API error: busy"
    assert len(http.payloads) == 3
