#!/usr/bin/env python3
"""LLM Client for Turing Tumble Benchmark.

Supports:
- vLLM
- Mock (for testing)
"""

from __future__ import annotations

import json
import os
import threading
import time
import logging
from copy import deepcopy
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Project root (for callers that need the path; env loading is opt-in).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def load_env() -> None:
    """Load environment variables from the project-level .env file.

    Called explicitly by ``run_benchmark.py`` rather than at import time
    so that importing this module does not mutate ``os.environ`` globally.
    """
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Response from an LLM."""

    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    latency_ms: int = 0
    raw_response: Optional[Dict[str, Any]] = None
    logprobs: Optional[List[Dict[str, Any]]] = None
    """Token-level log probabilities, when ``capture_logprobs`` is enabled.
    Each dict has keys: ``token``, ``logprob``, ``top_logprobs`` (list of
    {token, logprob} dicts).  ``None`` when logprobs were not requested or
    the provider does not support them."""


@dataclass
class ToolCall:
    """Represents a single tool call.

    `turn_index` is the 0-based index of the assistant turn that produced this
    call. `assistant_text` is any free-form text the assistant emitted in that
    same turn (alongside the tool_call); useful for narrating the model's
    reasoning when rendering transcripts beside the board.
    """

    name: str
    arguments: Dict[str, Any]
    tool_call_id: str = ""
    turn_index: int = 0
    assistant_text: str = ""


@dataclass
class ToolResult:
    """Result from executing a tool.

    `tool_call_id` and `turn_index` let callers pair a result back to the
    ToolCall that produced it (1:1 in the same order they were appended).
    """

    tool_name: str
    result: Any
    error: Optional[str] = None
    tool_call_id: str = ""
    turn_index: int = 0


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    provider: str = "mock"  # vllm, mock
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2048
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 300
    max_retries: int = 3
    capture_logprobs: bool = False
    """Request token-level log probabilities from the LLM provider."""
    seed: Optional[int] = None
    """Sampling seed forwarded to the provider. Only meaningful with
    temperature > 0; with greedy decoding the sampler is deterministic and the
    remaining run-to-run variation comes from vLLM's continuous batching, which
    a seed does not control."""


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------


def parse_json_text(content: str) -> Any:
    """Parse JSON from model output: bare, inside a ``` fence, or embedded in prose.

    Raises ``json.JSONDecodeError`` when no JSON object can be recovered.
    """
    content = content.strip()
    if content.startswith("```"):
        inner = content.split("```")[1] if "```" in content[3:] else content[3:]
        if inner.startswith("json"):
            inner = inner[4:]
        content = inner.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            data = json.loads(content[first_brace : last_brace + 1])
            logger.info(
                f"Successfully extracted JSON from position {first_brace}-{last_brace}"
            )
            return data
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("No valid JSON found", content, 0)


class LLMClient(ABC):
    """Abstract LLM client."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    def unload_model(self):
        """Unload the model from memory when running local models after benchmark completion.

        Override in subclasses to implement provider-specific cleanup. Default implementation
        does nothing (no-op).
        """
        pass

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, int], Optional[List[Dict[str, Any]]]]:
        """Generate a JSON response with retry logic.

        Returns (result, error, usage, logprobs) where usage has
        'prompt_tokens' and 'completion_tokens', and logprobs is a
        list of per-token log probability dicts (or None if not captured).
        """
        json_prompt = (
            prompt + "\n\nRespond ONLY with valid JSON. No markdown, no explanations."
        )

        if system_prompt is None:
            system_prompt = "You are a helpful assistant. Output valid JSON only."
        else:
            system_prompt += " Output valid JSON only. No markdown."

        last_error = ""
        # Every attempt costs tokens, not just the one that parsed.
        spent = {"prompt_tokens": 0, "completion_tokens": 0}
        for attempt in range(self.config.max_retries):
            content = None
            usage = {}
            try:
                response = self.generate(json_prompt, system_prompt, **kwargs)
                usage = response.usage if hasattr(response, 'usage') and response.usage else {}
                if isinstance(usage, dict):
                    spent["prompt_tokens"] += usage.get("prompt_tokens", usage.get("total_tokens", 0))
                    spent["completion_tokens"] += usage.get("completion_tokens", 0)

                raw_content = response.content.strip()

                if not raw_content:
                    logger.warning(
                        f"Empty or too-short response (len={len(raw_content)})"
                    )
                    last_error = "empty content returned by provider"
                    time.sleep(1 * (attempt + 1))
                    continue

                content = raw_content
                data = parse_json_text(content)
                return (data, "", spent,
                        response.logprobs if hasattr(response, 'logprobs') else None)

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse attempt {attempt + 1} failed: {e}")
                last_error = f"no valid JSON in response: {e}"
                preview = (
                    content[:500]
                    if content and len(content) > 500
                    else content
                    if content
                    else "N/A"
                )
                logger.warning(f"Content preview that failed: {repr(preview)}")
                time.sleep(1 * (attempt + 1))
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                # Same prefix the tool loops use, so the runner can tell a
                # provider failure from a wrong answer.
                last_error = f"Exception: {e}"
                time.sleep(1 * (attempt + 1))

        return (None, last_error or f"Failed after {self.config.max_retries} attempts",
                spent, None)

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        tool_executor,
        system_prompt: Optional[str] = None,
        max_turns: int = 10,
        **kwargs,
    ) -> Tuple[Optional[Dict[str, Any]], str, List[ToolCall], List[ToolResult], Dict[str, int], Optional[List[Optional[List[Dict[str, Any]]]]]]:
        """Generate using function calling.

        Returns: (final_response, error, tool_calls, tool_results, usage, turn_logprobs)
        turn_logprobs is a list of per-turn logprob lists (one entry per API call),
        or None if logprobs were not captured.
        """
        raise NotImplementedError("Subclasses must implement generate_with_tools")


# ---------------------------------------------------------------------------
# Tool Schemas for Turing Tumble
# ---------------------------------------------------------------------------

# Tool schemas for function calling
TURING_TUMBLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "place_component",
            "description": "Place a component on the Turing Tumble board at the specified position",
            "parameters": {
                "type": "object",
                "properties": {
                    "component_type": {
                        "type": "string",
                        "enum": [
                            "ramp_left",
                            "ramp_right",
                            "crossover",
                            "bit",
                            "gear_bit",
                            "gear",
                            "interceptor",
                            "trigger",
                        ],
                        "description": "Type of component to place",
                    },
                    "x": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10,
                        "description": "Column position (x-coordinate)",
                    },
                    "y": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10,
                        "description": "Row position (y-coordinate)",
                    },
                    "state": {
                        "type": "integer",
                        "default": 0,
                        "description": "Initial state (0 or 1) for bit/gear_bit components",
                    },
                },
                "required": ["component_type", "x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_component",
            "description": "Remove a component YOU previously placed from the board. IMPORTANT: only components with source='user' (shown in get_board_state) can be removed. Components with source='fixed' are part of the original board layout and CANNOT be removed — attempting to remove them will fail.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10,
                        "description": "Column position (x-coordinate)",
                    },
                    "y": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10,
                        "description": "Row position (y-coordinate)",
                    },
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_simulation",
            "description": "Execute the current board configuration with the given input sequence and return the simulation results",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_sequence": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["blue", "red"]},
                        "default": ["blue"],
                        "description": "Sequence of marble colors to drop",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_board_state",
            "description": "Get the current board configuration including all placed components and their positions. Each component has a 'source' field: 'fixed' (original board, cannot be removed) or 'user' (you placed it, can be removed).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def turing_tumble_tools(rows: int, cols: int) -> List[Dict[str, Any]]:
    """Return tool schemas constrained to a specific board's dimensions."""
    schemas = deepcopy(TURING_TUMBLE_TOOLS)
    for tool in schemas:
        name = tool.get("function", {}).get("name")
        if name not in {"place_component", "remove_component"}:
            continue
        properties = tool["function"]["parameters"]["properties"]
        properties["x"]["maximum"] = max(cols - 1, 0)
        properties["y"]["maximum"] = max(rows - 1, 0)
    return schemas


# ---------------------------------------------------------------------------
# Provider Implementations
# ---------------------------------------------------------------------------


class _ChatToolLoop(LLMClient):
    """Shared tool-calling loop for OpenAI-compatible ``/v1/chat/completions``.

    Subclasses set ``api_key``/``base_url`` and implement :meth:`_tool_payload`;
    the remaining hooks default to the common behaviour.
    """

    # Stop as soon as the executor reports a validated board.
    _STOP_ON_SOLUTION = True
    # Copy the assistant's free text of a tool-call turn onto its ToolCalls.
    _RECORD_ASSISTANT_TEXT = False

    def _tool_payload(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Model, token budget and sampling fields; messages/tools are added per turn."""
        raise NotImplementedError

    def _sampling(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"temperature": kwargs.get("temperature", self.config.temperature)}
        if self.config.seed is not None:
            out["seed"] = self.config.seed
        return out

    def _before_request(self) -> None:
        """Called once per turn before posting (rate limiting)."""

    def _recover_tool_calls(self, msg: Dict[str, Any], turn: int) -> None:
        """Repair ``msg`` in place when the server mangled its tool calls."""

    def _tool_error_result(self, error: Exception) -> Any:
        """``ToolResult.result`` recorded when a tool call fails."""
        return None

    def _tool_turn_messages(
        self, msg: Dict[str, Any], tool_msgs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Messages appended to the conversation after a tool-call turn."""
        return [msg, *tool_msgs]

    def _final_result(self, content: str, error: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """Shape of the final answer returned to the caller."""
        return {"content": content}, error

    def _post_chat(self, payload: Dict[str, Any]) -> requests.Response:
        """POST with bounded retries: network errors, 5xx and 429 up to 3
        attempts, plus one ``tool_choice=required`` fallback for vLLM servers
        started without ``--enable-auto-tool-choice``. Non-200 replies that
        are not retried are returned to the caller."""
        name = type(self).__name__
        post = getattr(self, "http_post", requests.post)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        retries = 0
        while True:
            try:
                response = post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout,
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if retries < 2:
                    wait = 2**retries
                    logger.warning(
                        "%s: %s (attempt %d/3), retrying in %ds…",
                        name, type(e).__name__, retries + 1, wait,
                    )
                    time.sleep(wait)
                    retries += 1
                    continue
                raise

            if response.status_code == 200:
                return response
            if (
                response.status_code == 400
                and "enable-auto-tool-choice" in response.text
                and payload.get("tool_choice") != "required"
            ):
                logger.warning(
                    "%s: tool_choice=auto unsupported by server, retrying with tool_choice=required…",
                    name,
                )
                payload["tool_choice"] = "required"
                continue
            if (response.status_code >= 500 or response.status_code == 429) and retries < 2:
                wait = 2**retries + 4  # extra wait for rate-limit recovery
                logger.warning(
                    "%s: HTTP %d (attempt %d/3), retrying in %ds…",
                    name, response.status_code, retries + 1, wait,
                )
                time.sleep(wait)
                retries += 1
                continue
            logger.error("%s API error: %s - %s", name, response.status_code, response.text)
            return response

    def _run_tool_calls(
        self,
        msg: Dict[str, Any],
        turn: int,
        tool_executor,
        tool_calls_made: List[ToolCall],
        tool_results: List[ToolResult],
    ) -> List[Dict[str, Any]]:
        """Execute every call in ``msg``; return the tool reply messages."""
        assistant_text = (msg.get("content") or "").strip() if self._RECORD_ASSISTANT_TEXT else ""
        tool_msgs = []
        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            tc_id = tc.get("id", f"call_{turn}")
            raw_args = tc["function"]["arguments"]
            try:
                args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                args = raw_args
            tool_calls_made.append(
                ToolCall(
                    name=name,
                    arguments=args,
                    tool_call_id=tc_id,
                    turn_index=turn,
                    assistant_text=assistant_text,
                )
            )
            # A malformed call is reported back to the model as a tool error.
            try:
                if not isinstance(args, dict):
                    raise ValueError(f"tool arguments are not a JSON object: {raw_args!r}")
                result, error = tool_executor.execute(name, args), None
            except Exception as e:
                result, error = self._tool_error_result(e), str(e)
            tool_results.append(
                ToolResult(
                    tool_name=name,
                    result=result,
                    error=error,
                    tool_call_id=tc_id,
                    turn_index=turn,
                )
            )
            tool_msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": error if error and result is None else json.dumps(result),
                }
            )
        return tool_msgs

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        tool_executor,
        system_prompt: Optional[str] = None,
        max_turns: int = 10,
        *,
        on_final_answer: Optional[Callable[[str], Optional[str]]] = None,
        revision_rounds: int = 0,
        **kwargs,
    ) -> Tuple[Optional[Dict[str, Any]], str, List[ToolCall], List[ToolResult], Dict[str, int], Optional[List[Optional[List[Dict[str, Any]]]]]]:
        """Generate using OpenAI-style tool calling.

        ``on_final_answer`` implements the harness revision arms. When given, a
        submitted answer is passed to it instead of ending the episode; if it
        returns a message, that message is appended and the loop continues, for
        at most ``revision_rounds`` rounds and always inside ``max_turns``.
        Returning ``None`` accepts the answer and returns as usual. Validation
        lives in the caller, beside the scorer, so this loop never decides
        whether an answer is correct.
        """
        name = type(self).__name__
        revisions_left = revision_rounds if on_final_answer is not None else 0
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        tool_calls_made: List[ToolCall] = []
        tool_results: List[ToolResult] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        turn_logprobs: List[Optional[List[Dict[str, Any]]]] = []

        def done(final: Optional[Dict[str, Any]], error: str):
            # Usage is the running total on every path, failures included.
            return final, error, tool_calls_made, tool_results, dict(usage), turn_logprobs or None

        base_payload = self._tool_payload(kwargs)
        if self.config.capture_logprobs:
            base_payload["logprobs"] = True
            base_payload["top_logprobs"] = 5
        max_tokens_val = base_payload.get("max_tokens", base_payload.get("max_completion_tokens"))
        tool_choice = base_payload.get("tool_choice")

        for turn in range(max_turns):
            payload = {**base_payload, "messages": messages, "tools": tools}
            if tool_choice:
                payload["tool_choice"] = tool_choice

            try:
                self._before_request()
                response = self._post_chat(payload)
                tool_choice = payload.get("tool_choice")  # keep a required-fallback
                if response.status_code != 200:
                    return done(None, f"API error: {response.text}")

                data = response.json()
                turn_usage = data.get("usage") or {}
                usage["prompt_tokens"] += turn_usage.get("prompt_tokens", 0)
                usage["completion_tokens"] += turn_usage.get("completion_tokens", 0)

                choice = data["choices"][0]
                lp_data = choice.get("logprobs") if self.config.capture_logprobs else None
                turn_logprobs.append(lp_data.get("content") if isinstance(lp_data, dict) else None)

                msg = choice["message"]
                finish_reason = choice.get("finish_reason", "")
                if finish_reason == "length":
                    logger.warning(
                        "%s turn %d: response truncated (finish_reason=length). "
                        "max_tokens=%s may be too low.",
                        name, turn, max_tokens_val,
                    )

                self._recover_tool_calls(msg, turn)

                if msg.get("tool_calls"):
                    tool_msgs = self._run_tool_calls(
                        msg, turn, tool_executor, tool_calls_made, tool_results
                    )
                    messages.extend(self._tool_turn_messages(msg, tool_msgs))
                    if (
                        self._STOP_ON_SOLUTION
                        and hasattr(tool_executor, "is_solution_found")
                        and tool_executor.is_solution_found()
                    ):
                        logger.info(
                            "%s turn %d: valid solution detected — terminating agentic loop early.",
                            name, turn,
                        )
                        return done({"content": "", "solution_found": True}, "")
                    continue

                # A cut-off answer is never graded from the reasoning channel.
                content = (msg.get("content") or "").strip()
                error_msg = ""
                if not content:
                    if msg.get("reasoning_content"):
                        logger.warning(
                            "%s turn %d: empty content with %d chars of reasoning_content "
                            "(finish_reason=%s). Not substituting reasoning for the answer.",
                            name, turn, len(msg["reasoning_content"]), finish_reason,
                        )
                    if finish_reason == "length":
                        error_msg = (
                            f"Response truncated: max_tokens={max_tokens_val} "
                            f"exhausted before content could be generated."
                        )
                        logger.warning("%s turn %d: %s", name, turn, error_msg)

                # Harness arms rev-retry / rev-structured: an answer the
                # scorer would reject does not end the episode. Revisions
                # draw on the same turn budget, so the arm is not
                # compute-matched against the baseline — which is what
                # rev-retry exists to control for.
                if content and revisions_left > 0:
                    critique = on_final_answer(content)
                    if critique:
                        revisions_left -= 1
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": critique})
                        logger.info(
                            "%s turn %d: submitted answer rejected — "
                            "injected critique, %d revision round(s) left.",
                            name, turn, revisions_left,
                        )
                        continue

                return done(*self._final_result(content, error_msg))

            except Exception as e:
                logger.exception("%s API exception: %s", name, e)
                return done(None, f"Exception: {e}")

        return done(None, f"Max turns ({max_turns}) reached")


class VLLMClient(_ChatToolLoop):
    """vLLM local server client (OpenAI-compatible API).

    vLLM serves an OpenAI-compatible ``/v1/chat/completions`` endpoint on
    ``http://localhost:8000`` by default.  No API key is required — a dummy
    value is sent to satisfy the ``Authorization`` header convention.

    ``generate_with_tools`` uses the OpenAI-style tool-calling protocol.
    This client adds vLLM-specific handling:

    * an explicit ``tool_choice="auto"`` in the payload, with a fallback to
      ``tool_choice="required"`` when the server is started without
      ``--enable-auto-tool-choice`` (a 400 mentioning that flag);
    * content-based tool-call extraction — when a misconfigured tool-call
      parser returns ``tool_calls: []`` but embeds the call JSON in the
      message ``content`` (hermes ``<tool_call>…</tool_call>`` or bare
      ``{"name": …}`` from ``llama3_json``);
    * no aggressive ``max_tokens`` cap — vLLM enforces its own
      ``--max-model-len`` server-side.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        # vLLM doesn't require a real API key; send a dummy value so callers
        # don't have to conditionally omit the header.
        self.api_key = config.api_key or "vllm"
        self.base_url = (
            config.base_url
            or os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
        )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            **self._sampling(kwargs),
        }

        if self.config.capture_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        start_time = time.time()

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.config.timeout,
        )
        if response.status_code != 200:
            logger.error(
                "vLLM API error: %s - %s", response.status_code, response.text
            )
            response.raise_for_status()

        data = response.json()
        latency_ms = int((time.time() - start_time) * 1000)

        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content") or ""
        if not content and msg.get("reasoning_content"):
            logger.warning(
                "vLLM: empty content with %d chars of reasoning_content "
                "(finish_reason=%s). Not substituting reasoning for the answer.",
                len(msg["reasoning_content"]),
                choice.get("finish_reason", ""),
            )

        logprobs_list: Optional[List[Dict[str, Any]]] = None
        if self.config.capture_logprobs:
            logprobs_data = choice.get("logprobs")
            if logprobs_data and isinstance(logprobs_data, dict):
                logprobs_list = logprobs_data.get("content")

        return LLMResponse(
            content=content,
            model=data.get("model", self.config.model),
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", ""),
            latency_ms=latency_ms,
            raw_response=data,
            logprobs=logprobs_list,
        )

    def unload_model(self):
        logger.info("Signaling completion for vLLM model: %s", self.config.model)

    @staticmethod
    def _extract_tool_call_from_content(content: str, turn: int) -> Optional[Dict[str, Any]]:
        """Recover an OpenAI-style tool call embedded in message content.

        vLLM with ``--enable-auto-tool-choice`` but a misconfigured parser can
        return ``tool_calls: []`` while placing the call JSON inside
        ``content``.  The exact wrapper varies by model/parser:

        * hermes: ``<tool_call>{"name": "...", "arguments": {...}}</tool_call>``
        * llama3_json: bare ``{"name": "...", "parameters": {...}}``
        * markdown-fenced: ````json\\n{"name": "...", "arguments": {...}}\\n`````

        Rather than pattern-match each wrapper, scan for the first valid JSON
        object carrying a ``name`` key — ``raw_decode`` stops at that object's
        real closing brace, so surrounding prose/fences/tags are ignored.

        Returns a single tool-call dict (OpenAI schema) or ``None``.
        """
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(content):
            brace = content.find("{", idx)
            if brace == -1:
                return None
            try:
                parsed, end = decoder.raw_decode(content, brace)
            except json.JSONDecodeError:
                idx = brace + 1
                continue
            if isinstance(parsed, dict) and "name" in parsed:
                # Normalise: hermes uses 'parameters', OpenAI uses 'arguments'.
                args = parsed.get("arguments", parsed.get("parameters", {}))
                return {
                    "id": f"call_{turn}",
                    "type": "function",
                    "function": {
                        "name": parsed["name"],
                        "arguments": args if isinstance(args, str) else json.dumps(args),
                    },
                }
            idx = end  # valid JSON but not a tool call — skip past it
        return None

    def _tool_payload(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        # vLLM enforces its own --max-model-len server-side, so pass the
        # configured value through without the aggressive local-model cap.
        return {
            "model": self.config.model,
            "tool_choice": "auto",
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            **self._sampling(kwargs),
        }

    def _recover_tool_calls(self, msg: Dict[str, Any], turn: int) -> None:
        # When the tool-call parser is misconfigured, vLLM returns an
        # empty tool_calls list with the call JSON inside content.
        if not msg.get("tool_calls") and (msg.get("content") or "").strip():
            recovered = self._extract_tool_call_from_content(msg["content"].strip(), turn)
            if recovered:
                msg["tool_calls"] = [recovered]


class MockClient(LLMClient):
    """Mock client for testing."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        self.call_count += 1
        content = (
            '{"result": "mock_response", "call_number": ' + str(self.call_count) + "}"
        )
        return LLMResponse(
            content=content,
            model="mock",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            finish_reason="stop",
            latency_ms=10,
        )

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        tool_executor,
        system_prompt: Optional[str] = None,
        max_turns: int = 10,
        **kwargs,
    ) -> Tuple[Optional[Dict[str, Any]], str, List[ToolCall], List[ToolResult], Dict[str, int], Optional[List[Optional[List[Dict[str, Any]]]]]]:
        self.call_count += 1
        return None, "MockClient does not support generate_with_tools", [], [], {"prompt_tokens": 0, "completion_tokens": 0}, None


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Create an LLM client based on configuration."""
    clients = {
        "vllm": VLLMClient,
        "mock": MockClient,
    }

    client_class = clients.get(config.provider.lower())
    if client_class is None:
        raise ValueError(f"Unknown provider: {config.provider}")

    return client_class(config)
