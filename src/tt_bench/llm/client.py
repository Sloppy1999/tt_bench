#!/usr/bin/env python3
"""LLM Client for Turing Tumble Benchmark.

Supports multiple LLM providers:
- OpenAI
- Anthropic
- Ollama
- LMStudio
- DeepSeek
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

    provider: str = "mock"  # openai, anthropic, ollama, lmstudio, deepseek, mock
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
        for attempt in range(self.config.max_retries):
            content = None
            usage = {}
            try:
                response = self.generate(json_prompt, system_prompt, **kwargs)
                usage = response.usage if hasattr(response, 'usage') and response.usage else {}

                raw_content = response.content.strip()

                if not raw_content:
                    logger.warning(
                        f"Empty or too-short response (len={len(raw_content)})"
                    )
                    last_error = "empty content returned by provider"
                    time.sleep(1 * (attempt + 1))
                    continue

                content = raw_content

                if content.startswith("```"):
                    inner = content.split("```")[1] if "```" in content[3:] else content[3:]
                    if inner.startswith("json"):
                        inner = inner[4:]
                    content = inner.strip()

                try:
                    data = json.loads(content)
                    total_tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
                    return (data, "",
                            {"prompt_tokens": usage.get("prompt_tokens", total_tokens),
                             "completion_tokens": usage.get("completion_tokens", 0)},
                            response.logprobs if hasattr(response, 'logprobs') else None)
                except json.JSONDecodeError:
                    pass

                first_brace = content.find("{")
                last_brace = content.rfind("}")

                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_candidate = content[first_brace : last_brace + 1]
                    try:
                        data = json.loads(json_candidate)
                        logger.info(
                            f"Successfully extracted JSON from position {first_brace}-{last_brace}"
                        )
                        total_tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
                        return (data, "",
                                {"prompt_tokens": usage.get("prompt_tokens", total_tokens),
                                 "completion_tokens": usage.get("completion_tokens", 0)},
                                response.logprobs if hasattr(response, 'logprobs') else None)
                    except json.JSONDecodeError:
                        pass

                raise json.JSONDecodeError("No valid JSON found", content, 0)

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
                last_error = str(e)
                time.sleep(1 * (attempt + 1))

        return (None, last_error or f"Failed after {self.config.max_retries} attempts",
                {"prompt_tokens": 0, "completion_tokens": 0}, None)

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


def _parse_final_json(content: str) -> Dict[str, Any]:
    """JSON-parse a final answer, wrapping plain text as ``final_answer``."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"final_answer": content}


class OpenAIClient(_ChatToolLoop):
    """OpenAI API client."""

    # Canonical model options (modern models only, no legacy snapshots).
    MODEL_REPLACEMENTS: Dict[str, Tuple[str, str, str]] = {
        "gpt-5": ("gpt-5.4", "gpt-5.4-mini", "gpt-5-codex-mini"),
    }

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = config.base_url or "https://api.openai.com"

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY in your environment or .env file, or pass api_key."
            )

    def _resolve_model_id(self, model: str) -> str:
        """Resolve legacy/deprecated model names to current OpenAI model IDs."""
        replacement_options = self.MODEL_REPLACEMENTS.get(model)
        if replacement_options:
            resolved = replacement_options[0]
            logger.warning(
                "OpenAI model '%s' is legacy/deprecated; options=%s; using '%s'.",
                model,
                ", ".join(replacement_options),
                resolved,
            )
            return resolved
        return model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        model = self._resolve_model_id(self.config.model)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Use Chat Completions API (GPT-5 has stricter parameters, eg. no temperature)
        is_gpt5 = "gpt-5" in model.lower()

        payload = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        # GPT-5 reasoning models need this to actually output text
        if is_gpt5:
            payload["reasoning_effort"] = "low"

        # Only add temperature for non-GPT-5 models (GPT-5 doesn't support it)
        if not is_gpt5:
            payload["temperature"] = kwargs.get("temperature", self.config.temperature)

        # Request token-level log probabilities when enabled
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
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            response.raise_for_status()

        data = response.json()
        latency_ms = int((time.time() - start_time) * 1000)

        # Parse Chat Completions format
        choice = data["choices"][0]
        msg = choice["message"]

        # For reasoning models, content might be empty but could be in refusal or annotations
        content = msg.get("content", "")
        if not content and msg.get("refusal"):
            content = msg.get("refusal")
        if not content and msg.get("annotations"):
            # Annotations might contain the actual response
            content = str(msg.get("annotations"))

        # Extract logprobs when available
        logprobs_list: Optional[List[Dict[str, Any]]] = None
        if self.config.capture_logprobs:
            logprobs_data = choice.get("logprobs")
            if logprobs_data and isinstance(logprobs_data, dict):
                logprobs_list = logprobs_data.get("content")

        return LLMResponse(
            content=content,
            model=data["model"],
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", ""),
            latency_ms=latency_ms,
            raw_response=data,
            logprobs=logprobs_list,
        )

    # OpenAI runs keep their original protocol: no early stop, assistant text
    # on each ToolCall, tool errors recorded as {"error": ...} results.
    _STOP_ON_SOLUTION = False
    _RECORD_ASSISTANT_TEXT = True

    def _tool_payload(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        model = self._resolve_model_id(self.config.model)
        payload: Dict[str, Any] = {
            "model": model,
            "max_completion_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        if "gpt-5" in model.lower():
            payload["reasoning_effort"] = "low"
        else:
            payload["temperature"] = kwargs.get("temperature", self.config.temperature)
        return payload

    def _tool_error_result(self, error: Exception) -> Any:
        return {"error": str(error)}

    def _tool_turn_messages(self, msg, tool_msgs):
        # One assistant message per call, each followed by its tool reply.
        out: List[Dict[str, Any]] = []
        for tc, tool_msg in zip(msg["tool_calls"], tool_msgs):
            out += [{"role": "assistant", "tool_calls": [tc]}, tool_msg]
        return out

    def _final_result(self, content, error):
        if not content:
            return None, "Empty response"
        return _parse_final_json(content), ""


class AnthropicClient(LLMClient):
    """Anthropic API client."""

    # Canonical model options (modern models only, no legacy snapshots).
    MODEL_REPLACEMENTS: Dict[str, Tuple[str, str, str]] = {
        "claude-4": ("claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"),
    }

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = config.base_url or "https://api.anthropic.com"

        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key."
            )

    def _resolve_model_id(self, model: str) -> str:
        """Resolve legacy/deprecated model names to active Anthropic model IDs."""
        replacement_options = self.MODEL_REPLACEMENTS.get(model)
        if replacement_options:
            resolved = replacement_options[0]
            logger.warning(
                "Anthropic model '%s' is legacy/deprecated; options=%s; using '%s'.",
                model,
                ", ".join(replacement_options),
                resolved,
            )
            return resolved
        return model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        model = self._resolve_model_id(self.config.model)

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        start_time = time.time()

        response = requests.post(
            f"{self.base_url}/v1/messages",
            json=payload,
            headers=headers,
            timeout=self.config.timeout,
        )
        response.raise_for_status()

        data = response.json()
        latency_ms = int((time.time() - start_time) * 1000)

        return LLMResponse(
            content=data["content"][0]["text"],
            model=data["model"],
            usage=data.get("usage", {}),
            finish_reason=data.get("stop_reason", ""),
            latency_ms=latency_ms,
            raw_response=data,
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
        """Generate using Anthropic's native ``tool_use`` protocol."""
        model = self._resolve_model_id(self.config.model)
        messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
        anthropic_tools = []
        for tool in tools:
            function = tool.get("function", {})
            anthropic_tools.append(
                {
                    "name": function.get("name", ""),
                    "description": function.get("description", ""),
                    "input_schema": function.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
            )

        tool_calls_made: List[ToolCall] = []
        tool_results: List[ToolResult] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        for turn in range(max_turns):
            payload: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "tools": anthropic_tools,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "temperature": kwargs.get("temperature", self.config.temperature),
            }
            if system_prompt:
                payload["system"] = system_prompt

            try:
                response = requests.post(
                    f"{self.base_url}/v1/messages",
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                data = response.json()
            except Exception as exc:
                logger.exception("Anthropic tool call failed: %s", exc)
                return (
                    None,
                    str(exc),
                    tool_calls_made,
                    tool_results,
                    {
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                    },
                    None,
                )

            usage = data.get("usage", {})
            total_prompt_tokens += usage.get("input_tokens", 0)
            total_completion_tokens += usage.get("output_tokens", 0)
            blocks = data.get("content", [])
            text = "\n".join(
                block.get("text", "")
                for block in blocks
                if block.get("type") == "text"
            ).strip()
            tool_use_blocks = [
                block for block in blocks if block.get("type") == "tool_use"
            ]

            if not tool_use_blocks:
                try:
                    final_result = json.loads(text) if text else {"final_answer": ""}
                except json.JSONDecodeError:
                    final_result = {"final_answer": text}
                return (
                    final_result,
                    "" if text else "Empty response",
                    tool_calls_made,
                    tool_results,
                    {
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                    },
                    None,
                )

            messages.append({"role": "assistant", "content": blocks})
            result_blocks = []
            for block in tool_use_blocks:
                tool_call_id = block.get("id", f"call_{turn}")
                tool_name = block.get("name", "")
                arguments = block.get("input", {})
                tool_call = ToolCall(
                    name=tool_name,
                    arguments=arguments,
                    tool_call_id=tool_call_id,
                    turn_index=turn,
                    assistant_text=text,
                )
                tool_calls_made.append(tool_call)

                try:
                    result = tool_executor.execute(tool_name, arguments)
                    tool_result = ToolResult(
                        tool_name=tool_name,
                        result=result,
                        tool_call_id=tool_call_id,
                        turn_index=turn,
                    )
                    is_error = False
                except Exception as exc:
                    result = {"error": str(exc)}
                    tool_result = ToolResult(
                        tool_name=tool_name,
                        result=result,
                        error=str(exc),
                        tool_call_id=tool_call_id,
                        turn_index=turn,
                    )
                    is_error = True
                tool_results.append(tool_result)
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": json.dumps(result),
                        "is_error": is_error,
                    }
                )

            messages.append({"role": "user", "content": result_blocks})
            if (
                hasattr(tool_executor, "is_solution_found")
                and tool_executor.is_solution_found()
            ):
                return (
                    {"content": "", "solution_found": True},
                    "",
                    tool_calls_made,
                    tool_results,
                    {
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                    },
                    None,
                )

        return (
            None,
            f"Max turns ({max_turns}) reached",
            tool_calls_made,
            tool_results,
            {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
            },
            None,
        )


def _extract_ollama_logprobs(data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Extract logprobs from an Ollama /api/chat or /api/generate response.

    Ollama returns logprobs as a top-level list of {token, logprob, bytes}
    dicts when ``logprobs: true`` is set in the request.
    """
    lp = data.get("logprobs")
    if lp and isinstance(lp, list):
        return lp
    return None


class OllamaClient(LLMClient):
    """Ollama local model client."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": prompt},
            ],
            "temperature": kwargs.get("temperature", self.config.temperature),
            "options": {
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                "no_thinking": True,
            },
            "stream": False,
        }

        # Request token-level log probabilities when enabled
        if self.config.capture_logprobs:
            payload["logprobs"] = True

        start_time = time.time()

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract logprobs when available (Ollama returns them at top level)
            logprobs_list = _extract_ollama_logprobs(data) if self.config.capture_logprobs else None

            # Some models (lfm, deepseek-r1, etc.) put their answer in the
            # ``thinking`` field and leave ``content`` empty.
            msg = data.get("message", {})
            content = msg.get("content", "")
            if not content:
                thinking = msg.get("thinking", "")
                if thinking:
                    logger.warning(
                        "Ollama: content is empty but thinking has %d chars. "
                        "Using thinking as fallback.", len(thinking),
                    )
                    content = thinking

            return LLMResponse(
                content=content,
                model=self.config.model,
                usage={},
                finish_reason=data.get("done", False),
                latency_ms=latency_ms,
                raw_response=data,
                logprobs=logprobs_list,
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                payload = {
                    "model": self.config.model,
                    "prompt": prompt,
                    "system": system_prompt or "",
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "options": {
                        "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                    },
                }
                if self.config.capture_logprobs:
                    payload["logprobs"] = True
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                data = response.json()
                latency_ms = int((time.time() - start_time) * 1000)

                logprobs_list = _extract_ollama_logprobs(data) if self.config.capture_logprobs else None

                return LLMResponse(
                    content=data.get("response", ""),
                    model=self.config.model,
                    usage={},
                    finish_reason=data.get("done", False),
                    latency_ms=latency_ms,
                    raw_response=data,
                    logprobs=logprobs_list,
                )
            raise

    def unload_model(self):
        logger.info(f"Signaling completion for Ollama model: {self.config.model}")

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        tool_executor,
        system_prompt: Optional[str] = None,
        max_turns: int = 10,
        **kwargs,
    ) -> Tuple[Optional[Dict[str, Any]], str, List[ToolCall], List[ToolResult], Dict[str, int], Optional[List[Optional[List[Dict[str, Any]]]]]]:
        """Generate using Ollama native tool calling.

        Ollama /api/chat supports tools natively.  Response format differs
        from OpenAI: the message is at the top level (no ``choices`` wrapper)
        and tool-call arguments are already parsed as dicts.
        """
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        tool_calls_made: List[ToolCall] = []
        tool_results: List[ToolResult] = []
        total_eval_count = 0
        total_prompt_eval = 0
        turn_logprobs: List[Optional[List[Dict[str, Any]]]] = []

        max_tokens_val = kwargs.get("max_tokens", self.config.max_tokens)

        for turn in range(max_turns):
            payload: Dict[str, Any] = {
                "model": self.config.model,
                "messages": messages,
                "tools": tools,
                "stream": False,
                "options": {"num_predict": max_tokens_val},
            }

            if self.config.capture_logprobs:
                payload["logprobs"] = True

            start_time = time.time()

            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
            except Exception as e:
                logger.error(f"Ollama tool call failed: {e}")
                return (
                    None, str(e), tool_calls_made, tool_results,
                    {"prompt_tokens": total_prompt_eval, "completion_tokens": total_eval_count},
                    turn_logprobs if turn_logprobs else None,
                )

            data = response.json()
            latency_ms = int((time.time() - start_time) * 1000)

            # Accumulate proxy token counts (Ollama doesn't have standard usage)
            total_prompt_eval += data.get("prompt_eval_count", 0)
            total_eval_count += data.get("eval_count", 0)

            # Capture per-turn logprobs (Ollama returns them at top level)
            if self.config.capture_logprobs:
                lp = data.get("logprobs")
                turn_logprobs.append(lp if isinstance(lp, list) else None)
            else:
                turn_logprobs.append(None)

            msg = data.get("message", {})
            content = (msg.get("content") or "").strip()
            # Some models (lfm, deepseek-r1 style) put reasoning in the
            # ``thinking`` field and leave ``content`` empty.  Capture it
            # as assistant_text so transcripts carry the model's reasoning.
            thinking = (msg.get("thinking") or "").strip()
            assistant_text = content if content else thinking
            tool_calls_raw = msg.get("tool_calls") or []

            if tool_calls_raw:
                # ── Tool call turn ──────────────────────────────────
                # Build assistant message with tool_calls for context
                assistant_msg = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tool_calls_raw,
                }
                # If the model uses thinking, preserve it in the
                # assistant message so the context stays coherent.
                if thinking and not content:
                    assistant_msg["thinking"] = thinking
                messages.append(assistant_msg)

                for tc in tool_calls_raw:
                    fn = tc.get("function", {})
                    tool_name = fn.get("name", "")
                    # Ollama returns arguments as a dict, not a JSON string
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            pass

                    tc_id = tc.get("id", f"call_{turn}")
                    tool_call = ToolCall(
                        name=tool_name,
                        arguments=args if isinstance(args, dict) else {},
                        tool_call_id=tc_id,
                        turn_index=turn,
                        assistant_text=assistant_text,
                    )
                    tool_calls_made.append(tool_call)

                    # Execute the tool
                    try:
                        result = tool_executor.execute(tool_name, args if isinstance(args, dict) else {})
                        tool_results.append(ToolResult(
                            tool_name=tool_name,
                            result=result,
                            tool_call_id=tc_id,
                            turn_index=turn,
                        ))
                        result_content = json.dumps(result)
                    except Exception as e:
                        error_result = {"error": str(e)}
                        tool_results.append(ToolResult(
                            tool_name=tool_name,
                            result=error_result,
                            error=str(e),
                            tool_call_id=tc_id,
                            turn_index=turn,
                        ))
                        result_content = json.dumps(error_result)

                    messages.append({
                        "role": "tool",
                        "content": result_content,
                    })

                # Early termination: executor detected a valid solution
                if (
                    hasattr(tool_executor, "is_solution_found")
                    and tool_executor.is_solution_found()
                ):
                    logger.info(
                        "Ollama turn %d: valid solution detected — "
                        "terminating agentic loop early.", turn,
                    )
                    return (
                        {"content": "", "solution_found": True},
                        "",
                        tool_calls_made,
                        tool_results,
                        {"prompt_tokens": total_prompt_eval, "completion_tokens": total_eval_count},
                        turn_logprobs if turn_logprobs else None,
                    )

            elif assistant_text:
                # ── Final answer turn ───────────────────────────────
                try:
                    final_result = json.loads(assistant_text)
                except json.JSONDecodeError:
                    final_result = {"final_answer": assistant_text}

                return (
                    final_result,
                    "",
                    tool_calls_made,
                    tool_results,
                    {"prompt_tokens": total_prompt_eval, "completion_tokens": total_eval_count},
                    turn_logprobs if turn_logprobs else None,
                )

            else:
                # Empty response — model didn't produce tools or content
                return (
                    None,
                    "Empty response from model",
                    tool_calls_made,
                    tool_results,
                    {"prompt_tokens": total_prompt_eval, "completion_tokens": total_eval_count},
                    turn_logprobs if turn_logprobs else None,
                )

        # Max turns reached
        return (
            None,
            f"Max turns ({max_turns}) reached",
            tool_calls_made,
            tool_results,
            {"prompt_tokens": total_prompt_eval, "completion_tokens": total_eval_count},
            turn_logprobs if turn_logprobs else None,
        )


class DeepSeekClient(_ChatToolLoop):
    """DeepSeek API client."""

    # Canonical model options (v4 reasoning models).
    MODEL_REPLACEMENTS: Dict[str, Tuple[str, ...]] = {
        "deepseek-v4": ("deepseek-v4-pro", "deepseek-v4-flash"),
    }

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = config.base_url or "https://api.deepseek.com"

        if not self.api_key:
            raise ValueError(
                "DeepSeek API key required. Set DEEPSEEK_API_KEY in your environment or .env file, or pass api_key."
            )

    def _resolve_model_id(self, model: str) -> str:
        """Resolve shorthand model names to full DeepSeek model IDs."""
        replacement_options = self.MODEL_REPLACEMENTS.get(model)
        if replacement_options:
            resolved = replacement_options[0]
            logger.warning(
                "DeepSeek model '%s' is a shorthand; options=%s; using '%s'.",
                model,
                ", ".join(replacement_options),
                resolved,
            )
            return resolved
        return model

    def _is_v4_reasoning_model(self, model: str) -> bool:
        """Check if the model is a DeepSeek v4 reasoning model."""
        return "deepseek-v4" in model or "deepseek-v4" in self.config.model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        model = self._resolve_model_id(self.config.model)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        is_v4 = self._is_v4_reasoning_model(model)

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        # v4 reasoning models: skip temperature, use reasoning_effort instead
        if is_v4:
            payload["reasoning_effort"] = "low"
            # Disable thinking mode so the model writes directly to content
            # instead of reasoning_content.  Required for JSON parsing since
            # reasoning_content contains internal monologue, not structured output.
            payload["thinking"] = {"type": "disabled"}
        else:
            payload["temperature"] = kwargs.get("temperature", self.config.temperature)

        # Request token-level log probabilities when enabled
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
            logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
            response.raise_for_status()

        data = response.json()
        latency_ms = int((time.time() - start_time) * 1000)

        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content", "")

        # Safety net: reasoning models sometimes put the answer in
        # reasoning_content and leave content empty — even with
        # reasoning_effort="low".
        if not content:
            reasoning = msg.get("reasoning_content", "")
            if reasoning:
                logger.warning(
                    "DeepSeek: empty content with %d chars of reasoning_content "
                    "(finish_reason=%s). Not substituting reasoning for the answer.",
                    len(reasoning),
                    choice.get("finish_reason", ""),
                )

        # Extract logprobs when available
        logprobs_list: Optional[List[Dict[str, Any]]] = None
        if self.config.capture_logprobs:
            logprobs_data = choice.get("logprobs")
            if logprobs_data and isinstance(logprobs_data, dict):
                logprobs_list = logprobs_data.get("content")

        return LLMResponse(
            content=content,
            model=data["model"],
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", ""),
            latency_ms=latency_ms,
            raw_response=data,
            logprobs=logprobs_list,
        )

    def _tool_payload(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        model = self._resolve_model_id(self.config.model)
        # The tool-calling loop grows context every turn and many
        # models (especially DeepSeek) generate verbose analysis
        # alongside tool_calls.  2048 is too low — 32768 gives
        # headroom for reasoning + tool_call arguments + content,
        # even on complex multi-component challenges.
        max_tokens_val = kwargs.get("max_tokens")
        if max_tokens_val is None:
            max_tokens_val = 32768
        payload: Dict[str, Any] = {"model": model, "max_tokens": max_tokens_val}
        if self._is_v4_reasoning_model(model):
            # v4 reasoning models: disable thinking mode for tool calls.
            # Thinking mode splits output into reasoning_content (internal)
            # and content (user-facing). The model often puts the answer in
            # reasoning_content and leaves content empty, causing the agent
            # loop to return no final solution. Non-thinking mode merges
            # everything into content — simpler and faster for tool-calling.
            payload["thinking"] = {"type": "disabled"}
        else:
            payload["temperature"] = kwargs.get("temperature", self.config.temperature)
        return payload


class LMStudioClient(_ChatToolLoop):
    """LMStudio local model client (OpenAI-compatible API).

    LMStudio serves an OpenAI-compatible ``/v1/chat/completions`` endpoint
    on ``http://localhost:1234`` by default.  No API key is required — a
    dummy value is sent to satisfy the ``Authorization`` header convention.

    ``generate_with_tools`` uses the OpenAI-style tool-calling protocol,
    which works when the loaded model supports function calling.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        # LMStudio doesn't require a real API key; send a dummy value
        # so callers don't have to conditionally omit the header.
        self.api_key = config.api_key or "lm-studio"
        self.base_url = (
            config.base_url
            or os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234")
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
            "max_tokens": min(kwargs.get("max_tokens", self.config.max_tokens), 8192),
            "temperature": kwargs.get("temperature", self.config.temperature),
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
                "LMStudio API error: %s - %s", response.status_code, response.text
            )
            response.raise_for_status()

        data = response.json()
        latency_ms = int((time.time() - start_time) * 1000)

        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content", "")

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
        logger.info(
            "Signaling completion for LMStudio model: %s", self.config.model
        )

    def _tool_payload(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        # LMStudio local models have much lower output token limits than
        # cloud providers (typically 4k-8k).  Capping prevents the server
        # from truncating mid-generation with finish_reason=length.
        max_tokens_val = kwargs.get("max_tokens", self.config.max_tokens)
        if max_tokens_val > 8192:
            logger.info(
                "LMStudio: capping max_tokens from %d to 8192 for local model.",
                max_tokens_val,
            )
            max_tokens_val = 8192
        return {"model": self.config.model, "max_tokens": max_tokens_val, **self._sampling(kwargs)}


class VLLMClient(_ChatToolLoop):
    """vLLM local server client (OpenAI-compatible API).

    vLLM serves an OpenAI-compatible ``/v1/chat/completions`` endpoint on
    ``http://localhost:8000`` by default.  No API key is required — a dummy
    value is sent to satisfy the ``Authorization`` header convention.

    ``generate_with_tools`` uses the OpenAI-style tool-calling protocol.
    Compared with :class:`LMStudioClient`, this client adds vLLM-specific
    handling:

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


class CloudClient(_ChatToolLoop):
    """Cloud API client for OpenAI-compatible endpoints (AcademicCloud).

    Uses the same /v1/chat/completions format as OpenAI with Bearer auth.
    No model-id rewriting — cloud model IDs pass through as-is.
    """

    # Rate limits: 2 req/s, 15 req/min (AcademicCloud via Kong gateway).
    # 15 req/min ≈ 4 s/req, but 429s still happen — 6 s gives safe margin.
    _MIN_INTERVAL: float = 6.0

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.environ.get("CLOUD_API_KEY")
        self.base_url = config.base_url or "https://chat-ai.academiccloud.de"
        self._last_request: float = 0.0
        self._rate_lock = threading.Lock()

        if not self.api_key:
            raise ValueError(
                "Cloud API key required. Set CLOUD_API_KEY in your environment "
                "or .env file, or pass api_key."
            )

    def _wait_for_rate_limit(self) -> None:
        """Sleep if needed to stay within API rate limits (2 req/s, 15 req/min).

        Locked: benchmark worker threads share one client, and without the
        lock they all read the same ``_last_request`` and fire together.
        """
        with self._rate_lock:
            wait = max(self._MIN_INTERVAL - (time.time() - self._last_request), 0)
            if wait > 0:
                logger.debug("Rate limit: waiting %.1fs", wait)
                time.sleep(wait)
            self._last_request = time.time()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        model = self.config.model

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }

        if self.config.capture_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self._wait_for_rate_limit()
        start_time = time.time()

        for attempt in range(3):
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout,
                )
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ) as e:
                if attempt < 2:
                    wait = 2 ** attempt
                    logger.warning(
                        "Cloud API %s (attempt %d/3), retrying in %ds…",
                        type(e).__name__, attempt + 1, wait,
                    )
                    time.sleep(wait)
                    continue
                raise

            if response.status_code == 200:
                break  # success — exit retry loop

            if 500 <= response.status_code < 600 or response.status_code == 429:
                if attempt < 2:
                    wait = 2 ** attempt + 4
                    logger.warning(
                        "Cloud API 5xx (attempt %d/3, status=%d), retrying in %ds…",
                        attempt + 1, response.status_code, wait,
                    )
                    time.sleep(wait)
                    continue
                logger.error(
                    "Cloud API error after 3 retries: %s - %s",
                    response.status_code, response.text,
                )
                response.raise_for_status()
            else:
                logger.error(
                    "Cloud API error: %s - %s", response.status_code, response.text
                )
                response.raise_for_status()

        data = response.json()
        latency_ms = int((time.time() - start_time) * 1000)

        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content", "")

        logprobs_list: Optional[List[Dict[str, Any]]] = None
        if self.config.capture_logprobs:
            logprobs_data = choice.get("logprobs")
            if logprobs_data and isinstance(logprobs_data, dict):
                logprobs_list = logprobs_data.get("content")

        return LLMResponse(
            content=content,
            model=data["model"],
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", ""),
            latency_ms=latency_ms,
            raw_response=data,
            logprobs=logprobs_list,
        )

    def _tool_payload(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        max_tokens_val = kwargs.get("max_tokens")
        if max_tokens_val is None:
            max_tokens_val = 32768
        return {"model": self.config.model, "max_tokens": max_tokens_val, **self._sampling(kwargs)}

    def _before_request(self) -> None:
        self._wait_for_rate_limit()

    def _final_result(self, content, error):
        return (_parse_final_json(content) if content else {"final_answer": ""}), error


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
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "ollama": OllamaClient,
        "lmstudio": LMStudioClient,
        "vllm": VLLMClient,
        "deepseek": DeepSeekClient,
        "cloud": CloudClient,
        "mock": MockClient,
    }

    client_class = clients.get(config.provider.lower())
    if client_class is None:
        raise ValueError(f"Unknown provider: {config.provider}")

    return client_class(config)
