#!/usr/bin/env python3
"""LLM Client for Turing Tumble Benchmark.

Supports multiple LLM providers:
- OpenAI
- Anthropic
- Ollama
- DeepSeek
- Mock (for testing)
"""

from __future__ import annotations

import json
import os
import time
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Project root (for callers that need the path; env loading is opt-in).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


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

    provider: str = "mock"  # openai, anthropic, ollama, deepseek, mock
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2048
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 300
    max_retries: int = 3


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
    ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, int]]:
        """Generate a JSON response with retry logic.

        Returns (result, error, usage) where usage has 'prompt_tokens' and 'completion_tokens'.
        """
        json_prompt = (
            prompt + "\n\nRespond ONLY with valid JSON. No markdown, no explanations."
        )

        if system_prompt is None:
            system_prompt = "You are a helpful assistant. Output valid JSON only."
        else:
            system_prompt += " Output valid JSON only. No markdown."

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
                    return data, "", {"prompt_tokens": usage.get("prompt_tokens", total_tokens), "completion_tokens": usage.get("completion_tokens", 0)}
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
                        return data, "", {"prompt_tokens": usage.get("prompt_tokens", total_tokens), "completion_tokens": usage.get("completion_tokens", 0)}
                    except json.JSONDecodeError:
                        pass

                raise json.JSONDecodeError("No valid JSON found", content, 0)

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse attempt {attempt + 1} failed: {e}")
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
                time.sleep(1 * (attempt + 1))

        return None, f"Failed after {self.config.max_retries} attempts", {"prompt_tokens": 0, "completion_tokens": 0}

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        tool_executor,
        system_prompt: Optional[str] = None,
        max_turns: int = 10,
        **kwargs,
    ) -> Tuple[Optional[Dict[str, Any]], str, List[ToolCall], List[ToolResult], Dict[str, int]]:
        """Generate using function calling.

        Returns: (final_response, error, tool_calls, tool_results, usage)
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
            "description": "Remove a component from the board at the specified position",
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
            "description": "Get the current board configuration including all placed components and their positions",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ---------------------------------------------------------------------------
# Provider Implementations
# ---------------------------------------------------------------------------


class OpenAIClient(LLMClient):
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

        return LLMResponse(
            content=content,
            model=data["model"],
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", ""),
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
    ) -> Tuple[Optional[Dict[str, Any]], str, List[ToolCall], List[ToolResult], Dict[str, int]]:
        """Generate using OpenAI function calling."""
        model = self._resolve_model_id(self.config.model)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        tool_calls_made: List[ToolCall] = []
        tool_results: List[ToolResult] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        is_gpt5 = "gpt-5" in model.lower()

        for turn in range(max_turns):
            payload = {
                "model": model,
                "messages": messages,
                "tools": tools,
            }

            if is_gpt5:
                pass

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            start_time = time.time()

            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout,
                )
                if response.status_code != 200:
                    logger.error(
                        f"OpenAI API error: {response.status_code} - {response.text}"
                    )
                    return (
                        None,
                        f"API error: {response.text}",
                        tool_calls_made,
                        tool_results,
                        {},
                    )

                data = response.json()
                latency_ms = int((time.time() - start_time) * 1000)

                usage = data.get("usage", {})
                total_prompt_tokens += usage.get("prompt_tokens", 0)
                total_completion_tokens += usage.get("completion_tokens", 0)

            except Exception as e:
                logger.error(f"API call failed: {e}")
                return None, str(e), tool_calls_made, tool_results, {}

            choice = data["choices"][0]
            msg = choice["message"]

            assistant_text = (msg.get("content") or "").strip()

            if "tool_calls" in msg and msg["tool_calls"]:
                for tc in msg["tool_calls"]:
                    tool_name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        args = tc["function"]["arguments"]

                    tc_id = tc.get("id", f"call_{turn}")
                    tool_call = ToolCall(
                        name=tool_name,
                        arguments=args,
                        tool_call_id=tc_id,
                        turn_index=turn,
                        assistant_text=assistant_text,
                    )
                    tool_calls_made.append(tool_call)

                    try:
                        result = tool_executor.execute(tool_name, args)
                        tool_results.append(
                            ToolResult(
                                tool_name=tool_name,
                                result=result,
                                tool_call_id=tc_id,
                                turn_index=turn,
                            )
                        )

                        messages.append({"role": "assistant", "tool_calls": [tc]})
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": json.dumps(result),
                            }
                        )

                    except Exception as e:
                        error_result = {"error": str(e)}
                        tool_results.append(
                            ToolResult(
                                tool_name=tool_name,
                                result=error_result,
                                error=str(e),
                                tool_call_id=tc_id,
                                turn_index=turn,
                            )
                        )
                        messages.append({"role": "assistant", "tool_calls": [tc]})
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": json.dumps(error_result),
                            }
                        )

            elif msg.get("content"):
                content = msg["content"].strip()
                try:
                    final_result = json.loads(content)
                except json.JSONDecodeError:
                    final_result = {"final_answer": content}

                return (
                    final_result,
                    "",
                    tool_calls_made,
                    tool_results,
                    {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens},
                )

            else:
                return (
                    None,
                    "Empty response",
                    tool_calls_made,
                    tool_results,
                    {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens},
                )

        return (
            None,
            f"Max turns ({max_turns}) reached",
            tool_calls_made,
            tool_results,
            {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens},
        )


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

            return LLMResponse(
                content=data["message"]["content"],
                model=self.config.model,
                usage={},
                finish_reason=data.get("done", False),
                latency_ms=latency_ms,
                raw_response=data,
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
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                data = response.json()
                latency_ms = int((time.time() - start_time) * 1000)

                return LLMResponse(
                    content=data.get("response", ""),
                    model=self.config.model,
                    usage={},
                    finish_reason=data.get("done", False),
                    latency_ms=latency_ms,
                    raw_response=data,
                )
            raise

    def unload_model(self):
        logger.info(f"Signaling completion for Ollama model: {self.config.model}")


class DeepSeekClient(LLMClient):
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
        else:
            payload["temperature"] = kwargs.get("temperature", self.config.temperature)

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

        return LLMResponse(
            content=content,
            model=data["model"],
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", ""),
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
    ) -> Tuple[Optional[Dict[str, Any]], str, List[ToolCall], List[ToolResult], Dict[str, int]]:
        """Generate using DeepSeek function calling."""
        model = self._resolve_model_id(self.config.model)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        tool_calls_made: List[ToolCall] = []
        tool_results: List[ToolResult] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        is_v4 = self._is_v4_reasoning_model(model)

        # The tool-calling loop grows context every turn and many
        # models (especially DeepSeek) generate verbose analysis
        # alongside tool_calls.  2048 is too low — 8192 gives
        # headroom for reasoning + tool_call arguments + content.
        max_tokens_val = kwargs.get("max_tokens")
        if max_tokens_val is None:
            max_tokens_val = 8192

        for turn in range(max_turns):
            payload = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "max_tokens": max_tokens_val,
            }

            # v4 reasoning models: disable thinking mode for tool calls.
            # Thinking mode splits output into reasoning_content (internal)
            # and content (user-facing). The model often puts the answer in
            # reasoning_content and leaves content empty, causing the agent
            # loop to return no final solution. Non-thinking mode merges
            # everything into content — simpler and faster for tool-calling.
            if is_v4:
                payload["thinking"] = {"type": "disabled"}

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            start_time = time.time()

            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout,
                )
                if response.status_code != 200:
                    logger.error(
                        f"DeepSeek API error: {response.status_code} - {response.text}"
                    )
                    return (
                        None,
                        f"API error: {response.text}",
                        tool_calls_made,
                        tool_results,
                        {},
                    )

                data = response.json()
                latency_ms = int((time.time() - start_time) * 1000)

                usage = data.get("usage", {})
                total_prompt_tokens += usage.get("prompt_tokens", 0)
                total_completion_tokens += usage.get("completion_tokens", 0)

                choice = data["choices"][0]
                msg = choice["message"]
                finish_reason = choice.get("finish_reason", "")

                # Warn if the model output was truncated (token limit hit).
                # Reasoning models can burn the entire budget on
                # reasoning_content before producing content or tool_calls.
                if finish_reason == "length":
                    logger.warning(
                        "DeepSeek turn %d: response truncated (finish_reason=length). "
                        "max_tokens=%d may be too low for this reasoning model. "
                        "Increase max_tokens to leave room for content after reasoning.",
                        turn,
                        max_tokens_val,
                    )

                # Check for tool calls
                if msg.get("tool_calls"):
                    # Add assistant message with tool_calls first
                    messages.append(msg)

                    # Then process each tool call and add tool result messages
                    for tc in msg["tool_calls"]:
                        tool_call_id = tc.get("id", "")
                        tool_call = ToolCall(
                            name=tc["function"]["name"],
                            arguments=json.loads(tc["function"]["arguments"]),
                            tool_call_id=tool_call_id,
                            turn_index=turn,
                        )
                        tool_calls_made.append(tool_call)

                        # Execute tool
                        try:
                            result = tool_executor.execute(tool_call.name, tool_call.arguments)
                            tool_result = ToolResult(
                                tool_name=tool_call.name,
                                result=result,
                                tool_call_id=tool_call.tool_call_id,
                                turn_index=turn,
                            )
                        except Exception as e:
                            tool_result = ToolResult(
                                tool_name=tool_call.name,
                                result=None,
                                error=str(e),
                                tool_call_id=tool_call.tool_call_id,
                                turn_index=turn,
                            )
                        tool_results.append(tool_result)

                        # Add tool result message
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.tool_call_id,
                            "content": json.dumps(tool_result.result) if tool_result.result else tool_result.error,
                        })

                    # Early termination: if the executor detected a valid
                    # solution (run_simulation with no free-fall and marbles
                    # reaching catchers), stop the loop immediately.  Saves
                    # tokens and prevents the LLM from second-guessing a
                    # correct board.
                    if (
                        hasattr(tool_executor, "is_solution_found")
                        and tool_executor.is_solution_found()
                    ):
                        logger.info(
                            "DeepSeek turn %d: valid solution detected — "
                            "terminating agentic loop early.",
                            turn,
                        )
                        return (
                            {"content": "", "solution_found": True},
                            "",
                            tool_calls_made,
                            tool_results,
                            {
                                "prompt_tokens": total_prompt_tokens,
                                "completion_tokens": total_completion_tokens,
                            },
                        )
                else:
                    # No more tool calls, return the final response.
                    content = msg.get("content", "")
                    error_msg = ""

                    # Safety net: reasoning models sometimes put the answer
                    # in reasoning_content and leave content empty.
                    if not content:
                        reasoning = msg.get("reasoning_content", "")
                        if reasoning:
                            logger.warning(
                                "DeepSeek turn %d: content is empty but "
                                "reasoning_content has %d chars. Using "
                                "reasoning_content as fallback.",
                                turn,
                                len(reasoning),
                            )
                            content = reasoning

                    if not content and finish_reason == "length":
                        error_msg = (
                            f"Response truncated: max_tokens={max_tokens_val} exhausted "
                            f"before content could be generated. reasoning_content consumed "
                            f"the entire token budget."
                        )
                        logger.warning("DeepSeek turn %d: %s", turn, error_msg)
                    return (
                        {"content": content},
                        error_msg,
                        tool_calls_made,
                        tool_results,
                        {
                            "prompt_tokens": total_prompt_tokens,
                            "completion_tokens": total_completion_tokens,
                        },
                    )

            except Exception as e:
                logger.exception(f"DeepSeek API exception: {e}")
                return (
                    None,
                    f"Exception: {str(e)}",
                    tool_calls_made,
                    tool_results,
                    {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens},
                )

        # Max turns reached
        return (
            None,
            "Max turns reached",
            tool_calls_made,
            tool_results,
            {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens},
        )


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
    ) -> Tuple[Optional[Dict[str, Any]], str, List[ToolCall], List[ToolResult], Dict[str, int]]:
        self.call_count += 1
        return None, "MockClient does not support generate_with_tools", [], [], {"prompt_tokens": 0, "completion_tokens": 0}


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Create an LLM client based on configuration."""
    clients = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "ollama": OllamaClient,
        "deepseek": DeepSeekClient,
        "mock": MockClient,
    }

    client_class = clients.get(config.provider.lower())
    if client_class is None:
        raise ValueError(f"Unknown provider: {config.provider}")

    return client_class(config)


# ---------------------------------------------------------------------------
# Default Configurations
# ---------------------------------------------------------------------------


def default_config(provider: str = "mock", model: str = "gpt-4") -> LLMConfig:
    """Create a default configuration."""
    return LLMConfig(provider=provider, model=model)
