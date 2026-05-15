#!/usr/bin/env python3
"""Availability checks for provider models used by scorer.llm_client.

These tests intentionally validate whether at least one configured model option
is actually available from each provider, instead of validating option-count
structure.
"""

import os
from typing import Set

import pytest
import requests

from scorer.llm_client import AnthropicClient, OpenAIClient


def _openai_available_models(api_key: str) -> Set[str]:
    response = requests.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return {m.get("id", "") for m in data.get("data", []) if m.get("id")}


def _anthropic_available_models(api_key: str) -> Set[str]:
    response = requests.get(
        "https://api.anthropic.com/v1/models",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return {m.get("id", "") for m in data.get("data", []) if m.get("id")}


def test_openai_has_usable_configured_model():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set; skipping OpenAI availability test")

    try:
        available_models = _openai_available_models(api_key)
    except requests.RequestException as exc:
        pytest.skip(f"Unable to query OpenAI models endpoint: {exc}")

    configured_options = OpenAIClient.MODEL_REPLACEMENTS["gpt-5"]
    usable = [m for m in configured_options if m in available_models]

    assert usable, (
        "None of the configured OpenAI models are available. "
        f"Configured={configured_options}"
    )


def test_anthropic_has_usable_configured_model():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set; skipping Anthropic availability test")

    try:
        available_models = _anthropic_available_models(api_key)
    except requests.RequestException as exc:
        pytest.skip(f"Unable to query Anthropic models endpoint: {exc}")

    configured_options = AnthropicClient.MODEL_REPLACEMENTS["claude-4"]
    usable = [m for m in configured_options if m in available_models]

    assert usable, (
        "None of the configured Anthropic models are available. "
        f"Configured={configured_options}"
    )


def test_ollama_has_at_least_one_local_model():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        pytest.skip(f"Ollama not reachable locally: {exc}")

    data = response.json()
    models = data.get("models", [])
    assert len(models) > 0, "Ollama is reachable but has no local models installed"
