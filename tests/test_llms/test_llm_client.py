#!/usr/bin/env python3
"""Availability checks for provider models used by scorer.llm_client.

These tests intentionally validate whether at least one configured model option
is actually available from each provider, instead of validating option-count
structure.
"""


import pytest
import requests



def test_ollama_has_at_least_one_local_model():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        pytest.skip(f"Ollama not reachable locally: {exc}")

    data = response.json()
    models = data.get("models", [])
    assert len(models) > 0, "Ollama is reachable but has no local models installed"


def test_lmstudio_available():
    try:
        # LMStudio serves an OpenAI-compatible /v1/models endpoint
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        pytest.skip(f"LMStudio not reachable locally: {exc}")

    data = response.json()
    models = data.get("data", [])
    assert len(models) > 0, "LMStudio is reachable but has no models loaded"
