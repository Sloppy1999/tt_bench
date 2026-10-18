"""Runner scoring and reporting: provider errors, hedges, tiers, max_tokens."""

from __future__ import annotations

import json
from pathlib import Path

from tt_bench.benchmark.runner import TuringTumbleBenchmark
from tt_bench.llm.client import LLMConfig
from tt_bench.simulator import Board

CHALLENGES = Path(__file__).resolve().parent.parent / "data/tasks/official/challenges/json"
TASK = "tt-official-ch09-pA"  # tier 2, three questions


class FakeLLM:
    """Records kwargs; every call fails the way the clients report a 503."""

    def __init__(self, raise_in_json: bool = False):
        self.config = LLMConfig(provider="fake", model="fake")
        self.raise_in_json = raise_in_json
        self.json_kwargs: list = []

    def generate_json(self, prompt, system_prompt=None, **kwargs):
        self.json_kwargs.append(kwargs)
        if self.raise_in_json:
            raise RuntimeError("boom")
        return None, "Exception: 503 Service Unavailable", {}, None

    def generate_with_tools(self, prompt, tools, tool_executor, system_prompt=None, max_turns=10, **kwargs):
        return None, "API error: 503", [], [], {}, None


def _bench(llm, tmp_path) -> TuringTumbleBenchmark:
    return TuringTumbleBenchmark(llm, CHALLENGES, tmp_path, max_tokens=1234)


def test_understanding_forwards_max_tokens_and_marks_provider_error_ungraded(tmp_path):
    llm = FakeLLM()
    results = _bench(llm, tmp_path).run_understanding_task(CHALLENGES / f"{TASK}.json")

    assert results and all(kw["max_tokens"] == 1234 for kw in llm.json_kwargs)
    assert all(r.success is None and "503" in r.error and r.tier == 2 for r in results)


def test_understanding_exception_path_uses_unique_qid(tmp_path):
    results = _bench(FakeLLM(raise_in_json=True), tmp_path).run_understanding_task(
        CHALLENGES / f"{TASK}.json"
    )
    ids = [r.task_id for r in results]
    assert len(ids) == len(set(ids)) == 3
    assert all(i.startswith(f"{TASK}_{TASK}-q") for i in ids)


def test_agentic_provider_error_is_ungraded_and_keeps_both_messages(tmp_path):
    r = _bench(FakeLLM(), tmp_path).run_agentic_task(CHALLENGES / f"{TASK}.json")
    assert r.success is None
    assert "No solution found" in r.error and "API error: 503" in r.error


def test_hedged_answers_are_not_correct():
    b = TuringTumbleBenchmark
    assert b._text_answer_matches("left or right", "left") is False
    assert b._text_answer_matches("left", "left") is True
    bench = b.__new__(b)
    expected = "The blue ball exits on the left side."
    hedged = bench._validate_understanding(
        Board(), "output_sequence", "?", {"answer": "blue exits left or right"}, expected
    )
    assert hedged["correct"] is False


def test_unsolvable_declaration_in_fence_prose_and_final_answer_wrapper():
    declares = TuringTumbleBenchmark._model_declares_unsolvable
    fenced = '```json\n{"success": false, "reason": "x"}\n```'
    prose = 'I checked every option. {"success": false} That is my answer.'
    assert declares({"content": fenced})
    assert declares({"content": prose})
    assert declares({"final_answer": fenced})
    assert not declares({"final_answer": '{"success": true}'})


def test_report_aggregates_by_tier_and_type_excluding_ungraded(tmp_path):
    bench = _bench(FakeLLM(), tmp_path)
    report = bench.run_benchmark(pattern=f"{TASK}.json")

    assert report.successful == report.failed == 0
    assert report.ungraded == report.total_tasks == 4
    assert report.per_tier[2]["understanding"]["ungraded"] == 3
    assert report.per_tier[2]["agentic_synthesis"]["ungraded"] == 1

    data = json.loads(bench.save_report(report, "r.json").read_text())
    assert data["success_rate"] == 0 and data["ungraded"] == 4
    assert data["per_tier"]["2"]["understanding"]["success_rate"] == 0
    assert data["results"][0]["tier"] == 2
