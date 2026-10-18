"""Regression tests for the 2026-10-18 audit: each one fails on the old code."""

from __future__ import annotations

import json
from pathlib import Path

from tests.test_harness_arms import (
    _assistant_turn,
    _install,
    _one_ramp_board,
    _run_loop,
    _vllm_client,
)
from tt_bench.benchmark.runner import TuringTumbleBenchmark
from tt_bench.simulator import Board, Crossover, GearBit, Side, verify_task
from tt_bench.tools.executor import TuringTumbleToolExecutor

OFFICIAL = Path(__file__).resolve().parent.parent / "data/tasks/official/challenges/json"


def _entry_gap_task(**extra) -> dict:
    """The blue marble enters through an empty cell, then two ramps take it left."""
    return {
        "board": {
            "width": 5, "height": 3, "hopper_entry_mode": "column",
            "ball_hoppers": {"blue": {"x": 2, "count": 1}, "red": {"x": 4, "count": 0}},
            "fixed_components": [],
        },
        "solution": {"placed_components": [
            {"type": "ramp_left", "x": 2, "y": 1},
            {"type": "ramp_left", "x": 1, "y": 2},
        ]},
        "input_sequence": ["blue"],
        "expected_output": {"left_catcher": 1},
        **extra,
    }


def test_trials_allow_the_empty_entry_cell_like_every_other_scorer():
    assert verify_task(_entry_gap_task())
    assert verify_task(_entry_gap_task(trials=[{"expect": {"left_catcher": 1}}]))


def test_inward_red_marble_enters_a_crossover_moving_left():
    board = Board(rows=3, cols=11, red_hopper_x=8, red_hopper_count=1,
                  hopper_entry_mode="inward")
    board.place(7, 0, Crossover(7, 0))

    path = board.release_marble(Side.RED).path

    assert path[:3] == [(8, -1), (7, 0), (6, 1)]


def test_meshed_gear_bits_in_different_states_are_rejected():
    task = _entry_gap_task()
    task["board"]["fixed_components"] = [
        {"type": "gear_bit", "x": 3, "y": 0, "state": 0},
        {"type": "gear_bit", "x": 4, "y": 0, "state": 1},
    ]
    assert not verify_task(task)
    task["board"]["fixed_components"][1]["state"] = 0
    assert verify_task(task)


def test_executor_rejects_parts_no_player_can_place():
    executor = TuringTumbleToolExecutor(_one_ramp_board())

    assert executor.place_component("bit", 2, 0, state=2)["success"] is False
    assert executor.place_component("ramp_left", 2.5, 0)["success"] is False
    assert executor.place_component("bit", 2, 0, state="1")["success"] is False
    assert executor.place_component("bit", 2, 0, state=1)["success"] is True


def test_malformed_tool_arguments_cost_a_turn_not_the_episode(monkeypatch):
    executor = TuringTumbleToolExecutor(_one_ramp_board())
    broken = {"id": "c1", "type": "function",
              "function": {"name": "place_component", "arguments": '{"x": 2, "y": 0,'}}
    script = [
        _assistant_turn(tool_calls=[broken], content="placing a ramp"),
        _assistant_turn(content="done"),
    ]

    (final, error, tool_calls, results, _usage, _lp), stub = _run_loop(
        _vllm_client(), executor, script, monkeypatch
    )

    assert final == {"content": "done"} and error == ""
    assert "not a JSON object" in results[0].error
    assert stub.calls == 2


def test_cut_off_reasoning_is_not_scored_as_the_answer(monkeypatch):
    truncated = {
        "choices": [{"message": {"role": "assistant", "content": None,
                                 "reasoning_content": 'Maybe {"answer": "left"}? No wait'},
                     "finish_reason": "length"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    _install(monkeypatch, [truncated] * 3)
    client = _vllm_client()
    client.config.max_retries = 3
    monkeypatch.setattr("time.sleep", lambda _s: None)

    data, error, usage, _lp = client.generate_json("which catcher?")

    assert data is None and error
    # All three attempts were paid for.
    assert usage == {"prompt_tokens": 30, "completion_tokens": 15}


def test_trigger_answer_in_plain_prose_is_accepted():
    benchmark = TuringTumbleBenchmark.__new__(TuringTumbleBenchmark)
    expected = "The first blue ball triggers the left lever, releasing a red ball."
    predicted = {"answer": "It triggers the left lever, which releases a red ball."}

    verdict = benchmark._validate_understanding(
        Board(), "trigger_sequence", "q", predicted, expected_answer=expected
    )

    assert verdict["correct"] is True


def test_controlled_ladder_accepts_every_official_reference_solution():
    from scripts.run_controlled_ladder import evaluate_candidate

    benchmark = TuringTumbleBenchmark.__new__(TuringTumbleBenchmark)
    checked = 0
    for path in sorted(OFFICIAL.glob("*.json")):
        info, _ = benchmark.load_task(path)
        reference = info.get("solution", {}).get("placed_components", [])
        if not reference:
            continue
        evaluation = evaluate_candidate(benchmark, info, reference)
        ladder = bool(evaluation["inventory_compliance"] and evaluation["physical_legality"]
                      and evaluation["functional_correctness"])
        assert ladder == benchmark.validate_synthesis(info, reference)[0], path.name
        assert ladder, path.name
        checked += 1
    assert checked >= 50
