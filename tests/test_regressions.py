"""Regression coverage for benchmark and agentic execution invariants."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from scripts.generate_scaled import (
    ch01_pattern,
    generate_height_padded_variants,
    generate_padded_variants,
)
from scripts.generate_scl import generate_scl
from tt_bench.benchmark.runner import TuringTumbleBenchmark
from tt_bench.cli.simulate import run_cli
from tt_bench.llm.client import (
    AnthropicClient,
    LLMConfig,
    OpenAIClient,
    turing_tumble_tools,
)
from tt_bench.simulator import Bit, Board, verify_task
from tt_bench.simulator.targets import validate_targets
from tt_bench.tools.executor import TuringTumbleToolExecutor


def test_tool_simulation_restores_mutable_board_state():
    # One marble isolates restoration from the lever's automatic second release.
    board = Board(rows=2, cols=5, blue_hopper_x=2, blue_hopper_count=1)
    board.place(2, 0, Bit(2, 0, state=0))
    executor = TuringTumbleToolExecutor(board)

    first = executor.run_simulation(["blue"])
    second = executor.run_simulation(["blue"])

    assert first["final_bit_states"] == second["final_bit_states"] == {"bit_2_0": 1}
    assert board.get_all_states() == {"bit_2_0": 0}
    assert board.blue_balls_remaining == 1
    assert board.marble_history == []


def test_solution_found_requires_explicit_target_match():
    wrong = TuringTumbleToolExecutor(
        Board(rows=1, cols=5, blue_hopper_x=2, blue_hopper_count=1),
        target_sequence=["blue"],
        target_final_state=["red"],
    )
    wrong.run_simulation(["blue"])
    assert wrong.is_solution_found() is False

    correct = TuringTumbleToolExecutor(
        Board(rows=1, cols=5, blue_hopper_x=2, blue_hopper_count=1),
        target_sequence=["blue"],
        target_final_state=["blue"],
    )
    correct.run_simulation(["blue"])
    assert correct.is_solution_found() is True


def test_tool_placed_gear_network_is_connected():
    board = Board(rows=3, cols=6, blue_hopper_x=2)
    executor = TuringTumbleToolExecutor(
        board, {"gear_bit": 2, "gear": 1}
    )

    assert executor.place_component("gear_bit", 2, 0)["success"]
    assert executor.place_component("gear", 3, 0)["success"]
    assert executor.place_component("gear_bit", 4, 0)["success"]
    board.release_marble("blue")

    assert board.get_all_states() == {
        "gear_bit_2_0": 1,
        "gear_bit_4_0": 1,
    }


def test_tool_schema_uses_actual_board_bounds():
    schemas = turing_tumble_tools(rows=16, cols=13)
    placement = next(
        tool for tool in schemas if tool["function"]["name"] == "place_component"
    )
    properties = placement["function"]["parameters"]["properties"]
    assert properties["x"]["maximum"] == 12
    assert properties["y"]["maximum"] == 15


def test_question_schema_normalization_covers_all_variants():
    normalize = TuringTumbleBenchmark._normalise_question
    assert normalize({"qid": "a", "type": "numeric", "question": "?", "answer": "1"}, 1) == (
        "a", "numeric", "?", "1"
    )
    assert normalize(
        {"question_id": "b", "type": "select", "question": "?", "answer": "A"}, 2
    ) == ("b", "select", "?", "A")
    assert normalize(
        {"id": "c", "type": "calculation", "question": "?", "expected_answer": "5"}, 3
    ) == ("c", "calculation", "?", "5")


def test_understanding_validation_rejects_partial_trace_answers():
    benchmark = TuringTumbleBenchmark.__new__(TuringTumbleBenchmark)
    expected = "The blue ball exits on the left side."
    partial = benchmark._validate_understanding(
        Board(), "output_sequence", "?", {"answer": "blue"}, expected
    )
    correct = benchmark._validate_understanding(
        Board(), "output_sequence", "?", {"answer": "blue exits left"}, expected
    )
    numeric = benchmark._validate_understanding(
        Board(), "calculation", "?", {"answer": 5}, "5"
    )
    wrong_select = benchmark._validate_understanding(
        Board(), "select", "?", {"answer": "not right"}, "right"
    )

    assert partial["correct"] is False
    assert correct["correct"] is True
    assert numeric["correct"] is True
    assert wrong_select["correct"] is False


def test_understanding_prompt_includes_options_and_hints():
    benchmark = TuringTumbleBenchmark.__new__(TuringTumbleBenchmark)
    benchmark._board_for_prompt = lambda *a, **k: Board()

    with_extras = benchmark.build_understanding_prompt(
        {"task_id": "x"},
        "select",
        "Where does the 2nd ball go?",
        "Return the selected option exactly.",
        options=["bottom (output)", "interceptor"],
        hints=["Each bit represents 1, 2, 4, 8"],
    )
    assert "## Options" in with_extras
    assert "- interceptor" in with_extras
    assert "## Hints" in with_extras

    without_extras = benchmark.build_understanding_prompt(
        {"task_id": "x"}, "numeric", "How many?", "Provide a number."
    )
    assert "## Options" not in without_extras
    assert "## Hints" not in without_extras


def test_verify_task_reuses_single_simulation_result():
    task = {
        "board": {
            "width": 5,
            "height": 1,
            "hopper_entry_mode": "column",
            "ball_hoppers": {
                "blue": {"x": 2, "count": 1},
                "red": {"x": 4, "count": 0},
            },
            "trigger_levers": {"left": {"x": 2}, "right": {"x": 4}},
            "fixed_components": [],
        },
        "solution": {"placed_components": []},
        "input_sequence": ["blue"],
        "expected_output": {"left_catcher": 1, "right_catcher": 0},
    }
    assert verify_task(task) is True


def test_padded_generator_only_widens_verified_board(tmp_path):
    task = {
        "task_id": "tiny",
        "objective": "Route blue to the left catcher",
        "board": {
            "width": 5,
            "height": 1,
            "hopper_entry_mode": "column",
            "ball_hoppers": {
                "blue": {"x": 2, "count": 1},
                "red": {"x": 4, "count": 0},
            },
            "trigger_levers": {"left": {"x": 2}, "right": {"x": 4}},
            "fixed_components": [],
        },
        "available_parts": {},
        "solution": {"placed_components": [], "final_marble_state": ["blue"]},
        "input_sequence": ["blue"],
        "expected_output": {},
    }

    assert generate_padded_variants(task, tmp_path, [2]) == 1
    generated = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert generated["board"]["width"] == 7
    assert generated["board"]["height"] == 1
    assert verify_task(generated) is True


def test_simulator_cli_imports_all_runtime_helpers():
    assert callable(run_cli)


class _FakeResponse:
    def __init__(self, data):
        self._data = data
        self.status_code = 200
        self.text = json.dumps(data)

    def json(self):
        return self._data

    def raise_for_status(self):
        return None


class _FakeExecutor:
    def execute(self, name, arguments):
        return {"success": True, "name": name, "arguments": arguments}

    def is_solution_found(self):
        return False


def test_anthropic_client_executes_native_tool_calls(monkeypatch):
    responses = iter(
        [
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "get_board_state",
                        "input": {},
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
            {
                "content": [{"type": "text", "text": '{"success": true}'}],
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        ]
    )
    monkeypatch.setattr(
        "tt_bench.llm.client.requests.post",
        lambda *args, **kwargs: _FakeResponse(next(responses)),
    )
    client = AnthropicClient(
        LLMConfig(provider="anthropic", model="claude-test", api_key="test")
    )
    result, error, calls, tool_results, usage, _ = client.generate_with_tools(
        "prompt",
        turing_tumble_tools(11, 11),
        _FakeExecutor(),
        max_turns=2,
    )

    assert error == ""
    assert result == {"success": True}
    assert calls[0].name == "get_board_state"
    assert tool_results[0].result["success"] is True
    assert usage == {"prompt_tokens": 7, "completion_tokens": 5}


def test_openai_agentic_payload_honors_token_limit(monkeypatch):
    captured = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return _FakeResponse(
            {
                "model": "gpt-test",
                "choices": [{"message": {"content": '{"done": true}'}}],
                "usage": {},
            }
        )

    monkeypatch.setattr("tt_bench.llm.client.requests.post", fake_post)
    client = OpenAIClient(
        LLMConfig(provider="openai", model="gpt-test", api_key="test")
    )
    client.generate_with_tools(
        "prompt", [], _FakeExecutor(), max_turns=1, max_tokens=1234
    )

    assert captured["max_completion_tokens"] == 1234


def test_scaled_pattern_puts_catcher_under_the_actual_exit():
    # The zig-zag exits one column further right after an odd number of rows;
    # a catcher fixed at column 2 loses every marble on those boards.
    for rows in range(4, 17):
        _, _, triggers, _ = ch01_pattern(rows)
        assert triggers["left"]["x"] == (2 if rows % 2 == 0 else 3)


def test_height_padding_extends_the_marble_path(tmp_path):
    source = json.loads(
        Path("data/tasks/challenges_1comp/tt-official-ch01-1comp.json").read_text()
    )
    assert generate_height_padded_variants(source, tmp_path, [2]) == 1

    generated = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert generated["board"]["height"] == source["board"]["height"] + 2
    assert generated["board"]["width"] == source["board"]["width"]
    # Taller board, still no free-fall: the added rows carry the marble.
    assert verify_task(generated) is True


def test_scl_generator_derives_ground_truth_and_rejects_degenerate():
    source = json.loads(
        Path("data/tasks/challenges_1comp/tt-official-ch01-1comp.json").read_text()
    )
    built = generate_scl(source, 8, "probe-scl8", "1comp")

    assert built is not None
    total = len(built["board"]["fixed_components"]) + len(
        built["solution"]["placed_components"]
    )
    assert total == 8
    assert verify_task(built) is True

    # Ground truth is read off a real run, not copied from the source, whose
    # outcome a rescaled board does not reproduce.
    assert built["solution"]["final_marble_state"]
    assert built["expected_output"]["left_catcher"] or built["expected_output"]["right_catcher"]

    # The board must not already work without the component to be placed.
    stripped = deepcopy(built)
    stripped["solution"]["placed_components"] = []
    assert verify_task(stripped) is False


def test_tier_filter_applies_before_max_tasks(tmp_path):
    # Truncating first meant a tier-limited run sampled from the wrong tier and
    # could come back with nothing to do.
    from tt_bench.benchmark.runner import TuringTumbleBenchmark

    for index, tier in enumerate([1, 1, 1, 2, 2]):
        (tmp_path / f"task{index}.json").write_text(json.dumps({
            "task_id": f"task{index}", "tier": tier,
            "board": {"width": 5, "height": 1, "fixed_components": [],
                      "ball_hoppers": {"blue": {"x": 2, "count": 1}},
                      "trigger_levers": {"left": {"x": 2}, "right": {"x": 4}}},
            "solution": {"placed_components": []}, "input_sequence": ["blue"],
        }))

    benchmark = TuringTumbleBenchmark.__new__(TuringTumbleBenchmark)
    benchmark.challenges_dir = tmp_path
    files = sorted(tmp_path.glob("*.json"))
    tier_two = [f for f in files if json.loads(f.read_text())["tier"] == 2]
    assert len(tier_two) == 2
    # The tier-2 files sort last, so a pre-filter truncation to 3 would drop them.
    assert files[:3] == [f for f in files if json.loads(f.read_text())["tier"] == 1][:3]


def test_intercepted_marbles_appear_in_the_scored_sequence():
    # The catcher-to-sequence mapping lives in three places. Two of them used to
    # drop interceptor hits, so a task whose ground truth ends in "intercepted"
    # could never be scored correct — not even by its own reference solution.
    task = json.loads(
        (Path(__file__).resolve().parent.parent
         / "data/tasks/official/challenges/json/tt-official-ch16-pB.json").read_text()
    )
    target = task["solution"]["final_marble_state"]
    assert "intercepted" in target

    board = Board.from_task_dict(task)
    results = board.run(task["input_sequence"])

    # The dataset-side check (validation.py) always agreed with the ground truth.
    assert verify_task(task) is True

    # The scorer must reach the same verdict, through the shared contract.
    assert validate_targets(task, board, results)[0]
    benchmark = TuringTumbleBenchmark.__new__(TuringTumbleBenchmark)
    solved, message = benchmark._validate_simulation_results(board, task, results)
    assert solved, message

    # So must the agent's early-stop signal, or it keeps working on a solved board.
    executor = TuringTumbleToolExecutor(
        board,
        target_final_state=target,
        expected_output=task.get("expected_output"),
    )
    assert executor._results_match_target(
        results, left_count=1, right_count=0, interceptor_count=1, free_fall_errors=[]
    )


def test_required_output_reads_ball_colours_not_catchers():
    """The two sequence targets describe different readings of one run.

    ``final_marble_state`` names the catcher each marble reached;
    ``required_output`` is the guide's printed strip, which is the colour of
    each ball. Comparing the strip against catchers rejected correct boards --
    on ch09-pA a blue ball leaves on the right, so the two sequences differ.
    """
    task = json.loads(
        (Path(__file__).resolve().parent.parent
         / "data/tasks/official/challenges/json/tt-official-ch09-pA.json").read_text()
    )
    catchers = task["solution"]["final_marble_state"]
    printed = task["required_output"]
    assert catchers != printed, "this board must exercise the divergence"

    board = Board.from_task_dict(task)
    results = board.run(task["input_sequence"])
    assert validate_targets(task, board, results)[0]

    # Swapping the two targets must be rejected, or they are interchangeable
    # and the contract means nothing.
    swapped = deepcopy(task)
    swapped["solution"]["final_marble_state"] = printed
    swapped["required_output"] = catchers
    board = Board.from_task_dict(swapped)
    results = board.run(swapped["input_sequence"])
    assert not validate_targets(swapped, board, results)[0]


def test_bit_state_objectives_are_scored():
    """A "flip bits X and Y" goal has to be a target, not a note.

    Both balls reaching the left catcher says nothing about the bits, so the
    bit configuration must be declared where scoring reads it.
    """
    task = json.loads(
        (Path(__file__).resolve().parent.parent
         / "data/tasks/official/challenges/json/tt-official-ch11.json").read_text()
    )
    goal = task["expected_output"]["final_bit_states"]
    assert any(state == 1 for state in goal.values())
    assert verify_task(task)

    wrong = deepcopy(task)
    for key, state in list(wrong["expected_output"]["final_bit_states"].items()):
        if state == 1:
            wrong["expected_output"]["final_bit_states"][key] = 0
    assert not verify_task(wrong)


def test_trigger_levers_span_their_half_of_the_board():
    """Each lever is a bar, not a single cell.

    On ch09-pA the red balls leave at column 3 and the blue ones at 4 or 6, so
    an exact-column catcher could never catch all three. The ball return sits
    between the two lever cups, not at the middle of the board -- a padded
    board keeps its levers where they were while the board grows wider.
    """
    board = Board(rows=11, cols=11, blue_hopper_x=2, red_hopper_x=8,
                  left_catcher_x=3, right_catcher_x=7)
    assert board.catcher_at(3) == "left_catcher"
    assert board.catcher_at(4) == "left_catcher"
    assert board.catcher_at(6) == "right_catcher"
    assert board.catcher_at(7) == "right_catcher"
    assert board.catcher_at(5) is None          # straight into the return

    # A width-padded board must keep its own lever columns catching.
    wide = Board(rows=11, cols=15, blue_hopper_x=2, red_hopper_x=8,
                 left_catcher_x=3, right_catcher_x=7)
    assert wide.catcher_at(3) == "left_catcher"
    assert wide.catcher_at(7) == "right_catcher"
