"""Coverage for the harness-ablation arms (see HARNESS_ABLATION_PLAN.md).

The arms are additive and off by default. The first three tests are the
regression guard of plan step 6.10: they pin the *baseline* configuration, which
is the comparator for the five stored greedy repetitions on `scaled_1comp`. If
one of them fails, the new flags have changed the baseline and those stored
results are no longer comparable.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from tt_bench.benchmark import prompts
from tt_bench.benchmark.runner import TuringTumbleBenchmark
from tt_bench.llm import client as client_module
from tt_bench.llm.client import (
    LLMConfig,
    VLLMClient,
    create_llm_client,
    turing_tumble_tools,
)
from tt_bench.simulator import Board
from tt_bench.tools.executor import TuringTumbleToolExecutor

# ---------------------------------------------------------------------------
# Scripted vLLM transport
# ---------------------------------------------------------------------------
# The arms change the agent loop, so the tests drive the real VLLMClient loop —
# the code path used on JURECA — against a stubbed HTTP layer, rather than a
# stand-in loop that could diverge from it.


class _FakeResponse:
    def __init__(self, payload: dict):
        self.status_code = 200
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        pass


def _assistant_turn(*, tool_calls=None, content=None) -> dict:
    """One scripted /v1/chat/completions response."""
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "choices": [
            {
                "message": msg,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _tool_call(name: str, arguments: dict, call_id: str = "c1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class _ScriptedVLLM:
    """Replaces requests.post with a fixed script of assistant turns."""

    def __init__(self, turns: list[dict]):
        self.turns = list(turns)
        self.payloads: list[dict] = []

    def __call__(self, url, **kwargs):
        self.payloads.append(kwargs.get("json"))
        if not self.turns:
            raise AssertionError(
                "script exhausted: the loop made more API calls than were scripted"
            )
        return _FakeResponse(self.turns.pop(0))

    @property
    def calls(self) -> int:
        return len(self.payloads)


def _install(monkeypatch, turns: list[dict]) -> _ScriptedVLLM:
    stub = _ScriptedVLLM(turns)
    monkeypatch.setattr(client_module.requests, "post", stub)
    return stub


def _vllm_client() -> VLLMClient:
    return VLLMClient(LLMConfig(provider="vllm", model="stub", temperature=0.0))


def _one_ramp_board() -> Board:
    """A 2x5 board a single ramp at (2,0) routes to the right catcher."""
    return Board(rows=2, cols=5, blue_hopper_x=2, blue_hopper_count=1)


def _run_loop(client, executor, turns, monkeypatch, **kwargs):
    stub = _install(monkeypatch, turns)
    result = client.generate_with_tools(
        prompt="solve it",
        tools=turing_tumble_tools(2, 5),
        tool_executor=executor,
        system_prompt=prompts.AGENTIC_SYSTEM_PROMPT,
        max_turns=kwargs.pop("max_turns", 10),
        **kwargs,
    )
    return result, stub


# ---------------------------------------------------------------------------
# 6.10 Regression guard — the baseline must stay observably identical
# ---------------------------------------------------------------------------

# Pinned 2026-09-18 against commit b61678b, the configuration behind
# docs/results_analysis.md §2. Changing a prompt string invalidates the
# comparison with the stored runs; add a *variant* instead.
BASELINE_PROMPT_HASHES = {
    "AGENTIC_SYSTEM_PROMPT": "a65b51dd72dca4db",
    "AGENTIC_PROMPT_TEMPLATE": "bf310fe69e4f1da5",
    "COMPONENT_RULES": "e386c2363b2a74d2",
}


def test_baseline_prompts_are_byte_identical():
    for name, expected in BASELINE_PROMPT_HASHES.items():
        text = getattr(prompts, name)
        actual = hashlib.sha256(text.encode()).hexdigest()[:16]
        assert actual == expected, (
            f"{name} changed (hash {actual}, expected {expected}). The five stored "
            "greedy repetitions on scaled_1comp were produced with the pinned text; "
            "add a prompt variant for an arm instead of editing the baseline."
        )


def test_placement_carries_no_simulation_by_default():
    executor = TuringTumbleToolExecutor(_one_ramp_board())

    placed = executor.place_component("ramp_right", 2, 0)

    assert placed["success"] is True
    assert "simulation" not in placed
    removed = executor.remove_component(2, 0)
    assert "simulation" not in removed


def test_loop_returns_on_content_without_a_callback(monkeypatch):
    executor = TuringTumbleToolExecutor(_one_ramp_board())
    script = [
        _assistant_turn(tool_calls=[_tool_call("get_board_state", {})]),
        _assistant_turn(content='{"success": false}'),
    ]

    (final, error, tool_calls, _results, _usage, turn_lp), stub = _run_loop(
        _vllm_client(), executor, script, monkeypatch
    )

    assert final == {"content": '{"success": false}'}
    assert error == ""
    assert len(tool_calls) == 1
    assert stub.calls == 2
    # One turn_logprobs entry per API call, with or without capture_logprobs —
    # this is what runner.py reports as `turns` (turns_source: api_calls).
    assert turn_lp is not None and len(turn_lp) == 2


# ---------------------------------------------------------------------------
# 6.1 / 6.2  fb-auto: simulator feedback attached at zero turn cost
# ---------------------------------------------------------------------------


def test_auto_simulate_attaches_simulation_to_placement_and_removal():
    executor = TuringTumbleToolExecutor(
        _one_ramp_board(), target_sequence=["blue"], auto_simulate=True
    )

    placed = executor.place_component("ramp_right", 2, 0)
    assert placed["success"] is True
    sim = placed["simulation"]
    assert sim["success"] is True
    assert sim["input_sequence"] == ["blue"]
    assert {"left_catcher", "right_catcher", "total_marbles", "free_fall_errors"} <= set(sim)

    removed = executor.remove_component(2, 0)
    assert removed["simulation"]["success"] is True


def test_auto_simulate_leaves_board_state_untouched():
    """The injected run must be as non-destructive as an agent-initiated one."""
    manual = TuringTumbleToolExecutor(_one_ramp_board(), target_sequence=["blue"])
    auto = TuringTumbleToolExecutor(
        _one_ramp_board(), target_sequence=["blue"], auto_simulate=True
    )

    manual.place_component("ramp_right", 2, 0)
    auto.place_component("ramp_right", 2, 0)

    assert auto.board.blue_balls_remaining == manual.board.blue_balls_remaining
    assert auto.board.get_all_states() == manual.board.get_all_states()
    assert auto.board.marble_history == manual.board.marble_history
    assert auto.get_placed_components() == manual.get_placed_components()


def test_auto_simulate_payload_drops_traces_and_bounds_free_fall():
    """Plan 6.2: the payload must stay small on the biggest board in the corpus."""
    board = Board(rows=15, cols=15, blue_hopper_x=7, blue_hopper_count=8)
    executor = TuringTumbleToolExecutor(
        board, target_sequence=["blue"] * 8, auto_simulate=True
    )

    sim = executor.place_component("crossover", 7, 0)["simulation"]

    assert "execution_traces" not in sim
    assert len(sim["free_fall_errors"]) <= executor.AUTO_SIM_MAX_FREE_FALL_CELLS
    if "free_fall_errors_omitted" in sim:
        assert sim["free_fall_errors_omitted"] > 0
    # 2 KB leaves room for 40 turns of injection inside a 131072-token context.
    assert len(json.dumps(sim)) < 2048, len(json.dumps(sim))


def test_auto_simulate_rejected_placement_carries_no_simulation():
    """A rejected action did not change the board, so feedback would be stale."""
    executor = TuringTumbleToolExecutor(
        _one_ramp_board(),
        available_parts={"ramp_right": 1},
        target_sequence=["blue"],
        auto_simulate=True,
    )
    executor.place_component("ramp_right", 2, 0)

    exhausted = executor.place_component("ramp_right", 3, 1)
    out_of_bounds = executor.place_component("ramp_left", 99, 99)

    assert exhausted["success"] is False and "simulation" not in exhausted
    assert out_of_bounds["success"] is False and "simulation" not in out_of_bounds


# ---------------------------------------------------------------------------
# 6.3  rev-structured: the critique must agree with the scorer
# ---------------------------------------------------------------------------

OFFICIAL_DIR = Path("data/tasks/official/challenges/json")


def _benchmark(**kwargs) -> TuringTumbleBenchmark:
    return TuringTumbleBenchmark(
        llm_client=create_llm_client(LLMConfig(provider="mock", model="stub")),
        challenges_dir=OFFICIAL_DIR,
        output_dir=Path(tempfile.mkdtemp()),
        **kwargs,
    )


def _ch01():
    bench = _benchmark()
    task_info, _raw = bench.load_task(OFFICIAL_DIR / "tt-official-ch01.json")
    return bench, task_info


def test_critique_is_none_when_the_solution_validates():
    bench, task_info = _ch01()
    reference = task_info["solution"]["placed_components"]

    assert bench.validate_synthesis(task_info, reference)[0] is True
    assert bench.build_revision_critique(task_info, reference) is None


def test_critique_names_the_cell_the_scorer_rejected():
    """The critique and the rejection must not describe different rules."""
    bench, task_info = _ch01()
    incomplete = task_info["solution"]["placed_components"][:-1]

    is_valid, reason = bench.validate_synthesis(task_info, incomplete)
    critique = bench.build_revision_critique(task_info, incomplete)

    assert is_valid is False
    assert critique is not None
    assert reason in critique
    # "Illegal free fall: marble 1 traversed empty cell (4, 9)"
    cell = reason.split("empty cell ")[1]
    assert cell in critique.split("Empty cells a marble traversed")[1]


def test_critique_reports_distinct_cells_not_one_entry_per_marble():
    bench, task_info = _ch01()
    incomplete = task_info["solution"]["placed_components"][:-1]

    critique = bench.build_revision_critique(task_info, incomplete)

    listing = critique.split("Empty cells a marble traversed")[1].split("\n")[0]
    assert "(1 distinct" in listing
    assert listing.count("(4, 9)") == 1
    assert "traversed by 8 marble(s)" in listing


def test_critique_carries_the_grounding_rev_retry_withholds():
    bench, task_info = _ch01()
    incomplete = task_info["solution"]["placed_components"][:-1]

    critique = bench.build_revision_critique(task_info, incomplete)

    # Structured: simulator-derived detail the model could act on.
    assert "Observed for input_sequence" in critique
    assert "Declared targets:" in critique
    assert "Remaining inventory:" in critique
    assert "Components you have placed:" in critique
    # The control says none of it.
    retry = bench.REVISION_RETRY_MESSAGE
    for marker in ("Observed", "Declared", "inventory", "(4, 9)"):
        assert marker not in retry


def test_critique_bounds_the_cells_it_names():
    """A 15x15 board can free-fall through dozens of cells."""
    bench = _benchmark()
    task_info = {
        "board": {"width": 15, "height": 15, "ball_hoppers": {"blue": {"x": 7, "count": 8}}},
        "input_sequence": ["blue"] * 8,
        "solution": {"final_marble_state": ["blue"] * 8, "placed_components": []},
        "available_parts": {"ramp_left": 1},
    }

    critique = bench.build_revision_critique(task_info, [])

    assert critique is not None
    listing = critique.split("Empty cells a marble traversed")[1].split("\n")[0]
    assert listing.count("traversed by") <= bench.REVISION_MAX_FREE_FALL_CELLS


# ---------------------------------------------------------------------------
# 6.4  the loop's revision hook
# ---------------------------------------------------------------------------


def test_revision_continues_after_a_rejected_answer(monkeypatch):
    executor = TuringTumbleToolExecutor(_one_ramp_board())
    script = [
        _assistant_turn(content="attempt 1"),
        _assistant_turn(content="attempt 2"),
    ]
    seen: list[str] = []

    def reject_first(content: str):
        seen.append(content)
        return "fix it" if content == "attempt 1" else None

    (final, error, _tc, _tr, _usage, turn_lp), stub = _run_loop(
        _vllm_client(),
        executor,
        script,
        monkeypatch,
        on_final_answer=reject_first,
        revision_rounds=3,
    )

    assert seen == ["attempt 1", "attempt 2"]
    assert final == {"content": "attempt 2"}
    assert error == ""
    assert stub.calls == 2
    # The critique is appended as a user message after the rejected answer.
    second_call_messages = stub.payloads[1]["messages"]
    assert second_call_messages[-2] == {"role": "assistant", "content": "attempt 1"}
    assert second_call_messages[-1] == {"role": "user", "content": "fix it"}
    # Revisions draw on the turn budget: two API calls, two turn entries.
    assert turn_lp is not None and len(turn_lp) == 2


def test_revision_stops_after_the_configured_rounds(monkeypatch):
    executor = TuringTumbleToolExecutor(_one_ramp_board())
    script = [_assistant_turn(content=f"attempt {i}") for i in range(1, 6)]

    (final, _e, _tc, _tr, _u, _lp), stub = _run_loop(
        _vllm_client(),
        executor,
        script,
        monkeypatch,
        on_final_answer=lambda _content: "still wrong",
        revision_rounds=2,
    )

    # 1 initial answer + 2 revisions = 3 calls, then the third answer is returned
    # even though the callback still rejects it.
    assert stub.calls == 3
    assert final == {"content": "attempt 3"}


def test_revision_does_not_fire_on_an_accepted_answer(monkeypatch):
    executor = TuringTumbleToolExecutor(_one_ramp_board())
    calls: list[str] = []

    (final, _e, _tc, _tr, _u, _lp), stub = _run_loop(
        _vllm_client(),
        executor,
        [_assistant_turn(content="correct")],
        monkeypatch,
        on_final_answer=lambda c: calls.append(c) or None,
        revision_rounds=3,
    )

    assert calls == ["correct"]
    assert final == {"content": "correct"}
    assert stub.calls == 1


def test_revision_respects_the_turn_budget(monkeypatch):
    """A revision arm must not buy turns the budget did not grant."""
    executor = TuringTumbleToolExecutor(_one_ramp_board())
    script = [_assistant_turn(content=f"attempt {i}") for i in range(1, 4)]

    (final, error, _tc, _tr, _u, _lp), stub = _run_loop(
        _vllm_client(),
        executor,
        script,
        monkeypatch,
        max_turns=2,
        on_final_answer=lambda _content: "still wrong",
        revision_rounds=10,
    )

    assert stub.calls == 2
    assert final is None
    assert error == "Max turns (2) reached"


def test_revision_is_inert_without_a_callback(monkeypatch):
    """revision_rounds alone must not change the baseline path."""
    executor = TuringTumbleToolExecutor(_one_ramp_board())

    (final, _e, _tc, _tr, _u, _lp), stub = _run_loop(
        _vllm_client(),
        executor,
        [_assistant_turn(content="attempt 1")],
        monkeypatch,
        revision_rounds=3,
    )

    assert stub.calls == 1
    assert final == {"content": "attempt 1"}


def test_revision_skips_an_empty_truncated_answer(monkeypatch):
    """There is no submitted solution to reject, so the episode ends as before."""
    executor = TuringTumbleToolExecutor(_one_ramp_board())
    truncated = {
        "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    calls: list[str] = []

    (final, error, _tc, _tr, _u, _lp), stub = _run_loop(
        _vllm_client(),
        executor,
        [truncated],
        monkeypatch,
        on_final_answer=lambda c: calls.append(c) or "fix it",
        revision_rounds=3,
    )

    assert calls == []
    assert stub.calls == 1
    assert final == {"content": ""}
    assert "truncated" in error


# ---------------------------------------------------------------------------
# 6.5  prompt variants: the feedback axis must stay the feedback axis
# ---------------------------------------------------------------------------

# Blocks that must be byte-identical in all three arms. If an edit to a variant
# touches one of these, the arm has become a general prompt rewrite and its
# effect is no longer attributable to simulation feedback.
SHARED_PROMPT_BLOCKS = [
    """You are a Turing Tumble solver agent.
You MUST use the provided tools to solve this puzzle. You cannot solve it by just thinking, 
you MUST call the tools.""",
    """CRITICAL CONSTRAINT: Marbles may NOT fall through empty cells. Every cell a marble visits
between entering the board and reaching a catcher/interceptor MUST contain a component.
Solutions with any empty-cell traversal will be rejected even if the catcher counts are correct.""",
    """FIXED vs USER COMPONENTS: The board comes with pre-placed components. In get_board_state,
each component has a "source" field:
- "fixed" = part of the original board layout — you CANNOT remove these.
- "user"  = you placed it via place_component — you CAN remove/replace these.
Never attempt to remove a "fixed" component; it will fail and waste a turn.""",
    """Be CONCISE — your analysis should be 2-3 sentences, not paragraphs.
Do not just think about the solution - you must USE the tools to build and test it.""",
    """In your final answer, the "explanation" field must describe your reasoning: what you observed
from the simulations, why you placed each component where you did, and whether you believe the
puzzle is solvable or unsolvable with the given inventory.""",
]


def test_feedback_variants_share_every_non_feedback_block():
    assert sorted(prompts.AGENTIC_SYSTEM_PROMPTS) == ["autosim", "optional", "urged"]
    for arm, text in prompts.AGENTIC_SYSTEM_PROMPTS.items():
        for block in SHARED_PROMPT_BLOCKS:
            assert block in text, f"{arm} prompt no longer carries a shared block"


def test_urged_arm_is_the_pinned_baseline():
    assert prompts.AGENTIC_SYSTEM_PROMPTS["urged"] is prompts.AGENTIC_SYSTEM_PROMPT


def test_optional_arm_drops_the_simulation_mandate():
    text = prompts.AGENTIC_SYSTEM_PROMPTS["optional"]

    assert "REQUIRED WORKFLOW" not in text
    assert "MUST call run_simulation" not in text
    assert "run_simulation" in text  # the tool is still offered, just not ordered


def test_autosim_arm_announces_the_injection_and_stops_ordering_the_call():
    text = prompts.AGENTIC_SYSTEM_PROMPTS["autosim"]

    assert "AUTOMATIC SIMULATION" in text
    assert "MUST call run_simulation" not in text
    assert '"simulation" field' in text


# ---------------------------------------------------------------------------
# 6.6  end-to-end wiring and provenance
# ---------------------------------------------------------------------------


def _vllm_benchmark(tmp_path, **kwargs) -> TuringTumbleBenchmark:
    return TuringTumbleBenchmark(
        llm_client=create_llm_client(
            LLMConfig(provider="vllm", model="stub", temperature=0.0)
        ),
        challenges_dir=OFFICIAL_DIR,
        output_dir=tmp_path,
        **kwargs,
    )


def test_revision_arm_rejects_a_provider_that_would_ignore_it(tmp_path):
    """A silently-baseline run reported under an arm label is the worst outcome."""
    with pytest.raises(ValueError, match="needs the vllm provider"):
        TuringTumbleBenchmark(
            llm_client=create_llm_client(LLMConfig(provider="mock", model="m")),
            challenges_dir=OFFICIAL_DIR,
            output_dir=tmp_path,
            revision_style="structured",
        )


def test_unknown_arm_labels_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown prompt_variant"):
        _vllm_benchmark(tmp_path, prompt_variant="nonsense")
    with pytest.raises(ValueError, match="Unknown revision_style"):
        _vllm_benchmark(tmp_path, revision_style="nonsense")


def test_arm_settings_round_trip_into_the_report(tmp_path, monkeypatch):
    bench = _vllm_benchmark(
        tmp_path,
        auto_simulate=True,
        prompt_variant="autosim",
        revision_style="structured",
        revision_rounds=2,
        max_turns=7,
    )
    _install(monkeypatch, [_assistant_turn(content='{"success": false}')])

    report = bench.run_benchmark(
        pattern="tt-official-ch01.json", task_types=["agentic_synthesis"]
    )
    bench.save_report(report, filename="r.json")
    saved = json.loads((tmp_path / "r.json").read_text())

    assert saved["harness"] == {
        "prompt_variant": "autosim",
        "auto_simulate": True,
        "revision_style": "structured",
        "revision_rounds": 2,
        "declare_zero_parts": False,
        "observable_inventory": False,
        "max_tokens": 32768,
    }
    assert saved["sampling"]["max_turns"] == 7


def test_baseline_report_records_the_baseline_arm(tmp_path, monkeypatch):
    bench = _vllm_benchmark(tmp_path)
    _install(monkeypatch, [_assistant_turn(content='{"success": false}')])

    report = bench.run_benchmark(
        pattern="tt-official-ch01.json", task_types=["agentic_synthesis"]
    )
    bench.save_report(report, filename="r.json")
    saved = json.loads((tmp_path / "r.json").read_text())

    assert saved["harness"]["prompt_variant"] == "urged"
    assert saved["harness"]["auto_simulate"] is False
    assert saved["harness"]["revision_style"] == "off"


def test_fb_auto_reaches_the_model_as_tool_output(tmp_path, monkeypatch):
    """The injected simulation must actually arrive in the conversation."""
    bench = _vllm_benchmark(tmp_path, auto_simulate=True, prompt_variant="autosim")
    script = [
        _assistant_turn(
            tool_calls=[
                _tool_call("place_component", {"component_type": "ramp_left", "x": 4, "y": 9})
            ]
        ),
        _assistant_turn(content='{"success": false}'),
    ]
    stub = _install(monkeypatch, script)

    bench.run_benchmark(
        pattern="tt-official-ch01.json", task_types=["agentic_synthesis"]
    )

    tool_messages = [
        m for m in stub.payloads[-1]["messages"] if m.get("role") == "tool"
    ]
    assert len(tool_messages) == 1
    payload = json.loads(tool_messages[0]["content"])
    assert payload["success"] is True
    assert payload["simulation"]["success"] is True
    # ch01 needs four components; one placement leaves gaps, and naming them
    # unprompted is the whole point of the arm.
    assert payload["simulation"]["free_fall_errors"]
    assert "execution_traces" not in payload["simulation"]
    # The autosim prompt, not the baseline, was sent.
    assert "AUTOMATIC SIMULATION" in stub.payloads[0]["messages"][0]["content"]


def test_rev_structured_feeds_the_critique_back_end_to_end(tmp_path, monkeypatch):
    bench = _vllm_benchmark(tmp_path, revision_style="structured", revision_rounds=1)
    script = [
        _assistant_turn(content='{"final_solution": [], "success": true}'),
        _assistant_turn(content='{"final_solution": [], "success": true}'),
    ]
    stub = _install(monkeypatch, script)

    report = bench.run_benchmark(
        pattern="tt-official-ch01.json", task_types=["agentic_synthesis"]
    )

    assert stub.calls == 2
    injected = stub.payloads[1]["messages"][-1]
    assert injected["role"] == "user"
    assert "REJECTED by the scorer" in injected["content"]
    assert "Empty cells a marble traversed" in injected["content"]
    assert report.task_results[0].metrics["revisions"] == 1


def test_rev_retry_fires_on_the_same_trigger_but_says_nothing(tmp_path, monkeypatch):
    bench = _vllm_benchmark(tmp_path, revision_style="retry", revision_rounds=1)
    script = [
        _assistant_turn(content='{"final_solution": [], "success": true}'),
        _assistant_turn(content='{"final_solution": [], "success": true}'),
    ]
    stub = _install(monkeypatch, script)

    report = bench.run_benchmark(
        pattern="tt-official-ch01.json", task_types=["agentic_synthesis"]
    )

    injected = stub.payloads[1]["messages"][-1]
    assert injected["content"] == bench.REVISION_RETRY_MESSAGE
    assert "REJECTED by the scorer" not in injected["content"]
    # Same trigger, same count — only the message differs.
    assert report.task_results[0].metrics["revisions"] == 1


def test_baseline_records_zero_revisions(tmp_path, monkeypatch):
    bench = _vllm_benchmark(tmp_path)
    _install(monkeypatch, [_assistant_turn(content='{"success": false}')])

    report = bench.run_benchmark(
        pattern="tt-official-ch01.json", task_types=["agentic_synthesis"]
    )

    assert report.task_results[0].metrics["revisions"] == 0
