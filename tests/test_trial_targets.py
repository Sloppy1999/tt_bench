"""The multi-trial scoring contract.

Tiers 3 and 4 of the official guide state goals that one simulation cannot
decide: "use register A to count the number of blue balls" is judged over
several ball counts, and "reverse the direction of each bit, regardless of the
direction it starts in" over several starting configurations. These tests pin
the contract that expresses them.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from tt_bench.simulator import Board, Gear, GearBit, build_gear_connections, verify_task
from tt_bench.simulator.targets import validate_targets

CHALLENGES = (
    Path(__file__).resolve().parent.parent
    / "data/tasks/official/challenges/json"
)


def _load(stem: str) -> dict:
    return json.loads((CHALLENGES / f"{stem}.json").read_text())


def _ch16pB() -> tuple[dict, Board, list]:
    """A board whose single-run targets are known good, reused as a fixture."""
    task = _load("tt-official-ch16-pB")
    board = Board.from_task_dict(task)
    results = board.run(task["input_sequence"])
    return task, board, results


# ── the single-run contract is unchanged ────────────────────────────────────


def test_task_without_trials_keeps_single_run_semantics():
    task, board, results = _ch16pB()
    assert "trials" not in task
    assert validate_targets(task, board, results)[0]


def test_state_only_declaration_is_still_not_a_marble_target():
    """Unchanged behaviour: a bit-state target alone does not score a task."""
    task, board, results = _ch16pB()
    stripped = deepcopy(task)
    stripped["solution"].pop("final_marble_state", None)
    stripped.pop("required_output", None)
    stripped["expected_output"] = {"final_bit_states": board.get_all_states()}
    ok, why = validate_targets(stripped, board, results)
    assert not ok
    assert "No explicit marble output target" in why


# ── trials ──────────────────────────────────────────────────────────────────


def test_trials_replace_the_single_run_verdict():
    """A trial table decides the task; the passed-in results are ignored."""
    task, board, results = _ch16pB()
    trial_task = deepcopy(task)
    trial_task["trials"] = [
        {
            "name": "as-declared",
            "input_sequence": task["input_sequence"],
            "expect": {"final_marble_state": task["solution"]["final_marble_state"]},
        }
    ]
    ok, why = validate_targets(trial_task, board, results)
    assert ok, why
    assert "1 declared trials" in why


def test_every_trial_must_pass():
    task, board, results = _ch16pB()
    trial_task = deepcopy(task)
    good = task["solution"]["final_marble_state"]
    trial_task["trials"] = [
        {"name": "good", "input_sequence": task["input_sequence"],
         "expect": {"final_marble_state": good}},
        {"name": "impossible", "input_sequence": task["input_sequence"],
         "expect": {"final_marble_state": list(reversed(good)) + ["blue"]}},
    ]
    ok, why = validate_targets(trial_task, board, results)
    assert not ok
    assert "trial impossible" in why


def test_trial_with_no_expectations_is_rejected():
    """An empty trial would otherwise pass vacuously and score a broken board."""
    task, board, results = _ch16pB()
    trial_task = deepcopy(task)
    trial_task["trials"] = [{"name": "empty", "expect": {}}]
    ok, why = validate_targets(trial_task, board, results)
    assert not ok
    assert "no expectations" in why


def test_empty_trial_list_is_rejected_not_treated_as_absent():
    task, board, results = _ch16pB()
    trial_task = deepcopy(task)
    trial_task["trials"] = []
    ok, why = validate_targets(trial_task, board, results)
    assert not ok
    assert "nonempty list" in why


# ── initial bit states ──────────────────────────────────────────────────────


def _bit_task(min_bits: int = 1) -> tuple[dict, Board, list, list[str]]:
    """A task carrying at least ``min_bits`` plain bits, plus their keys."""
    for stem in sorted(p.stem for p in CHALLENGES.glob("tt-official-ch*.json")):
        task = _load(stem)
        try:
            board = Board.from_task_dict(task)
        except Exception:
            continue
        bits = sorted(k for k in board.get_all_states() if k.startswith("bit_"))
        if len(bits) >= min_bits and task.get("input_sequence"):
            return task, board, board.run(task["input_sequence"]), bits
    pytest.skip(f"no official board with {min_bits} plain bit(s)")


def test_initial_bit_states_are_applied_before_the_run():
    task, board, results, bits = _bit_task()
    key = bits[0]
    trial_task = deepcopy(task)
    trial_task["trials"] = [
        {
            "name": "forced",
            "input_sequence": task["input_sequence"],
            "initial_bit_states": {key: 1},
            # Asserting the bit is *readable* as set is enough here; what the run
            # does to it afterwards is the board's business.
            "expect": {"left_catcher": 0, "right_catcher": 0, "intercepted": 0},
        }
    ]
    # The expectation above is deliberately wrong, so the failure names the
    # catcher target -- proving the trial ran rather than erroring on the bit.
    ok, why = validate_targets(trial_task, board, results)
    assert not ok
    assert "expected_output.left_catcher" in why or "expected_output.right_catcher" in why


def test_unknown_bit_key_is_rejected():
    task, board, results, _ = _bit_task()
    trial_task = deepcopy(task)
    trial_task["trials"] = [
        {"name": "bad", "initial_bit_states": {"bit_99_99": 1},
         "expect": {"left_catcher": 0}}
    ]
    ok, why = validate_targets(trial_task, board, results)
    assert not ok
    assert "No bit at bit_99_99" in why


def test_malformed_bit_key_is_rejected():
    task, board, results, _ = _bit_task()
    trial_task = deepcopy(task)
    trial_task["trials"] = [
        {"name": "bad", "initial_bit_states": {"ramp_right_3_4": 1},
         "expect": {"left_catcher": 0}}
    ]
    ok, why = validate_targets(trial_task, board, results)
    assert not ok
    assert "Invalid bit key" in why


# ── registers ───────────────────────────────────────────────────────────────


def test_register_reads_bits_most_significant_first():
    task, board, results, bits = _bit_task(min_bits=2)
    keys = bits[:2]

    trial_task = deepcopy(task)
    trial_task["registers"] = {"A": keys}
    trial_task["trials"] = [
        {
            "name": "one-zero",
            "input_sequence": [],
            "initial_bit_states": {keys[0]: 1, keys[1]: 0},
            "expect": {"registers": {"A": 0b10}},
        },
        {
            "name": "zero-one",
            "input_sequence": [],
            "initial_bit_states": {keys[0]: 0, keys[1]: 1},
            "expect": {"registers": {"A": 0b01}},
        },
    ]
    ok, why = validate_targets(trial_task, board, results)
    assert ok, why


def test_register_must_be_declared():
    task, board, results, _ = _bit_task()
    trial_task = deepcopy(task)
    trial_task["trials"] = [
        {"name": "t", "input_sequence": [], "expect": {"registers": {"A": 1}}}
    ]
    ok, why = validate_targets(trial_task, board, results)
    assert not ok
    assert "not declared" in why


def test_register_naming_absent_bits_is_rejected():
    task, board, results, _ = _bit_task()
    trial_task = deepcopy(task)
    trial_task["registers"] = {"A": ["bit_99_99"]}
    trial_task["trials"] = [
        {"name": "t", "input_sequence": [], "expect": {"registers": {"A": 0}}}
    ]
    ok, why = validate_targets(trial_task, board, results)
    assert not ok
    assert "absent bits" in why


# ── the board must survive being scored ─────────────────────────────────────


def test_scoring_trials_leaves_the_board_untouched():
    """run_simulation reports final_bit_states *after* asking whether it is done.

    If trial evaluation reset and re-ran the board in place, the agent would be
    handed the last trial's board as its own simulation result.
    """
    task, board, results, bits = _bit_task()
    key = bits[0]
    before_states = board.get_all_states()
    before_hoppers = (board.blue_balls_remaining, board.red_balls_remaining)
    before_released = board.marble_count_released

    trial_task = deepcopy(task)
    trial_task["trials"] = [
        {
            "name": "disturbing",
            "input_sequence": task["input_sequence"],
            "initial_bit_states": {key: 1 - before_states[key]},
            "expect": {"left_catcher": 0},
        }
    ]
    validate_targets(trial_task, board, results)

    assert board.get_all_states() == before_states
    assert (board.blue_balls_remaining, board.red_balls_remaining) == before_hoppers
    assert board.marble_count_released == before_released


# ── gear bits turn as one group ─────────────────────────────────────────────


def test_initial_state_on_a_gear_bit_carries_to_its_whole_group():
    """Two gear bits meshed through a gear turn as one.

    Setting one and leaving the other would describe a configuration the
    physical board cannot hold, so the trial must move the whole group. The
    board is built here rather than borrowed from the corpus: an earlier
    version of this test pinned itself to whichever official board happened to
    mesh two gear bits, and broke the moment that board was re-transcribed.
    """
    board = Board(rows=11, cols=11)
    board.place(3, 4, GearBit(3, 4, state=0))
    board.place(4, 4, Gear(4, 4))
    board.place(5, 4, GearBit(5, 4, state=0))
    build_gear_connections(board)
    group = ["gear_bit_3_4", "gear_bit_5_4"]
    assert all(k in board.get_all_states() for k in group)

    trial_task = {"registers": {"G": group}}
    trial_task["trials"] = [
        {
            "name": "both-right",
            "input_sequence": [],
            "initial_bit_states": {"gear_bit_3_4": 1},
            "expect": {"registers": {"G": 0b11}},
        },
        {
            "name": "both-left",
            "input_sequence": [],
            "initial_bit_states": {"gear_bit_3_4": 0},
            "expect": {"registers": {"G": 0b00}},
        },
    ]
    ok, why = validate_targets(trial_task, board, [])
    assert ok, why


# ── the register boards transcribed from the guide ──────────────────────────


@pytest.mark.parametrize(
    "stem,table",
    [
        # Challenge 21 "Quantum Number": count the blue balls.
        ("tt-official-ch21", {5: 5, 7: 7, 11: 11, 14: 14}),
        # Challenge 22 "Depletion": A starts at 15, subtract the blue balls.
        ("tt-official-ch22", {1: 14, 4: 11, 10: 5, 15: 0}),
    ],
)
def test_register_board_reproduces_the_guides_table(stem, table):
    """The board must agree with every row the guide prints, not just one."""
    task = _load(stem)
    assert task["registers"]["A"], "register must be declared"
    assert {t["hoppers"]["blue"]: t["expect"]["registers"]["A"] for t in task["trials"]} == table

    board = Board.from_task_dict(task)
    ok, why = validate_targets(task, board, [])
    assert ok, why


@pytest.mark.parametrize("stem", ["tt-official-ch21", "tt-official-ch22"])
def test_register_board_actually_needs_its_placed_parts(stem):
    """A puzzle that scores without its solution would score an empty board.

    The gate is ``verify_task``, not ``validate_targets``: on these two boards
    the fixed components already do the counting, and the parts the solver has
    to place are what make the marble path physically legal. Stripping them
    leaves the register reading right and the board free-falling -- so it is
    the free-fall check, which the scorer runs first, that rejects it.
    """
    task = _load(stem)
    stripped = deepcopy(task)
    stripped["solution"]["placed_components"] = []
    assert not verify_task(stripped)


@pytest.mark.parametrize("stem", ["tt-official-ch21", "tt-official-ch22"])
def test_register_board_rejects_a_wrong_count(stem):
    """Guard the read itself: shifting one expected value must fail."""
    task = _load(stem)
    wrong = deepcopy(task)
    wrong["trials"][0]["expect"]["registers"]["A"] += 1
    board = Board.from_task_dict(wrong)
    ok, why = validate_targets(wrong, board, [])
    assert not ok
    assert "register A" in why


# ── the logic and reversal boards transcribed from the guide ────────────────


@pytest.mark.parametrize("stem", ["tt-official-ch18-pA", "tt-official-ch18"])
def test_and_board_routes_only_the_true_row_to_its_own_interceptor(stem):
    """Three rows take the "otherwise" branch, so T and F cannot be swapped."""
    task = _load(stem)
    assert len(task["trials"]) == 4

    targets = {t["name"]: tuple(t["expect"]["intercepted_at"]) for t in task["trials"]}
    true_row = targets["11"]
    others = {name: at for name, at in targets.items() if name != "11"}
    assert len(set(others.values())) == 1, "the three false rows must share one interceptor"
    assert true_row not in set(others.values()), "T and F must be different interceptors"

    board = Board.from_task_dict(task)
    ok, why = validate_targets(task, board, [])
    assert ok, why


@pytest.mark.parametrize(
    "stem,bits",
    [("tt-official-ch27-pA", 2), ("tt-official-ch27-pB", 3), ("tt-official-ch27", 9)],
)
def test_reversal_board_covers_every_starting_configuration(stem, bits):
    """"Regardless of how they start" means all 2^n of them, not a sample."""
    task = _load(stem)
    assert len(task["trials"]) == 2 ** bits

    for trial in task["trials"]:
        start = trial["initial_bit_states"]
        assert trial["expect"]["final_bit_states"] == {k: 1 - v for k, v in start.items()}
    assert len({t["name"] for t in task["trials"]}) == 2 ** bits

    board = Board.from_task_dict(task)
    ok, why = validate_targets(task, board, [])
    assert ok, why


@pytest.mark.parametrize(
    "stem", ["tt-official-ch18", "tt-official-ch27-pA", "tt-official-ch27-pB"]
)
def test_trial_board_still_needs_its_placed_parts(stem):
    task = _load(stem)
    stripped = deepcopy(task)
    stripped["solution"]["placed_components"] = []
    assert not verify_task(stripped)


def test_objective_is_captured_whole_not_cut_at_the_line_break():
    """The guide wraps its objectives; a truncated one is an unanswerable prompt."""
    for stem in ("tt-official-ch18", "tt-official-ch27", "tt-official-ch21"):
        objective = _load(stem)["objective"]
        assert not objective.rstrip().endswith("-"), f"{stem} cut mid-word"
        # ch21 ends on a parenthetical: "... blue balls. (Use 15 or less balls.)"
        assert objective.rstrip().endswith((".", ")")), f"{stem} cut mid-sentence"


# ── the overflow board ──────────────────────────────────────────────────────


def test_overflow_board_counts_and_latches_its_flag():
    """ch30 asserts BOTH the register value and the flag, at each worked count.

    Scored on one run it would pass on any board that happened to end at the
    same state, which is what the state-goal fallback would have produced.
    """
    task = _load("tt-official-ch30")
    rows = {t["hoppers"]["blue"]: t["expect"] for t in task["trials"]}
    assert set(rows) == {7, 8, 15}

    width = 2 ** len(task["registers"]["A"])
    for count, expect in rows.items():
        assert expect["registers"]["A"] == count % width
        raised = 1 if count > 7 else 0
        assert set(expect["final_bit_states"].values()) == {raised}, (
            "a meshed latch moves as one, so every flag bit takes the same value"
        )

    board = Board.from_task_dict(task)
    ok, why = validate_targets(task, board, [])
    assert ok, why


def test_overflow_board_rejects_a_flag_that_never_rises():
    """The flag is the point of the puzzle, so it has to be able to fail."""
    task = _load("tt-official-ch30")
    broken = deepcopy(task)
    for trial in broken["trials"]:
        trial["expect"]["final_bit_states"] = {
            k: 0 for k in trial["expect"]["final_bit_states"]
        }
    board = Board.from_task_dict(broken)
    assert not validate_targets(broken, board, [])[0]
