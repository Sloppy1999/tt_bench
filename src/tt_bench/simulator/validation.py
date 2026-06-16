"""Challenge loading and solution verification."""

from __future__ import annotations

import json
from typing import Optional

from tt_bench.simulator.board import Board

# Challenge Loader
# =============================================================================


def load_challenge(challenge_path: str) -> tuple[Board, dict]:
    """
    Load a challenge from JSON file.

    Returns:
        tuple: (Board with solution placed, task dictionary)
    """
    with open(challenge_path) as fp:
        task = json.load(fp)

    board = Board.from_task_json(challenge_path)
    return board, task


def verify_solution(challenge_path: str, sequence: list[str] | None = None) -> bool:
    """
    Verify that a solution solves the challenge.

    Args:
        challenge_path: Path to challenge JSON file
        sequence: Optional marble release sequence (defaults to challenge's input_sequence)

    Returns:
        bool: True if solution is valid
    """
    board, task = load_challenge(challenge_path)

    # Use challenge's input_sequence if no explicit sequence provided
    if sequence is None:
        raw_seq = task.get("input_sequence", ["blue"])
        if isinstance(raw_seq, str):
            sequence = [s.strip() for s in raw_seq.split(",") if s.strip()]
        else:
            sequence = [str(s).strip() for s in raw_seq if str(s).strip()]

    # Run simulation
    results = board.run(sequence)

    # --- Free-fall check (no empty in-board cell traversal) ---
    for marble_idx, result in enumerate(results, start=1):
        path = result.path or []
        for path_idx, curr in enumerate(path[1:], start=1):
            prev = path[path_idx - 1]
            x, y = curr
            if prev[1] < 0 and y >= 0:
                continue
            next_pos = path[path_idx + 1] if path_idx + 1 < len(path) else None
            if (
                y == board.rows - 1
                and next_pos is not None
                and next_pos[1] >= board.rows
                and x in (board.left_catcher_x, board.right_catcher_x)
            ):
                continue
            if 0 <= x < board.cols and 0 <= y < board.rows and curr not in board.components:
                return False  # Free-fall through empty cell

    # --- Check final_marble_state (primary ground truth) ---
    final_marble_state = task.get("solution", {}).get("final_marble_state")
    if final_marble_state is not None:
        actual_colours = []
        for r in results:
            if r.caught_by == "left_catcher":
                actual_colours.append("blue")
            elif r.caught_by == "right_catcher":
                actual_colours.append("red")
            elif r.caught_by and "interceptor" in str(r.caught_by):
                actual_colours.append("intercepted")
        if actual_colours != final_marble_state:
            return False
        return True

    # --- Check expected_output counts (explicit numeric fields) ---
    expected_output = task.get("expected_output")
    if expected_output:
        has_numeric = any(
            k in expected_output
            for k in ("left_catcher", "right_catcher", "intercepted")
        )
        if has_numeric:
            return _verify_against_expected_output(board, expected_output, sequence)

    # --- Fallback to heuristic-based verification ---
    objective = task.get("objective", "").lower()
    return _verify_heuristic(board, objective, sequence)


def _verify_against_expected_output(
    board: Board, expected: dict, sequence: list[str] | None
) -> bool:
    """Verify solution against explicit expected_output declaration."""
    results = board.run(sequence)

    # Get actual output
    left_catcher = sum(1 for r in results if r.caught_by == "left_catcher")
    right_catcher = sum(1 for r in results if r.caught_by == "right_catcher")
    intercepted = sum(1 for r in results if r.caught_by == "interceptor")

    # Check each expected field
    if "left_catcher" in expected:
        if left_catcher != expected["left_catcher"]:
            return False
    if "right_catcher" in expected:
        if right_catcher != expected["right_catcher"]:
            return False
    if "intercepted" in expected:
        if intercepted != expected["intercepted"]:
            return False

    # Check bit states if specified
    if "final_bit_states" in expected:
        actual_states = board.get_all_states()
        for pos, expected_state in expected["final_bit_states"].items():
            # Parse position string like "bit_3_5" or "gear_bit_2_4"
            actual_state = actual_states.get(pos)
            if actual_state != expected_state:
                return False

    return True


def _verify_heuristic(board: Board, objective: str, sequence: list[str] | None) -> bool:
    """Fallback heuristic-based verification."""
    # Run simulation
    results = board.run(sequence)

    # Analyze results
    blue_reached_left = 0
    blue_reached_right = 0
    red_reached_left = 0
    red_reached_right = 0
    intercepted = 0

    for i, result in enumerate(results):
        if result.caught_by == "left_catcher":
            if i % 2 == 0:  # Blue
                blue_reached_left += 1
            else:
                red_reached_left += 1
        elif result.caught_by == "right_catcher":
            if i % 2 == 0:
                blue_reached_right += 1
            else:
                red_reached_right += 1
        elif result.caught_by == "interceptor":
            intercepted += 1

    # Simple verification based on objective
    if "blue" in objective and "left" in objective:
        # Blue should reach left catcher
        expected_blue = board.blue_hopper_count_initial
        return blue_reached_left == expected_blue and blue_reached_right == 0

    if "red" in objective and "right" in objective:
        expected_red = board.red_hopper_count_initial
        return red_reached_right == expected_red

    # If no clear objective, return False (can't verify)
    return False


# =============================================================================
