"""Challenge loading and solution verification."""

from __future__ import annotations

import json

from tt_bench.simulator.board import Board
from tt_bench.simulator.targets import (
    free_fall_cells,
    lost_marbles,
    normalize_sequence,
    validate_targets,
)

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

    board = Board.from_task_dict(task)
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
    return _verify_task_board(board, task, sequence)


def verify_task(task: dict, sequence: list[str] | None = None) -> bool:
    """Verify an already-loaded challenge dictionary.

    This is useful to generators, which should reject invalid candidates before
    writing them into a benchmark dataset.
    """
    board = Board.from_task_dict(task)
    return _verify_task_board(board, task, sequence)


def _verify_task_board(
    board: Board, task: dict, sequence: list[str] | None = None
) -> bool:
    """Verify one task using a single simulator run."""

    specified = {(p['x'], p['y']) for p in task.get('solution', {}).get('placed_components', [])
                 if 'state' in p}
    if board.editable_bit_states - specified:
        return False

    # Use challenge's input_sequence if no explicit sequence provided
    if sequence is None:
        sequence = normalize_sequence(task.get("input_sequence"))

    # Run simulation
    results = board.run(sequence)

    # Free fall through an empty cell is illegal, and a matching subset of
    # catcher outputs is not sufficient if another marble was lost elsewhere.
    if free_fall_cells(board, results) or lost_marbles(results):
        return False

    return validate_targets(task, board, results)[0]


# =============================================================================
