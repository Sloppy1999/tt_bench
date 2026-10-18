import json
from pathlib import Path

from tt_bench.simulator import Board
from tt_bench.simulator.renderer import _build_board_from_task


def test_renderer_builds_boards_with_editable_fixed_bits():
    """ch08 sets the state of a fixed bit from its solution; the renderer must not
    treat that as a second part on the same cell, and must enter like the scorer."""
    task = json.loads(
        (Path(__file__).resolve().parent.parent
         / "data/tasks/official/challenges/json/tt-official-ch08.json").read_text()
    )
    board, origins = _build_board_from_task(task)
    scored = Board.from_task_dict(task)
    assert board.hopper_entry_mode == scored.hopper_entry_mode
    assert board.get_all_states() == scored.get_all_states()
    assert origins[(5, 2)] == "fixed"
