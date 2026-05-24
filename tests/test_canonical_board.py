"""Tests for the canonical LLM-facing board representation.

Covers:
- Board.to_llm_dict shape and values on an official challenge.
- Board.to_dict/from_dict round-trip with custom catchers and pre-flipped bits.
- tool_executor.get_board_state parity with board.to_llm_dict.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "simulator"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scorer"))

from tt_sim import Board, Bit, GearBit, Gear, Ramp, Direction, Side  # noqa: E402

CH01 = os.path.join(REPO_ROOT, "tasks/official/challenges/json/tt-official-ch01.json")


class TestToLlmDictOnOfficialChallenge:
    def test_shape_and_core_fields(self):
        board = Board.from_task_json(CH01)
        d = board.to_llm_dict()

        assert d["dimensions"] == {"width": 11, "height": 11}
        assert d["hopper_entry_mode"] == "inward"
        # Official ch01 hoppers at x=2/8, catchers at x=2/8.
        assert d["ball_hoppers"]["blue"]["x"] == 2
        assert d["ball_hoppers"]["red"]["x"] == 8
        assert d["trigger_levers"]["left"]["x"] == 2
        assert d["trigger_levers"]["right"]["x"] == 8

    def test_entry_x_precomputed_for_inward_mode(self):
        board = Board.from_task_json(CH01)
        d = board.to_llm_dict()
        # Inward mode shifts blue one column right, red one column left.
        assert d["ball_hoppers"]["blue"]["entry_x"] == 3
        assert d["ball_hoppers"]["red"]["entry_x"] == 7

    def test_entry_x_collapses_to_hopper_in_column_mode(self):
        board = Board(rows=10, cols=10, blue_hopper_x=2, red_hopper_x=8,
                      hopper_entry_mode="column")
        d = board.to_llm_dict()
        assert d["ball_hoppers"]["blue"]["entry_x"] == 2
        assert d["ball_hoppers"]["red"]["entry_x"] == 8

    def test_components_and_bit_states_present(self):
        board = Board.from_task_json(CH01)
        d = board.to_llm_dict()
        # ch01 has 6 fixed + 4 solution ramps.
        assert len(d["components"]) == 10
        # No bits in ch01 so bit_states is empty.
        assert d["bit_states"] == {}

    def test_gear_groups_empty_when_no_gears(self):
        board = Board.from_task_json(CH01)
        d = board.to_llm_dict()
        assert d["gear_groups"] == []

    def test_ball_counts_reflect_initial_and_remaining(self):
        board = Board.from_task_json(CH01)
        d = board.to_llm_dict()
        assert d["ball_hoppers"]["blue"]["balls_initial"] == 8
        assert d["ball_hoppers"]["blue"]["balls_remaining"] == 8
        board.release_marble(Side.BLUE)
        d2 = board.to_llm_dict()
        assert d2["ball_hoppers"]["blue"]["balls_remaining"] == 7


class TestToLlmDictWithBitsAndGears:
    def _mk_board(self) -> Board:
        b = Board(rows=6, cols=6, blue_hopper_x=1, red_hopper_x=4,
                  hopper_entry_mode="column")
        b.place(2, 2, Bit(2, 2, state=1))
        b.place(3, 2, GearBit(3, 2, state=0))
        b.place(4, 2, GearBit(4, 2, state=0))
        # Force gear adjacency graph.
        from tt_sim import build_gear_connections
        build_gear_connections(b)
        return b

    def test_bit_states_keys_match_get_all_states(self):
        b = self._mk_board()
        d = b.to_llm_dict()
        assert d["bit_states"] == b.get_all_states()
        assert "bit_2_2" in d["bit_states"]
        assert "gear_bit_3_2" in d["bit_states"]

    def test_gear_groups_group_connected_gearbits(self):
        b = self._mk_board()
        d = b.to_llm_dict()
        groups = d["gear_groups"]
        assert len(groups) == 1
        positions = {tuple(p) for p in groups[0]}
        assert (3, 2) in positions
        assert (4, 2) in positions


class TestRoundTrip:
    def test_custom_catchers_survive_to_dict_from_dict(self):
        b = Board(rows=7, cols=7, blue_hopper_x=1, red_hopper_x=5,
                  left_catcher_x=3, right_catcher_x=4)
        restored = Board.from_dict(b.to_dict())
        assert restored.left_catcher_x == 3
        assert restored.right_catcher_x == 4

    def test_preflipped_bit_initial_state_preserved(self):
        b = Board(rows=6, cols=6)
        bit = Bit(2, 2, state=0)
        bit.state = 1  # simulate an external flip before serialization
        b.place(2, 2, bit)
        restored = Board.from_dict(b.to_dict())
        restored_bit = restored.get(2, 2)
        assert restored_bit.state == 1
        assert restored_bit._initial_state == 0
        # After reset, the bit must return to its original state 0.
        restored.reset()
        assert restored.get(2, 2).state == 0


class TestToolExecutorParity:
    def test_get_board_state_matches_to_llm_dict(self):
        from tool_executor import TuringTumbleToolExecutor

        board = Board.from_task_json(CH01)
        executor = TuringTumbleToolExecutor(board)
        state = executor.get_board_state()
        canonical = board.to_llm_dict()

        assert state["success"] is True
        for key in ("dimensions", "hopper_entry_mode", "ball_hoppers",
                    "trigger_levers", "components", "bit_states", "gear_groups"):
            actual = state[key]
            expected = canonical[key]
            # ToolExecutor augments components with a 'source' field
            # ("fixed"/"user"). Strip it for parity comparison against
            # the raw to_llm_dict which doesn't carry agentic metadata.
            if key == "components":
                actual = [{k: v for k, v in c.items() if k != "source"} for c in actual]
                expected = [{k: v for k, v in c.items() if k != "source"} for c in expected]
            assert actual == expected, f"{key} diverged"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
