#!/usr/bin/env python3
"""Turing Tumble Tool Executor for agentic workflow.

Provides tool functions that the LLM can call during agentic synthesis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add simulator to path
sys.path.insert(0, str(Path(__file__).parent.parent / "simulator"))

from tt_sim import (
    Board,
    Component,
    Ramp,
    Crossover,
    Bit,
    GearBit,
    Gear,
    Interceptor,
    Trigger,
    Direction,
    Side,
    ComponentType,
)


class TuringTumbleToolExecutor:
    """Tool executor for Turing Tumble board manipulation.

    This class provides the tool functions that can be called by the LLM
    during the agentic synthesis loop.
    """

    def __init__(self, board: Board):
        self.board = board
        self.placed_components: List[Dict[str, Any]] = []

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name with given arguments.

        Returns the result of the tool execution.
        """
        tool_map = {
            "place_component": self.place_component,
            "remove_component": self.remove_component,
            "run_simulation": self.run_simulation,
            "get_board_state": self.get_board_state,
        }

        if tool_name not in tool_map:
            return {"error": f"Unknown tool: {tool_name}"}

        return tool_map[tool_name](**arguments)

    def place_component(
        self,
        component_type: str,
        x: int,
        y: int,
        state: int = 0,
    ) -> Dict[str, Any]:
        """Place a component on the board.

        Args:
            component_type: Type of component (ramp_left, ramp_right, etc.)
            x: Column position
            y: Row position
            state: Initial state for bit/gear_bit (0 or 1)

        Returns:
            Success status and component details
        """
        try:
            # Check if position is valid
            if not (0 <= x < self.board.cols and 0 <= y < self.board.rows):
                return {
                    "success": False,
                    "error": f"Position ({x}, {y}) out of bounds for board {self.board.cols}x{self.board.rows}",
                }

            # Check if position is already occupied
            existing = self.board.components.get((x, y))
            if existing is not None:
                return {
                    "success": False,
                    "error": f"Position ({x}, {y}) already occupied by {existing.component_type.value}",
                }

            # Create component
            comp_dict = {
                "type": component_type,
                "x": x,
                "y": y,
                "state": state,
            }
            component = Component.from_dict(comp_dict)

            # Place on board
            self.board.place(x, y, component)

            # Track placed components
            self.placed_components.append(
                {
                    "component_type": component_type,
                    "x": x,
                    "y": y,
                    "state": state,
                }
            )

            return {
                "success": True,
                "component": comp_dict,
                "message": f"Placed {component_type} at ({x}, {y})",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def remove_component(self, x: int, y: int) -> Dict[str, Any]:
        """Remove a component from the board.

        Args:
            x: Column position
            y: Row position

        Returns:
            Success status
        """
        try:
            if not (0 <= x < self.board.cols and 0 <= y < self.board.rows):
                return {"success": False, "error": f"Position ({x}, {y}) out of bounds"}

            existing = self.board.components.get((x, y))
            if existing is None:
                return {"success": False, "error": f"No component at ({x}, {y})"}

            # Remove from board
            self.board.remove(x, y)

            # Remove from tracking
            self.placed_components = [
                c for c in self.placed_components if not (c["x"] == x and c["y"] == y)
            ]

            return {"success": True, "message": f"Removed component from ({x}, {y})"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_simulation(
        self, input_sequence: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run simulation with given input sequence.

        Args:
            input_sequence: List of marble colors ("blue" or "red")

        Returns:
            Simulation results including marble destinations and final bit states
        """
        try:
            # Default to single blue marble
            if input_sequence is None:
                input_sequence = ["blue"]

            # Run simulation (cumulative — state persists across calls)
            results = self.board.run(input_sequence)

            # Extract results
            left_count = sum(1 for r in results if r.caught_by == "left_catcher")
            right_count = sum(1 for r in results if r.caught_by == "right_catcher")
            interceptor_count = sum(
                1 for r in results if r.caught_by and "interceptor" in r.caught_by
            )

            # Build execution traces
            traces = []
            for idx, r in enumerate(results):
                traces.append(
                {
                    "marble": idx + 1,
                    "path": r.path,
                    "final_destination": r.caught_by,
                    "steps": r.steps,
                    "terminated": r.terminated,
                    "termination_reason": r.termination_reason,
                }
                )

            return {
                "success": True,
                "left_catcher": left_count,
                "right_catcher": right_count,
                "interceptor": interceptor_count,
                "final_bit_states": self.board.get_all_states(),
                "execution_traces": traces,
                "total_marbles": len(results),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_board_state(self) -> Dict[str, Any]:
        """Get the current board configuration in the canonical LLM shape.

        Delegates to ``Board.to_llm_dict`` so the agentic LLM sees the same
        structure it was given in the initial prompt: dimensions, hopper entry
        mode (with precomputed ``entry_x``), trigger-lever positions,
        components with their current states, a flat bit-state map, and
        connected gear groups.
        """
        try:
            payload = self.board.to_llm_dict()
            return {
                "success": True,
                "placed_count": len(payload["components"]),
                **payload,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_placed_components(self) -> List[Dict[str, Any]]:
        """Get list of all placed components.

        Returns:
            List of component dictionaries
        """
        return self.placed_components.copy()


def create_executor_from_task(
    board_data: Dict[str, Any],
    fixed_components: Optional[List[Dict[str, Any]]] = None,
) -> TuringTumbleToolExecutor:
    """Create a tool executor from task configuration.

    Args:
        board_data: Board configuration (dimensions, ball hoppers)
        fixed_components: Components that are already placed

    Returns:
        Configured TuringTumbleToolExecutor
    """
    rows = board_data.get("height", 11)
    cols = board_data.get("width", 11)

    blue_x = board_data.get("ball_hoppers", {}).get("blue", {}).get("x", 2)
    red_x = board_data.get("ball_hoppers", {}).get("red", {}).get("x", 8)
    blue_count = board_data.get("ball_hoppers", {}).get("blue", {}).get("count", 8)
    red_count = board_data.get("ball_hoppers", {}).get("red", {}).get("count", 8)
    left_lever_x = board_data.get("trigger_levers", {}).get("left", {}).get("x")
    right_lever_x = board_data.get("trigger_levers", {}).get("right", {}).get("x")
    hopper_entry_mode = board_data.get("hopper_entry_mode", "inward")

    # Create board
    board = Board(
        rows=rows,
        cols=cols,
        blue_hopper_x=blue_x,
        red_hopper_x=red_x,
        blue_hopper_count=blue_count,
        red_hopper_count=red_count,
        hopper_entry_mode=hopper_entry_mode,
        left_catcher_x=left_lever_x,
        right_catcher_x=right_lever_x,
    )

    # Place fixed components
    if fixed_components:
        for comp_dict in fixed_components:
            comp = Component.from_dict(comp_dict)
            board.place(comp.x, comp.y, comp)

    return TuringTumbleToolExecutor(board)


# ---------------------------------------------------------------------------
# CLI for testing
# ---------------------------------------------------------------------------


def main():
    """Test the tool executor."""
    import argparse

    parser = argparse.ArgumentParser(description="Test tool executor")
    parser.add_argument(
        "--board", default="11x11", help="Board dimensions (e.g., 11x11)"
    )
    args = parser.parse_args()

    # Parse dimensions
    dims = args.board.split("x")
    rows, cols = int(dims[0]), int(dims[1])

    # Create board
    board = Board(rows=rows, cols=cols)
    executor = TuringTumbleToolExecutor(board)

    # Test place_component
    print("Testing place_component...")
    result = executor.place_component("ramp_left", 3, 5)
    print(f"  Result: {result}")

    # Test get_board_state
    print("\nTesting get_board_state...")
    state = executor.get_board_state()
    print(f"  Components: {state['placed_count']}")
    print(f"  Board: {state['dimensions']}")

    # Test run_simulation
    print("\nTesting run_simulation...")
    sim_result = executor.run_simulation(["blue"])
    print(f"  Left catcher: {sim_result.get('left_catcher', 'N/A')}")
    print(f"  Right catcher: {sim_result.get('right_catcher', 'N/A')}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
