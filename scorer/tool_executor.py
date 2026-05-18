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

    def __init__(
        self,
        board: Board,
        available_parts: Optional[Dict[str, int]] = None,
        *,
        fixed_positions: Optional[set] = None,
    ):
        self.board = board
        self.placed_components: List[Dict[str, Any]] = []
        self.available_parts: Dict[str, int] = dict(available_parts) if available_parts else {}
        # Tracks how many of each type have been placed (for inventory enforcement).
        self._used_parts: Dict[str, int] = {}
        # Set of (x, y) tuples for components that were pre-placed (fixed) and
        # must not be removed by the LLM.
        self._fixed_positions: set = fixed_positions or set()
        # Best board state seen so far, saved whenever run_simulation
        # completes without free-fall errors.  Used as a fallback when the
        # LLM places a correct component, verifies it, but then removes it
        # before submitting a final_solution.
        self._best_placement: Optional[List[Dict[str, Any]]] = None
        # Flag set to True when run_simulation completes with no free-fall
        # errors and at least one marble reaches a catcher.  The agentic
        # loop can check this to terminate early instead of burning turns.
        self._solution_found: bool = False

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

            # Enforce available-parts inventory when declared.
            # If available_parts is empty (not provided), all types are allowed
            # (backward-compatible with tasks that don't declare inventory).
            if self.available_parts:
                allowed = self.available_parts.get(component_type, 0)
                used = self._used_parts.get(component_type, 0)
                if used >= allowed:
                    return {
                        "success": False,
                        "error": (
                            f"No {component_type} available: "
                            f"inventory allows {allowed}, already used {used}. "
                            f"Available types: {[k for k, v in self.available_parts.items() if v > 0]}"
                        ),
                    }
                self._used_parts[component_type] = used + 1

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
            # Board changed — the last solution_found state is now stale.
            self._solution_found = False

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

            # Reject removal of fixed (pre-placed) components — the LLM may
            # only remove components it placed itself.
            if (x, y) in self._fixed_positions:
                return {
                    "success": False,
                    "error": (
                        f"Cannot remove fixed component {existing.component_type.value} "
                        f"at ({x}, {y}) — only components you placed may be removed."
                    ),
                }

            # Remove from board
            self.board.remove(x, y)

            # Remove from tracking
            removed = [c for c in self.placed_components if c["x"] == x and c["y"] == y]
            self.placed_components = [
                c for c in self.placed_components if not (c["x"] == x and c["y"] == y)
            ]

            # Decrement inventory usage counter if applicable
            for c in removed:
                ct = c.get("component_type", "")
                if ct in self._used_parts and self._used_parts[ct] > 0:
                    self._used_parts[ct] -= 1

            # Board changed — the last solution_found state is now stale.
            self._solution_found = False

            return {"success": True, "message": f"Removed component from ({x}, {y})"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_simulation(
        self, input_sequence: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run simulation with given input sequence — non-destructive.

        Marble counts and pending trigger releases are saved before the run
        and restored afterwards so exploratory simulations do not permanently
        drain hoppers or leak queued trigger releases.

        Args:
            input_sequence: List of marble colors ("blue" or "red")

        Returns:
            Simulation results including marble destinations and final bit states
        """
        # Default to single blue marble
        if input_sequence is None:
            input_sequence = ["blue"]

        # Save hopper state so the run is non-destructive
        saved_blue = self.board.blue_balls_remaining
        saved_red = self.board.red_balls_remaining
        saved_triggers = list(self.board._pending_trigger_releases)

        try:
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

            # Detect illegal in-board free-fall through empty cells.
            # The simulator allows it, but valid Turing Tumble solutions
            # require a component at every in-board cell a marble visits.
            free_fall_errors: List[str] = []
            for marble_idx, result in enumerate(results, start=1):
                path = result.path or []
                for path_idx, curr in enumerate(path[1:], start=1):
                    prev = path[path_idx - 1]
                    x, y = curr

                    # Hopper-to-board entry may be empty; subsequent cells
                    # inside the board may not.
                    if prev[1] < 0 and y >= 0:
                        continue

                    # Last cell just above a catcher is a valid approach slot.
                    next_pos = path[path_idx + 1] if path_idx + 1 < len(path) else None
                    if (
                        y == self.board.rows - 1
                        and next_pos is not None
                        and next_pos[1] >= self.board.rows
                        and x in (
                            self.board.left_catcher_x,
                            self.board.right_catcher_x,
                        )
                    ):
                        continue

                    if (
                        0 <= x < self.board.cols
                        and 0 <= y < self.board.rows
                        and curr not in self.board.components
                    ):
                        free_fall_errors.append(
                            f"marble {marble_idx} traversed empty cell {curr}"
                        )

            # Snapshot best board when simulation succeeds without free-fall.
            # Used as fallback when the LLM finds a correct placement but
            # then removes it before submitting a final_solution.
            if (
                not free_fall_errors
                and (left_count > 0 or right_count > 0)
            ):
                self._best_placement = [
                    dict(p) for p in self.placed_components
                ]
                self._solution_found = True

            return {
                "success": True,
                "left_catcher": left_count,
                "right_catcher": right_count,
                "interceptor": interceptor_count,
                "final_bit_states": self.board.get_all_states(),
                "execution_traces": traces,
                "total_marbles": len(results),
                "free_fall_errors": free_fall_errors,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            # Restore hopper counts so exploratory runs don't drain them
            self.board.blue_balls_remaining = saved_blue
            self.board.red_balls_remaining = saved_red
            self.board._pending_trigger_releases = saved_triggers

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

    def get_best_placement(self) -> List[Dict[str, Any]]:
        """Get the best placement snapshot from any successful simulation.

        Falls back to current placed_components if no successful
        simulation was recorded.

        Returns:
            List of component dictionaries
        """
        if self._best_placement is not None:
            return [dict(p) for p in self._best_placement]
        return self.placed_components.copy()

    def is_solution_found(self) -> bool:
        """Check whether a valid solution has been found.

        Set to True when run_simulation completes without free-fall
        errors and at least one marble reaches a catcher.  Cleared
        whenever the board is modified (place_component/remove_component).

        The agentic loop can use this to terminate early instead of
        waiting for the LLM to submit a final_solution.
        """
        return self._solution_found


def create_executor_from_task(
    board_data: Dict[str, Any],
    fixed_components: Optional[List[Dict[str, Any]]] = None,
    available_parts: Optional[Dict[str, int]] = None,
) -> TuringTumbleToolExecutor:
    """Create a tool executor from task configuration.

    Args:
        board_data: Board configuration (dimensions, ball hoppers)
        fixed_components: Components that are already placed
        available_parts: Inventory of placeable component types (count per type).
            When provided, ``place_component`` rejects placements exceeding the
            declared inventory.

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

    # Place fixed components and track their positions
    fixed_positions = set()
    if fixed_components:
        for comp_dict in fixed_components:
            comp = Component.from_dict(comp_dict)
            board.place(comp.x, comp.y, comp)
            fixed_positions.add((comp.x, comp.y))

    return TuringTumbleToolExecutor(board, available_parts=available_parts, fixed_positions=fixed_positions)


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
