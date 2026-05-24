"""
Turing Tumble Simulator
=======================
A fully functional simulation of the Turing Tumble board game in Python.

This module provides:
- Board simulation with all component types
- Marble physics and path tracing
- Gear bit propagation via gear connections
- Text-based board rendering
- JSON serialization/deserialization
- CLI interface for interactive play
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Callable, Optional
from dataclasses import dataclass, field
from collections import defaultdict


# =============================================================================
# Enums and Constants
# =============================================================================


class Direction(Enum):
    """Direction a marble exits a component."""

    LEFT = "left"
    RIGHT = "right"


class Side(Enum):
    """Which side of the board (blue/left or red/right)."""

    BLUE = "blue"
    RED = "red"


class ComponentType(Enum):
    """All available component types in Turing Tumble."""

    RAMP_RIGHT = "ramp_right"
    RAMP_LEFT = "ramp_left"
    CROSSOVER = "crossover"
    BIT = "bit"
    GEAR_BIT = "gear_bit"
    GEAR = "gear"
    INTERCEPTOR = "interceptor"
    TRIGGER = "trigger"


# Symbol mapping for ASCII rendering
COMPONENT_SYMBOLS = {
    ComponentType.RAMP_RIGHT: ">",
    ComponentType.RAMP_LEFT: "<",
    ComponentType.CROSSOVER: "X",
    ComponentType.BIT: "B", 
    ComponentType.GEAR_BIT: "G", 
    ComponentType.GEAR: "O",
    ComponentType.INTERCEPTOR: "I",
    ComponentType.TRIGGER: "T",
}

BIT_SYMBOLS = {
    0: ">",  # state 0: points right (exits right, then flips to 1)
    1: "<",  # state 1: points left (exits left, then flips to 0)
}

GEAR_BIT_SYMBOLS = {
    0: "g",
    1: "G",
}


# =============================================================================
# Component Classes
# =============================================================================


@dataclass
class Component:
    """Base class for all Turing Tumble components."""

    component_type: ComponentType
    x: int
    y: int

    def get_symbol(self, state: int = 0) -> str:
        """Get the ASCII symbol for this component."""
        return COMPONENT_SYMBOLS.get(self.component_type, "?")

    def to_dict(self) -> dict:
        """Serialize component to dictionary."""
        return {
            "type": self.component_type.value,
            "x": self.x,
            "y": self.y,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Component:
        """Deserialize component from dictionary."""
        comp_type = ComponentType(data["type"])
        x, y = data["x"], data["y"]

        if comp_type == ComponentType.RAMP_RIGHT:
            return Ramp(x, y, Direction.RIGHT)
        elif comp_type == ComponentType.RAMP_LEFT:
            return Ramp(x, y, Direction.LEFT)
        elif comp_type == ComponentType.CROSSOVER:
            return Crossover(x, y)
        elif comp_type == ComponentType.BIT:
            bit = Bit(x, y, state=data.get("state", 0))
            if "initial_state" in data:
                bit._initial_state = data["initial_state"]
            return bit
        elif comp_type == ComponentType.GEAR_BIT:
            gbit = GearBit(x, y, state=data.get("state", 0))
            if "initial_state" in data:
                gbit._initial_state = data["initial_state"]
            return gbit
        elif comp_type == ComponentType.GEAR:
            return Gear(x, y)
        elif comp_type == ComponentType.INTERCEPTOR:
            return Interceptor(x, y, side=data.get("side", "left"))
        elif comp_type == ComponentType.TRIGGER:
            return Trigger(x, y, side=data.get("side", "blue"))
        else:
            raise ValueError(f"Unknown component type: {comp_type}")


@dataclass
class Ramp(Component):
    """Ramp component - sends marble diagonally left or right."""

    direction: Direction = Direction.RIGHT

    def __init__(self, x: int, y: int, direction: Direction | str = Direction.RIGHT):
        super().__init__(
            ComponentType.RAMP_RIGHT
            if direction == Direction.RIGHT
            else ComponentType.RAMP_LEFT,
            x,
            y,
        )
        if isinstance(direction, str):
            self.direction = Direction(direction)
        else:
            self.direction = direction

    def get_exit_direction(self, entry_side: str) -> Direction:
        """Ramp always exits in its configured direction."""
        return self.direction

    def get_symbol(self, state: int = 0) -> str:
        return ">" if self.direction == Direction.RIGHT else "<"

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["direction"] = self.direction.value
        return d


@dataclass
class Crossover(Component):
    """Crossover component - lets marbles cross without interfering."""

    def __init__(self, x: int, y: int):
        super().__init__(ComponentType.CROSSOVER, x, y)

    def get_exit_direction(self, entry_side: str) -> Direction:
        """
        - Entry from upper-left exits lower-right
        - Entry from upper-right exits lower-left
        """
        return Direction.RIGHT if entry_side == "left" else Direction.LEFT

    def get_symbol(self, state: int = 0) -> str:
        return "X"


@dataclass
class Bit(Component):
    """
    Bit component - stores binary state (0=left, 1=right).
    When hit, marble exits in current direction and bit flips.
    """

    state: int = 0  # 0 = left, 1 = right
    _initial_state: int = 0  # Stored initial state for reset()

    def __init__(
        self, x: int, y: int, state: int = 0, direction: Direction | None = None
    ):
        comp_type = ComponentType.BIT
        super().__init__(comp_type, x, y)
        self.state = state
        self._initial_state = state  # Remember initial state
        # If direction is provided, derive state from it
        if direction is not None:
            self.state = 1 if direction == Direction.RIGHT else 0
            self._initial_state = self.state

    def get_exit_direction(self, entry_side: str) -> Direction:
        """Exit in the direction the bit is currently pointing, then flip.
        
        Canonical Turing Tumble rules:
        - state 0: exits to lower-right, flips to 1
        - state 1: exits to lower-left, flips to 0
        """
        exit_dir = Direction.RIGHT if self.state == 0 else Direction.LEFT
        self.state = 1 - self.state  # Flip state
        return exit_dir

    def get_symbol(self, state: int = 0) -> str:
        # Use the instance state if not provided
        s = self.state if state == 0 else state
        return BIT_SYMBOLS.get(s, "?")

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["state"] = self.state
        if self._initial_state != self.state:
            d["initial_state"] = self._initial_state
        return d


@dataclass
class GearBit(Component):
    """
    Gear Bit - like Bit but can be connected via Gears.
    When triggered, all connected Gear Bits flip together.
    """

    state: int = 0  # 0 = left, 1 = right
    gear_group: int = -1  # Group ID for connected gear bits
    _initial_state: int = 0  # Stored initial state for reset()

    def __init__(self, x: int, y: int, state: int = 0, gear_group: int = -1):
        super().__init__(ComponentType.GEAR_BIT, x, y)
        self.state = state
        self.gear_group = gear_group
        self._initial_state = state  # Remember initial state

    def get_exit_direction(self, entry_side: str) -> Direction:
        """Exit in current direction, flip happens externally via gear propagation.
        
        Canonical Turing Tumble rules (same as Bit):
        - state 0: exits to lower-right, flips to 1
        - state 1: exits to lower-left, flips to 0
        """
        return Direction.RIGHT if self.state == 0 else Direction.LEFT

    def flip(self) -> None:
        """Flip the bit state (called during gear propagation)."""
        self.state = 1 - self.state

    def get_symbol(self, state: int = 0) -> str:
        s = self.state if state == 0 else state
        return GEAR_BIT_SYMBOLS.get(s, "?")

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["state"] = self.state
        d["gear_group"] = self.gear_group
        if self._initial_state != self.state:
            d["initial_state"] = self._initial_state
        return d


@dataclass
class Gear(Component):
    """Gear - connects adjacent Gear Bits so they flip together."""

    def __init__(self, x: int, y: int):
        super().__init__(ComponentType.GEAR, x, y)

    def get_exit_direction(self, entry_side: str) -> Direction:
        """Marbles pass straight through a gear (entry side = exit side)."""
        return Direction(entry_side)

    def get_symbol(self, state: int = 0) -> str:
        return "O"


@dataclass
class Interceptor(Component):
    """Interceptor - catches marble and removes it from play."""

    side: str = "left"  # left or right catcher

    def __init__(self, x: int, y: int, side: str = "left"):
        super().__init__(ComponentType.INTERCEPTOR, x, y)
        self.side = side

    def get_exit_direction(self, entry_side: str) -> Direction | None:
        """Interceptor always catches - return None to indicate termination."""
        return None

    def get_symbol(self, state: int = 0) -> str:
        return "I"

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["side"] = self.side
        return d


@dataclass
class Trigger(Component):
    """Trigger - when marble passes through, releases one ball from paired hopper."""

    side: str = "blue"  # blue or red

    def __init__(self, x: int, y: int, side: str = "blue"):
        super().__init__(ComponentType.TRIGGER, x, y)
        self.side = side

    def get_exit_direction(self, entry_side: str) -> Direction:
        """Trigger passes through and releases next ball.
        
        When triggered, releases one ball from the opposite-colored hopper.
        (Blue trigger releases red ball, red trigger releases blue ball)
        """
        # Trigger just passes through - marble continues in same direction
        # The actual ball release is handled by the Board in release_marble
        return Direction.LEFT if entry_side == "left" else Direction.RIGHT

    def get_paired_side(self) -> Side:
        """Returns the opposite side that this trigger releases."""
        return Side.RED if self.side == "blue" else Side.BLUE

    def get_symbol(self, state: int = 0) -> str:
        return "⌐"

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["side"] = self.side
        return d


# =============================================================================
# Simulation Result
# =============================================================================


@dataclass
class MarbleResult:
    """Result of releasing a single marble."""

    path: list[tuple[int, int]]  # List of (x, y) positions
    caught_by: (
        str | None
    )  # "left_catcher", "right_catcher", "interceptor", or None if still running
    final_state: dict[tuple[int, int], int]  # Final states of all bits/gear bits
    steps: int  # Number of steps taken
    terminated: bool  # Whether simulation terminated
    termination_reason: str | None  # Reason for termination

    @property
    def success(self) -> bool:
        """Check if marble was successfully caught."""
        return self.caught_by is not None


# =============================================================================
# Board Class
# =============================================================================


class Board:
    """
    Turing Tumble game board.

    Manages component placement, marble physics, and simulation.
    """

    def __init__(
        self,
        rows: int = 11,
        cols: int = 11,
        blue_hopper_x: int = 2,
        red_hopper_x: int = 8,
        blue_hopper_count: int = 8,
        red_hopper_count: int = 8,
        hopper_entry_mode: str = "column",
        left_catcher_x: Optional[int] = None,
        right_catcher_x: Optional[int] = None,
    ):
        self.rows = rows
        self.cols = cols

        if hopper_entry_mode not in ("column", "inward"):
            raise ValueError(
                "hopper_entry_mode must be 'column' or 'inward'"
            )

        # Ball hoppers
        self.blue_hopper_x = blue_hopper_x
        self.red_hopper_x = red_hopper_x
        self.blue_balls_remaining = blue_hopper_count
        self.red_balls_remaining = red_hopper_count
        self.blue_hopper_count_initial = blue_hopper_count
        self.red_hopper_count_initial = red_hopper_count
        self.hopper_entry_mode = hopper_entry_mode

        # Catchers at bottom (trigger levers). Default aligned with hopper columns
        # so left catcher releases blue and right catcher releases red — same-
        # color semantics per the physical Turing Tumble board.
        self.left_catcher_x = left_catcher_x if left_catcher_x is not None else blue_hopper_x
        self.right_catcher_x = right_catcher_x if right_catcher_x is not None else red_hopper_x

        # Components: dict of (x, y) -> Component
        self.components: dict[tuple[int, int], Component] = {}

        # Gear connections: adjacency list of (x, y) positions
        self.gear_connections: dict[tuple[int, int], set[tuple[int, int]]] = (
            defaultdict(set)
        )

        # Track gear bit groups
        self._gear_bit_groups: dict[int, set[tuple[int, int]]] = defaultdict(set)

        # Current ball in play
        self.current_marble_side: Side | None = None
        self.marble_count_released = 0

        # History
        self.marble_history: list[MarbleResult] = []

        # Pending trigger releases (marbles to release after current marble finishes)
        self._pending_trigger_releases: list[Side] = []

        # Maximum steps to prevent infinite loops
        self.max_steps = 500

    # -------------------------------------------------------------------------
    # Component Placement
    # -------------------------------------------------------------------------

    def place(self, x: int, y: int, component: Component) -> None:
        """Place a component at the given position."""
        if not (0 <= x < self.cols and 0 <= y < self.rows):
            raise ValueError(f"Position ({x}, {y}) out of bounds")
        if (x, y) in self.components:
            raise ValueError(f"Position ({x}, {y}) already occupied")

        self.components[(x, y)] = component

    def remove(self, x: int, y: int) -> Component | None:
        """Remove and return component at position."""
        return self.components.pop((x, y), None)

    def get(self, x: int, y: int) -> Component | None:
        """Get component at position."""
        return self.components.get((x, y))

    def connect_gears(self, positions: list[tuple[int, int]]) -> None:
        """Connect multiple gear bits with gears."""
        # Create gear components at intermediate positions
        for i in range(len(positions) - 1):
            x1, y1 = positions[i]
            x2, y2 = positions[i + 1]

            # Add bidirectional connection
            self.gear_connections[(x1, y1)].add((x2, y2))
            self.gear_connections[(x2, y2)].add((x1, y1))

            # If either is a gear bit, ensure we track the connection
            comp1 = self.get(x1, y1)
            comp2 = self.get(x2, y2)

            # Place gear components between gear bits if needed
            # Check if there's already a component at midpoint
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2

            if (mid_x, mid_y) not in self.components:
                gear = Gear(mid_x, mid_y)
                self.components[(mid_x, mid_y)] = gear

    def _propagate_gear_flip(self, triggered_pos: tuple[int, int]) -> None:
        """Propagate flip to all connected gear bits via BFS."""
        if triggered_pos not in self.gear_connections:
            return

        visited: set[tuple[int, int]] = {triggered_pos}
        queue = list(self.gear_connections[triggered_pos])

        while queue:
            pos = queue.pop(0)
            if pos in visited:
                continue
            visited.add(pos)

            comp = self.get(pos[0], pos[1])
            if isinstance(comp, GearBit):
                comp.flip()

            queue.extend(self.gear_connections.get(pos, []))

    # -------------------------------------------------------------------------
    # Marble Release and Simulation
    # -------------------------------------------------------------------------

    def release_marble(
        self,
        side: Side | str,
        step_callback: Optional[Callable[[Board, tuple[int, int], int], None]] = None,
    ) -> MarbleResult:
        """
        Release a marble from the specified side and simulate its path.

        Returns a MarbleResult with the path, final state, and termination info.

        Args:
            side: Which hopper to release from.
            step_callback: Optional callback invoked once per simulation step with
                (board, current_position, step_number). Useful for real-time rendering.
        """
        if isinstance(side, str):
            side = Side(side)

        # Check if balls available
        if side == Side.BLUE and self.blue_balls_remaining <= 0:
            return MarbleResult(
                path=[],
                caught_by=None,
                final_state=self._get_bit_states(),
                steps=0,
                terminated=True,
                termination_reason="no_blue_balls",
            )
        if side == Side.RED and self.red_balls_remaining <= 0:
            return MarbleResult(
                path=[],
                caught_by=None,
                final_state=self._get_bit_states(),
                steps=0,
                terminated=True,
                termination_reason="no_red_balls",
            )

        # Decrement ball count
        if side == Side.BLUE:
            self.blue_balls_remaining -= 1
        else:
            self.red_balls_remaining -= 1

        self.current_marble_side = side
        self.marble_count_released += 1

        # Start at hopper slot above the board.
        hopper_x = self.blue_hopper_x if side == Side.BLUE else self.red_hopper_x
        path = [(hopper_x, -1)]

        if self.hopper_entry_mode == "inward":
            # Official task JSON encodes hopper x as a slot above the board.
            # Marbles enter one column inward from that slot.
            entry_offset = 1 if side == Side.BLUE else -1
            entry_x = hopper_x + entry_offset
            if not (0 <= entry_x < self.cols):
                # Fallback for edge-case custom boards with boundary hoppers.
                entry_x = hopper_x
        else:
            # Legacy/custom board behavior: enter directly under hopper x.
            entry_x = hopper_x

        # Current position (about to enter board)
        curr_x = entry_x
        curr_y = 0  # Enter at top row

        # Track visited positions to detect loops
        visited_positions: set[tuple[int, int]] = set()

        # Marble's current horizontal direction; None means vertical fall.
        last_exit_dir: Direction | None = None

        # Simulate marble path
        steps = 0
        terminated = False
        termination_reason = None
        caught_by = None

        while not terminated and steps < self.max_steps:
            steps += 1

            # Add current position to path
            path.append((curr_x, curr_y))

            if step_callback is not None:
                step_callback(self, (curr_x, curr_y), steps)

            # Check for loop
            if (curr_x, curr_y) in visited_positions:
                terminated = True
                termination_reason = "infinite_loop"
                caught_by = None
                break

            visited_positions.add((curr_x, curr_y))

            # Check if marble is off the board
            if curr_x < 0:
                # Fell off the left side
                terminated = True
                termination_reason = "fell_off_side"
                caught_by = None
                break

            if curr_x >= self.cols:
                # Fell off the right side
                terminated = True
                termination_reason = "fell_off_side"
                caught_by = None
                break

            if curr_y >= self.rows:
                # Reached bottom - landing on a trigger lever releases the
                # same-coloured hopper: left lever → blue, right lever → red.
                if curr_x == self.left_catcher_x:
                    caught_by = "left_catcher"
                    if self.blue_balls_remaining > 0:
                        self._pending_trigger_releases.append(Side.BLUE)
                elif curr_x == self.right_catcher_x:
                    caught_by = "right_catcher"
                    if self.red_balls_remaining > 0:
                        self._pending_trigger_releases.append(Side.RED)
                else:
                    caught_by = None
                    termination_reason = "fell_off_bottom"
                terminated = True
                break

            # Check if there's a component at current position
            comp = self.get(curr_x, curr_y)

            if comp is None:
                # No component - marble falls straight down, loses horizontal direction
                curr_y += 1
                last_exit_dir = None
                continue

            # Determine entry side based on marble's horizontal direction.
            # entry_from == "left"  means marble arrived from upper-left (was moving right)
            # entry_from == "right" means marble arrived from upper-right (was moving left)
            if last_exit_dir == Direction.LEFT:
                entry_from = "right"
            else:
                entry_from = "left"
            exit_dir = comp.get_exit_direction(entry_from)

            # Handle special cases
            if exit_dir is None:
                # Interceptor caught the marble
                caught_by = "interceptor"
                terminated = True
                break

            # Handle Bit flip - happens when marble exits
            if isinstance(comp, Bit):
                pass  # Bit flips automatically in get_exit_direction

            # Handle GearBit flip - need to propagate to connected gear bits
            if isinstance(comp, GearBit):
                # Flip this gear bit (get_exit_direction returns current state without flipping)
                comp.flip()
                # Propagate to connected gear bits
                self._propagate_gear_flip((comp.x, comp.y))

            # Handle Trigger - release paired ball after current marble terminates
            if isinstance(comp, Trigger):
                # Schedule the paired ball for release (will be released after current marble finishes)
                paired_side = comp.get_paired_side()
                self._pending_trigger_releases.append(paired_side)

            # Record horizontal direction for next iteration's entry_from
            last_exit_dir = exit_dir

            # Move to exit position (diagonal)
            if exit_dir == Direction.LEFT:
                curr_x -= 1
            else:
                curr_x += 1

            # After exiting component, marble falls one step
            curr_y += 1

        if steps >= self.max_steps:
            terminated = True
            termination_reason = "max_steps_exceeded"

        result = MarbleResult(
            path=path,
            caught_by=caught_by,
            final_state=self._get_bit_states(),
            steps=steps,
            terminated=terminated,
            termination_reason=termination_reason,
        )

        self.marble_history.append(result)
        return result

    def _get_bit_states(self) -> dict[tuple[int, int], int]:
        """Get current states of all bits and gear bits."""
        states = {}
        for (x, y), comp in self.components.items():
            if isinstance(comp, (Bit, GearBit)):
                states[(x, y)] = comp.state
        return states

    def get_all_states(self) -> dict[str, int]:
        """Get all bit states as a dictionary with string keys for JSON serialization.

        Returns dict with keys like 'bit_3_5' or 'gear_bit_4_2' and their states.
        """
        states = {}
        for (x, y), comp in self.components.items():
            if isinstance(comp, Bit):
                states[f"bit_{x}_{y}"] = comp.state
            elif isinstance(comp, GearBit):
                states[f"gear_bit_{x}_{y}"] = comp.state
        return states

    def run(self, sequence: list[str] | None = None) -> list[MarbleResult]:
        """
        Run a sequence of marble releases.

        If sequence is None, releases alternating blue/red balls until one hopper is empty.
        """
        results = []

        if sequence is None:
            # Default: alternate blue/red
            i = 0
            while self.blue_balls_remaining > 0 or self.red_balls_remaining > 0:
                side = Side.BLUE if i % 2 == 0 else Side.RED
                result = self.release_marble(side)
                results.append(result)

                # Check if we should stop (ball caught by interceptor or fell off)
                if result.terminated and result.termination_reason in (
                    "infinite_loop",
                    "max_steps_exceeded",
                    "fell_off_side",
                ):
                    break

                i += 1
        else:
            for side_str in sequence:
                result = self.release_marble(side_str)
                results.append(result)

                # Process any pending trigger releases before next marble
                while self._pending_trigger_releases:
                    paired_side = self._pending_trigger_releases.pop(0)
                    if paired_side == Side.BLUE and self.blue_balls_remaining > 0:
                        paired_result = self.release_marble(Side.BLUE)
                        results.append(paired_result)
                    elif paired_side == Side.RED and self.red_balls_remaining > 0:
                        paired_result = self.release_marble(Side.RED)
                        results.append(paired_result)

                if result.terminated and result.termination_reason in (
                    "infinite_loop",
                    "max_steps_exceeded",
                    "fell_off_side",
                ):
                    break

        return results

    def reset(self) -> None:
        """Reset the board to initial state."""
        self.blue_balls_remaining = self.blue_hopper_count_initial
        self.red_balls_remaining = self.red_hopper_count_initial
        self.current_marble_side = None
        self.marble_count_released = 0
        self.marble_history = []
        self._pending_trigger_releases = []  # Clear pending trigger releases

        # Reset all bits to their stored initial state
        for comp in self.components.values():
            if isinstance(comp, (Bit, GearBit)):
                comp.state = comp._initial_state

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize board to dictionary."""
        components = []
        for (x, y), comp in self.components.items():
            comp_dict = comp.to_dict()
            components.append(comp_dict)

        return {
            "width": self.cols,
            "height": self.rows,
            "hopper_entry_mode": self.hopper_entry_mode,
            "blue_hopper": {
                "x": self.blue_hopper_x,
                "count": self.blue_hopper_count_initial,
            },
            "red_hopper": {
                "x": self.red_hopper_x,
                "count": self.red_hopper_count_initial,
            },
            "trigger_levers": {
                "left": {"x": self.left_catcher_x},
                "right": {"x": self.right_catcher_x},
            },
            "components": components,
        }

    def to_json(self, fp) -> None:
        """Serialize board to JSON file."""
        json.dump(self.to_dict(), fp, indent=2)

    def _entry_x_for(self, side: Side) -> int:
        """Resolve the column at which a marble actually enters the playfield.

        Mirrors the entry-column logic in ``release_marble`` so the value can be
        surfaced to the LLM without it having to reapply the inward-offset rule.
        """
        hopper_x = self.blue_hopper_x if side == Side.BLUE else self.red_hopper_x
        if self.hopper_entry_mode == "inward":
            offset = 1 if side == Side.BLUE else -1
            candidate = hopper_x + offset
            if 0 <= candidate < self.cols:
                return candidate
        return hopper_x

    def to_llm_dict(self) -> dict:
        """Canonical, simulator-aligned board snapshot.

        Used by both prompt rendering and the ``get_board_state`` tool so the
        model sees an identical shape at turn 0 and after every tool call.
        Includes everything the simulator actually consults at runtime:
        geometry, hopper entry semantics (with precomputed ``entry_x``),
        trigger-lever positions, components with their current states, a
        flat map of bit states, and connected gear groups.
        """
        components = [comp.to_dict() for _, comp in sorted(self.components.items())]

        gear_groups: list[list[list[int]]] = []
        seen: set[tuple[int, int]] = set()
        for pos, comp in self.components.items():
            if not isinstance(comp, (GearBit, Gear)):
                continue
            if pos in seen:
                continue
            group: list[tuple[int, int]] = []
            stack = [pos]
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                group.append(cur)
                stack.extend(self.gear_connections.get(cur, ()))
            if any(isinstance(self.get(x, y), GearBit) for (x, y) in group):
                gear_groups.append([[x, y] for (x, y) in sorted(group)])

        return {
            "dimensions": {"width": self.cols, "height": self.rows},
            "hopper_entry_mode": self.hopper_entry_mode,
            "ball_hoppers": {
                "blue": {
                    "x": self.blue_hopper_x,
                    "entry_x": self._entry_x_for(Side.BLUE),
                    "balls_remaining": self.blue_balls_remaining,
                    "balls_initial": self.blue_hopper_count_initial,
                },
                "red": {
                    "x": self.red_hopper_x,
                    "entry_x": self._entry_x_for(Side.RED),
                    "balls_remaining": self.red_balls_remaining,
                    "balls_initial": self.red_hopper_count_initial,
                },
            },
            "trigger_levers": {
                "left": {"x": self.left_catcher_x},
                "right": {"x": self.right_catcher_x},
            },
            "components": components,
            "bit_states": self.get_all_states(),
            "gear_groups": gear_groups,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Board:
        """Deserialize board from dictionary."""
        levers = data.get("trigger_levers", {})
        board = cls(
            rows=data.get("height", 11),
            cols=data.get("width", 11),
            blue_hopper_x=data.get("blue_hopper", {}).get("x", 2),
            red_hopper_x=data.get("red_hopper", {}).get("x", 8),
            blue_hopper_count=data.get("blue_hopper", {}).get("count", 8),
            red_hopper_count=data.get("red_hopper", {}).get("count", 8),
            hopper_entry_mode=data.get("hopper_entry_mode", "column"),
            left_catcher_x=levers.get("left", {}).get("x"),
            right_catcher_x=levers.get("right", {}).get("x"),
        )

        # Load components
        for comp_dict in data.get("components", []):
            comp = Component.from_dict(comp_dict)
            board.place(comp.x, comp.y, comp)

        # Auto-detect and build gear connections
        build_gear_connections(board)

        return board

    @classmethod
    def from_json(cls, fp) -> Board:
        """Deserialize board from JSON file."""
        data = json.load(fp)
        return cls.from_dict(data)

    @classmethod
    def from_task_json(cls, task_path: str) -> Board:
        """Load board from a challenge task JSON file."""
        with open(task_path) as fp:
            task = json.load(fp)

        board_data = task.get("board", {})
        available = task.get("available_parts", {})
        solution = task.get("solution", {})

        # Create board
        board = cls(
            rows=board_data.get("height", 11),
            cols=board_data.get("width", 11),
            blue_hopper_x=board_data.get("ball_hoppers", {})
            .get("blue", {})
            .get("x", 2),
            red_hopper_x=board_data.get("ball_hoppers", {}).get("red", {}).get("x", 8),
            blue_hopper_count=board_data.get("ball_hoppers", {})
            .get("blue", {})
            .get("count", 8),
            red_hopper_count=board_data.get("ball_hoppers", {})
            .get("red", {})
            .get("count", 8),
            hopper_entry_mode="inward",
            left_catcher_x=board_data.get("trigger_levers", {})
            .get("left", {})
            .get("x"),
            right_catcher_x=board_data.get("trigger_levers", {})
            .get("right", {})
            .get("x"),
        )

        # Place fixed components
        for comp_dict in board_data.get("fixed_components", []):
            comp = Component.from_dict(comp_dict)
            board.place(comp.x, comp.y, comp)

        # Place solution components if provided
        for comp_dict in solution.get("placed_components", []):
            comp = Component.from_dict(comp_dict)
            board.place(comp.x, comp.y, comp)

        # Auto-detect and build gear connections
        build_gear_connections(board)

        return board

    # -------------------------------------------------------------------------
    # Rendering
    # -------------------------------------------------------------------------

    def render(self, show_marble_path: tuple[int, int] | None = None) -> str:
        """
        Render the board as ASCII art.

        Args:
            show_marble_path: Optional tuple (x, y) to show marble position

        Returns:
            String representation of the board
        """
        lines = []

        # Top header
        lines.append("=" * (self.cols * 2 + 1))

        # Ball counters
        blue_balls = "o" * min(self.blue_balls_remaining, 8)
        red_balls = "o" * min(self.red_balls_remaining, 8)
        lines.append(f"Blue: {blue_balls or 'none'} ({self.blue_balls_remaining})")
        lines.append(f"Red:  {red_balls or 'none'} ({self.red_balls_remaining})")
        lines.append("=" * (self.cols * 2 + 1))

        # Hopper row (y = -1 conceptually)
        # Clamp hopper positions to board bounds
        blue_x = min(self.blue_hopper_x, self.cols - 1)
        red_x = min(self.red_hopper_x, self.cols - 1)
        hopper_row = [" "] * self.cols
        if 0 <= blue_x < self.cols:
            hopper_row[blue_x] = "b"
        if 0 <= red_x < self.cols:
            hopper_row[red_x] = "r"
        lines.append("  " + " ".join(hopper_row))
        lines.append("-" * (self.cols * 2 + 1))

        # Board rows
        for y in range(self.rows):
            row_chars = [" "] * self.cols

            for x in range(self.cols):
                comp = self.get(x, y)
                if comp:
                    row_chars[x] = comp.get_symbol()
                else:
                    row_chars[x] = "."

                # Show marble if present
                if (
                    show_marble_path
                    and show_marble_path[0] == x
                    and show_marble_path[1] == y
                ):
                    row_chars[x] = "*"

            # Add row label
            lines.append(f"{y:2d} " + " ".join(row_chars))

        # Bottom row - catchers
        lines.append("-" * (self.cols * 2 + 1))
        catcher_row = [" "] * self.cols
        left_catcher_x = min(self.left_catcher_x, self.cols - 1)
        right_catcher_x = min(self.right_catcher_x, self.cols - 1)
        if 0 <= left_catcher_x < self.cols:
            catcher_row[left_catcher_x] = "L"
        if 0 <= right_catcher_x < self.cols:
            catcher_row[right_catcher_x] = "R"
        lines.append("   " + " ".join(catcher_row))

        # Bit states
        bit_states = []
        for (x, y), state in sorted(self._get_bit_states().items()):
            bit_states.append(f"({x},{y}):{state}")

        if bit_states:
            lines.append("Bit states: " + ", ".join(bit_states))

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


# =============================================================================
# Gear Connection Builder
# =============================================================================


def build_gear_connections(board: Board) -> None:
    """
    Auto-detect and build gear connections based on spatial adjacency.

    This function scans the board for adjacent GearBits and Gear components,
    then establishes bidirectional connections in board.gear_connections.

    Rules for gear connectivity:
    1. GearBits at adjacent positions (horizontal, vertical, or diagonal) are connected
    2. Gear components connect to any adjacent GearBits
    3. Multiple connected GearBits form a gear group

    Args:
        board: The Board to analyze and update
    """
    # Find all gear-related components
    gear_bits: list[tuple[int, int]] = []
    gears: list[tuple[int, int]] = []

    for (x, y), comp in board.components.items():
        if isinstance(comp, GearBit):
            gear_bits.append((x, y))
        elif isinstance(comp, Gear):
            gears.append((x, y))

    # Build a set of all positions with gears (gear bits + gear components)
    all_gear_positions = set(gear_bits + gears)

    # For each pair of adjacent gear-related positions, create a connection
    for i, pos1 in enumerate(all_gear_positions):
        x1, y1 = pos1
        for pos2 in list(all_gear_positions)[i + 1:]:
            x2, y2 = pos2
            # Check adjacency (including diagonals)
            if abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1:
                # These are adjacent - create connection
                board.gear_connections[pos1].add(pos2)
                board.gear_connections[pos2].add(pos1)

    # Also create gear components at intermediate positions between connected gear bits if needed
    # This allows gear bits to be connected through intermediate gear components
    for pos1 in gear_bits:
        for pos2 in gear_bits:
            if pos1 >= pos2:
                continue
            x1, y1 = pos1
            x2, y2 = pos2
            # If gear bits are within 2 positions in any direction
            if abs(x1 - x2) <= 2 and abs(y1 - y2) <= 2:
                # Check if there's already a path through existing gears
                if pos2 in board.gear_connections.get(pos1, set()):
                    continue  # Already connected
                # Add intermediate position if on board
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                if 0 <= mid_x < board.cols and 0 <= mid_y < board.rows:
                    if (mid_x, mid_y) not in board.components:
                        # Place a gear at midpoint
                        gear = Gear(mid_x, mid_y)
                        board.components[(mid_x, mid_y)] = gear
                        gears.append((mid_x, mid_y))
                    # Connect all three
                    board.gear_connections[pos1].add((mid_x, mid_y))
                    board.gear_connections[(mid_x, mid_y)].add(pos1)
                    board.gear_connections[pos2].add((mid_x, mid_y))
                    board.gear_connections[(mid_x, mid_y)].add(pos2)


# =============================================================================
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
        sequence: Optional marble release sequence

    Returns:
        bool: True if solution is valid
    """
    board, task = load_challenge(challenge_path)

    # First try the preferred method: expected_output from task
    expected_output = task.get("expected_output")
    if expected_output:
        return _verify_against_expected_output(board, expected_output, sequence)

    # Fallback to heuristic-based verification
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
# CLI Interface
# =============================================================================


def run_cli():
    """Interactive CLI for the Turing Tumble simulator."""
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Turing Tumble Simulator")
    parser.add_argument(
        "--load",
        type=str,
        help="Load a challenge JSON file",
    )
    parser.add_argument(
        "--run",
        type=str,
        help="Run a sequence of marbles (e.g., 'blue,red,blue' or 'b,r,b')",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify solution after loading",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset board before running",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Maximum steps per marble (default: 500)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Render simulation in real time in the terminal",
    )
    parser.add_argument(
        "--tick",
        type=float,
        default=0.2,
        help="Seconds between real-time frames (default: 0.2)",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear terminal between real-time frames",
    )

    args = parser.parse_args()

    board = None

    if args.load:
        board, task = load_challenge(args.load)
        board.max_steps = args.steps
        print(f"Loaded: {task.get('task_id', 'unknown')}")
        print(f"Objective: {task.get('objective', 'none')}")
        print()
        print(board.render())

        if args.verify:
            print("\nVerifying solution...")
            is_valid = verify_solution(args.load)
            print(f"Solution valid: {is_valid}")

    if board:
        if args.reset:
            board.reset()
            print("\nBoard reset.")

        if args.run:
            # Parse sequence
            seq = [s.strip().lower() for s in args.run.split(",")]
            seq = [
                "blue" if s in ("b", "blue") else "red" if s in ("r", "red") else s
                for s in seq
            ]

            def infer_side(result: MarbleResult) -> str:
                if not result.path:
                    return "unknown"
                start_x = result.path[0][0]
                if start_x == board.blue_hopper_x:
                    return "blue"
                if start_x == board.red_hopper_x:
                    return "red"
                return "unknown"

            print(f"\nRunning sequence: {seq}")
            if args.realtime:
                print(f"Real-time mode enabled (tick={max(args.tick, 0.0):.3f}s)")

                remaining = list(seq)
                results = []

                marble_idx = 0
                while remaining:
                    side_str = remaining.pop(0)
                    if side_str not in ("blue", "red"):
                        print(f"Skipping unknown marble side: {side_str}")
                        continue

                    current_side = Side(side_str)

                    def show_step(
                        step_board: Board,
                        position: tuple[int, int],
                        step_number: int,
                    ) -> None:
                        if not args.no_clear:
                            print("\033[2J\033[H", end="")
                        marker = None
                        if 0 <= position[0] < step_board.cols and 0 <= position[1] < step_board.rows:
                            marker = position
                        print(
                            f"Real-time simulation | Marble {marble_idx + 1} "
                            f"({current_side.value}) | Step {step_number}"
                        )
                        print(step_board.render(show_marble_path=marker))
                        if args.tick > 0:
                            time.sleep(args.tick)

                    result = board.release_marble(current_side, step_callback=show_step)
                    results.append(result)

                    trigger_releases: list[str] = []
                    while board._pending_trigger_releases:
                        paired_side = board._pending_trigger_releases.pop(0)
                        if paired_side == Side.BLUE and board.blue_balls_remaining > 0:
                            trigger_releases.append("blue")
                        elif paired_side == Side.RED and board.red_balls_remaining > 0:
                            trigger_releases.append("red")
                    remaining = trigger_releases + remaining

                    if result.terminated and result.termination_reason in (
                        "infinite_loop",
                        "max_steps_exceeded",
                        "fell_off_side",
                    ):
                        break

                    marble_idx += 1
            else:
                results = board.run(seq)

            for i, result in enumerate(results):
                side_label = infer_side(result)
                print(f"\nMarble {i + 1} ({side_label}):")
                print(f"  Path length: {len(result.path)} steps")
                print(f"  Caught by: {result.caught_by}")
                print(
                    f"  Terminated: {result.terminated} ({result.termination_reason})"
                )

            print("\n" + board.render())

    if not args.load and not args.run:
        # Interactive mode
        print("Turing Tumble Simulator - Interactive Mode")
        print("Commands: load <file>, run <seq>, reset, render, quit")

        board = None
        while True:
            try:
                cmd = input("\n> ").strip().split(maxsplit=1)
                if not cmd:
                    continue

                if cmd[0] == "quit" or cmd[0] == "q":
                    break
                elif cmd[0] == "load" and len(cmd) > 1:
                    board, task = load_challenge(cmd[1])
                    print(f"Loaded: {task.get('task_id', 'unknown')}")
                    print(board.render())
                elif cmd[0] == "run" and len(cmd) > 1:
                    if board:
                        seq = [s.strip() for s in cmd[1].split(",")]
                        results = board.run(seq)
                        print(f"Released {len(results)} marbles")
                        print(board.render())
                    else:
                        print("No board loaded")
                elif cmd[0] == "reset" and board:
                    board.reset()
                    print("Board reset")
                elif cmd[0] == "render" and board:
                    print(board.render())
                elif cmd[0] == "help":
                    print("Commands: load <file>, run <seq>, reset, render, quit")
                else:
                    print("Unknown command. Type 'help' for available commands.")
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    run_cli()
