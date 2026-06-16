"""Turing Tumble component types and marble result."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from enum import Enum
from dataclasses import dataclass, field

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
