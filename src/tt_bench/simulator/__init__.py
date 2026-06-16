"""Turing Tumble physics engine."""

from tt_bench.simulator.components import (
    Bit,
    Component,
    ComponentType,
    Crossover,
    Direction,
    Gear,
    GearBit,
    Interceptor,
    MarbleResult,
    Ramp,
    Side,
    Trigger,
)
from tt_bench.simulator.board import Board, build_gear_connections
from tt_bench.simulator.validation import load_challenge, verify_solution

__all__ = [
    "Bit",
    "Board",
    "build_gear_connections",
    "Component",
    "ComponentType",
    "Crossover",
    "Direction",
    "Gear",
    "GearBit",
    "Interceptor",
    "load_challenge",
    "MarbleResult",
    "Ramp",
    "Side",
    "Trigger",
    "verify_solution",
]
