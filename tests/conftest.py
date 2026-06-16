"""Pytest configuration for tt_bench tests.

With ``pip install -e .`` the package is importable directly.
The ``pythonpath = ["src"]`` setting in pyproject.toml handles
the case where you run pytest without an editable install.
"""

import pytest
from tt_bench.simulator import Board, Component, ComponentType


@pytest.fixture
def empty_board():
    """An 11×11 board with no components placed."""
    return Board()


@pytest.fixture
def simple_board():
    """Board with two ramps: one right, one left."""
    from tt_bench.simulator import Ramp, Direction

    b = Board()
    b.place(2, 0, Ramp(2, 0, Direction.RIGHT))
    b.place(3, 1, Ramp(3, 1, Direction.LEFT))
    return b
