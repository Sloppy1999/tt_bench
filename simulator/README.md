# Turing Tumble Simulator

A fully functional Python simulation of the Turing Tumble board game.

## Installation

No external dependencies required. Uses only Python standard library.

```bash
cd simulator
```

## Quick Start

```python
from tt_sim import Board, Ramp, Bit, Direction, Side

# Create a board
board = Board()

# Place components
board.place(3, 3, Ramp(x=3, y=3, direction=Direction.RIGHT))
board.place(4, 4, Ramp(x=4, y=4, direction=Direction.LEFT))
board.place(5, 5, Bit(x=5, y=5, state=0))

# Release a marble
result = board.release_marble(Side.BLUE)

print(f"Path: {result.path}")
print(f"Caught by: {result.caught_by}")
print(f"Steps: {result.steps}")

# Render the board
print(board)
```

## Loading Challenges

Load and solve official Turing Tumble challenges:

```python
from tt_sim import load_challenge, verify_solution

# Load a challenge
board, task = load_challenge("../tasks/official/challenges/json/tt-official-ch01.json")

print(f"Objective: {task['objective']}")
print(board)

# Verify solution
is_valid = verify_solution("../tasks/official/challenges/json/tt-official-ch01.json")
print(f"Solution valid: {is_valid}")
```

## CLI Usage

```bash
# Load and render a challenge
python tt_sim.py --load ../tasks/official/challenges/json/tt-official-ch01.json

# Load, run sequence, and verify
python tt_sim.py --load ../tasks/official/challenges/json/tt-official-ch01.json --run blue,blue,blue --verify

# Real-time terminal simulation (step-by-step)
python tt_sim.py --load ../tasks/official/challenges/json/tt-official-ch01.json --run blue,red,blue --realtime --tick 0.15

# Interactive mode
python tt_sim.py
```

## MP4 Animation Export

You can now export a real simulation animation (MP4) using the board renderer:

```bash
# Export one task as MP4
python board_renderer.py --task tt-official-ch02 --animate --run blue,blue,blue --fps 12

# Let sequence default to task input_sequence (or alternating fallback)
python board_renderer.py --task tt-official-ch02 --animate
```

Animation flags:

- `--animate` enables MP4 export mode (requires `--task`)
- `--run "blue,red,..."` sets a custom release sequence (`b/r` also accepted)
- `--fps` sets video frame rate
- `--hold-frames` holds still frames at start/end for readability

If `--run` is omitted, animation uses the task's `input_sequence` first.
If no task sequence exists, it falls back to alternating blue/red by hopper counts.

### Hopper Entry Semantics

Official task boards treat hopper coordinates as slot positions above the board.
For those boards, marbles enter one column inward on row `y=0`:

- Blue hopper enters at `x + 1`
- Red hopper enters at `x - 1`

Generic/custom boards created directly with `Board(...)` keep legacy behavior:
marbles enter directly under hopper `x` (same-column entry).

### Real-time Mode

Use `--realtime` to animate marble motion frame-by-frame in the terminal:

- `--tick <seconds>` controls frame delay (default `0.2`)
- `--no-clear` keeps all frames instead of clearing the terminal
- `--steps <N>` still applies as max steps per marble

## Component Types

| Component | Symbol | Description |
|-----------|--------|-------------|
| Ramp Right | `>` | Sends marble down-right |
| Ramp Left | `<` | Sends marble down-left |
| Crossover | `X` | Lets marbles cross paths |
| Bit (state 0) | `<` | Binary state pointing left |
| Bit (state 1) | `>` | Binary state pointing right |
| Gear Bit (state 0) | `g` | Gear-connected bit (lowercase) |
| Gear Bit (state 1) | `G` | Gear-connected bit (uppercase) |
| Gear | `O` | Connects gear bits |
| Interceptor | `I` | Catches marble |
| Trigger | `T` | Releases next ball |

## Running Tests

```bash
# Run all tests
python -m pytest tests/test_tt_sim.py -v

# Run specific test
python -m pytest tests/test_tt_sim.py::TestBit -v
```

## Board Rendering

The `render()` method produces ASCII art:

```
=======================
Blue: oooooooo (8)
Red:  oooooooo (8)
=======================
      b           r    
-----------------------
 0 . . . . . . . . . . .
 1 . . . > . . . . . . .
 2 . . . . < . . . . . .
 3 . . . > . . . . . . .
 4 . . . . < . . . . . .
 5 . . . > . . . . . . .
 6 . . . . < . . . . . .
 7 . . . > . . . . . . .
 8 . . . . < . . . . . .
 9 . . . . . . . . . . .
10 . . . . . . . . . . .
-----------------------
   L                   R
```

### Symbol Legend

| Symbol | Meaning |
|--------|---------|
| `.` | Empty cell |
| `*` | Marble in flight |
| `o` | Ball in hopper |
| `b` | Blue hopper |
| `r` | Red hopper |
| `L` | Left catcher |
| `R` | Right catcher |

## File Structure

```
simulator/
├── tt_sim.py           # Main simulator module
├── tests/
│   └── test_tt_sim.py # Unit tests
└── README.md           # This file
```

## API Reference

### Board Class

- `Board(rows=11, cols=11, ...)` - Create a new board
- `place(x, y, component)` - Place a component
- `remove(x, y)` - Remove a component
- `get(x, y)` - Get component at position
- `connect_gears(positions)` - Connect gear bits
- `release_marble(side)` - Release a marble
- `release_marble(side, step_callback=None)` - Release a marble with optional per-step callback
- `run(sequence)` - Run a sequence of marbles
- `reset()` - Reset board to initial state
- `to_dict()` / `from_dict()` - Serialization
- `render()` - ASCII rendering

### Component Classes

- `Ramp(x, y, direction)` - Direction.RIGHT or Direction.LEFT
- `Crossover(x, y)` - Crossover component
- `Bit(x, y, state=0)` - Bit component (0 or 1)
- `GearBit(x, y, state=0)` - Gear bit component
- `Gear(x, y)` - Gear connector
- `Interceptor(x, y, side="left")` - Interceptor
- `Trigger(x, y, side="blue")` - Trigger

### MarbleResult

- `path` - List of (x, y) positions
- `caught_by` - "left_catcher", "right_catcher", "interceptor", or None
- `final_state` - Dict of bit states {(x,y): state}
- `steps` - Number of steps taken
- `terminated` - Whether simulation terminated
- `termination_reason` - Reason for termination

## Limitations

- Maximum 500 steps per marble (to prevent infinite loops)
- Standard board is 11x11 but can be customized
- Balls that fall off the side are considered terminated
