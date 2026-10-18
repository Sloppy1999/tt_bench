"""One explicit-output contract shared by validation, scoring and agent tools."""
from __future__ import annotations

from collections import Counter

from tt_bench.simulator.components import GearBit

_OUTCOMES = {"left_catcher": "blue", "right_catcher": "red", "interceptor": "intercepted"}
_CATCHER_FIELDS = (
    ("left_catcher", "left_catcher"),
    ("right_catcher", "right_catcher"),
    ("intercepted", "interceptor"),
)


def validate_targets(task: dict, board, results: list) -> tuple[bool, str]:
    """Compare every declared target against one completed simulation.

    Physical path and inventory checks belong to callers. Descriptive expected_output
    metadata is not a target. Missing targets never fall back to objective heuristics.

    A task carrying ``trials`` is scored differently: one simulation cannot decide a
    goal quantified over several inputs or starting configurations ("count the blue
    balls", "reverse each bit regardless of the direction it starts in"), so the
    board is re-run once per trial and ``results`` is ignored.
    """
    if task.get("trials") is not None:
        return _validate_trials(task, board)

    expected_output = task.get("expected_output", {})
    if expected_output is None:
        expected_output = {}
    if not isinstance(expected_output, dict):
        return False, "Invalid expected_output: expected an object"

    spec = dict(expected_output)
    spec["final_marble_state"] = task.get("solution", {}).get("final_marble_state")
    spec["required_output"] = task.get("required_output")

    ok, why, compared = _compare(spec, board, results)
    if not ok:
        return False, why
    # A state-only declaration does not supply a marble-output target.
    if "marble" not in compared:
        return False, "No explicit marble output target available"
    return True, "Matched all declared output targets"


def _validate_trials(task: dict, board) -> tuple[bool, str]:
    """Run each declared trial on its own and require every one to match.

    The board is left exactly as it was found. Callers hand us a board that has
    already been run and then read state off it -- the agent's ``run_simulation``
    reports ``final_bit_states`` after asking whether it is done -- so trials
    that reset and re-ran it in place would report the last trial's board back
    as the agent's own result.
    """
    trials = task.get("trials")
    if not isinstance(trials, list) or not trials:
        return False, "Invalid trials: expected a nonempty list"

    registers = task.get("registers") or {}
    if not isinstance(registers, dict):
        return False, "Invalid registers: expected an object"

    snapshot = _snapshot(board)
    try:
        return _run_trials(trials, task, board, registers)
    finally:
        _restore(board, snapshot)


def _run_trials(trials: list, task: dict, board, registers: dict) -> tuple[bool, str]:
    default_sequence = task.get("input_sequence")
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            return False, f"trial {index}: expected an object"
        label = trial.get("name", index)

        expect = trial.get("expect")
        if not isinstance(expect, dict) or not expect:
            return False, f"trial {label}: declares no expectations"

        board.reset()
        ok, why = _apply_hoppers(board, trial.get("hoppers"))
        if not ok:
            return False, f"trial {label}: {why}"
        ok, why = _apply_initial_states(board, trial.get("initial_bit_states") or {})
        if not ok:
            return False, f"trial {label}: {why}"

        sequence = normalize_sequence(trial.get("input_sequence", default_sequence))
        results = board.run(sequence)

        # Each trial must be physically legal, including branches not taken
        # by the task's default state. An output match cannot excuse a gap.
        cells = free_fall_cells(board, results)
        if cells:
            return False, f"trial {label}: free fall at {cells[0][1]}"

        lost = lost_marbles(results)
        if lost:
            return False, f"trial {label}: {len(lost)} marble(s) reached no catcher"

        ok, why, compared = _compare(expect, board, results, registers)
        if not ok:
            return False, f"trial {label}: {why}"
        if not compared:
            return False, f"trial {label}: declares no recognised expectations"

    return True, f"Matched all {len(trials)} declared trials"


def _compare(
    spec: dict, board, results: list, registers: dict | None = None
) -> tuple[bool, str, set[str]]:
    """Check one finished run against a target spec.

    Returns the verdict, a reason, and which kinds of target were actually
    compared -- ``"marble"`` for a catcher/colour target, ``"state"`` for a bit
    or register one. Callers decide which kinds they require.
    """
    compared: set[str] = set()

    # The two sequence targets describe different things and must be compared
    # against different readings of the same run. final_marble_state names the
    # catcher each marble reached; required_output is the guide's printed strip,
    # which is the colour of each ball. A blue ball can leave on the right, so
    # comparing the printed strip against catchers rejects correct boards.
    caught = [r for r in results if r.caught_by in _OUTCOMES]
    by_catcher = [_OUTCOMES[r.caught_by] for r in caught]
    by_colour = [
        "intercepted" if r.caught_by == "interceptor" else (r.colour or _OUTCOMES[r.caught_by])
        for r in caught
    ]

    for field, expected, actual in (
        ("solution.final_marble_state", spec.get("final_marble_state"), by_catcher),
        ("required_output", spec.get("required_output"), by_colour),
    ):
        if expected is None:
            continue
        if not isinstance(expected, list) or not expected or any(
            not isinstance(value, str) or value not in _OUTCOMES.values() for value in expected
        ):
            return False, f"Invalid {field}: expected a nonempty blue/red/intercepted sequence", compared
        compared.add("marble")
        if actual != expected:
            return False, f"{field}: expected {expected}, got {actual}", compared

    counts = Counter(r.caught_by for r in results)
    for field, catcher in _CATCHER_FIELDS:
        if field not in spec:
            continue
        expected = spec[field]
        if type(expected) is not int or expected < 0:
            return False, f"Invalid expected_output.{field}: expected a nonnegative integer", compared
        compared.add("marble")
        if counts[catcher] != expected:
            return False, f"expected_output.{field}: expected {expected}, got {counts[catcher]}", compared

    if "final_bit_states" in spec:
        expected = spec["final_bit_states"]
        if not isinstance(expected, dict):
            return False, "Invalid expected_output.final_bit_states: expected an object", compared
        states = board.get_all_states()
        for key, value in expected.items():
            if type(value) is not int or value not in (0, 1) or key not in states:
                return False, f"Invalid final bit target {key}={value}", compared
            compared.add("state")
            if states[key] != value:
                return False, (
                    f"expected_output.final_bit_states.{key}: "
                    f"expected {value}, got {states[key]}"
                ), compared

    if "intercepted_at" in spec:
        expected = spec["intercepted_at"]
        if (not isinstance(expected, (list, tuple)) or len(expected) != 2
                or any(type(v) is not int for v in expected)):
            return False, "Invalid expect.intercepted_at: expected [x, y]", compared
        # "Put a ball in interceptor T" names one of several interceptors, and
        # caught_by says only that some interceptor caught it. A marble stops on
        # the cell that caught it, so the end of its path is which one.
        where = [list(r.path[-1]) for r in results
                 if r.caught_by == "interceptor" and r.path]
        compared.add("marble")
        if not where:
            return False, "expect.intercepted_at: no marble was intercepted", compared
        if any(cell != list(expected) for cell in where):
            return False, (
                f"expect.intercepted_at: expected {list(expected)}, got {where}"
            ), compared

    if "intercepted_colours" in spec:
        expected = spec['intercepted_colours']
        if (not isinstance(expected, list) or not expected
                or any(c not in ('blue', 'red') for c in expected)):
            return False, 'Invalid expect.intercepted_colours', compared
        actual = [r.colour for r in results if r.caught_by == 'interceptor']
        compared.add('marble')
        if actual != expected:
            return False, f'intercepted_colours: expected {expected}, got {actual}', compared

    if "registers" in spec:
        expected = spec["registers"]
        if not isinstance(expected, dict):
            return False, "Invalid expect.registers: expected an object", compared
        states = board.get_all_states()
        for name, value in expected.items():
            bits = (registers or {}).get(name)
            if not isinstance(bits, list) or not bits:
                return False, f"Register {name!r} is not declared in registers", compared
            if type(value) is not int or value < 0:
                return False, f"Invalid register target {name}={value}", compared
            absent = [b for b in bits if b not in states]
            if absent:
                return False, f"Register {name!r} names absent bits: {absent}", compared
            actual = 0
            for bit in bits:
                actual = (actual << 1) | states[bit]
            compared.add("state")
            if actual != value:
                return False, f"register {name}: expected {value}, got {actual}", compared

    return True, "Matched all declared output targets", compared


def _snapshot(board) -> dict:
    """Everything a trial run disturbs: bit states and hopper bookkeeping."""
    return {
        "bits": {
            pos: comp.state
            for pos, comp in board.components.items()
            if hasattr(comp, "state")
        },
        "blue": board.blue_balls_remaining,
        "red": board.red_balls_remaining,
        "side": board.current_marble_side,
        "released": board.marble_count_released,
        "history": list(board.marble_history),
        "pending": list(board._pending_trigger_releases),
    }


def _restore(board, snapshot: dict) -> None:
    for pos, state in snapshot["bits"].items():
        comp = board.components.get(pos)
        if comp is not None and hasattr(comp, "state"):
            comp.state = state
    board.blue_balls_remaining = snapshot["blue"]
    board.red_balls_remaining = snapshot["red"]
    board.current_marble_side = snapshot["side"]
    board.marble_count_released = snapshot["released"]
    board.marble_history = snapshot["history"]
    board._pending_trigger_releases = snapshot["pending"]


def _apply_hoppers(board, hoppers) -> tuple[bool, str]:
    """Load the hoppers for this trial.

    A counter board feeds itself: each marble trips the lever that releases the
    next, so the run length is set by how many balls the hopper holds, not by
    the input sequence. The guide's "x5 => A = 5" examples vary exactly this.
    Only the remaining count is touched, so the reset between trials -- which
    restores from the board's stored initial count -- still undoes it.
    """
    if hoppers is None:
        return True, ""
    if not isinstance(hoppers, dict):
        return False, "Invalid hoppers: expected an object"
    for colour, attribute in (("blue", "blue_balls_remaining"),
                              ("red", "red_balls_remaining")):
        if colour not in hoppers:
            continue
        count = hoppers[colour]
        if type(count) is not int or count < 0:
            return False, f"Invalid hoppers.{colour}: expected a nonnegative integer"
        setattr(board, attribute, count)
    return True, ""


def _apply_initial_states(board, initial: dict) -> tuple[bool, str]:
    """Point named bits a given way before the trial runs."""
    if not isinstance(initial, dict):
        return False, "Invalid initial_bit_states: expected an object"

    for key, value in initial.items():
        position = _parse_bit_key(key)
        if position is None:
            return False, f"Invalid bit key {key!r}"
        if type(value) is not int or value not in (0, 1):
            return False, f"Invalid initial state {key}={value}"
        component = board.get(*position)
        if component is None or not hasattr(component, "state"):
            return False, f"No bit at {key}"
        # Gear bits turn as one group, so an initial state applies to every gear
        # bit meshed with the named one -- setting just the one named would
        # describe a configuration the physical board cannot hold.
        for linked in _gear_group(board, position):
            other = board.get(*linked)
            if other is not None and hasattr(other, "state"):
                other.state = value
        component.state = value

    return True, ""


def _gear_group(board, position: tuple[int, int]) -> set[tuple[int, int]]:
    """Every position mechanically linked to this one, itself included."""
    connections = getattr(board, "gear_connections", None)
    if not connections or position not in connections:
        return {position}
    seen = {position}
    queue = [position]
    while queue:
        current = queue.pop()
        for neighbour in connections.get(current, ()):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return seen


def gear_state_conflict(board) -> tuple[int, int] | None:
    """A gear bit whose meshed gear bits start in another state, or None.

    Meshed gear bits turn as one, so a physical board cannot hold them apart.
    """
    for position, component in board.components.items():
        if not isinstance(component, GearBit):
            continue
        for linked in _gear_group(board, position):
            other = board.get(*linked)
            if isinstance(other, GearBit) and other.state != component.state:
                return position
    return None


def _parse_bit_key(key) -> tuple[int, int] | None:
    """``bit_3_5`` or ``gear_bit_4_2`` -> ``(x, y)``, matching get_all_states."""
    parts = str(key).rsplit("_", 2)
    if len(parts) != 3:
        return None
    name, x, y = parts
    if name not in ("bit", "gear_bit"):
        return None
    try:
        return int(x), int(y)
    except ValueError:
        return None


def normalize_sequence(raw) -> list[str]:
    """Normalise an input sequence; a missing one releases a single blue ball."""
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(",") if s.strip()]
    if isinstance(raw, list):
        return [str(s).strip() for s in raw if str(s).strip()]
    return ["blue"]


def free_fall_cells(board, results: list) -> list[tuple[int, tuple[int, int]]]:
    """Every (marble index, cell) pair that traverses an empty in-board cell.

    The simulator can carry a marble through empty cells, but a Turing Tumble
    solution is only legal when the marble lands on a component at every
    in-board step after it enters from the hopper.
    """
    offending: list[tuple[int, tuple[int, int]]] = []
    for marble_idx, result in enumerate(results, start=1):
        path = result.path or []
        for path_idx, curr in enumerate(path[1:], start=1):
            prev = path[path_idx - 1]
            x, y = curr

            # The hopper-to-board transition may enter an empty coordinate;
            # subsequent in-board motion may not.
            if prev[1] < 0 and y >= 0:
                continue

            next_pos = path[path_idx + 1] if path_idx + 1 < len(path) else None
            if (
                y == board.rows - 1
                and next_pos is not None
                and next_pos[1] >= board.rows
                and board.catcher_at(x) is not None
            ):
                # The final coordinate just above a trigger lever is a
                # catcher approach slot in several official encodings, not
                # an illegal mid-board gap.
                continue

            if 0 <= x < board.cols and 0 <= y < board.rows and curr not in board.components:
                offending.append((marble_idx, tuple(curr)))
    return offending


def lost_marbles(results: list) -> list:
    """Marbles that stopped without reaching a catcher or interceptor."""
    return [
        r
        for r in results
        if r.caught_by is None
        and r.termination_reason not in ("no_blue_balls", "no_red_balls")
    ]
