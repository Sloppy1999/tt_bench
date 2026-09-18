"""One explicit-output contract shared by validation, scoring and agent tools."""
from __future__ import annotations

from collections import Counter


def validate_targets(task: dict, board, results: list) -> tuple[bool, str]:
    """Compare every declared target against one completed simulation.

    Physical path and inventory checks belong to callers. Descriptive expected_output
    metadata is not a target. Missing targets never fall back to objective heuristics.
    """
    outcomes = {"left_catcher": "blue", "right_catcher": "red", "interceptor": "intercepted"}
    caught = [r for r in results if r.caught_by in outcomes]
    # The two sequence targets describe different things and must be compared
    # against different readings of the same run. final_marble_state names the
    # catcher each marble reached; required_output is the guide's printed strip,
    # which is the colour of each ball. A blue ball can leave on the right, so
    # comparing the printed strip against catchers rejects correct boards.
    by_catcher = [outcomes[r.caught_by] for r in caught]
    by_colour = [
        "intercepted" if r.caught_by == "interceptor" else (r.colour or outcomes[r.caught_by])
        for r in caught
    ]
    compared = False
    for field, expected, actual in (
        ("solution.final_marble_state",
         task.get("solution", {}).get("final_marble_state"), by_catcher),
        ("required_output", task.get("required_output"), by_colour),
    ):
        if expected is None:
            continue
        if not isinstance(expected, list) or not expected or any(
            not isinstance(value, str) or value not in outcomes.values() for value in expected
        ):
            return False, f"Invalid {field}: expected a nonempty blue/red/intercepted sequence"
        compared = True
        if actual != expected:
            return False, f"{field}: expected {expected}, got {actual}"

    expected_output = task.get("expected_output", {})
    if expected_output is None:
        expected_output = {}
    if not isinstance(expected_output, dict):
        return False, "Invalid expected_output: expected an object"
    counts = Counter(r.caught_by for r in results)
    for field, catcher in (("left_catcher", "left_catcher"),
                           ("right_catcher", "right_catcher"),
                           ("intercepted", "interceptor")):
        if field not in expected_output:
            continue
        expected = expected_output[field]
        if type(expected) is not int or expected < 0:
            return False, f"Invalid expected_output.{field}: expected a nonnegative integer"
        compared = True
        if counts[catcher] != expected:
            return False, f"expected_output.{field}: expected {expected}, got {counts[catcher]}"

    if "final_bit_states" in expected_output:
        expected = expected_output["final_bit_states"]
        if not isinstance(expected, dict):
            return False, "Invalid expected_output.final_bit_states: expected an object"
        states = board.get_all_states()
        for key, value in expected.items():
            if type(value) is not int or value not in (0, 1) or key not in states:
                return False, f"Invalid final bit target {key}={value}"
            if states[key] != value:
                return False, f"expected_output.final_bit_states.{key}: expected {value}, got {states[key]}"
        # State-only declarations do not supply a marble-output target.

    if not compared:
        return False, "No explicit marble output target available"
    return True, "Matched all declared output targets"
