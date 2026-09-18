#!/usr/bin/env python3
"""Rebuild official challenge boards so they realize their documented behaviour.

The encoded boards in ``data/tasks/official/challenges/json`` largely do not
work: marbles free-fall through gaps or end up somewhere other than the
documented ``final_marble_state``. Only ch01/ch02 verify.

These rebuilds are RECONSTRUCTIONS, not transcriptions of the physical puzzles:
each board is constructed to reproduce the target output the challenge already
documents (its ``final_marble_state``, or the pattern named in its objective),
and is stamped as such in ``provenance`` so reconstructed tasks stay
distinguishable from encoded ones.
"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path

from tt_bench.simulator import Board, verify_task

REPO = Path(__file__).resolve().parent.parent
CHALLENGE_DIR = REPO / "data" / "tasks" / "official" / "challenges" / "json"

WIDTH = 11
HEIGHT = 11
BLUE_X = 2
RED_X = 8
LEFT_X = 2
RIGHT_X = 8

ALL_PART_TYPES = [
    "ramp_right", "ramp_left", "crossover",
    "bit", "gear_bit", "gear", "interceptor", "trigger",
]

# "blue" reads out of the left catcher, "red" out of the right one.
CATCHER_X = {"blue": LEFT_X, "red": RIGHT_X}


# ── Target derivation ─────────────────────────────────────────────────────

def target_pattern(challenge: dict) -> list[str] | None:
    """The output the challenge says it should produce, if it says."""
    documented = challenge.get("solution", {}).get("final_marble_state")
    if documented:
        # Some encodings carry placeholder entries such as "..." rather than a
        # complete output, which is not a target that can be reproduced.
        if all(str(c) in ("blue", "red", "intercepted") for c in documented):
            return list(documented)
        return None

    objective = (challenge.get("objective") or "").lower()
    match = re.search(r"pattern[^a-z]*((?:\b(?:blue|red)\b[,\s]*)+)", objective)
    if match:
        colours = re.findall(r"blue|red", match.group(1))
        if colours:
            return colours
    return None


# ── Path construction ─────────────────────────────────────────────────────

def route(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int, int]]:
    """Ramp steps carrying a marble from (x0,y0) to (x1,y1).

    Returns (x, y, step) per occupied row. A ramp moves the marble exactly one
    column per row, so the trip is only possible when the horizontal distance
    and the number of rows share a parity.
    """
    rows = y1 - y0
    shift = x1 - x0
    if rows < 0 or abs(shift) > rows or (rows - shift) % 2:
        return []

    placed: list[tuple[int, int, int]] = []
    x = x0
    for i in range(rows):
        left = rows - i - 1  # steps still available after this one
        # Prefer heading toward the target, but only take a step that leaves
        # the destination reachable and stays on the board.
        options = (1, -1) if x1 >= x else (-1, 1)
        for step in options:
            nxt = x + step
            if 0 <= nxt < WIDTH and abs(x1 - nxt) <= left:
                placed.append((x, y0 + i, step))
                x = nxt
                break
        else:
            return []
    return placed if x == x1 else []


def ramps(steps: list[tuple[int, int, int]]) -> list[dict]:
    return [
        {"type": "ramp_right" if s > 0 else "ramp_left", "x": x, "y": y}
        for x, y, s in steps
    ]


# ── Board builders ────────────────────────────────────────────────────────

def build_uniform(colour: str, entry_x: int) -> list[dict] | None:
    """Every marble ends in the same catcher."""
    steps = route(entry_x, 0, CATCHER_X[colour], HEIGHT - 1)
    return ramps(steps) if steps else None


def build_alternating(pattern: list[str], entry_x: int) -> list[dict] | None:
    """A bit alternates its exit each marble, feeding one catcher then the other.

    A bit in state 0 leaves to the lower right and flips, so the first marble
    follows the right branch.
    """
    first, second = pattern[0], pattern[1]
    # state 0 exits right; pick the state that sends marble 1 the documented way.
    for state in (0, 1):
        right_colour = first if state == 0 else second
        left_colour = second if state == 0 else first
        for bit_y in range(0, HEIGHT - 2):
            head = route(entry_x, 0, entry_x, bit_y)
            if bit_y and not head:
                continue
            bit = {"type": "bit", "x": entry_x, "y": bit_y, "state": state}
            right_arm = route(entry_x + 1, bit_y + 1, CATCHER_X[right_colour], HEIGHT - 1)
            left_arm = route(entry_x - 1, bit_y + 1, CATCHER_X[left_colour], HEIGHT - 1)
            if not right_arm or not left_arm:
                continue
            comps = ramps(head) + [bit] + ramps(right_arm) + ramps(left_arm)
            if len({(c["x"], c["y"]) for c in comps}) == len(comps):
                return comps
    return None


def build_intercept(entry_x: int, drop_before: int) -> list[dict] | None:
    """Let `drop_before` marbles through, then catch one in an interceptor.

    Not expressible with a plain bit chain for arbitrary counts, so this only
    covers the "intercept every other marble" shape.
    """
    if drop_before != 1:
        return None
    for state in (0, 1):
        for bit_y in range(0, HEIGHT - 2):
            head = route(entry_x, 0, entry_x, bit_y)
            if bit_y and not head:
                continue
            bit = {"type": "bit", "x": entry_x, "y": bit_y, "state": state}
            through = route(entry_x + 1, bit_y + 1, CATCHER_X["blue"], HEIGHT - 1)
            if not through:
                continue
            comps = ramps(head) + [bit] + ramps(through)
            comps.append({"type": "interceptor", "x": entry_x - 1, "y": bit_y + 1})
            if len({(c["x"], c["y"]) for c in comps}) == len(comps):
                return comps
    return None


def build_for_target(pattern: list[str], entry_x: int) -> list[dict] | None:
    distinct = set(pattern)
    if "intercepted" in distinct:
        return None
    if len(distinct) == 1:
        return build_uniform(pattern[0], entry_x)
    if len(pattern) >= 2 and all(
        pattern[i] != pattern[i + 1] for i in range(len(pattern) - 1)
    ):
        return build_alternating(pattern, entry_x)
    return None


# ── Assembly ──────────────────────────────────────────────────────────────

def available_parts(placed: list[dict]) -> dict[str, int]:
    parts = {t: 0 for t in ALL_PART_TYPES}
    for comp in placed:
        if comp["type"] in parts:
            parts[comp["type"]] += 1
    return parts


def simulate(task: dict) -> list:
    board = Board.from_task_dict(task)
    sequence = task.get("input_sequence") or ["blue"]
    if isinstance(sequence, str):
        sequence = [s.strip() for s in sequence.split(",") if s.strip()]
    return board.run(sequence)


def observed(results: list) -> list[str]:
    out = []
    for r in results:
        if r.caught_by == "left_catcher":
            out.append("blue")
        elif r.caught_by == "right_catcher":
            out.append("red")
        elif r.caught_by and "interceptor" in str(r.caught_by):
            out.append("intercepted")
    return out


def assemble(original: dict, components: list[dict], pattern: list[str]) -> dict | None:
    """Wrap constructed components into a challenge and confirm the behaviour."""
    rebuilt = deepcopy(original)
    board = rebuilt["board"]
    board["width"] = WIDTH
    board["height"] = HEIGHT
    board["hopper_entry_mode"] = "column"
    board["trigger_levers"] = {
        "left": {"x": LEFT_X, "y": HEIGHT},
        "right": {"x": RIGHT_X, "y": HEIGHT},
    }

    ordered = sorted(components, key=lambda c: (-c["y"], c["x"]))
    n_solution = 1 if len(ordered) > 1 else 0
    placed = [dict(c) for c in ordered[:n_solution]]
    keys = {(c["x"], c["y"]) for c in placed}
    board["fixed_components"] = [
        dict(c) for c in components if (c["x"], c["y"]) not in keys
    ]

    rebuilt["available_parts"] = available_parts(placed)
    rebuilt["solution"]["placed_components"] = placed
    rebuilt["solution"]["final_marble_state"] = list(pattern)

    try:
        results = simulate(rebuilt)
    except Exception:
        return None
    if observed(results) != list(pattern):
        return None

    counts = {
        "left_catcher": sum(1 for r in results if r.caught_by == "left_catcher"),
        "right_catcher": sum(1 for r in results if r.caught_by == "right_catcher"),
        "intercepted": sum(1 for r in results if r.caught_by == "interceptor"),
    }
    rebuilt["expected_output"] = counts

    try:
        if not verify_task(rebuilt):
            return None
    except Exception:
        return None

    if placed:
        stripped = deepcopy(rebuilt)
        stripped["solution"]["placed_components"] = []
        try:
            if verify_task(stripped):
                return None
        except Exception:
            pass

    rebuilt["solution"]["verified"] = True
    rebuilt["solution"]["position_verified"] = True
    rebuilt["provenance"] = {
        "board": "reconstructed",
        "note": (
            "Board reconstructed to reproduce the challenge's documented output; "
            "not transcribed from the physical puzzle."
        ),
    }
    return rebuilt


def rebuild(original: dict) -> tuple[dict | None, str]:
    pattern = target_pattern(original)
    if not pattern:
        return None, "no documented target output"

    sequence = original.get("input_sequence") or []
    entry_x = BLUE_X if (not sequence or sequence[0] == "blue") else RED_X
    components = build_for_target(pattern, entry_x)
    if not components:
        return None, f"no constructor for pattern {'/'.join(pattern[:4])}..."

    rebuilt = assemble(original, components, pattern)
    if rebuilt is None:
        return None, "construction did not reproduce the target"
    return rebuilt, "rebuilt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write rebuilt boards")
    args = parser.parse_args()

    files = sorted(p for p in CHALLENGE_DIR.rglob("*.json") if "_questions" not in p.stem)
    already = rebuilt_n = 0
    failures: list[tuple[str, str]] = []

    for path in files:
        original = json.loads(path.read_text())
        try:
            if verify_task(original):
                already += 1
                continue
        except Exception:
            pass

        rebuilt, reason = rebuild(original)
        if rebuilt is None:
            failures.append((path.stem, reason))
            continue
        rebuilt_n += 1
        if args.write:
            path.write_text(json.dumps(rebuilt, indent=2) + "\n")

    print(f"{len(files)} official boards")
    print(f"  already verifying : {already}")
    print(f"  rebuilt           : {rebuilt_n}")
    print(f"  still broken      : {len(failures)}")
    reasons: dict[str, int] = {}
    for _, reason in failures:
        key = reason.split(" for pattern")[0]
        reasons[key] = reasons.get(key, 0) + 1
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"      {n:2d}  {reason}")
    if not args.write:
        print("\n(dry run — pass --write to apply)")


if __name__ == "__main__":
    main()
