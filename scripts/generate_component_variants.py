#!/usr/bin/env python3
"""Build the 1comp / 2comp challenge families from verified official boards.

Mirrors the tier-1 layout: each official challenge yields a reduced puzzle where
only one (or two) components are left for the solver to place, plus one position
variant per alternative component.

Unlike the original variant generator this writes nothing it has not simulated.
A variant is kept only when the board still runs cleanly and still needs the
component being asked for — a puzzle that solves itself cannot separate a good
solver from a bad one.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from tt_bench.simulator import Board, verify_task

REPO = Path(__file__).resolve().parent.parent
CHALLENGE_DIR = REPO / "data" / "tasks" / "official" / "challenges" / "json"
ALL_PART_TYPES = [
    "ramp_right", "ramp_left", "crossover",
    "bit", "gear_bit", "gear", "interceptor", "trigger",
]


def key(comp: dict) -> tuple:
    return (comp["type"], comp["x"], comp["y"])


def available_parts(placed: list[dict]) -> dict[str, int]:
    parts = {t: 0 for t in ALL_PART_TYPES}
    for comp in placed:
        if comp["type"] in parts:
            parts[comp["type"]] += 1
    return parts


def touched_cells(task: dict) -> set[tuple[int, int]]:
    """Cells the marbles actually roll over."""
    board = Board.from_task_dict(task)
    sequence = task.get("input_sequence") or ["blue"]
    cells: set[tuple[int, int]] = set()
    for result in board.run(sequence):
        for cell in (result.path or []):
            cells.add(cell)
    return cells


def build(original: dict, solution: list[dict], suffix: str) -> dict | None:
    """A challenge asking for exactly `solution`, or None if that is no puzzle."""
    everything = []
    seen = set()
    for comp in (original["board"]["fixed_components"]
                 + original["solution"]["placed_components"]):
        if key(comp) not in seen:
            seen.add(key(comp))
            everything.append(comp)

    wanted = {key(c) for c in solution}
    task = deepcopy(original)
    task["task_id"] = f"{original['task_id']}-{suffix}"
    task["board"]["fixed_components"] = [c for c in everything if key(c) not in wanted]
    task["solution"]["placed_components"] = [dict(c) for c in solution]
    task["available_parts"] = available_parts(solution)
    task["solution"]["explanation"] = (
        f"Place {len(solution)} component(s) to complete the board: "
        + ", ".join(f"{c['type']} at ({c['x']}, {c['y']})" for c in solution) + "."
    )
    task.pop("provenance", None)

    try:
        if not verify_task(task):
            return None
    except Exception:
        return None

    stripped = deepcopy(task)
    stripped["solution"]["placed_components"] = []
    try:
        if verify_task(stripped):
            return None          # the board works without the parts: not a puzzle
    except Exception:
        pass

    task["solution"]["verified"] = True
    task["solution"]["position_verified"] = True
    return task


def family(original: dict, size: int) -> tuple[dict | None, list[dict]]:
    """The base puzzle of the given size plus its position variants."""
    everything = []
    seen = set()
    for comp in (original["board"]["fixed_components"]
                 + original["solution"]["placed_components"]):
        if key(comp) not in seen:
            seen.add(key(comp))
            everything.append(comp)
    if len(everything) <= size:
        return None, []

    try:
        live = touched_cells(original)
    except Exception:
        return None, []
    # Only a component the marbles actually reach can make a real puzzle.
    ordered = sorted(
        (c for c in everything if (c["x"], c["y"]) in live),
        key=lambda c: (-c["y"], c["x"]),
    )
    if len(ordered) <= size:
        return None, []

    base = None
    for start in range(len(ordered) - size + 1):
        candidate = build(original, ordered[start:start + size], f"{size}comp")
        if candidate:
            base = candidate
            break
    if base is None:
        return None, []

    chosen = base["solution"]["placed_components"]
    variants = []
    for comp in ordered:
        if key(comp) in {key(c) for c in chosen}:
            continue
        swapped = [comp] + chosen[1:]
        variant = build(original, swapped, f"{size}comp_var_{len(variants) + 1}")
        if variant:
            variants.append(variant)
    return base, variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", type=int, default=2)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    totals = {"base": 0, "variants": 0, "skipped": 0}
    for path in sorted(CHALLENGE_DIR.glob("*.json")):
        original = json.loads(path.read_text())
        if original.get("tier") != args.tier:
            continue
        try:
            if not verify_task(original):
                totals["skipped"] += 1
                continue
        except Exception:
            totals["skipped"] += 1
            continue

        line = [f"  {path.stem:24s}"]
        for size in (1, 2):
            base, variants = family(original, size)
            out_dir = REPO / "data" / "tasks" / f"challenges_{size}comp"
            if base is None:
                line.append(f"{size}comp: -")
                continue
            line.append(f"{size}comp: 1+{len(variants)}")
            totals["base"] += 1
            totals["variants"] += len(variants)
            if args.write:
                out_dir.mkdir(parents=True, exist_ok=True)
                # Clear what a previous run produced for this challenge so an
                # older, longer variant list cannot leave orphans behind.
                stem = f"{original['task_id']}-{size}comp"
                for old in list(out_dir.glob(f"{stem}.json")) + \
                        list((out_dir / "variants").glob(f"{stem}_var_*.json")):
                    old.unlink()
                (out_dir / f"{base['task_id']}.json").write_text(
                    json.dumps(base, indent=2) + "\n")
                var_dir = out_dir / "variants"
                var_dir.mkdir(parents=True, exist_ok=True)
                for variant in variants:
                    (var_dir / f"{variant['task_id']}.json").write_text(
                        json.dumps(variant, indent=2) + "\n")
        print("  ".join(line))

    print(f"\ntier {args.tier}: {totals['base']} base puzzles, "
          f"{totals['variants']} position variants, "
          f"{totals['skipped']} challenges skipped (board does not verify)")
    if not args.write:
        print("(dry run — pass --write to apply)")


if __name__ == "__main__":
    main()
