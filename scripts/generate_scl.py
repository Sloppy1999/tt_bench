#!/usr/bin/env python3
"""
Generate verified scl (scaled component-count) board files for ch02-ch10.

Mimics the ch01 scl pattern: for each 1comp/2comp challenge and each of its
variants, create truncated board variants at sizes 4, 6, 8, 12, 14, 16
(only for sizes ≤ original total component count).

Output: ``data/tasks/scaled/``
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from tt_bench.simulator import Board, verify_task

SCL_SIZES = [4, 6, 8, 12, 14, 16]
CHALLENGES_T1 = [2, 3, 4, 5]
CHALLENGES_T2 = [6, 7, 8, 9, 10]
CHALLENGES_ALL = [2, 3, 4, 5, 6, 7, 8, 9, 10]
CATEGORIES = ["1comp", "2comp"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHALLENGES_1COMP = PROJECT_ROOT / "data" / "tasks" / "challenges_1comp"
CHALLENGES_2COMP = PROJECT_ROOT / "data" / "tasks" / "challenges_2comp"
OUTPUT_DIR = PROJECT_ROOT / "data" / "tasks" / "scaled"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


ALL_PART_TYPES = [
    "ramp_right", "ramp_left", "crossover",
    "bit", "gear_bit", "gear", "interceptor", "trigger",
]


def _chute(start_x: int, rows: int, width: int, first_step: int) -> tuple[list[dict], int]:
    """Zig-zag ramps carrying a marble down `rows` rows, plus its exit column.

    Every row needs a component or the marble free-falls, and a ramp shifts the
    marble exactly one column, so the path has to alternate.
    """
    components: list[dict] = []
    x, step = start_x, first_step
    for y in range(rows):
        if not 0 <= x + step < width:
            step = -step
        components.append({
            "type": "ramp_right" if step > 0 else "ramp_left",
            "x": x,
            "y": y,
        })
        x += step
        step = -step
    return components, x


def _available_parts(placed: list[dict]) -> dict[str, int]:
    parts = {t: 0 for t in ALL_PART_TYPES}
    for comp in placed:
        if comp["type"] in parts:
            parts[comp["type"]] += 1
    return parts


def _simulate(task: dict) -> list:
    board = Board.from_task_dict(task)
    sequence = task.get("input_sequence") or ["blue"]
    if isinstance(sequence, str):
        sequence = [s.strip() for s in sequence.split(",") if s.strip()]
    return board.run(sequence)


def _ground_truth(results: list) -> tuple[list[str], dict]:
    """Derive the expected outcome from an actual run of the board.

    The simulator is the benchmark's oracle, so a generated board's ground
    truth has to be read off a real run rather than copied from the source
    challenge, whose behaviour a rescaled board does not reproduce.
    """
    colours = []
    for result in results:
        if result.caught_by == "left_catcher":
            colours.append("blue")
        elif result.caught_by == "right_catcher":
            colours.append("red")
        elif result.caught_by and "interceptor" in str(result.caught_by):
            colours.append("intercepted")
    counts = {
        "left_catcher": sum(1 for r in results if r.caught_by == "left_catcher"),
        "right_catcher": sum(1 for r in results if r.caught_by == "right_catcher"),
        "intercepted": sum(1 for r in results if r.caught_by == "interceptor"),
    }
    return colours, counts


def generate_scl(
    source_data: dict,
    scl_size: int,
    task_id: str,
    category: str,
) -> dict | None:
    """Build a challenge with exactly `scl_size` components.

    Returns None when no valid board of that size exists for this source.
    """
    board = source_data["board"]
    width = board.get("width", 11)
    hoppers = copy.deepcopy(board.get("ball_hoppers", {}))
    sequence = source_data.get("input_sequence", [])
    n_solution = 2 if category == "2comp" else 1
    if scl_size <= n_solution:
        return None

    colours_used = set(sequence)
    blue_x = hoppers.get("blue", {}).get("x", 2)
    red_x = hoppers.get("red", {}).get("x", 8)

    if colours_used <= {"blue"}:
        rows = scl_size
        components, blue_exit = _chute(blue_x, rows, width, 1)
        red_exit = red_x
    else:
        # Both hoppers are used, so both need a supported path down.
        if scl_size % 2:
            return None
        rows = scl_size // 2
        blue_components, blue_exit = _chute(blue_x, rows, width, 1)
        red_components, red_exit = _chute(red_x, rows, width, -1)
        components = blue_components + red_components

    if len({(c["x"], c["y"]) for c in components}) != len(components):
        return None

    height = rows + 1
    triggers = {
        "left": {"x": blue_exit, "y": height},
        "right": {"x": red_exit, "y": height},
    }

    # The deepest components make the puzzle: they sit at the end of the path.
    ordered = sorted(components, key=lambda c: (-c["y"], c["x"]))
    placed = [dict(c) for c in ordered[:n_solution]]
    placed_keys = {(c["x"], c["y"]) for c in placed}
    fixed = [dict(c) for c in components if (c["x"], c["y"]) not in placed_keys]

    result = {
        "task_id": task_id,
        "challenge_number": source_data["challenge_number"],
        "tier": source_data.get("tier", 1),
        "title": source_data.get("title", ""),
        "objective": source_data.get("objective", ""),
        "board": {
            "width": width,
            "height": height,
            "fixed_components": fixed,
            "ball_hoppers": hoppers,
            "trigger_levers": triggers,
        },
        "available_parts": _available_parts(placed),
        "solution": {
            "placed_components": placed,
            "explanation": (
                f"Scaled variant — {scl_size} total components, "
                f"{n_solution} to place."
            ),
            "verified": False,
            "position_verified": False,
            "final_marble_state": [],
        },
        "input_sequence": copy.deepcopy(sequence),
        "expected_output": {},
    }
    if board.get("hopper_entry_mode"):
        result["board"]["hopper_entry_mode"] = board["hopper_entry_mode"]

    try:
        colours, counts = _ground_truth(_simulate(result))
    except Exception:
        return None

    result["solution"]["final_marble_state"] = colours
    result["expected_output"] = counts

    try:
        if not verify_task(result):
            return None
    except Exception:
        return None

    # A puzzle whose board already works without the components the solver is
    # asked to place is not a puzzle.
    degenerate = copy.deepcopy(result)
    degenerate["solution"]["placed_components"] = []
    try:
        if verify_task(degenerate):
            return None
    except Exception:
        pass

    result["solution"]["verified"] = True
    result["solution"]["position_verified"] = True
    return result


def get_source_path(ch: int, cat: str, variant: int | None = None) -> Path:
    """Return path to source challenge JSON."""
    base_dir = CHALLENGES_1COMP if cat == "1comp" else CHALLENGES_2COMP

    if variant is None:
        return base_dir / f"tt-official-ch{ch:02d}-{cat}.json"
    return base_dir / "variants" / f"tt-official-ch{ch:02d}-{cat}_var_{variant}.json"


def get_output_path(ch: int, cat: str, scl_size: int, variant: int | None = None) -> Path:
    """Return output path for scl JSON."""
    base = f"tt-official-ch{ch:02d}-{cat}_scl{scl_size}"
    suffix = f"_var_{variant}" if variant is not None else ""
    return OUTPUT_DIR / f"{base}{suffix}.json"


def generate_all() -> dict[str, int]:
    """Generate all verified scl files for ch02-ch10."""
    removed = 0
    for ch in CHALLENGES_ALL:
        for path in OUTPUT_DIR.glob(f"tt-official-ch{ch:02d}-*comp_scl*.json"):
            path.unlink()
            removed += 1
    if removed:
        print(f"Removed {removed} stale ch02-ch10 scl files")

    counts: dict[str, int] = {}
    total = 0

    for ch in CHALLENGES_ALL:
        ch_counts = 0
        for cat in CATEGORIES:
            base_dir = CHALLENGES_1COMP if cat == "1comp" else CHALLENGES_2COMP

            # Determine available variants
            variants_dir = base_dir / "variants"
            if variants_dir.exists():
                variant_files = sorted(variants_dir.glob(
                    f"tt-official-ch{ch:02d}-{cat}_var_*.json"
                ))
                variant_nums = []
                for vf in variant_files:
                    try:
                        num = int(vf.stem.split("_var_")[-1])
                        variant_nums.append(num)
                    except ValueError:
                        continue
            else:
                variant_nums = []

            # Process base + all variants
            to_process: list[int | None] = [None] + sorted(variant_nums)

            for var in to_process:
                src_path = get_source_path(ch, cat, var)
                if not src_path.exists():
                    print(f"  ⚠ Missing source: {src_path}")
                    continue

                source = load_json(src_path)

                # Determine applicable scl sizes
                total_comps = len(source["board"].get("fixed_components", [])) + len(
                    source.get("solution", {}).get("placed_components", [])
                )
                applicable_sizes = [s for s in SCL_SIZES if s <= total_comps]

                for scl_size in applicable_sizes:
                    task_id_base = f"tt-official-ch{ch:02d}-{cat}_scl{scl_size}"
                    if var is not None:
                        task_id = f"{task_id_base}_var_{var}"
                    else:
                        task_id = task_id_base

                    result = generate_scl(source, scl_size, task_id, cat)
                    if result is None:
                        continue

                    out_path = get_output_path(ch, cat, scl_size, var)
                    save_json(result, out_path)
                    ch_counts += 1
                    total += 1

        counts[f"ch{ch:02d}"] = ch_counts
        print(f"  ch{ch:02d}: {ch_counts} files generated")

    counts["TOTAL"] = total
    return counts


def main() -> None:
    print(f"Source: {CHALLENGES_1COMP.parent.parent}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"SCL sizes: {SCL_SIZES}")
    print(f"Challenges: {CHALLENGES_ALL}")
    print()

    counts = generate_all()

    print(f"\n{'='*50}")
    print("SUMMARY:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
