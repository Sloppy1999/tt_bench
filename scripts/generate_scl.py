#!/usr/bin/env python3
"""
Generate scl (scaled component-count) board files for ch02-ch05.

Mimics the ch01 scl pattern: for each 1comp/2comp challenge and each of its
variants, create truncated board variants at sizes 4, 6, 8, 12, 14, 16
(only for sizes ≤ original total component count).

Output: ``data/tasks/scaled/``
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

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


def remap_placed_y(placed: list[dict], start_y: int, end_y: int) -> list[dict]:
    """Remap placed-component y-coordinates into the range [start_y, end_y].

    Components keep their original x and type. y is assigned sequentially
    (preserving original relative y-order), starting at start_y.
    """
    sorted_placed = sorted(placed, key=lambda c: c.get("y", 0))
    remapped = []
    for i, comp in enumerate(sorted_placed):
        new_y = start_y + i
        if new_y > end_y:
            new_y = end_y - (len(sorted_placed) - 1 - i)
        remapped.append({**comp, "y": new_y})
    return remapped


def generate_scl(
    source_data: dict,
    scl_size: int,
    task_id: str,
    category: str,
) -> dict | None:
    """Generate a scaled-down board for a given scl_size.

    Returns None if scl_size > original total components.
    """
    board = source_data["board"]
    original_fixed = board.get("fixed_components", [])
    original_placed = source_data.get("solution", {}).get("placed_components", [])
    num_placed = len(original_placed)
    original_total = len(original_fixed) + num_placed

    if scl_size > original_total or scl_size < num_placed:
        return None

    num_fixed = scl_size - num_placed

    # Truncate fixed components (keep first num_fixed)
    new_fixed = copy.deepcopy(original_fixed[:num_fixed])

    # Remap placed components' y-coordinates
    new_height = scl_size + 1
    placed_start_y = num_fixed
    placed_end_y = new_height - 2  # row above trigger levers
    new_placed = remap_placed_y(copy.deepcopy(original_placed), placed_start_y, placed_end_y)

    # Build new board
    new_hoppers = copy.deepcopy(board.get("ball_hoppers", {}))
    new_triggers = {
        "left": {"x": 2, "y": new_height},
        "right": {"x": 8, "y": new_height},
    }

    result = {
        "task_id": task_id,
        "challenge_number": source_data["challenge_number"],
        "tier": source_data.get("tier", 1),
        "title": source_data.get("title", ""),
        "objective": source_data.get("objective", ""),
        "board": {
            "width": board.get("width", 11),
            "height": new_height,
            "fixed_components": new_fixed,
            "ball_hoppers": new_hoppers,
            "trigger_levers": new_triggers,
        },
        "available_parts": copy.deepcopy(source_data.get("available_parts", {})),
        "solution": {
            "placed_components": new_placed,
            "explanation": (
                f"Scaled variant — {scl_size} total components, "
                f"{num_placed} to place."
            ),
            "verified": False,
            "position_verified": False,
            "final_marble_state": copy.deepcopy(
                source_data.get("solution", {}).get("final_marble_state", [])
            ),
        },
        "input_sequence": copy.deepcopy(source_data.get("input_sequence", [])),
        "expected_output": copy.deepcopy(source_data.get("expected_output", {})),
    }

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
    """Generate all scl files for ch02-ch05. Returns counts per challenge."""
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
