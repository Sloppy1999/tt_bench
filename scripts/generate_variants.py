#!/usr/bin/env python3
"""Generate position variants for N-component challenges (1comp, 2comp, etc.).

For each challenge, builds the complete component set (fixed + solution),
then produces one variant per component where that component replaces the
first solution component.
"""

import json
import os
import sys
from pathlib import Path
from copy import deepcopy

ALL_PART_TYPES = [
    "ramp_right", "ramp_left", "crossover",
    "bit", "gear_bit", "gear", "interceptor", "trigger",
]


def component_key(c):
    """Stable identity for a component: (type, x, y)."""
    return (c["type"], c["x"], c["y"])


def build_available_parts(placed_components: list) -> dict:
    """Return available_parts based on the solution components."""
    parts = {t: 0 for t in ALL_PART_TYPES}
    for pc in placed_components:
        t = pc["type"]
        if t in parts:
            parts[t] += 1
    return parts


def generate_variants(challenge_path: Path, output_dir: Path) -> int:
    """Generate all position variants for one challenge file.

    For each component NOT in the original solution, creates a variant
    where that component replaces solution[0].

    Returns the number of variants written.
    """
    with open(challenge_path) as f:
        original = json.load(f)

    task_id = original["task_id"]
    fixed = original["board"]["fixed_components"]
    placed = original["solution"]["placed_components"]

    # Complete component set — deduplicate by (type, x, y)
    seen = set()
    all_components = []
    for c in fixed + placed:
        key = component_key(c)
        if key not in seen:
            seen.add(key)
            all_components.append(c)

    if not all_components:
        print(f"  WARNING: {task_id} has no components, skipping")
        return 0

    # Identify original solution component keys
    original_solution_keys = {component_key(pc) for pc in placed}

    # Components NOT in original solution (candidates for swapping)
    candidates = [c for c in all_components if component_key(c) not in original_solution_keys]

    if not candidates:
        # Edge case: all components are in solution → no variants possible
        return 0

    written = 0
    for variant_num, comp in enumerate(candidates, start=1):
        key = component_key(comp)

        variant = deepcopy(original)
        variant["task_id"] = f"{task_id}_var_{variant_num}"

        # New solution: replace solution[0] with comp, keep the rest
        new_placed = [comp] + placed[1:]

        # Fixed components = everything except the new solution components
        new_solution_keys = {component_key(pc) for pc in new_placed}
        variant["board"]["fixed_components"] = [
            c for c in all_components if component_key(c) not in new_solution_keys
        ]

        variant["solution"]["placed_components"] = new_placed

        # Update available_parts based on new solution
        variant["available_parts"] = build_available_parts(new_placed)

        # Reset verification
        variant["solution"]["verified"] = False
        variant["solution"]["position_verified"] = False

        variant["solution"]["explanation"] = (
            f"Variant — removed {comp['type']} at ({comp['x']}, {comp['y']}). "
            "This variant swaps one solution component."
        )

        out_path = output_dir / f"{variant['task_id']}.json"
        with open(out_path, "w") as f:
            json.dump(variant, f, indent=2)
            f.write("\n")

        written += 1

    return written


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    # Determine which comp directory to process from command line or default
    if len(sys.argv) > 1:
        comp = sys.argv[1]
    else:
        print("Usage: python generate_variants.py <1comp|2comp>")
        print("   or: python generate_variants.py <dir_path>")
        sys.exit(1)

    # Accept either "1comp"/"2comp" shorthand or a full path
    if comp in ("1comp", "2comp"):
        input_dir = repo_root / "data" / "tasks" / f"challenges_{comp}"
    else:
        input_dir = Path(comp)

    if not input_dir.is_dir():
        print(f"ERROR: directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = input_dir / "variants"
    output_dir.mkdir(parents=True, exist_ok=True)

    challenge_files = sorted(input_dir.glob("tt-official-*.json"))
    if not challenge_files:
        print(f"ERROR: no challenge files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    total_variants = 0
    for cf in challenge_files:
        n = generate_variants(cf, output_dir)
        with open(cf) as f:
            d = json.load(f)
        total_comp = len(d["board"]["fixed_components"]) + len(d["solution"]["placed_components"])
        sol_count = len(d["solution"]["placed_components"])
        print(f"{cf.name}: {n} variants ({total_comp} total comps, {sol_count} in solution → {total_comp - sol_count} candidates)")
        total_variants += n

    print(f"\nTotal: {total_variants} variant files written to {output_dir}")


if __name__ == "__main__":
    main()
