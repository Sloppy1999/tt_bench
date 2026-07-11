#!/usr/bin/env python3
"""
Mechanism substitution generator for Turing Tumble benchmark.

Generates alternative implementations for official challenges by:
  1. bit → gear_bit (independent, same behavior)
  2. bit → gear_bit (adjacent/coupled, different behavior)
  3. Counter direction reversal (up ↔ down)

Output: data/tasks/mech_sub/{substitution_type}/
"""

import copy
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

REPO = Path(__file__).resolve().parent.parent
OFFICIAL = REPO / "data" / "tasks" / "official" / "challenges" / "json"
OUTBASE = REPO / "data" / "tasks" / "mech_sub"

# ── Challenges with bits good for substitution ──
# (task_id, #bits, description, has_solution)
# Only Tier 1 & 2 challenges (excluding Tier 3+: ch18, ch22, ch23, ch24, ch30)
TARGETS = [
    ("ch08", 1, "Bit alternator → gear_bit alternator", False),
    ("ch12", 1, "Conditional intercept → gear_bit intercept", False),
    ("ch13", 1, "Dual-color intercept → gear_bit intercept", False),
    ("ch14", 1, "State-dependent intercept → gear_bit", False),
    ("ch16", 2, "2-bit counter → gear_bit counter", True),
]

# All component types that could appear
ALL_PART_TYPES = [
    "ramp_right", "ramp_left", "crossover", "bit",
    "gear_bit", "gear", "interceptor", "trigger",
]

BIT_TYPES = {"bit", "gear_bit"}


def load_challenge(ch_id: str) -> dict:
    """Load an official challenge JSON by its short ID (e.g. 'ch08')."""
    path = OFFICIAL / f"tt-official-{ch_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Challenge not found: {path}")
    with open(path) as f:
        return json.load(f)


def ensure_parts(d: dict) -> dict:
    """Ensure all part types exist in available_parts with at least 0."""
    parts = d.get("available_parts", {})
    for pt in ALL_PART_TYPES:
        parts.setdefault(pt, 0)
    d["available_parts"] = parts
    return d


def count_bits_in(components: list[dict]) -> int:
    return sum(1 for c in components if c["type"] in BIT_TYPES)


def apply_bit_to_gearbit(data: dict, coupled: bool = False) -> dict:
    """
    Replace all 'bit' components with 'gear_bit'.
    
    If coupled=True, also auto-adds intermediate gear components
    between adjacent gear_bits to create coupling via build_gear_connections.
    """
    r = copy.deepcopy(data)
    r = ensure_parts(r)

    bit_count = 0
    gear_needed = 0

    # 1. Change fixed components
    for c in r["board"]["fixed_components"]:
        if c["type"] == "bit":
            c["type"] = "gear_bit"
            c["gear_group"] = 1 if coupled else -1
            bit_count += 1

    # 2. Change solution components
    for c in r.get("solution", {}).get("placed_components", []):
        if c["type"] == "bit":
            c["type"] = "gear_bit"

    # 3. Update available_parts
    parts = r["available_parts"]
    parts["bit"] = max(0, parts.get("bit", 0) - bit_count)
    parts["gear_bit"] = parts.get("gear_bit", 0) + bit_count

    if coupled and bit_count >= 2:
        # Add gear components between adjacent gear_bits
        # Find all gear_bit positions
        gb_positions = []
        for c in r["board"]["fixed_components"]:
            if c["type"] == "gear_bit":
                gb_positions.append((c["x"], c["y"]))

        # Add intermediate gears for bits within 2 cells vertically or horizontally
        existing_positions = {(c["x"], c["y"]) for c in r["board"]["fixed_components"]}
        for i, (x1, y1) in enumerate(gb_positions):
            for (x2, y2) in gb_positions[i + 1:]:
                if abs(x1 - x2) <= 2 and abs(y1 - y2) <= 2:
                    mid_x = (x1 + x2) // 2
                    mid_y = (y1 + y2) // 2
                    if (mid_x, mid_y) not in existing_positions:
                        r["board"]["fixed_components"].append({
                            "type": "gear",
                            "x": mid_x,
                            "y": mid_y,
                        })
                        existing_positions.add((mid_x, mid_y))
                        gear_needed += 1

        parts["gear"] = parts.get("gear", 0) + gear_needed

    # 4. Metadata
    r["_meta"] = {
        **(r.get("_meta", {})),
        "mechanism_substitution": "bit_to_gearbit",
        "substitution_subtype": "coupled" if coupled else "independent",
        "original_bit_count": bit_count,
        "gears_added": gear_needed,
    }

    # 5. Update task_id and title
    suffix = "gearbit-coup" if coupled else "gearbit-indep"
    original_id = r["task_id"]
    r["task_id"] = f"{original_id}-mechsub-{suffix}"
    r["title"] = f"{r['title']} (GearBit {'Coupled' if coupled else 'Independent'})"

    return r


def apply_counter_direction_flip(data: dict) -> dict:
    """
    Reverse counting direction: mirror ramp topology.
    
    For challenges with binary counters (bits in column + ramps around them):
    - Map ramps from left-side to right-side and vice-versa
    - This changes countUP ↔ countDOWN
    """
    r = copy.deepcopy(data)
    r = ensure_parts(r)

    # Find bit column (the x-coordinate where most bits live)
    bit_xs = Counter()
    for c in r["board"]["fixed_components"]:
        if c["type"] in BIT_TYPES:
            bit_xs[c["x"]] += 1

    if not bit_xs:
        return r

    bit_column = bit_xs.most_common(1)[0][0]

    # Mirror ALL components that affect routing around the bit column
    # ramps, interceptors (catchers flip sides)
    mirrorable = {"ramp_right", "ramp_left", "interceptor"}
    
    for c in r["board"]["fixed_components"]:
        if c["type"] in mirrorable:
            c["x"] = 2 * bit_column - c["x"]
            if c["type"] == "ramp_right":
                c["type"] = "ramp_left"
            elif c["type"] == "ramp_left":
                c["type"] = "ramp_right"

    # Mirror solution components too
    for c in r.get("solution", {}).get("placed_components", []):
        if c["type"] in mirrorable:
            c["x"] = 2 * bit_column - c["x"]
            if c["type"] == "ramp_right":
                c["type"] = "ramp_left"
            elif c["type"] == "ramp_left":
                c["type"] = "ramp_right"

    r["_meta"] = {
        **(r.get("_meta", {})),
        "mechanism_substitution": "counter_direction_flip",
        "bit_column": bit_column,
        "description": "Ramps mirrored around bit column; reverses count direction",
    }

    original_id = r["task_id"]
    r["task_id"] = f"{original_id}-mechsub-dirflip"
    r["title"] = f"{r['title']} (Direction Flipped)"

    return r


def generate_all():
    """Generate all mechanism substitution variants."""
    OUTBASE.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)

    for ch_id, nbits, desc, has_solution in TARGETS:
        try:
            data = load_challenge(ch_id)
        except FileNotFoundError as e:
            print(f"  SKIP {ch_id}: {e}", file=sys.stderr)
            continue

        print(f"  Processing {ch_id} ({nbits} bits): {desc}")

        # ── Substitution 1: bit → gear_bit (independent) ──
        indep = apply_bit_to_gearbit(data, coupled=False)
        outdir = OUTBASE / "gearbit_independent"
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / f"{indep['task_id']}.json"
        with open(outfile, "w") as f:
            json.dump(indep, f, indent=2)
        stats["gearbit_independent"] += 1

        # ── Substitution 2: bit → gear_bit (coupled) ──
        # Only makes sense for challenges with 2+ bits
        if nbits >= 2:
            coup = apply_bit_to_gearbit(data, coupled=True)
            outdir = OUTBASE / "gearbit_coupled"
            outdir.mkdir(parents=True, exist_ok=True)
            outfile = outdir / f"{coup['task_id']}.json"
            with open(outfile, "w") as f:
                json.dump(coup, f, indent=2)
            stats["gearbit_coupled"] += 1

        # ── Substitution 3: counter direction flip ──
        # Only for challenges that are binary counters
        if ch_id in ("ch16", "ch22"):
            flip = apply_counter_direction_flip(data)
            outdir = OUTBASE / "counter_direction"
            outdir.mkdir(parents=True, exist_ok=True)
            outfile = outdir / f"{flip['task_id']}.json"
            with open(outfile, "w") as f:
                json.dump(flip, f, indent=2)
            stats["counter_direction"] += 1

    return stats


def main():
    print("Generating mechanism substitution variants...\n")
    stats = generate_all()

    print(f"\n{'='*50}")
    print(f"  GENERATION SUMMARY")
    print(f"{'='*50}")
    for sub_type, count in sorted(stats.items()):
        print(f"  {sub_type:<25} {count:>5} tasks")
    print(f"  {'─'*32}")
    print(f"  {'TOTAL':<25} {sum(stats.values()):>5} tasks")
    print(f"\n  Output: {OUTBASE}/")


if __name__ == "__main__":
    main()
