#!/usr/bin/env python3
"""Generate Insight (C1) and Unsolvable (C2) variants for all challenges.

C1 — Insight: Add a distractor component type to available_parts.
     (_insight suffix)

C2 — Unsolvable (4 sub-types):
     _unsolvable_t1 — Type mismatch, same category (e.g. ramp_left → ramp_right)
     _unsolvable_t2 — Type mismatch, different category (e.g. ramp → crossover)
     _unsolvable_g1 — Extra gap: N+1 gaps, only N parts available
     _unsolvable_g2 — Double extra gap: N+2 gaps, only N parts available

Processes base files + existing position variants.
"""

import json
import sys
from pathlib import Path
from copy import deepcopy

ALL_PART_TYPES = [
    "ramp_right", "ramp_left", "crossover",
    "bit", "gear_bit", "gear", "interceptor", "trigger",
]

# ── Type mismatch mappings ─────────────────────────────────────────────────

# T1: same category, different direction/function
TYPE_SWAP_T1 = {
    "ramp_left": "ramp_right",
    "ramp_right": "ramp_left",
    "crossover": "bit",
    "bit": "crossover",
    "gear_bit": "gear",
    "gear": "gear_bit",
    "interceptor": "trigger",
    "trigger": "interceptor",
}

# T2: completely different category
TYPE_SWAP_T2 = {
    "ramp_left": "crossover",
    "ramp_right": "crossover",
    "crossover": "gear_bit",
    "bit": "gear",
    "gear_bit": "ramp_right",
    "gear": "ramp_left",
    "interceptor": "bit",
    "trigger": "crossover",
}

# C1 distractor mapping
DISTRACTOR_MAP = {
    "ramp_left": "ramp_right",
    "ramp_right": "ramp_left",
    "crossover": "bit",
    "bit": "crossover",
    "gear_bit": "gear",
    "gear": "gear_bit",
    "interceptor": "trigger",
    "trigger": "interceptor",
}


def component_key(c):
    return (c["type"], c["x"], c["y"])


def build_available_parts(placed):
    parts = {t: 0 for t in ALL_PART_TYPES}
    for pc in placed:
        t = pc["type"]
        if t in parts:
            parts[t] += 1
    return parts


# ── C1: Insight ────────────────────────────────────────────────────────────

def generate_insight(challenge: dict) -> dict:
    variant = deepcopy(challenge)
    variant["task_id"] = f"{challenge['task_id']}_insight"
    variant["solution"]["explanation"] = (
        f"Insight variant — available_parts includes a distractor type. "
        f"{challenge.get('solution', {}).get('explanation', '')}"
    )
    variant["solution"]["verified"] = False
    variant["solution"]["position_verified"] = False

    placed = variant["solution"]["placed_components"]
    for pc in placed:
        dist_type = DISTRACTOR_MAP.get(pc["type"])
        if dist_type and dist_type in variant["available_parts"]:
            variant["available_parts"][dist_type] += 1

    variant["_meta"] = variant.get("_meta", {})
    variant["_meta"]["variant_type"] = "insight"
    variant["_meta"]["distractor_types"] = list(set(
        DISTRACTOR_MAP[pc["type"]] for pc in placed
        if pc["type"] in DISTRACTOR_MAP
    ))
    return variant


# ── C2.1: Type mismatch T1 (same category) ─────────────────────────────────

def generate_unsolvable_t1(challenge: dict) -> dict:
    return _type_swap_variant(challenge, TYPE_SWAP_T1, "_unsolvable_t1",
                              "T1 type-mismatch (same category)")


# ── C2.2: Type mismatch T2 (different category) ────────────────────────────

def generate_unsolvable_t2(challenge: dict) -> dict:
    return _type_swap_variant(challenge, TYPE_SWAP_T2, "_unsolvable_t2",
                              "T2 type-mismatch (different category)")


def _type_swap_variant(challenge: dict, swap_map: dict, suffix: str, label: str) -> dict:
    variant = deepcopy(challenge)
    variant["task_id"] = f"{challenge['task_id']}{suffix}"
    variant["solution"]["verified"] = False
    variant["solution"]["position_verified"] = False

    swaps = []
    for pc in variant["solution"]["placed_components"]:
        orig = pc["type"]
        new = swap_map.get(orig, "crossover")
        pc["type"] = new
        swaps.append({"from": orig, "to": new})

    variant["solution"]["explanation"] = (
        f"Unsolvable {label} — {len(swaps)} type swap(s). "
        f"{challenge.get('solution', {}).get('explanation', '')}"
    )
    variant["available_parts"] = build_available_parts(
        variant["solution"]["placed_components"]
    )
    variant["_meta"] = variant.get("_meta", {})
    variant["_meta"]["variant_type"] = "unsolvable"
    variant["_meta"]["unsolvable_subtype"] = label
    variant["_meta"]["type_swaps"] = swaps
    return variant


# ── C2.3: Extra gap G1 (N+1 gaps, N parts) ─────────────────────────────────

def generate_unsolvable_g1(challenge: dict) -> dict | None:
    return _extra_gap_variant(challenge, 1, "_unsolvable_g1")


# ── C2.4: Extra gap G2 (N+2 gaps, N parts) ─────────────────────────────────

def generate_unsolvable_g2(challenge: dict) -> dict | None:
    return _extra_gap_variant(challenge, 2, "_unsolvable_g2")


def _extra_gap_variant(challenge: dict, extra_gaps: int, suffix: str) -> dict | None:
    """Remove extra_gaps additional components from fixed, keeping K parts.

    The board ends up with (original_gaps + extra_gaps) gaps but only K parts
    available — genuinely unsolvable because you can't fill all gaps.
    """
    variant = deepcopy(challenge)
    variant["task_id"] = f"{challenge['task_id']}{suffix}"
    variant["solution"]["verified"] = False
    variant["solution"]["position_verified"] = False

    fixed = variant["board"]["fixed_components"]
    solution = variant["solution"]["placed_components"]

    # Build complete set
    seen = set()
    all_components = []
    for c in fixed + solution:
        key = component_key(c)
        if key not in seen:
            seen.add(key)
            all_components.append(c)

    solution_keys = {component_key(pc) for pc in solution}

    # Find candidates for extra removal: components NOT already in solution,
    # preferring those adjacent (same column, nearby row) to existing gaps
    candidates = []
    for c in all_components:
        key = component_key(c)
        if key in solution_keys:
            continue
        # Proximity to nearest gap: lower = closer (prefer these)
        min_dist = min(
            abs(c["x"] - pc["x"]) + abs(c["y"] - pc["y"])
            for pc in solution
        )
        candidates.append((min_dist, c))

    # Sort by proximity (closest first), then pick extra_gaps
    candidates.sort(key=lambda x: x[0])
    removed = candidates[:extra_gaps]

    if len(removed) < extra_gaps:
        # Not enough non-solution components — skip
        return None

    removed_keys = {component_key(c) for _, c in removed}
    removed_types = [(c["type"], c["x"], c["y"]) for _, c in removed]

    # New fixed = all components minus (original solution gaps + extra removed)
    all_removed_keys = solution_keys | removed_keys
    variant["board"]["fixed_components"] = [
        c for c in all_components if component_key(c) not in all_removed_keys
    ]

    variant["solution"]["explanation"] = (
        f"Unsolvable G{extra_gaps} — {len(solution)} part(s) available, "
        f"{len(solution) + extra_gaps} gap(s) in board. "
        f"Extra gap(s) at: {removed_types}. "
        f"{challenge.get('solution', {}).get('explanation', '')}"
    )
    # available_parts and solution.placed_components stay unchanged
    # (K parts for K+extra_gaps gaps — unsolvable)

    variant["_meta"] = variant.get("_meta", {})
    variant["_meta"]["variant_type"] = "unsolvable"
    variant["_meta"]["unsolvable_subtype"] = f"extra_gap_{extra_gaps}"
    variant["_meta"]["extra_gaps"] = extra_gaps
    variant["_meta"]["parts_available"] = len(solution)
    variant["_meta"]["total_gaps"] = len(solution) + extra_gaps
    variant["_meta"]["extra_removed"] = removed_types
    return variant


# ── Orchestration ──────────────────────────────────────────────────────────

def process_directory(input_dir: Path, output_dir: Path) -> dict:
    """Generate all variant types for JSON files in input_dir.

    Returns counts by variant type.
    """
    insight_dir = output_dir / "insight"
    unsolvable_dir = output_dir / "unsolvable"
    insight_dir.mkdir(parents=True, exist_ok=True)
    unsolvable_dir.mkdir(parents=True, exist_ok=True)

    challenge_files = sorted(input_dir.glob("tt-official-*.json"))
    if not challenge_files:
        return {}

    counts = {"insight": 0, "unsolvable_t1": 0, "unsolvable_t2": 0,
              "unsolvable_g1": 0, "unsolvable_g2": 0}

    for cf in challenge_files:
        with open(cf) as f:
            challenge = json.load(f)

        # C1 Insight
        v = generate_insight(challenge)
        _write(v, insight_dir)
        counts["insight"] += 1

        # C2 T1 (same-category type mismatch)
        v = generate_unsolvable_t1(challenge)
        _write(v, unsolvable_dir)
        counts["unsolvable_t1"] += 1

        # C2 T2 (different-category type mismatch)
        v = generate_unsolvable_t2(challenge)
        _write(v, unsolvable_dir)
        counts["unsolvable_t2"] += 1

        # C2 G1 (1 extra gap)
        v = generate_unsolvable_g1(challenge)
        if v:
            _write(v, unsolvable_dir)
            counts["unsolvable_g1"] += 1

        # C2 G2 (2 extra gaps)
        v = generate_unsolvable_g2(challenge)
        if v:
            _write(v, unsolvable_dir)
            counts["unsolvable_g2"] += 1

    return counts


def _write(variant: dict, out_dir: Path):
    path = out_dir / f"{variant['task_id']}.json"
    with open(path, "w") as f:
        json.dump(variant, f, indent=2)
        f.write("\n")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data" / "tasks"

    totals = {"insight": 0, "unsolvable_t1": 0, "unsolvable_t2": 0,
              "unsolvable_g1": 0, "unsolvable_g2": 0}

    for comp in ("1comp", "2comp"):
        comp_dir = data_dir / f"challenges_{comp}"

        # Base files
        print(f"\n=== challenges_{comp}/ (base) ===")
        c = process_directory(comp_dir, comp_dir)
        _print_counts(c)
        for k in totals:
            totals[k] += c.get(k, 0)

        # Position variants
        variants_dir = comp_dir / "variants"
        if variants_dir.is_dir():
            flat = sorted(variants_dir.glob("tt-official-*.json"))
            flat = [f for f in flat if f.parent == variants_dir]
            if flat:
                print(f"\n=== challenges_{comp}/variants/ ===")
                c = process_directory(variants_dir, variants_dir)
                _print_counts(c)
                for k in totals:
                    totals[k] += c.get(k, 0)

    print(f"\n{'='*60}")
    for k, v in totals.items():
        print(f"  {k:20s}: {v}")
    print(f"  {'TOTAL':20s}: {sum(totals.values())}")


def _print_counts(c: dict):
    for k, v in c.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
