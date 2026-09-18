#!/usr/bin/env python3
"""Generate scaled and padded solvable variants.

Two strategies:
1. Linear-scalable challenges (zig-zag patterns): generate genuinely scaled
   versions with more/fewer component rows. Applies to ch01, ch03, ch04.
2. Other challenges: pad the board canvas while keeping components identical.

For scaled versions, also generates full position variants at each scale.
"""

import json
import sys
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

from tt_bench.simulator import Board, verify_task

ALL_PART_TYPES = [
    "ramp_right", "ramp_left", "crossover",
    "bit", "gear_bit", "gear", "interceptor", "trigger",
]

REPO = Path(__file__).resolve().parent.parent

# ── Scale definitions for linear-scalable challenges ─────────────────────

# ch01 Gravity: alternating ramps at cols 2-3, rows 0..N-1
# ch03 Ignition: two separated zig-zags (blue left, red right converging)
# ch04 Fusion: two zig-zags that merge in the middle

# Each entry: (challenge_number, generator_fn)
# The generator takes (num_rows) and returns (components, hoppers, triggers, width)

def ch01_pattern(rows: int):
    """Gravity: zig-zag at cols 2-3."""
    comps = []
    for y in range(rows):
        if y % 2 == 0:
            comps.append({"type": "ramp_right", "x": 2, "y": y})
        else:
            comps.append({"type": "ramp_left", "x": 3, "y": y})
    hoppers = {
        "blue": {"x": 2, "y": -1, "count": 8},
        "red": {"x": 8, "y": -1, "count": 8},
    }
    # The zig-zag alternates between columns 2 and 3, so it exits at column 2
    # after an even number of rows and column 3 after an odd number. The left
    # catcher has to sit under the actual exit or the marble is lost.
    triggers = {
        "left": {"x": 2 if rows % 2 == 0 else 3, "y": rows + 1},
        "right": {"x": 8, "y": rows + 1},
    }
    return comps, hoppers, triggers, 11


def ch03_pattern(rows: int):
    """Ignition: two mirrored zig-zags (left blue, right red)."""
    # Left side (blue): ramps going down-right from col 2
    # Right side (red): ramps going down-left from col 8
    comps = []
    for y in range(min(4, rows)):  # blue side: 4 fixed ramps then crossover
        if y == 0:
            comps.append({"type": "ramp_right", "x": 2, "y": y})
        else:
            comps.append({"type": "ramp_right", "x": 2 + y, "y": y})
    # Red side mirror
    for y in range(min(4, rows)):
        if y == 0:
            comps.append({"type": "ramp_left", "x": 8, "y": y})
        else:
            comps.append({"type": "ramp_left", "x": 8 - y, "y": y})
    # Bottom convergence (rows 4+)
    for y in range(4, rows):
        x = 4 + (y - 4)  # start at col 4, move right
        comps.append({"type": "ramp_right", "x": min(x, 10), "y": y})
    hoppers = {
        "blue": {"x": 2, "y": -1, "count": 1},
        "red": {"x": 8, "y": -1, "count": 1},
    }
    triggers = {
        "left": {"x": 2, "y": rows + 1},
        "right": {"x": 8, "y": rows + 1},
    }
    return comps, hoppers, triggers, 11


# Map challenge number → scaling function + min/max rows
SCALABLE = {
    1: (ch01_pattern, 4, 16),   # Gravity: zig-zag
    # ch02 is irregular — skip for now
    # 3: (ch03_pattern, 6, 14),  # Ignition (more complex, skip for now)
    # ch04 is merging — skip for now
}

# ── Helpers ───────────────────────────────────────────────────────────────

def component_key(c):
    return (c["type"], c["x"], c["y"])


def build_available_parts(placed):
    parts = {t: 0 for t in ALL_PART_TYPES}
    for pc in placed:
        t = pc["type"]
        if t in parts:
            parts[t] += 1
    return parts


def generate_challenge(components, solution_indices, hoppers, triggers,
                       width, height, base_task_id, challenge_number, tier,
                       title, objective, input_sequence, final_marble_state):
    """Build a challenge JSON given a component list and solution indices."""
    solution_keys = {component_key(components[i]) for i in solution_indices}
    fixed = [c for i, c in enumerate(components) if i not in solution_indices]
    placed = [components[i] for i in solution_indices]

    return {
        "task_id": base_task_id,
        "challenge_number": challenge_number,
        "tier": tier,
        "title": title,
        "objective": objective,
        "board": {
            "width": width,
            "height": height,
            "fixed_components": fixed,
            "ball_hoppers": hoppers,
            "trigger_levers": triggers,
        },
        "available_parts": build_available_parts(placed),
        "solution": {
            "placed_components": placed,
            "explanation": f"Scaled variant — {len(components)} total components, "
                           f"{len(placed)} to place.",
            "verified": False,
            "position_verified": False,
            "final_marble_state": final_marble_state,
        },
        "input_sequence": input_sequence,
        "expected_output": {
            "description": "Where each marble ends up after execution",
            "format": "left_catcher, right_catcher counts",
        },
    }


# ── Scaled variants for linear challenges ────────────────────────────────

def is_degenerate(challenge: dict) -> bool:
    """Whether the board already works without the components to be placed.

    Such a task cannot discriminate between solvers — it scores as solved
    whatever the model does — so it is worse than useless in a benchmark.
    """
    placed = challenge.get("solution", {}).get("placed_components") or []
    if not placed:
        return False
    stripped = deepcopy(challenge)
    stripped["solution"]["placed_components"] = []
    try:
        return bool(verify_task(stripped))
    except Exception:
        return False


def generate_scaled_variants(ch_num: int, out_dir: Path) -> int:
    """Generate genuinely scaled versions of a linear challenge."""
    if ch_num not in SCALABLE:
        return 0

    pattern_fn, min_rows, max_rows = SCALABLE[ch_num]
    original_1comp_file = REPO / f"data/tasks/challenges_1comp/tt-official-ch{ch_num:02d}-1comp.json"

    if not original_1comp_file.exists():
        print(f"  Warning: {original_1comp_file} not found, skipping scaling")
        return 0

    with open(original_1comp_file) as f:
        orig = json.load(f)

    tier = orig["tier"]
    title = orig["title"]
    objective = orig["objective"]
    input_sequence = orig["input_sequence"]
    final_marble_state = orig["solution"].get("final_marble_state", [])

    count = 0

    # Don't generate the original size (10 rows for ch01) — that already exists
    original_rows = len(orig["board"]["fixed_components"]) + len(orig["solution"]["placed_components"])

    for num_rows in range(min_rows, max_rows + 1):
        if num_rows == original_rows:
            continue  # skip original size

        comps, hoppers, triggers, width = pattern_fn(num_rows)
        height = num_rows + 1  # board height = rows + 1 for triggers

        # ── 1comp version: last component is the solution ──
        base_id = f"tt-official-ch{ch_num:02d}-1comp_scl{num_rows}"
        challenge = generate_challenge(
            comps, [num_rows - 1], hoppers, triggers,
            width, height, base_id, ch_num, tier,
            title, objective, input_sequence, final_marble_state,
        )
        # All variants at this num_rows share the same full component
        # layout (fixed + placed), so this one verification covers them all —
        # e.g. an odd row count leaves the marble one column off from either
        # catcher, which is invalid regardless of which component is "solution".
        if not verify_task(challenge):
            print(f"  Warning: skipping unsolvable ch{ch_num:02d} scl{num_rows} (invalid layout)")
            continue
        if not is_degenerate(challenge):
            if is_degenerate(challenge):
                continue
            _write(challenge, out_dir)
            count += 1

        # ── 1comp position variants: each component gets its turn ──
        for i in range(num_rows - 1):  # all except last (already done)
            var_id = f"tt-official-ch{ch_num:02d}-1comp_scl{num_rows}_var_{i + 1}"
            challenge = generate_challenge(
                comps, [i], hoppers, triggers,
                width, height, var_id, ch_num, tier,
                title, objective, input_sequence, final_marble_state,
            )
            if is_degenerate(challenge):
                continue
            _write(challenge, out_dir)
            count += 1

        # ── 2comp version: last two components ──
        if num_rows >= 3:
            base_id = f"tt-official-ch{ch_num:02d}-2comp_scl{num_rows}"
            challenge = generate_challenge(
                comps, [num_rows - 1, num_rows - 2], hoppers, triggers,
                width, height, base_id, ch_num, tier,
                title, objective, input_sequence, final_marble_state,
            )
            if is_degenerate(challenge):
                continue
            _write(challenge, out_dir)
            count += 1

            # ── 2comp position variants ──
            # Variant: swap first solution component with each fixed component
            var_count = 0
            for i in range(num_rows):
                sol_indices = [i, num_rows - 2]
                if i == num_rows - 1 or i == num_rows - 2:
                    continue  # skip original
                var_count += 1
                var_id = f"tt-official-ch{ch_num:02d}-2comp_scl{num_rows}_var_{var_count}"
                challenge = generate_challenge(
                    comps, sol_indices, hoppers, triggers,
                    width, height, var_id, ch_num, tier,
                    title, objective, input_sequence, final_marble_state,
                )
                _write(challenge, out_dir)
                count += 1

    return count


# ── Height-scaled variants for all solvable challenges ───────────────────

def _chute_starts(challenge: dict) -> set[tuple[int, int]]:
    """Cells where an extended marble path has to pick up.

    The bottom row is allowed to be empty on a catcher column, so once the
    board grows that cell stops being exempt and needs a component too.
    """
    board = Board.from_task_dict(challenge)
    sequence = challenge.get("input_sequence") or ["blue"]
    if isinstance(sequence, str):
        sequence = [s.strip() for s in sequence.split(",") if s.strip()]

    starts = set()
    for result in board.run(sequence):
        path = result.path or []
        inboard = [p for p in path if 0 <= p[1] < board.rows]
        below = [p for p in path if p[1] >= board.rows]
        if not inboard or not below:
            continue
        last = inboard[-1]
        if last in board.components:
            starts.add((below[0][0], board.rows))
        else:
            starts.add(last)
    return starts


def _chute(start_x: int, start_y: int, end_y: int, width: int,
           first_step: int) -> list[dict]:
    """Ramps carrying a marble from (start_x, start_y) down to row end_y.

    Every row needs a component or the marble free-falls, and each ramp moves
    it exactly one column, so the chute zig-zags.
    """
    components = []
    x, step = start_x, first_step
    for y in range(start_y, end_y):
        if not 0 <= x + step < width:
            step = -step
        components.append({
            "type": "ramp_right" if step > 0 else "ramp_left",
            "x": x,
            "y": y,
        })
        x += step
        step = -step
    return components


def generate_height_padded_variants(challenge: dict, out_dir: Path,
                                    pad_sizes: list[int]) -> int:
    """Create taller boards, extending the marble path through the new rows."""
    count = 0
    width = challenge["board"].get("width", 11)
    orig_height = challenge["board"].get("height", 11)

    try:
        starts = _chute_starts(challenge)
    except Exception:
        return 0
    if not starts:
        return 0

    for pad in pad_sizes:
        new_height = orig_height + pad
        built = None

        # The chute's length fixes the parity of its horizontal displacement,
        # so try both start directions and both bottom-row treatments rather
        # than assuming one lands on a catcher.
        for last_row_empty in (True, False):
            for first_step in (1, -1):
                variant = deepcopy(challenge)
                variant["board"]["height"] = new_height
                end_y = new_height - 1 if last_row_empty else new_height
                extra = []
                for start_x, start_y in sorted(starts):
                    extra.extend(
                        _chute(start_x, start_y, end_y, width, first_step)
                    )
                variant["board"]["fixed_components"] = (
                    deepcopy(challenge["board"].get("fixed_components", [])) + extra
                )
                for side in ("left", "right"):
                    lever = variant["board"].get("trigger_levers", {}).get(side)
                    if lever and "y" in lever:
                        lever["y"] += pad
                try:
                    if verify_task(variant):
                        built = variant
                        break
                except Exception:
                    continue
            if built:
                break

        if built is None:
            continue

        built["task_id"] = f"{challenge['task_id']}_sz{width}x{new_height}"
        built["solution"]["verified"] = True
        built["solution"]["position_verified"] = True
        built["solution"]["explanation"] = (
            f"Tall scaled variant — board expanded from {width}×{orig_height} "
            f"to {width}×{new_height}, extending the marble path through the "
            f"added rows. {challenge.get('solution', {}).get('explanation', '')}"
        )
        _write(built, out_dir)
        count += 1

    return count


# ── Padded variants for all solvable challenges ──────────────────────────

def generate_padded_variants(challenge: dict, out_dir: Path,
                              pad_sizes: list[int]) -> int:
    """Create valid wider-board variants without inserting free-fall rows."""
    count = 0
    orig_width = challenge["board"].get("width", 11)
    orig_height = challenge["board"].get("height", 11)

    for pad in pad_sizes:
        variant = deepcopy(challenge)
        new_width = orig_width + pad
        variant["board"]["width"] = new_width
        # Increasing height while leaving the path unchanged inserts empty rows
        # before the catchers.  Horizontal padding increases canvas area while
        # preserving the original, verified marble path.
        variant["board"]["height"] = orig_height
        variant["task_id"] = f"{challenge['task_id']}_sz{new_width}x{orig_height}"
        variant["solution"]["verified"] = True
        variant["solution"]["position_verified"] = True
        variant["solution"]["explanation"] = (
            f"Wide padded variant — board expanded from {orig_width}×{orig_height} "
            f"to {new_width}×{orig_height} without changing the marble path. "
            f"{challenge.get('solution', {}).get('explanation', '')}"
        )

        if verify_task(variant):
            _write(variant, out_dir)
            count += 1
        else:
            print(f"  Warning: rejected invalid padded variant {variant['task_id']}")

    return count


def _write(variant, out_dir):
    path = out_dir / f"{variant['task_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(variant, f, indent=2)
        f.write("\n")


def remove_stale_padded_variants(out_dir: Path) -> int:
    """Remove only padded files owned by this generator before rebuilding."""
    removed = 0
    for path in out_dir.glob("*_sz*.json"):
        path.unlink()
        removed += 1
    return removed


def remove_stale_scaled_variants(out_dir: Path, ch_nums) -> int:
    """Remove only linear-scaled files owned by this generator before rebuilding."""
    removed = 0
    for ch_num in ch_nums:
        for path in out_dir.glob(f"tt-official-ch{ch_num:02d}-*comp_scl*.json"):
            path.unlink()
            removed += 1
    return removed


# ── Collect solvable challenges ──────────────────────────────────────────

def collect_solvable() -> list[tuple[Path, dict]]:
    """Find all solvable challenge JSONs."""
    results = []
    roots = [
        REPO / "data/tasks/official/challenges/json",
        REPO / "data/tasks/challenges_1comp",
        REPO / "data/tasks/challenges_2comp",
    ]
    for root in roots:
        for f in sorted(root.rglob("tt-official-*.json")):
            if "_questions" in f.stem:
                continue
            if "insight" in f.stem or "unsolvable" in f.stem:
                continue
            try:
                d = json.load(open(f))
                # Already padded variants should not be re-padded
                if "_sz" in d.get("task_id", ""):
                    continue
                if "_scl" in d.get("task_id", ""):
                    continue
                # A source that already works without its solution parts would
                # produce padded variants with the same defect.
                if verify_task(d) and not is_degenerate(d):
                    results.append((f, d))
            except Exception:
                pass
    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    out_dir = REPO / "data/tasks/scaled"
    out_dir.mkdir(parents=True, exist_ok=True)
    removed = remove_stale_padded_variants(out_dir)
    if removed:
        print(f"Removed {removed} stale padded variants")
    removed = remove_stale_scaled_variants(out_dir, SCALABLE)
    if removed:
        print(f"Removed {removed} stale scaled-linear variants")

    scaled_total = 0
    padded_total = 0

    # ── Phase 1: Genuinely scaled linear challenges ──
    print("=== Phase 1: Scaled linear patterns ===")
    for ch_num in SCALABLE:
        n = generate_scaled_variants(ch_num, out_dir)
        print(f"  ch{ch_num:02d}: {n} scaled variants")
        scaled_total += n

    # ── Phase 2: Padded versions of all solvable challenges ──
    print("\n=== Phase 2: Padded boards ===")
    pad_sizes = [2, 4]  # +2 and +4 padding
    all_solvable = collect_solvable()
    print(f"  Found {len(all_solvable)} solvable challenges to pad")

    height_total = 0
    for fpath, challenge in all_solvable:
        padded_total += generate_padded_variants(challenge, out_dir, pad_sizes)
        height_total += generate_height_padded_variants(challenge, out_dir, pad_sizes)

    print(f"  {padded_total} wide variants generated")
    print(f"  {height_total} tall variants generated")

    print(f"\n{'='*60}")
    print(f"  Scaled (linear patterns): {scaled_total}")
    print(f"  Padded — wider boards:    {padded_total}")
    print(f"  Padded — taller boards:   {height_total}")
    print(f"  TOTAL new solvable:       {scaled_total + padded_total + height_total}")
    print(f"  Output: {out_dir}/")


if __name__ == "__main__":
    main()
