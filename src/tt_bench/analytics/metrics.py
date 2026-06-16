#!/usr/bin/env python3
"""Board complexity metrics for the Turing Tumble benchmark.

All functions accept a tt_sim.Board and optionally a task_info dict,
returning Dict[str, float] values keyed by metric name.

Metrics are organized into priority tiers:
  P0 — Primary: computable from board structure alone, strongest signals
  P1 — Composite/Aggregate/Domain: BICI, GCC, RPCC, IBR, HIC, SAC, PSDE
  P2 — Sub-domain specific: OSS (objective constraint density)
  P3 — Algorithmic complexity: K̃ via Block Decomposition Method (BDM)

Design constraint: metrics are computable without access to ground-truth
solutions.  Some P1/P2 metrics require optional task_info (available_parts,
objective text).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMPONENT_TYPES = frozenset({
    "ramp_right", "ramp_left", "crossover",
    "bit", "gear_bit", "gear", "interceptor", "trigger",
})


def _count_by_type(board) -> Counter[str]:
    """Return Counter mapping component-type-string -> count."""
    c: Counter[str] = Counter()
    for comp in board.components.values():
        c[comp.component_type.value] += 1
    return c


def _empty_cells(board) -> int:
    """Number of board cells with no component placed."""
    occupied = len(board.components)
    return board.rows * board.cols - occupied


def _trace_path(board, side: str = "blue") -> List[Tuple[int, int]]:
    """Trace the marble path from *side* hopper entry through the board.

    Runs one lightweight simulation (max 500 steps) and returns the list
    of (x, y) positions visited.  The board is reset afterward so side
    effects are not observable.
    """
    board.reset()
    result = board.release_marble(side)
    board.reset()
    return result.path or []


def _gear_groups_typed(board) -> List[List[Tuple[int, int]]]:
    """Return connected groups of GearBit + Gear components via BFS."""
    from tt_bench.simulator import GearBit, Gear

    visited: set[Tuple[int, int]] = set()
    groups: List[List[Tuple[int, int]]] = []

    for pos, comp in board.components.items():
        if not isinstance(comp, (GearBit, Gear)):
            continue
        if pos in visited:
            continue

        group: List[Tuple[int, int]] = []
        queue = [pos]
        while queue:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            group.append(cur)
            for nb in board.gear_connections.get(cur, []):
                if nb not in visited:
                    queue.append(nb)

        if group:
            groups.append(group)

    return groups


# ---------------------------------------------------------------------------
# P0 — Primary Metrics (board structure alone)
# ---------------------------------------------------------------------------


def scr(board) -> float:
    """Stateful Component Ratio.

    (bits + 2 × gear_bits) / total_components.  Gear bits are
    weighted 2× because networked, synchronized state transitions
    are strictly more complex than isolated bit flips.  Gears
    (structural connectors) appear in the denominator but not the
    numerator, reflecting that they add mechanical complexity
    without contributing state themselves.

    Range [0, 1].  0 = purely deterministic routing; 1 = every
    component is a stateful gear_bit.
    """
    counts = _count_by_type(board)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    stateful = counts.get("bit", 0) + 2 * counts.get("gear_bit", 0)
    return min(stateful / total, 1.0)


def ctd(board) -> float:
    """Component Type Diversity.

    Fraction of the 8 possible component types present on the board.
    Range [0, 1].  Higher values mean more physics rules to track.
    """
    types_present: set[str] = set()
    for comp in board.components.values():
        types_present.add(comp.component_type.value)
    return len(types_present) / len(_COMPONENT_TYPES)


def dependency_depth(board) -> int:
    """Dependency Depth — longest causal chain of state-dependent outcomes.

    Traces the marble path from each hopper entry and counts how many
    stateful components (bit, gear_bit) are encountered along the way.
    Returns the maximum count across both entry sides.  This is a
    structural proxy for how many sequential state transitions a single
    marble can trigger — measuring the length of the causal dependency
    chain in procedural reasoning.
    """
    max_bits = 0
    for side in ("blue", "red"):
        path = _trace_path(board, side)
        bits_seen = 0
        for x, y in path:
            if 0 <= x < board.cols and 0 <= y < board.rows:
                comp = board.get(x, y)
                if comp is not None and comp.component_type.value in ("bit", "gear_bit"):
                    bits_seen += 1
        if bits_seen > max_bits:
            max_bits = bits_seen
    return max_bits


# ---------------------------------------------------------------------------
# P1 — Composite / Domain-Specific Metrics
# ---------------------------------------------------------------------------


def gcc(board) -> float:
    """Gear Connectivity Complexity.

    Measures how heavily a board relies on networked gear mechanisms.
    For each connected gear group G containing at least one gear_bit:

        score(G) = n_gear_bits × (1.5 if gears present else 1.0)

    The 1.5× bonus reflects that intermediate Gear connectors add a
    layer of mechanical abstraction beyond direct gear_bit adjacency.

    Total is normalized by the number of components on the board, so
    GCC represents the gear-complexity fraction of the total board
    composition.  Boards with no gear_bits return 0.0.

    Range [0, 1].
    """
    from tt_bench.simulator import GearBit, Gear

    groups = _gear_groups_typed(board)
    total_weight = 0.0
    for group in groups:
        n_bits = sum(1 for pos in group
                     if isinstance(board.get(pos[0], pos[1]), GearBit))
        if n_bits == 0:
            continue  # gear-only groups carry no state complexity
        has_gears = any(isinstance(board.get(pos[0], pos[1]), Gear)
                        for pos in group)
        total_weight += n_bits * (1.5 if has_gears else 1.0)

    total_components = len(board.components)
    if total_components == 0:
        return 0.0
    return min(total_weight / total_components, 1.0)


def rpcc(board) -> float:
    """Routing Path Crossover Count (normalized).

    Number of crossover components divided by total board cells.
    Range [0, 1].
    """
    counts = _count_by_type(board)
    n_cross = counts.get("crossover", 0)
    max_cells = board.rows * board.cols
    if max_cells == 0:
        return 0.0
    return n_cross / max_cells


def ibr(board) -> float:
    """Interceptor-to-Bit Ratio (naturally bounded to [0, 1]).

    interceptors / max(interceptors + stateful_components, 1).

    A board with interceptors but no stateful components → IBR = 1.0
    (every component is a control-flow element).  As stateful components
    increase relative to interceptors, the ratio decreases, reflecting
    that control-flow logic is a smaller fraction of total reasoning load.

    High IBR → board implements conditional termination/decision logic
    rather than pure routing.  Range [0, 1].
    """
    counts = _count_by_type(board)
    n_int = counts.get("interceptor", 0)
    n_stateful = counts.get("bit", 0) + counts.get("gear_bit", 0)
    denom = max(n_int + n_stateful, 1)
    return n_int / denom


def hic(board) -> float:
    """Hopper Interaction Complexity.

    Measures cross-side coupling via trigger components.
    Range [0, 1]:
      0   = no triggers
      0.25 = one trigger present
      0.5  = both triggers present (bidirectional coupling)
    """
    counts = _count_by_type(board)
    n_trig = counts.get("trigger", 0)
    # Triggers typically come in pairs (blue/red); max meaningful is 2
    return min(n_trig / 4.0, 1.0)


def bici(board, weights: Optional[List[float]] = None) -> float:
    """Board Input Complexity Index — weighted composite.

    Combines SCR, CTD, GCC (norm), RPCC (norm), and IBR.
    Default weights are uniform [0.2, 0.2, 0.2, 0.2, 0.2].
    Range [0, 1].
    """
    if weights is None:
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]

    vals = [
        scr(board),
        ctd(board),
        gcc(board),   # already normalized
        rpcc(board),  # already normalized
        ibr(board),
    ]
    return sum(w * v for w, v in zip(weights, vals))


def sac(board, task_info: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """Synthesis Action Complexity.

    |empty cells| × (1 + H), where H is the Shannon entropy of the
    available-parts distribution.  The +1 ensures spatial search is
    captured even when only one part type is available (H = 0).

    Returns 0.0 for read-only boards with no available parts.
    Returns None if task_info is missing entirely.

    Range ≥ 0; unbounded above.
    """
    if task_info is None:
        return None

    available = task_info.get("available_parts", {})
    total_parts = sum(available.values())
    if total_parts == 0:
        return 0.0  # Nothing to place → no synthesis complexity

    # Shannon entropy of the part-type distribution
    entropy = 0.0
    for count in available.values():
        if count > 0:
            p = count / total_parts
            entropy -= p * math.log2(p)

    empty = _empty_cells(board)
    return empty * (1.0 + entropy)


def sac_norm(board, task_info: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """Synthesis Action Complexity — normalized to [0, 1].

    (1 + H) / 4, where H is Shannon entropy of the available-parts
    distribution and 4 = 1 + log₂(8) is the maximum possible (1+H)
    when all 8 part types are equally available.

    Isolates the *type-diversity* dimension of synthesis difficulty
    (how many different kinds of decisions the LLM faces).  The
    spatial dimension (how many places) is captured by other metrics.

    Returns 0.0 for read-only boards (no parts available).
    Returns None if task_info is missing.
    """
    if task_info is None:
        return None

    available = task_info.get("available_parts", {})
    total_parts = sum(available.values())
    if total_parts == 0:
        return 0.0  # Nothing to place

    entropy = 0.0
    for count in available.values():
        if count > 0:
            p = count / total_parts
            entropy -= p * math.log2(p)

    return (1.0 + entropy) / 4.0


def synthesis_load(board, task_info: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """Synthesis Load — how much of the board needs filling.

    total_available_parts / (rows × cols), normalized to [0, 1].
    Read-only boards (no parts available) return 0.0.

    Captures the *volume* of the synthesis task: placing 40 components
    on a 121-cell board (0.33) is objectively more work than placing
    4 (0.03), regardless of type diversity.
    """
    if task_info is None:
        return None

    available = task_info.get("available_parts", {})
    total = sum(available.values())
    if total == 0:
        return 0.0

    max_cells = board.rows * board.cols
    if max_cells == 0:
        return 0.0
    return min(total / max_cells, 1.0)


def psde(board, task_info: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """Program Synthesis Difficulty Estimate — composite [0, 1].

    Average of type-diversity (sac_norm) and placement-volume
    (synthesis_load), capturing both dimensions of synthesis
    difficulty.  Returns None when task_info is missing.
    """
    sn = sac_norm(board, task_info)
    sl = synthesis_load(board, task_info)
    if sn is None or sl is None:
        return None
    return (sn + sl) / 2.0


# ---------------------------------------------------------------------------
# P2 — Sub-Domain Specific Metrics
# ---------------------------------------------------------------------------


def oss(task_info: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """Objective Specificity Score.

    Parses objective text for quantifiers, conditionals, and pattern
    constraints.  Returns None if no task_info or no objective.

    Range [0, 1].  Higher = more precisely specified (may be harder to
    satisfy).
    """
    if task_info is None:
        return None

    objective = (task_info.get("objective", "") or "").lower()
    if not objective.strip():
        return None

    score = 0.0

    # Numeric quantifiers
    if re.search(r'\bexactly\b|\bonly\b|\bprecisely\b', objective):
        score += 1.0
    elif re.search(r'\bat least\b|\bat most\b|\bmore than\b|\bfewer than\b|'
                   r'\bgreater than\b|\bless than\b|\bno more than\b|'
                   r'\bover\b', objective):
        score += 0.5

    # Conditional logic — any use of "if", "when", "unless", "otherwise", "else"
    if re.search(r'\bif\b|\bwhen\b|\bunless\b|\botherwise\b|\belse\b',
                 objective):
        score += 1.0

    # Pattern / sequence constraints
    if re.search(r'pattern|sequence|alternating|blue.*red.*blue|red.*blue.*red',
                 objective):
        score += 1.0

    return score / 3.0


# ---------------------------------------------------------------------------
# P3 — Algorithmic Complexity (BDM)
# ---------------------------------------------------------------------------

_TYPE_ID: dict[str, int] = {
    "ramp_right": 1, "ramp_left": 2, "crossover": 3,
    "bit": 4, "gear_bit": 5, "gear": 6,
    "interceptor": 7, "trigger": 8,
}


def _grid_from_board(board) -> list[list[int]]:
    """Extract a type-ID matrix from the board (0 = empty)."""
    grid = [[0] * board.cols for _ in range(board.rows)]
    for (x, y), comp in board.components.items():
        if 0 <= x < board.cols and 0 <= y < board.rows:
            grid[y][x] = _TYPE_ID.get(comp.component_type.value, 1)
    return grid


def _ctm_block(block: list[list[int]]) -> float:
    """Approximate CTM (Coding Theorem Method) for a k×k block.

    Since precomputed CTM tables for 9-symbol alphabets are infeasible
    at practical block sizes (9^9 ≈ 387M patterns for 3×3), we use
    Block Shannon Entropy as a principled CTM proxy, following the
    standard BDM fallback described in Zenil et al. (2018, §4.2).

    CTM(block) ≈ k² · H(block)

    where H is the Shannon entropy of the symbol distribution within
    the block.  Highly repetitive blocks → near-zero CTM; diverse
    blocks → CTM approaches k² · log₂(min(k², 9)).
    """
    from collections import Counter
    cells = [cell for row in block for cell in row]
    counts = Counter(cells)
    n = len(cells)
    if n == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy * n  # scale by block size


def k_approx(board) -> float:
    """BDM approximation of Kolmogorov complexity (Zenil et al., 2018).

    Decomposes the board into overlapping k×k blocks and aggregates
    per-block algorithmic complexity via the Block Decomposition Method.
    Each block's CTM is approximated by Block Shannon Entropy — the
    standard fallback when precomputed CTM tables are unavailable for
    the target alphabet size.

    The 2D block decomposition preserves spatial locality that gzip
    compression (which flattens the grid to a 1D byte array) loses:

      BDM(X) = Σ CTM(block_i)  for all overlapping k×k blocks

    The result is normalized to [0, 1] relative to the theoretical
    maximum block complexity (all k² cells uniformly distributed across
    the 9 possible symbols).  Lower values indicate structural
    regularity (empty regions, repeated patterns); higher values
    indicate irregular/diverse component arrangements.

    Range (0, 1].
    """
    grid = _grid_from_board(board)
    rows, cols = board.rows, board.cols
    if rows == 0 or cols == 0:
        return 0.0

    # Block size: adapt to board dimensions
    k = 4 if min(rows, cols) >= 4 else min(rows, cols)
    if k < 2:
        # Board too small to decompose
        block = [row[:] for row in grid]
        return max(_ctm_block(block) / (k * k * math.log2(max(9, 1))), 1e-10)

    # Full overlap (stride = 1) for maximum resolution
    bdm_sum = 0.0
    block_count = 0

    for y in range(rows - k + 1):
        for x in range(cols - k + 1):
            block = [row[x:x + k] for row in grid[y:y + k]]
            bdm_sum += _ctm_block(block)
            block_count += 1

    if block_count == 0:
        return 0.0

    # Theoretical maximum per block: H_max = log₂(min(k², 9))
    # since at most 9 distinct symbols can appear in any block,
    # and at most k² cells
    n = k * k
    max_symbols = min(n, 9)
    max_per_block = n * math.log2(max_symbols) if max_symbols > 1 else n

    if max_per_block == 0:
        return 0.0

    avg_bdm = bdm_sum / block_count
    normalized = avg_bdm / max_per_block

    # Clamp to (0, 1] — K is strictly positive for any object
    return max(min(normalized, 1.0), 1e-10)


# ---------------------------------------------------------------------------
# Aggregate entry point
# ---------------------------------------------------------------------------


def compute_all_metrics(
    board,
    task_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Compute all available metrics in one call.

    Returns a dict keyed by metric name.  Metrics that cannot be computed
    (e.g. SAC without task_info) are omitted.
    """
    metrics: Dict[str, float] = {}

    # P0 — always available
    metrics["scr"] = scr(board)
    metrics["ctd"] = ctd(board)
    metrics["dependency_depth"] = float(dependency_depth(board))

    # P1
    metrics["gcc"] = gcc(board)
    metrics["rpcc"] = rpcc(board)
    metrics["ibr"] = ibr(board)
    metrics["hic"] = hic(board)
    metrics["bici"] = bici(board)

    sac_val = sac(board, task_info)
    if sac_val is not None:
        metrics["sac"] = sac_val

    sac_norm_val = sac_norm(board, task_info)
    if sac_norm_val is not None:
        metrics["sac_norm"] = sac_norm_val

    synth_load_val = synthesis_load(board, task_info)
    if synth_load_val is not None:
        metrics["synthesis_load"] = synth_load_val

    psde_val = psde(board, task_info)
    if psde_val is not None:
        metrics["psde"] = psde_val

    # P2
    oss_val = oss(task_info)
    if oss_val is not None:
        metrics["oss"] = oss_val

    # P3
    metrics["k_approx"] = k_approx(board)

    return metrics


# ---------------------------------------------------------------------------
# Quick CLI for ad-hoc metric inspection
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "simulator"))
    from tt_bench import simulator as tt_sim

    parser = argparse.ArgumentParser(description="Compute complexity metrics for one board")
    parser.add_argument("challenge_json", help="Path to a challenge JSON file")
    args = parser.parse_args()

    board = tt_sim.Board.from_task_json(args.challenge_json)
    with open(args.challenge_json) as f:
        task_info = json.load(f)

    metrics = compute_all_metrics(board, task_info)

    print(f"Task: {task_info.get('task_id', args.challenge_json)}")
    print("-" * 50)
    for name in sorted(metrics):
        val = metrics[name]
        if isinstance(val, float):
            print(f"  {name:20s} = {val:.4f}")
        else:
            print(f"  {name:20s} = {val}")
    print("-" * 50)
    print(f"BICI (overall input complexity): {metrics.get('bici', 0):.4f}")
