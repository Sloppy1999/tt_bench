#!/usr/bin/env python3
"""
Mechanism substitution visualization: side-by-side board comparison.

Shows the original challenge (right, large) alongside 3 substitution variants
(left, stacked) with highlighted differences for supervisor presentation.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Import renderer primitives ──
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tt_bench.simulator.renderer import (
    COLOURS,
    BOARD_W, BOARD_H,
    MARGIN_TOP, MARGIN_BOTTOM, MARGIN_SIDES,
    FIG_W, FIG_H,
    draw_peg_grid,
    draw_board_frame,
    draw_component,
    draw_hopper,
    draw_catcher,
    draw_legend,
    draw_ramp_right,
    draw_ramp_left,
    draw_bit,
    draw_gear_bit,
    draw_gear,
    draw_crossover,
    draw_interceptor,
    draw_trigger,
)
from tt_bench.simulator import Board, build_gear_connections

OUTDIR = REPO / "data" / "visualizations"

# ── Layout constants ──
# We'll embed each board in its own mini-axes with internal coordinates [0, FIG_W] × [0, FIG_H]
PANEL_SCALE = 0.48  # how much to scale variant panels vs original


def load_task(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def draw_board_in_axes(ax, task: dict, state: str = "start", show_title: bool = True,
                       highlight_changes: list | None = None):
    """
    Draw a TT board inside a matplotlib Axes using renderer primitives.
    
    highlight_changes: list of (x, y, old_type, new_type) tuples to outline.
    """
    board_data = task["board"]
    sol = task.get("solution", {})
    hoppers = board_data.get("ball_hoppers", {})
    tlevs = board_data.get("trigger_levers", {})

    fixed_comps = board_data.get("fixed_components", [])
    placed_comps = sol.get("placed_components", []) if state == "solution" else []
    avail = task.get("available_parts", {})

    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(COLOURS["bg"])

    draw_peg_grid(ax)

    ch_title = task.get("title", "") if show_title else ""
    state_label = "Starting setup" if state == "start" and show_title else ""
    draw_board_frame(ax, title=ch_title, subtitle=state_label)

    # Hoppers
    blue_h = hoppers.get("blue", {})
    red_h = hoppers.get("red", {})
    if blue_h.get("count", 0) > 0:
        draw_hopper(ax, blue_h["x"], "B", blue_h["count"], "blue", blue_h.get("y", -1))
    if red_h.get("count", 0) > 0:
        draw_hopper(ax, red_h["x"], "R", red_h["count"], "red", red_h.get("y", -1))

    # Catchers
    left_t = tlevs.get("left", {})
    right_t = tlevs.get("right", {})
    if left_t:
        draw_catcher(ax, left_t["x"], "blue", ball_count=0, active=False)
    if right_t:
        draw_catcher(ax, right_t["x"], "red", ball_count=0, active=False)

    # Fixed components
    for comp in fixed_comps:
        draw_component(ax, comp, COLOURS["fixed"], zorder=5)

    # Placed components
    for comp in placed_comps:
        draw_component(ax, comp, COLOURS["placed"], zorder=6)

    # Available parts legend
    if state == "start":
        draw_legend(ax, avail, zorder=9)

    # Highlight changed components
    if highlight_changes:
        _CELL = 1.0  # grid cell size in internal coords
        for x, y, old_type, new_type in highlight_changes:
            cx = MARGIN_SIDES + x * _CELL
            cy = MARGIN_BOTTOM + (BOARD_H - 1 - y) * _CELL
            rect = FancyBboxPatch(
                (cx - 0.45, cy - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.05",
                facecolor="none",
                edgecolor="#e74c3c",
                linewidth=2.5,
                linestyle="--",
                zorder=20,
            )
            ax.add_patch(rect)
            # Small label
            ax.annotate(
                f"was:\n{old_type}",
                xy=(cx, cy - 0.55),
                fontsize=5,
                color="#e74c3c",
                ha="center",
                va="top",
                zorder=21,
                fontweight="bold",
            )

    # Objective text
    obj = task.get("objective", "")
    if obj:
        ax.text(FIG_W - 0.2, 0.12, obj, ha="right", va="bottom",
                fontsize=6.5, color="#333333", zorder=9, style="italic")


def find_mech_sub_file(ch_id: str, subdir: str, suffix: str) -> Path | None:
    """Find a mech_sub file by scanning the directory for a task_id pattern match."""
    d = REPO / "data" / "tasks" / "mech_sub" / subdir
    if not d.exists():
        return None
    # Files are named {task_id}.json — task_id may be "tt-official-16" or "tt-official-ch16"
    for f in sorted(d.glob("*.json")):
        if suffix in f.stem and ch_id.replace("ch", "") in f.stem:
            return f
    return None


def build_comparison(ch_id: str):
    base_path = REPO / "data" / "tasks" / "official" / "challenges" / "json"
    orig_file = base_path / f"tt-official-{ch_id}.json"
    if not orig_file.exists():
        raise FileNotFoundError(f"Original not found: {orig_file}")
    original = load_task(orig_file)

    variants = []

    # gearbit_independent
    indep_file = find_mech_sub_file(ch_id, "gearbit_independent", "gearbit-indep")
    if indep_file:
        indep_task = load_task(indep_file)
        changes = []
        for c in indep_task["board"]["fixed_components"]:
            if c["type"] == "gear_bit":
                changes.append((c["x"], c["y"], "bit", "gear_bit\n(independent)"))
        variants.append(("GearBit\nIndependent", indep_task, changes))

    # gearbit_coupled
    coup_file = find_mech_sub_file(ch_id, "gearbit_coupled", "gearbit-coup")
    if coup_file:
        coup_task = load_task(coup_file)
        changes = []
        for c in coup_task["board"]["fixed_components"]:
            if c["type"] == "gear_bit":
                changes.append((c["x"], c["y"], "bit", "gear_bit\n(coupled)"))
            elif c["type"] == "gear":
                changes.append((c["x"], c["y"], "—", "gear\n(coupler)"))
        variants.append(("GearBit\nCoupled", coup_task, changes))

    # counter_direction
    dir_file = find_mech_sub_file(ch_id, "counter_direction", "dirflip")
    if dir_file:
        dir_task = load_task(dir_file)
        changes = []
        orig_fixed = {(c["x"], c["y"]): c for c in original["board"]["fixed_components"]}
        for c in dir_task["board"]["fixed_components"]:
            if c["type"] in ("ramp_right", "ramp_left"):
                orig_c = orig_fixed.get((c["x"], c["y"]))
                if orig_c and orig_c.get("type") != c["type"]:
                    changes.append((c["x"], c["y"], orig_c["type"], c["type"]))
            elif c["type"] == "interceptor":
                orig_c = orig_fixed.get((c["x"], c["y"]))
                if orig_c and orig_c.get("x") != c["x"]:
                    # Interceptor moved
                    for oc in original["board"]["fixed_components"]:
                        if oc["type"] == "interceptor":
                            changes.append((c["x"], c["y"],
                                          f"interceptor\nwas at x={oc['x']}",
                                          f"interceptor\nnow at x={c['x']}"))
        variants.append(("Counter\nDirection ↺", dir_task, changes))

    return original, variants


def create_comparison_figure(original: dict, variants: list, ch_id: str, outdir: Path):
    """
    Create the multi-panel comparison figure.
    
    Layout:
      ┌──────────────┬─────────────────────┐
      │  Variant 1   │                     │
      │  (small)     │     ORIGINAL        │
      ├──────────────┤     (large)         │
      │  Variant 2   │                     │
      │  (small)     │                     │
      ├──────────────┤                     │
      │  Variant 3   │                     │
      │  (small)     │                     │
      └──────────────┴─────────────────────┘
    """
    n_vars = len(variants)
    if n_vars == 0:
        return

    # Figure dimensions
    fig_w = 24
    fig_h = max(7 * n_vars, 16)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Grid: 2 columns, N rows. Right column spans all rows.
    gs = fig.add_gridspec(
        n_vars, 2,
        width_ratios=[1, 1.8],
        hspace=0.15,
        wspace=0.08,
        left=0.04, right=0.96,
        top=0.94, bottom=0.04,
    )

    # Right: original (spans all rows)
    ax_orig = fig.add_subplot(gs[:, 1])
    draw_board_in_axes(ax_orig, original, show_title=True)

    # Substitution type badge on original
    ax_orig.text(
        FIG_W / 2, FIG_H + 0.8,
        f"Original: {original.get('title', '')}",
        ha="center", va="bottom",
        fontsize=12, fontweight="bold",
        color="#1f77b4",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f0fe", edgecolor="#1f77b4", linewidth=1.5),
    )

    # Left: variants
    var_colors = ["#2ca02c", "#d62728", "#ff7f0e"]
    for i, (label, task, changes) in enumerate(variants):
        ax_var = fig.add_subplot(gs[i, 0])
        draw_board_in_axes(ax_var, task, show_title=True, highlight_changes=changes)

        # Colored left border
        color = var_colors[i % len(var_colors)]
        border = FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.02",
            transform=ax_var.transAxes,
            facecolor="none",
            edgecolor=color,
            linewidth=3,
            zorder=30,
        )
        ax_var.add_patch(border)

        # Variant type label (top-left badge)
        ax_var.text(
            0.05, 0.93,
            f"Variant: {label}",
            transform=ax_var.transAxes,
            fontsize=9, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="none"),
            zorder=31,
            va="top",
        )

        # Change count badge
        n_changes = len(changes) if changes else 0
        if n_changes > 0:
            ax_var.text(
                0.05, 0.82,
                f"{n_changes} component\n{'changes' if n_changes > 1 else 'change'}",
                transform=ax_var.transAxes,
                fontsize=7,
                color="#555",
                va="top",
                zorder=31,
            )

    # Title
    ch_num = ch_id.replace("ch", "").replace("-", "")
    fig.suptitle(
        f"Mechanism Substitution — Challenge {ch_num}: {original.get('title', '')}",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    # Footer
    fig.text(
        0.5, 0.01,
        "Dashed red outlines = substituted components. Each variant preserves the original objective "
        "while changing the physical implementation.",
        ha="center", fontsize=9, color="#888", style="italic",
    )

    outpath = outdir / f"mechsub_comparison_{ch_id}.png"
    fig.savefig(outpath, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return outpath


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Showcase challenges — ch16 is the best (has all 3 substitution types)
    showcase = ["ch16"]

    for ch_id in showcase:
        print(f"Building comparison for {ch_id}...")
        try:
            original, variants = build_comparison(ch_id)
            outpath = create_comparison_figure(original, variants, ch_id, OUTDIR)
            print(f"  ✓ Saved: {outpath}")
        except Exception as e:
            print(f"  ✗ {ch_id}: {e}", file=sys.stderr)

    # Also try ch14 (single bit substitution — simpler)
    for ch_id in ["ch14"]:
        print(f"Building comparison for {ch_id} (single-bit)...")
        try:
            original, variants = build_comparison(ch_id)
            if variants:
                outpath = create_comparison_figure(original, variants, ch_id, OUTDIR)
                print(f"  ✓ Saved: {outpath}")
            else:
                print(f"  ⊘ No variants found for {ch_id}")
        except Exception as e:
            print(f"  ✗ {ch_id}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
