#!/usr/bin/env python3
"""
Variant type comparison visualization.

Shows a base 1-component challenge alongside its insight, T1, T2, and G1 variants.
Highlights changed components with colored outlines and annotations.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tt_bench.simulator.renderer import (
    COLOURS, BOARD_W, BOARD_H,
    MARGIN_TOP, MARGIN_BOTTOM, MARGIN_SIDES, FIG_W, FIG_H,
    draw_peg_grid, draw_board_frame, draw_component,
    draw_hopper, draw_catcher, draw_legend,
)

OUTDIR = REPO / "data" / "visualizations"
TASKS = REPO / "data" / "tasks"

# ── Variant type colours ──
VT_BORDER = {
    "insight":      "#9467bd",   # purple
    "unsolvable_t1": "#d62728",  # red
    "unsolvable_t2": "#e377c2",  # pink
    "unsolvable_g1": "#ff7f0e",  # orange
}
VT_LABEL = {
    "insight":      "Insight\n(+distractor)",
    "unsolvable_t1": "Unsolvable T1\n(same-cat swap)",
    "unsolvable_t2": "Unsolvable T2\n(diff-cat swap)",
    "unsolvable_g1": "Unsolvable G1\n(N+1 gaps)",
}


def load_task(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def draw_board_in_axes(ax, task: dict, highlight_cells: list | None = None,
                       highlight_color: str = "#e74c3c"):
    """Draw board with optional cell highlights."""
    board_data = task["board"]
    hoppers = board_data.get("ball_hoppers", {})
    tlevs = board_data.get("trigger_levers", {})
    fixed_comps = board_data.get("fixed_components", [])
    sol = task.get("solution", {})
    placed_comps = sol.get("placed_components", [])
    avail = task.get("available_parts", {})

    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(COLOURS["bg"])

    draw_peg_grid(ax)
    draw_board_frame(ax, title=task.get("title", ""), subtitle="")

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

    # Build set of fixed positions for highlighting
    fixed_positions = {(c["x"], c["y"]) for c in fixed_comps}

    # Fixed components
    for comp in fixed_comps:
        draw_component(ax, comp, COLOURS["fixed"], zorder=5)

    # Placed components
    for comp in placed_comps:
        draw_component(ax, comp, COLOURS["placed"], zorder=6)

    # Legend
    draw_legend(ax, avail, zorder=9)

    # Highlight cells
    if highlight_cells:
        _CELL = 1.0
        for x, y, label in highlight_cells:
            cx = MARGIN_SIDES + x * _CELL
            cy = MARGIN_BOTTOM + (BOARD_H - 1 - y) * _CELL
            rect = FancyBboxPatch(
                (cx - 0.45, cy - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.05",
                facecolor="none",
                edgecolor=highlight_color,
                linewidth=2.5,
                linestyle="--",
                zorder=20,
            )
            ax.add_patch(rect)
            if label:
                ax.annotate(
                    label, xy=(cx, cy - 0.55),
                    fontsize=5.5, color=highlight_color,
                    ha="center", va="top", zorder=21, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              edgecolor="none", alpha=0.85),
                )

    # Objective
    obj = task.get("objective", "")
    if obj:
        ax.text(FIG_W - 0.2, 0.12, obj, ha="right", va="bottom",
                fontsize=6, color="#333333", zorder=9, style="italic")


def find_variant(ch_id: str, comp_count: int, vt: str) -> Path | None:
    """Find a canonical variant file (same board, different solution/parts)."""
    base = TASKS / f"challenges_{comp_count}comp"

    if vt == "insight":
        return base / "insight" / \
               f"tt-official-{ch_id}-{comp_count}comp_insight.json"
    elif vt.startswith("unsolvable"):
        subtype = vt.replace("unsolvable_", "")
        return base / "unsolvable" / \
               f"tt-official-{ch_id}-{comp_count}comp_unsolvable_{subtype}.json"

    return None


def build_highlights(base_task: dict, variant_task: dict, vt: str) -> list:
    """
    Compare base vs variant and return highlight annotations.
    Each highlight: (x, y, label)
    """
    highlights = []
    base_fixed = base_task["board"]["fixed_components"]
    var_fixed = variant_task["board"]["fixed_components"]
    base_sol = base_task.get("solution", {}).get("placed_components", [])
    var_sol = variant_task.get("solution", {}).get("placed_components", [])

    base_fixed_map = {(c["x"], c["y"]): c for c in base_fixed}
    var_fixed_map = {(c["x"], c["y"]): c for c in var_fixed}

    if vt == "insight":
        # Board and solution are identical to base.
        # The difference is only in available_parts (distractor added).
        # The legend already shows the extra part; add a callout.
        return []
    elif vt in ("unsolvable_t1", "unsolvable_t2"):
        # Solution type changed, available_parts changed
        for c in var_sol:
            orig = next((bc for bc in base_sol
                        if bc["x"] == c["x"] and bc["y"] == c["y"]), None)
            if orig and orig["type"] != c["type"]:
                highlights.append((c["x"], c["y"],
                                 f"was: {orig['type']}\nnow: {c['type']}"))
    elif vt in ("unsolvable_g1", "unsolvable_g2"):
        # Components removed from fixed
        for bc in base_fixed:
            if (bc["x"], bc["y"]) not in var_fixed_map:
                highlights.append((bc["x"], bc["y"], f"removed:\n{bc['type']}"))

    return highlights


def create_showcase(ch_id: str, comp_count: int = 1, var_num: int = 1):
    """
    Create a 5-panel figure: base (left, large) + 4 variants (2×2 right).
    """
    # Load base challenge (original, not the insight variant)
    base_path = TASKS / f"challenges_{comp_count}comp" / \
                f"tt-official-{ch_id}-{comp_count}comp.json"
    if not base_path.exists():
        print(f"  Base not found: {base_path}")
        return None
    base = load_task(base_path)

    # Load variants
    variant_types = ["insight", "unsolvable_t1", "unsolvable_t2", "unsolvable_g1"]
    variants = []
    for vt in variant_types:
        vpath = find_variant(ch_id, comp_count, vt)
        if vpath:
            variants.append((vt, load_task(vpath)))
        else:
            print(f"  Variant {vt} not found for {ch_id}")

    if not variants:
        print(f"  No variants found for {ch_id}")
        return None

    # Build figure
    n_vars = len(variants)
    fig_w = 28
    fig_h = 18
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.6, 1, 1],
        height_ratios=[1, 1],
        hspace=0.12, wspace=0.06,
        left=0.03, right=0.97, top=0.93, bottom=0.04,
    )

    # ── Left panel: BASE challenge (spans both rows) ──
    ax_base = fig.add_subplot(gs[:, 0])
    draw_board_in_axes(ax_base, base)

    # Base badge
    ax_base.text(
        FIG_W / 2, FIG_H + 0.6,
        f"Base: {base.get('title', '')}\n({comp_count} component to place)",
        ha="center", va="bottom", fontsize=11, fontweight="bold",
        color="#1f77b4",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f0fe",
                  edgecolor="#1f77b4", linewidth=2),
    )

    # ── Right panels: 2×2 grid of variants ──
    positions = [(0, 1), (0, 2), (1, 1), (1, 2)]
    for i, (vt, vtask) in enumerate(variants):
        row, col = positions[i]
        ax_var = fig.add_subplot(gs[row, col])

        highlights = build_highlights(base, vtask, vt)
        color = VT_BORDER.get(vt, "#999")
        draw_board_in_axes(ax_var, vtask, highlight_cells=highlights,
                          highlight_color=color)

        # Colored border
        border = FancyBboxPatch(
            (0.01, 0.01), 0.98, 0.98,
            boxstyle="round,pad=0.02",
            transform=ax_var.transAxes,
            facecolor="none", edgecolor=color, linewidth=3, zorder=30,
        )
        ax_var.add_patch(border)

        # Badge — top-left, inside board margin
        label = VT_LABEL.get(vt, vt)
        if label:
            ax_var.text(
                0.04, 0.97, label,
                transform=ax_var.transAxes, fontsize=8, fontweight="bold",
                color="white", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="none"),
                zorder=31,
            )

        # Insight variant: annotate that the distractor is in available_parts
        if vt == "insight":
            ax_var.text(
                FIG_W - 0.3, FIG_H / 2,
                "Distractor\nin available\nparts",
                ha="right", va="center",
                fontsize=6, color=color, fontweight="bold", style="italic",
                zorder=31,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=color, linewidth=1.2, alpha=0.92),
            )


    # ── Title ──
    ch_num = ch_id.replace("ch", "")
    fig.suptitle(
        f"Variant Decomposition — Challenge {ch_num}: {base.get('title', '')}",
        fontsize=18, fontweight="bold", y=0.97,
    )

    # Footer
    fig.text(
        0.5, 0.01,
        "Dashed outlines = changed components. Insight adds a distractor part (same board, harder choice). "
        "T1/T2 swap the solution component type (unsolvable). G1 removes extra fixed components (unsolvable).",
        ha="center", fontsize=8.5, color="#888", style="italic",
    )

    outpath = OUTDIR / f"variant_comparison_{ch_id}_{comp_count}comp.png"
    fig.savefig(outpath, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return outpath


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Showcase challenges — Tier 1-2, different mechanical types
    showcases = [
        ("ch01", 1, 1),   # Gravity — pure ramp routing (1 comp)
        ("ch06", 1, 1),   # Total Internal Reflection — crossover routing (1 comp)
        ("ch08", 1, 1),   # Depolarization — bit alternator (1 comp)
        ("ch01", 2, 1),   # Gravity — same board, 2 components to place (2 comp)
    ]

    for ch_id, comp_count, var_num in showcases:
        print(f"Building variant comparison for {ch_id} ({comp_count}comp)...")
        try:
            outpath = create_showcase(ch_id, comp_count, var_num)
            if outpath:
                print(f"  ✓ Saved: {outpath.name}")
        except Exception as e:
            import traceback
            print(f"  ✗ {ch_id}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
