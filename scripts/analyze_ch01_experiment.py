#!/usr/bin/env python3
"""
ch01 experiment analysis: heatmap of model performance by component position.
"""
import json, sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tt_bench.simulator.renderer import (
    COLOURS, BOARD_W, BOARD_H, MARGIN_TOP, MARGIN_BOTTOM, MARGIN_SIDES, FIG_W, FIG_H,
    draw_peg_grid, draw_board_frame, draw_component, draw_hopper, draw_catcher,
)

OUTDIR = REPO / "data" / "visualizations"

# ── Load experiment data ──
def load_results():
    base = REPO / "benchmark_results/ch01_experiment/2026-07-06T074028"
    reports = sorted(base.rglob("benchmark_*.json"))
    tasks = []
    for rp in reports:
        with open(rp) as f:
            data = json.load(f)
        for r in data.get("results", []):
            sol = r.get("expected", {}).get("solution", {})
            placed = sol.get("placed_components", [])
            tasks.append({
                "task_id": r.get("task_id"),
                "success": r.get("success"),
                "component_score": r.get("component_score", 0),
                "tool_calls": r.get("metrics", {}).get("tool_calls_count", 0),
                "turns": r.get("metrics", {}).get("turns", 0),
                "x": placed[0]["x"] if placed else None,
                "y": placed[0]["y"] if placed else None,
                "type": placed[0]["type"] if placed else None,
            })
    return tasks


def plot_position_analysis(tasks: list, outdir: Path):
    fig = plt.figure(figsize=(20, 9), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.25,
                          left=0.05, right=0.95, top=0.90, bottom=0.12)

    # ── LEFT: Board heatmap ──
    ax_board = fig.add_subplot(gs[0, 0])
    ax_board.set_xlim(0, FIG_W)
    ax_board.set_ylim(0, FIG_H)
    ax_board.set_aspect("equal")
    ax_board.axis("off")
    ax_board.set_facecolor(COLOURS["bg"])

    # Draw the ch01 board (base 1comp version)
    ch01_path = REPO / "data/tasks/challenges_1comp/tt-official-ch01-1comp.json"
    with open(ch01_path) as f:
        ch01 = json.load(f)

    draw_peg_grid(ax_board)
    draw_board_frame(ax_board, title="ch01 — Component Position Analysis", subtitle="")

    board_data = ch01["board"]
    for side, color in [("blue", "blue"), ("red", "red")]:
        hp = board_data["ball_hoppers"].get(side, {})
        if hp.get("count", 0) > 0:
            draw_hopper(ax_board, hp["x"], "B" if side == "blue" else "R",
                       hp["count"], color, hp.get("y", -1))
    for side in ("left", "right"):
        tlev = board_data["trigger_levers"].get(side, {})
        if tlev:
            draw_catcher(ax_board, tlev["x"], "blue" if side == "left" else "red")

    for c in board_data["fixed_components"]:
        draw_component(ax_board, c, COLOURS["fixed"], zorder=5)

    # Build score grid per (x,y) — average across base+insight
    cell_scores = {}
    cell_calls = {}
    for t in tasks:
        if t["x"] is not None and t["y"] is not None:
            key = (t["x"], t["y"])
            cell_scores.setdefault(key, []).append(t["component_score"])
            cell_calls.setdefault(key, []).append(t["tool_calls"])

    # Overlay heatmap
    _CELL = 1.0
    for (x, y), scores in cell_scores.items():
        avg_score = np.mean(scores)
        avg_calls = np.mean(cell_calls[(x, y)])
        cx = MARGIN_SIDES + x * _CELL
        cy = MARGIN_BOTTOM + (BOARD_H - 1 - y) * _CELL

        # Color: green=perfect, yellow=ok, red=poor
        color = plt.colormaps["RdYlGn"](avg_score)
        rect = mpatches.Rectangle((cx - 0.48, cy - 0.48), 0.96, 0.96,
                             facecolor=color, edgecolor="#333",
                             linewidth=1.5, alpha=0.7, zorder=10)
        ax_board.add_patch(rect)

        # Show score
        ax_board.text(cx, cy, f"{avg_score:.2f}\n{avg_calls:.0f}c",
                     ha="center", va="center", fontsize=6.5,
                     fontweight="bold", color="black", zorder=11)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.colormaps["RdYlGn"], norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_board, shrink=0.5, aspect=20,
                        location="bottom", pad=0.08)
    cbar.set_label("Avg Component Score (0=poor, 1=perfect)", fontsize=9)

    # ── RIGHT: Bar chart — tool calls per zone ──
    ax_bar = fig.add_subplot(gs[0, 1])

    zones_data = {"Top\n(y=0-3)": [], "Middle\n(y=4-6)": [], "Bottom\n(y=7-9)": []}
    for t in tasks:
        y = t.get("y", -1)
        if y is None: continue
        if y <= 3: zones_data["Top\n(y=0-3)"].append(t)
        elif y <= 6: zones_data["Middle\n(y=4-6)"].append(t)
        else: zones_data["Bottom\n(y=7-9)"].append(t)

    zone_names = list(zones_data.keys())
    avg_calls = [np.mean([t["tool_calls"] for t in zones_data[z]]) if zones_data[z] else 0 for z in zone_names]
    avg_scores = [np.mean([t["component_score"] for t in zones_data[z]]) if zones_data[z] else 0 for z in zone_names]
    counts = [len(zones_data[z]) for z in zone_names]

    x = np.arange(len(zone_names))
    width = 0.35

    # Bar chart: tool calls
    bars1 = ax_bar.bar(x - width/2, avg_calls, width, label="Avg Tool Calls",
                       color="#1f77b4", edgecolor="white")
    ax_bar.set_ylabel("Avg Tool Calls", color="#1f77b4", fontsize=12)
    ax_bar.tick_params(axis="y", labelcolor="#1f77b4")

    # Overlay score line
    ax_score = ax_bar.twinx()
    bars2 = ax_score.bar(x + width/2, avg_scores, width, label="Avg Component Score",
                         color="#2ca02c", edgecolor="white", alpha=0.7)
    ax_score.set_ylabel("Avg Component Score", color="#2ca02c", fontsize=12)
    ax_score.tick_params(axis="y", labelcolor="#2ca02c")
    ax_score.set_ylim(0, 1.15)

    # Labels on bars
    for bar, calls in zip(bars1, avg_calls):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f"{calls:.1f}", ha="center", fontsize=10, fontweight="bold", color="#1f77b4")
    for bar, score in zip(bars2, avg_scores):
        ax_score.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{score:.2f}", ha="center", fontsize=10, fontweight="bold", color="#2ca02c")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(zone_names, fontsize=11)
    ax_bar.set_title("Performance by Board Zone", fontsize=14, fontweight="bold")

    # Combined legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label=f"Tool Calls (n={len(tasks)})"),
        Patch(facecolor="#2ca02c", label="Component Score"),
    ]
    ax_bar.legend(handles=legend_elements, loc="upper left", fontsize=9)

    # Individual task dots
    for t in tasks:
        y = t.get("y", -1)
        if y is None: continue
        if y <= 3: zi = 0
        elif y <= 6: zi = 1
        else: zi = 2
        ax_bar.scatter(zi - width/2 + np.random.uniform(-0.05, 0.05),
                      t["tool_calls"], alpha=0.4, color="#1f77b4", s=30, zorder=5)

    # ── Title & footer ──
    fig.suptitle(
        f"ch01 Position Sensitivity — qwen3-coder-30b-a3b-instruct (20 tasks)",
        fontsize=16, fontweight="bold", y=0.96,
    )
    fig.text(0.5, 0.04,
             "Each cell shows avg component score + avg tool calls (base + insight variants). "
             "Green = perfect, Red = failed. Top zone (y=0-3) shows significantly worse performance.",
             ha="center", fontsize=9, color="#888", style="italic")

    outpath = outdir / "ch01_position_analysis.png"
    fig.savefig(outpath, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {outpath}")
    return outpath


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Loading experiment results...")
    tasks = load_results()
    print(f"  → {len(tasks)} tasks")

    plot_position_analysis(tasks, OUTDIR)


if __name__ == "__main__":
    main()
