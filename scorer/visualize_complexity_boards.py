#!/usr/bin/env python3
"""Generate board visualizations with complexity metrics as a legend panel.

Produces publication-quality two-panel figures: board rendering on the
left, complexity metrics table on the right.  One image per challenge,
plus an optional contact sheet of all boards sorted by tier.

Usage:
    PYTHONPATH=simulator uv run python scorer/visualize_complexity_boards.py \
        --challenges-dir tasks/official/challenges/json \
        --index tasks/official/INDEX.json \
        --output-dir experiments/complexity_boards

    # Single challenge
    PYTHONPATH=simulator uv run python scorer/visualize_complexity_boards.py \
        --challenges-dir tasks/official/challenges/json \
        --task tt-official-ch30 \
        --output-dir experiments/complexity_boards
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Path setup
sys.path.insert(0, str(Path(__file__).parent.parent / "simulator"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from board_renderer import render_board
from scorer.complexity_metrics import compute_all_metrics

# ---------------------------------------------------------------------------
# Colour palette — academic green, consistent with thesis table
# ---------------------------------------------------------------------------
COLOURS = {
    "bg": "#FAFAF7",
    "panel_bg": "#FFFFFF",
    "text_primary": "#1a2e1a",
    "text_secondary": "#3d5c3d",
    "text_muted": "#6b8e6b",
    "border": "#c8dcc8",
    "accent": "#2e7d32",
    "accent_light": "#e8f5e9",
    "metric_high": "#2e7d32",
    "metric_mid": "#f9a825",
    "metric_low": "#6b8e6b",
    "divider": "#d8e8d8",
    "tier_badge_1": "#e8f5e9",
    "tier_badge_2": "#c8e6c9",
    "tier_badge_3": "#fff9c4",
    "tier_badge_4": "#ffccbc",
    "tier_text": "#1a2e1a",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_board_to_image(task: dict, transparent: bool = False) -> np.ndarray:
    """Render a board using the existing renderer, return as RGBA numpy array."""
    buf = io.BytesIO()
    task_copy = dict(task)
    # Only render the start state (fixed components, no solution)
    fig = render_board(task_copy, state="start", show_title=True)
    bg = "none" if transparent else COLOURS["bg"]
    fig.savefig(buf, dpi=150, bbox_inches="tight", facecolor=bg, transparent=transparent)
    plt.close(fig)
    buf.seek(0)
    img = plt.imread(buf)
    buf.close()
    return img


def _metric_bar(value: float, vmin: float = 0.0, vmax: float = 1.0) -> Tuple[float, str]:
    """Return (normalized_value, color) for a metric bar."""
    if value is None:
        return (0.0, COLOURS["text_muted"])
    norm = max(0.0, min(1.0, (value - vmin) / max(vmax - vmin, 0.001)))
    if norm < 0.33:
        return (norm, COLOURS["metric_low"])
    elif norm < 0.67:
        return (norm, COLOURS["metric_mid"])
    else:
        return (norm, COLOURS["metric_high"])


def _tier_color(tier) -> str:
    return COLOURS.get(f"tier_badge_{tier}", COLOURS["tier_badge_1"])


def _load_index(index_path: Path) -> Dict[str, Dict[str, Any]]:
    if not index_path.exists():
        return {}
    with open(index_path) as f:
        data = json.load(f)
    return {e["task_id"]: e for e in data.get("tasks", [])}


# ---------------------------------------------------------------------------
# Main visualization function
# ---------------------------------------------------------------------------


def visualize_board_with_complexity(
    task: dict,
    task_id: str,
    metrics: Dict[str, float],
    tier: Optional[int] = None,
    title: str = "",
    transparent: bool = False,
) -> plt.Figure:
    """Create a two-panel figure: board (left) + complexity metrics (right).

    Parameters
    ----------
    task : dict
        Challenge task dictionary (with 'board', 'objective', etc.)
    task_id : str
        Identifier for the challenge (e.g. "tt-official-ch30")
    metrics : dict
        Complexity metrics dict from compute_all_metrics()
    tier : int or None
        Challenge tier (1-4)
    transparent : bool
        If True, render with transparent background (no fill behind board or panel)

    Returns
    -------
    matplotlib.figure.Figure
    """
    board_img = _render_board_to_image(task, transparent)

    # Figure layout: left panel (board) + right panel (metrics)
    fig_bg = "none" if transparent else COLOURS["bg"]
    fig = plt.figure(figsize=(14, 7.5), facecolor=fig_bg)

    # GridSpec: left=board, right=metrics panel
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 0.7],
                          wspace=0.04, left=0.02, right=0.98,
                          top=0.94, bottom=0.04)

    # --- Left panel: Board image ---
    ax_board = fig.add_subplot(gs[0, 0])
    ax_board.imshow(board_img)
    ax_board.axis("off")

    # --- Right panel: Complexity metrics ---
    ax_metrics = fig.add_subplot(gs[0, 1])
    ax_metrics.set_xlim(0, 10)
    ax_metrics.set_ylim(0, 10)
    ax_metrics.axis("off")
    panel_bg = "none" if transparent else COLOURS["panel_bg"]
    ax_metrics.set_facecolor(panel_bg)

    # Panel border
    border = FancyBboxPatch(
        (0.1, 0.1), 9.8, 9.8,
        boxstyle="round,pad=0.15", facecolor=panel_bg,
        edgecolor=COLOURS["border"], linewidth=1.2, zorder=0,
    )
    ax_metrics.add_patch(border)

    y = 9.4
    x_left = 0.8

    # Header
    ax_metrics.text(x_left, y, "Complexity Metrics",
                    fontsize=13, fontweight="bold",
                    color=COLOURS["text_primary"], va="top", zorder=5)
    y -= 0.55

    # Tier badge
    if tier is not None:
        badge = FancyBboxPatch(
            (x_left, y - 0.45), 3.0, 0.55,
            boxstyle="round,pad=0.1", facecolor=_tier_color(tier),
            edgecolor="none", zorder=3,
        )
        ax_metrics.add_patch(badge)
        ax_metrics.text(x_left + 1.5, y - 0.17, f"Tier {tier}",
                        fontsize=10, fontweight="bold",
                        color=COLOURS["text_primary"], ha="center", va="center",
                        zorder=4)
        y -= 0.75

    # Divider
    ax_metrics.plot([x_left, 9.2], [y, y], color=COLOURS["divider"],
                    lw=0.8, zorder=2)
    y -= 0.35

    y -= 0.30

    bici_sub_metrics = [
        ("SCR", "scr", 0.0, 1.0, "(bits + 2×gear_bits) / total — statefulness"),
        ("CTD", "ctd", 0.0, 1.0, "unique types / 8 — physics-rule diversity"),
        ("GCC", "gcc", 0.0, 1.0, "gear groups weighted by complexity"),
        ("RPCC", "rpcc", 0.0, 1.0, "crossovers / cells — spatial routing"),
        ("IBR", "ibr", 0.0, 1.0, "interceptors / max(I+S, 1) — control flow"),
    ]

    # Composite + other core metrics
    composite_metrics = [
        ("BICI", "bici", 0.0, 1.0,
         "Board Input Complexity Index — weighted composite", True),
        ("PSDE", "psde", 0.0, 1.0,
         "Program Synthesis Difficulty Estimate — type × load", False),
        ("K̃", "k_approx", 0.0, 1.0,
         "Kolmogorov approx — BDM algorithmic complexity", False),
    ]

    bar_left = x_left + 3.0
    bar_width = 5.3
    row_h = 0.72
    label_fontsize = 8.0
    value_fontsize = 9.0

    # ---- Render BICI sub-metrics ----
    for label, key, vmin, vmax, tooltip in bici_sub_metrics:
        val = metrics.get(key)
        if val is None or y < 0.8:
            continue

        # Indented label
        ax_metrics.text(x_left, y, f"  {label}",
                        fontsize=label_fontsize, fontweight="bold",
                        color=COLOURS["text_secondary"], va="center", zorder=5)

        # Mini bar background
        bar_bg = FancyBboxPatch(
            (bar_left, y - 0.14), bar_width, 0.28,
            boxstyle="round,pad=0.04", facecolor="#E8ECE8",
            edgecolor="none", zorder=2,
        )
        ax_metrics.add_patch(bar_bg)

        # Bar fill
        norm_val, bar_color = _metric_bar(val, vmin, vmax)
        if norm_val > 0.005:
            bar_fill = FancyBboxPatch(
                (bar_left, y - 0.14), bar_width * norm_val, 0.28,
                boxstyle="round,pad=0.04", facecolor=bar_color,
                edgecolor="none", alpha=0.85, zorder=3,
            )
            ax_metrics.add_patch(bar_fill)

        # Value
        val_str = f"{val:.3f}" if isinstance(val, float) else str(int(val))
        ax_metrics.text(bar_left + bar_width + 0.12, y, val_str,
                        fontsize=value_fontsize, fontweight="bold",
                        color=COLOURS["text_primary"], va="center", zorder=5)

        y -= row_h

    # ---- Separator before composite ----
    y -= 0.10
    ax_metrics.plot([x_left, 9.2], [y, y], color=COLOURS["divider"],
                    lw=1.2, zorder=2)
    y -= 0.40

    # ---- Render composite + other core metrics ----
    for label, key, vmin, vmax, tooltip, is_composite in composite_metrics:
        val = metrics.get(key)
        if val is None or y < 0.8:
            continue

        font_wt = "bold" if is_composite else "book"
        font_sz = label_fontsize + (0.5 if is_composite else 0.0)

        ax_metrics.text(x_left, y, label,
                        fontsize=font_sz, fontweight=font_wt,
                        color=COLOURS["text_primary"], va="center", zorder=5)

        bar_h = 0.32 if is_composite else 0.28
        bar_bg = FancyBboxPatch(
            (bar_left, y - bar_h/2), bar_width, bar_h,
            boxstyle="round,pad=0.04", facecolor="#E8ECE8",
            edgecolor=COLOURS["accent"] if is_composite else "none",
            linewidth=1.2 if is_composite else 0,
            zorder=2,
        )
        ax_metrics.add_patch(bar_bg)

        norm_val, bar_color = _metric_bar(val, vmin, vmax)
        if norm_val > 0.005:
            bar_fill = FancyBboxPatch(
                (bar_left, y - bar_h/2), bar_width * norm_val, bar_h,
                boxstyle="round,pad=0.04", facecolor=bar_color,
                edgecolor="none", alpha=0.85, zorder=3,
            )
            ax_metrics.add_patch(bar_fill)

        val_str = f"{val:.3f}" if isinstance(val, float) else str(int(val))
        ax_metrics.text(bar_left + bar_width + 0.12, y, val_str,
                        fontsize=value_fontsize + (1 if is_composite else 0),
                        fontweight="bold",
                        color=COLOURS["text_primary"], va="center", zorder=5)

        y -= row_h + (0.10 if is_composite else 0.0)

    return fig


# ---------------------------------------------------------------------------
# Batch generator
# ---------------------------------------------------------------------------


def generate_all(
    challenges_dir: Path,
    index_path: Path,
    output_dir: Path,
    pattern: str = "tt-official-ch*.json",
    single_task: Optional[str] = None,
    transparent: bool = False,
) -> List[Path]:
    """Generate complexity-annotated board images for all (or one) challenges."""
    output_dir.mkdir(parents=True, exist_ok=True)
    index = _load_index(index_path)

    if single_task:
        # Handle with or without .json extension
        stem = single_task.replace(".json", "")
        challenge_files = [challenges_dir / f"{stem}.json"]
    else:
        challenge_files = sorted(challenges_dir.glob(pattern))

    generated: List[Path] = []
    skipped = 0

    for ch_path in challenge_files:
        if not ch_path.exists():
            print(f"  SKIP: {ch_path} not found")
            skipped += 1
            continue

        task_id = ch_path.stem
        meta = index.get(task_id, {})
        tier = meta.get("tier")
        title = meta.get("title", "")

        try:
            # Load task and compute metrics
            with open(ch_path) as f:
                task = json.load(f)
            if ch_path.stem.startswith("tt-official-") and task.get("task_id") != ch_path.stem:
                task["task_id"] = ch_path.stem

            # Build board for metrics (skip solution-placed components)
            import tt_sim
            board = tt_sim.Board.from_task_json(str(ch_path))
            metrics = compute_all_metrics(board, task)
        except Exception as e:
            print(f"  ERROR {task_id}: {e}")
            skipped += 1
            continue

        # Generate figure
        try:
            fig = visualize_board_with_complexity(
                task, task_id, metrics, tier=tier, title=title,
                transparent=transparent,
            )
            out_path = output_dir / f"{task_id}_complexity.png"
            save_bg = "none" if transparent else COLOURS["bg"]
            fig.savefig(out_path, dpi=200, bbox_inches="tight",
                        facecolor=save_bg, edgecolor="none",
                        transparent=transparent)
            plt.close(fig)
            generated.append(out_path)
            print(f"  OK  {task_id:30s}  tier={tier or '?'}  BICI={metrics.get('bici', 0):.4f}")
        except Exception as e:
            print(f"  RENDER ERROR {task_id}: {e}")
            skipped += 1

    print(f"\nGenerated {len(generated)} images, skipped {skipped}, output → {output_dir}")
    return generated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate board visualizations with complexity metrics"
    )
    parser.add_argument(
        "--challenges-dir", type=Path,
        default=Path("tasks/official/challenges/json"),
    )
    parser.add_argument(
        "--index", type=Path,
        default=Path("tasks/official/INDEX.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("experiments/complexity_boards"),
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Render a single task (e.g. tt-official-ch30)",
    )
    parser.add_argument(
        "--pattern", default="tt-official-ch*.json",
    )
    parser.add_argument(
        "--transparent", action="store_true",
        help="Render with transparent background",
    )

    args = parser.parse_args()

    generate_all(
        challenges_dir=args.challenges_dir,
        index_path=args.index,
        output_dir=args.output_dir,
        pattern=args.pattern,
        single_task=args.task,
        transparent=args.transparent,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
