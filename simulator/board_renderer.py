"""
board_renderer.py
Renders Turing Tumble board states as PNG images and MP4 animations.

Usage:
    python board_renderer.py                      # render all official challenges
    python board_renderer.py --task tt-official-ch01  # render one task
    python board_renderer.py --task tt-official-ch01 --animate --run blue,red,blue
"""

import json
import os
import math
import argparse
import glob
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Arc, Wedge, Circle, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.image import imread
import numpy as np

from tt_sim import Board, Component, Side, build_gear_connections

# ---------------------------------------------------------------------------
# Load component assets
# ---------------------------------------------------------------------------
ASSETS_DIR = Path(__file__).parent.parent / "assets" / "components"

COMPONENT_IMAGES = {}
if ASSETS_DIR.exists():
    for img_file in ASSETS_DIR.glob("*.png"):
        COMPONENT_IMAGES[img_file.stem] = imread(str(img_file))
else:
    print(f"Warning: Assets directory not found: {ASSETS_DIR}")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLOURS = {
    "bg": "#F5F0E8",
    "grid_dot": "#C8BFA8",
    "grid_line": "#E0D8C8",
    "fixed": "#1A1A2E",
    "placed": "#2563EB",
    "blue_hopper": "#1E88E5",
    "red_hopper": "#E53935",
    "blue_trigger": "#0D47A1",
    "red_trigger": "#B71C1C",
    "frame": "#3D2B1F",
    "interceptor": "#6B21A8",
    "gear": "#D97706",
    "bit0": "#DC2626",
    "bit1": "#16A34A",
    "crossover": "#0891B2",
}

CELL = 1.0
BOARD_W = 11
BOARD_H = 11
MARGIN_TOP = 4.5
MARGIN_BOTTOM = 3.18  # tuned so 150 DPI MP4 height is even (x264 req)
MARGIN_SIDES = 1.5

FIG_W = BOARD_W + 2 * MARGIN_SIDES
FIG_H = BOARD_H + MARGIN_TOP + MARGIN_BOTTOM

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _ax_coord(gx, gy):
    """Grid (x,y) → matplotlib axes (x,y). y=0 is top, grows downward."""
    ax_x = MARGIN_SIDES + gx * CELL
    ax_y = MARGIN_BOTTOM + (BOARD_H - 1 - gy) * CELL
    return ax_x, ax_y


def draw_peg_grid(ax):
    """Draw the staggered peg grid (Turing Tumble uses offset rows)."""
    for gy in range(BOARD_H):
        for gx in range(BOARD_W):
            cx, cy = _ax_coord(gx, gy)
            # Stagger even rows by half a cell horizontally (visual only, coords unchanged)
            offset = 0.0  # pegs sit on integer grid positions per the JSON spec
            ax.plot(cx + offset, cy, "o",
                    color=COLOURS["grid_dot"], markersize=4, zorder=1,
                    markeredgewidth=0)


def draw_board_frame(ax, title="", subtitle=""):
    """Draw the outer board frame, title text, and top/bottom rails."""
    # Board inner background
    bx = MARGIN_SIDES - 0.5
    by = MARGIN_BOTTOM - 0.5
    bw = (BOARD_W - 1) + 1.0
    bh = (BOARD_H - 1) + 1.0
    rect = FancyBboxPatch(
        (bx, by), bw, bh,
        boxstyle="round,pad=0.2",
        linewidth=2.5,
        edgecolor=COLOURS["frame"],
        facecolor=COLOURS["bg"],
        zorder=0,
    )
    ax.add_patch(rect)

    # Top rail (hopper channel)
    rail_y_top = MARGIN_BOTTOM + BOARD_H - 0.5 + 0.05
    ax.plot([bx + 0.1, bx + bw - 0.1], [rail_y_top, rail_y_top],
            color=COLOURS["frame"], lw=2, zorder=2)

    # Bottom rail (catcher channel)
    rail_y_bot = MARGIN_BOTTOM - 0.5 + 0.05
    ax.plot([bx + 0.1, bx + bw - 0.1], [rail_y_bot, rail_y_bot],
            color=COLOURS["frame"], lw=2, zorder=2)

    if title:
        ax.text(FIG_W / 2, FIG_H - 0.45,
                title, ha="center", va="top",
                fontsize=12, fontweight="bold",
                color=COLOURS["frame"], zorder=10)
    if subtitle:
        ax.text(FIG_W / 2, FIG_H - 1.05,
                subtitle, ha="center", va="top",
                fontsize=9, color="#555555", style="italic", zorder=10)


# ---------------------------------------------------------------------------
# Component drawing
# ---------------------------------------------------------------------------

def draw_ramp_right(ax, gx, gy, color, zorder=5):
    """Ramp directing ball to lower-right (╲). Shelf + arrow tip."""
    cx, cy = _ax_coord(gx, gy)
    r = 0.38
    # Thick shelf line (the ramp body)
    ax.plot([cx - r, cx + r], [cy + r * 0.7, cy - r * 0.7],
            color=color, lw=4.5, solid_capstyle="round", zorder=zorder)
    # Arrow tip indicating direction (lower-right end)
    ax.annotate("",
                xy=(cx + r + 0.01, cy - r * 0.7 - 0.01),
                xytext=(cx + r - 0.18, cy - r * 0.7 + 0.13),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5, mutation_scale=16),
                zorder=zorder + 1)


def draw_ramp_left(ax, gx, gy, color, zorder=5):
    """Ramp directing ball to lower-left (╱). Shelf + arrow tip."""
    cx, cy = _ax_coord(gx, gy)
    r = 0.38
    ax.plot([cx - r, cx + r], [cy - r * 0.7, cy + r * 0.7],
            color=color, lw=4.5, solid_capstyle="round", zorder=zorder)
    ax.annotate("",
                xy=(cx - r - 0.01, cy - r * 0.7 - 0.01),
                xytext=(cx - r + 0.18, cy - r * 0.7 + 0.13),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5, mutation_scale=16),
                zorder=zorder + 1)


def draw_crossover(ax, gx, gy, color, zorder=5):
    """Crossover: two diagonal lines forming an X with a bridge gap."""
    cx, cy = _ax_coord(gx, gy)
    r = 0.34
    for x0, y0, x1, y1 in [
        (cx - r, cy + r, cx + r, cy - r),
        (cx - r, cy - r, cx + r, cy + r),
    ]:
        ax.plot([x0, x1], [y0, y1], color=color, lw=2.8,
                solid_capstyle="round", zorder=zorder)
    circ = Circle((cx, cy), 0.10, color="white", zorder=zorder + 1)
    ax.add_patch(circ)
    circ2 = Circle((cx, cy), 0.10, color=color, fill=False,
                   linewidth=1.5, zorder=zorder + 2)
    ax.add_patch(circ2)


def draw_bit(ax, gx, gy, state, color_fixed, zorder=5):
    """Bit: diamond body with directional arrow."""
    cx, cy = _ax_coord(gx, gy)
    r = 0.28
    diamond = plt.Polygon(
        [(cx, cy + r), (cx + r, cy), (cx, cy - r), (cx - r, cy)],
        closed=True, facecolor="white", edgecolor=color_fixed,
        linewidth=2, zorder=zorder,
    )
    ax.add_patch(diamond)
    bit_color = COLOURS["bit0"] if state == 0 else COLOURS["bit1"]
    dx = -0.18 if state == 0 else 0.18
    ax.annotate("",
                xy=(cx + dx, cy), xytext=(cx - dx, cy),
                arrowprops=dict(arrowstyle="-|>", color=bit_color,
                                lw=1.6, mutation_scale=10),
                zorder=zorder + 1)


def draw_gear_bit(ax, gx, gy, state, color, zorder=5):
    draw_bit(ax, gx, gy, state, color, zorder)
    cx, cy = _ax_coord(gx, gy)
    n_teeth = 8
    r_inner, r_outer = 0.30, 0.40
    for i in range(n_teeth):
        angle = 2 * math.pi * i / n_teeth
        x0 = cx + r_inner * math.cos(angle)
        y0 = cy + r_inner * math.sin(angle)
        x1 = cx + r_outer * math.cos(angle)
        y1 = cy + r_outer * math.sin(angle)
        ax.plot([x0, x1], [y0, y1], color=COLOURS["gear"],
                lw=2, solid_capstyle="round", zorder=zorder)


def draw_gear(ax, gx, gy, color, zorder=5):
    cx, cy = _ax_coord(gx, gy)
    circ = Circle((cx, cy), 0.28, color="white", zorder=zorder)
    ax.add_patch(circ)
    circ2 = Circle((cx, cy), 0.28, fill=False,
                   edgecolor=COLOURS["gear"], linewidth=2.2, zorder=zorder + 1)
    ax.add_patch(circ2)
    n_teeth = 8
    r_inner, r_outer = 0.28, 0.40
    for i in range(n_teeth):
        angle = 2 * math.pi * i / n_teeth
        x0 = cx + r_inner * math.cos(angle)
        y0 = cy + r_inner * math.sin(angle)
        x1 = cx + r_outer * math.cos(angle)
        y1 = cy + r_outer * math.sin(angle)
        ax.plot([x0, x1], [y0, y1], color=COLOURS["gear"],
                lw=2.2, solid_capstyle="round", zorder=zorder + 1)
    circ3 = Circle((cx, cy), 0.07, color=COLOURS["gear"], zorder=zorder + 2)
    ax.add_patch(circ3)


def draw_interceptor(ax, gx, gy, color, zorder=5):
    cx, cy = _ax_coord(gx, gy)
    r = 0.32
    bucket = mpatches.Arc(
        (cx, cy + 0.05), r * 2, r * 1.6,
        angle=0, theta1=180, theta2=360,
        color=COLOURS["interceptor"], lw=2.8, zorder=zorder,
    )
    ax.add_patch(bucket)
    ax.plot([cx - r, cx - r], [cy + 0.05, cy + r * 0.5],
            color=COLOURS["interceptor"], lw=2.8, zorder=zorder)
    ax.plot([cx + r, cx + r], [cy + 0.05, cy + r * 0.5],
            color=COLOURS["interceptor"], lw=2.8, zorder=zorder)
    ax.plot([cx - 0.14, cx + 0.14], [cy - 0.10, cy + 0.10],
            color=COLOURS["interceptor"], lw=1.8, zorder=zorder + 1)
    ax.plot([cx - 0.14, cx + 0.14], [cy + 0.10, cy - 0.10],
            color=COLOURS["interceptor"], lw=1.8, zorder=zorder + 1)


def draw_trigger(ax, gx, gy, color, zorder=5):
    cx, cy = _ax_coord(gx, gy)
    ax.plot([cx - 0.35, cx + 0.35], [cy, cy],
            color=color, lw=3.0, solid_capstyle="round", zorder=zorder)
    ax.plot([cx, cx], [cy, cy + 0.3],
            color=color, lw=2.2, solid_capstyle="round", zorder=zorder)


# ---------------------------------------------------------------------------
# Hopper and trigger lever
# ---------------------------------------------------------------------------

def draw_hopper(ax, x_grid, label, count, side, y_grid=-1, zorder=8):
    """Draw a ball hopper above the board.

    Hopper x is a SLOT coordinate (gap between pegs), so the tray is drawn
    centred at x + 0.5 in axis units. A vertical drop line runs from the
    tray down to the top rail, showing exactly which column the ball enters.
    y_grid determines the vertical offset above the board (negative = above).
    """
    cx = MARGIN_SIDES + (x_grid) * CELL
    base_y = MARGIN_BOTTOM + (BOARD_H - 1) * CELL
    tray_y = base_y - y_grid * CELL

    col = COLOURS["blue_hopper"] if side == "blue" else COLOURS["red_hopper"]

    # Tray: U-shape
    tray_w = 0.55
    tray_h = 0.30
    ax.plot([cx - tray_w, cx + tray_w], [tray_y, tray_y],
            color=col, lw=3, solid_capstyle="round", zorder=zorder)
    ax.plot([cx - tray_w, cx - tray_w], [tray_y, tray_y - tray_h],
            color=col, lw=2.5, solid_capstyle="round", zorder=zorder)
    ax.plot([cx + tray_w, cx + tray_w], [tray_y, tray_y - tray_h],
            color=col, lw=2.5, solid_capstyle="round", zorder=zorder)

    # Balls in hopper (up to 6 visible in 2 rows of 3)
    n_vis = min(count, 6)
    ball_r = 0.11
    spacing = 0.26
    for i in range(n_vis):
        col_i = i % 3
        row_i = i // 3
        bx = cx - spacing + col_i * spacing
        by = tray_y + ball_r + 0.04 + row_i * (ball_r * 2 + 0.04)
        ball = Circle((bx, by), ball_r, color=col, zorder=zorder + 1)
        ax.add_patch(ball)

    # Count label to the right of tray
    count_str = "×∞" if count >= 999 else f"×{count}"
    ax.text(cx + tray_w + 0.15, tray_y,
            count_str, ha="left", va="center",
            fontsize=8.5, color=col, fontweight="bold", zorder=zorder)

    # Column letter above hopper
    ax.text(cx, tray_y + (ball_r * 2 + 0.04) * 2 + 0.35,
            label, ha="center", va="bottom",
            fontsize=12, color=col, fontweight="bold", zorder=zorder)

    # Drop line: tray bottom → board top rail, centred on the slot
    drop_top = tray_y - tray_h - 0.02
    base_y = MARGIN_BOTTOM + (BOARD_H - 1) * CELL
    drop_bot = base_y + 0.52 - y_grid * CELL
    ax.plot([cx, cx], [drop_bot, drop_top],
            color=col, lw=1.8, alpha=0.55, zorder=zorder - 1)


def draw_catcher(ax, x_grid, side, ball_count=0, active=False, zorder=8):
    """Draw a catcher bucket below the board.

    x_grid is a SLOT coordinate → visual centre at x + 0.5.
    active=True highlights the bucket (thick coloured border + fill).
    ball_count fills the bucket with small ball circles.
    """
    cx = MARGIN_SIDES + x_grid * CELL
    # Bucket sits just below the bottom rail
    base_y  = MARGIN_BOTTOM - 2.00   # bottom of bucket
    mouth_y = MARGIN_BOTTOM - 0.70   # top opening of bucket
    half_w  = 0.70                   # half-width of bucket mouth

    col        = COLOURS["blue_trigger"] if side == "blue" else COLOURS["red_trigger"]
    fill_col   = "#D0E8FF" if side == "blue" else "#FFD0D0"
    label_name = "Left catcher" if side == "blue" else "Right catcher"

    lw_border = 3.5 if active else 1.8
    alpha_fill = 0.55 if active else 0.18

    # Bucket fill (trapezoid approximation via polygon)
    bucket_poly = plt.Polygon(
        [(cx - half_w, mouth_y),
         (cx + half_w, mouth_y),
         (cx + half_w * 0.6, base_y),
         (cx - half_w * 0.6, base_y)],
        closed=True,
        facecolor=fill_col,
        edgecolor="none",
        alpha=alpha_fill,
        zorder=zorder,
    )
    ax.add_patch(bucket_poly)

    # Bucket walls: left side, right side, bottom
    ax.plot([cx - half_w, cx - half_w * 0.6],
            [mouth_y,      base_y],
            color=col, lw=lw_border, solid_capstyle="round", zorder=zorder + 1)
    ax.plot([cx + half_w, cx + half_w * 0.6],
            [mouth_y,      base_y],
            color=col, lw=lw_border, solid_capstyle="round", zorder=zorder + 1)
    ax.plot([cx - half_w * 0.6, cx + half_w * 0.6],
            [base_y,             base_y],
            color=col, lw=lw_border, solid_capstyle="round", zorder=zorder + 1)

    # Drop-channel line from board bottom rail into the bucket mouth
    rail_y = MARGIN_BOTTOM - 0.45
    ax.plot([cx, cx], [rail_y, mouth_y],
            color=col, lw=1.8, alpha=0.6, zorder=zorder)

    # Ball stack inside the bucket
    if ball_count > 0:
        ball_r   = 0.13
        spacing  = ball_r * 2 + 0.04
        max_cols = 3
        for i in range(min(ball_count, 9)):
            col_i = i % max_cols
            row_i = i // max_cols
            bx = cx - spacing + col_i * spacing
            by = base_y + ball_r + 0.06 + row_i * (ball_r * 2 + 0.05)
            ball = Circle((bx, by), ball_r, color=col, zorder=zorder + 2)
            ax.add_patch(ball)

        # Overflow label if more than 9
        if ball_count > 9:
            ax.text(cx, base_y + 0.12, f"+{ball_count - 9}",
                    ha="center", va="bottom", fontsize=7,
                    color=col, fontweight="bold", zorder=zorder + 3)

    # Count badge on the right wall
    count_str = str(ball_count) if ball_count > 0 else "0"
    badge_x = cx + half_w + 0.18
    badge_y = base_y + (mouth_y - base_y) * 0.5
    ax.text(badge_x, badge_y, f"×{count_str}",
            ha="left", va="center", fontsize=9,
            color=col, fontweight="bold", zorder=zorder + 2)

    # Label below bucket
    ax.text(cx, base_y - 0.38,
            label_name, ha="center", va="top",
            fontsize=8, color=col,
            fontweight="bold" if active else "normal",
            zorder=zorder + 1)

    # Active indicator: glowing outline rect
    if active:
        highlight = FancyBboxPatch(
            (cx - half_w - 0.12, base_y - 0.10),
            (half_w + 0.12) * 2,
            mouth_y - base_y + 0.20,
            boxstyle="round,pad=0.08",
            linewidth=2.2,
            edgecolor=col,
            facecolor="none",
            linestyle="--",
            alpha=0.7,
            zorder=zorder,
        )
        ax.add_patch(highlight)


# ---------------------------------------------------------------------------
# Component dispatcher (using PNG assets)
# ---------------------------------------------------------------------------

def draw_component(ax, comp, color, zorder=5):
    gx, gy = comp["x"], comp["y"]
    t = comp["type"]
    cx, cy = _ax_coord(gx, gy)

    img = None
    flip_horizontal = False

    if t == "ramp_right":
        if "ramp" in COMPONENT_IMAGES:
            img = COMPONENT_IMAGES["ramp"]
            flip_horizontal = True
        else:
            draw_ramp_right(ax, gx, gy, color, zorder)
            return
    elif t == "ramp_left":
        if "ramp" in COMPONENT_IMAGES:
            img = COMPONENT_IMAGES["ramp"]
        else:
            draw_ramp_left(ax, gx, gy, color, zorder)
            return
    elif t == "crossover":
        img = COMPONENT_IMAGES.get("crossover")
        if img is None:
            draw_crossover(ax, gx, gy, color, zorder)
            return
    elif t == "bit":
        img = COMPONENT_IMAGES.get("bit")
        if img is None:
            draw_bit(ax, gx, gy, comp.get("state", 0), color, zorder)
            return
    elif t == "gear_bit":
        img = COMPONENT_IMAGES.get("gear_bit")
        if img is None:
            draw_gear_bit(ax, gx, gy, comp.get("state", 0), color, zorder)
            return
    elif t == "gear":
        img = COMPONENT_IMAGES.get("gear")
        if img is None:
            draw_gear(ax, gx, gy, color, zorder)
            return
    elif t == "interceptor":
        img = COMPONENT_IMAGES.get("interceptor")
        if img is None:
            draw_interceptor(ax, gx, gy, color, zorder)
            return
    elif t == "trigger":
        draw_trigger(ax, gx, gy, color, zorder)
        return
    else:
        return

    if img is not None:
        if flip_horizontal:
            img = np.fliplr(img)
        h, w = img.shape[:2]
        scale = CELL / max(h, w) * 0.85
        extent = (
            cx - w * scale / 2,
            cx + w * scale / 2,
            cy - h * scale / 2,
            cy + h * scale / 2,
        )
        ax.imshow(img, extent=extent, zorder=zorder, origin="lower")


# ---------------------------------------------------------------------------
# Available-parts legend
# ---------------------------------------------------------------------------

PART_SYMBOLS = {
    "ramp_right": "╲▶",
    "ramp_left":  "◀╱",
    "crossover":  "✕",
    "bit":        "◆",
    "gear_bit":   "⚙◆",
    "gear":       "⚙",
    "interceptor":"⌣",
    "trigger":    "⌐",
}


def draw_legend(ax, available_parts, zorder=9):
    entries = [(k, v) for k, v in available_parts.items()
               if isinstance(v, int) and v > 0]

    # Place legend in top-right corner (higher above the board)
    legend_x = MARGIN_SIDES + BOARD_W + 0.3
    legend_y_start = MARGIN_BOTTOM + BOARD_H + 3.8

    if not entries:
        ax.text(legend_x, legend_y_start,
                "No available parts",
                ha="right", va="top", fontsize=7.5,
                color="#888888", style="italic", zorder=zorder)
        return

    ax.text(legend_x, legend_y_start,
            "Available parts:",
            ha="right", va="top", fontsize=8,
            fontweight="bold", color=COLOURS["frame"], zorder=zorder)

    for i, (k, v) in enumerate(entries):
        count_str = "∞" if v >= 999 else str(v)
        y_pos = legend_y_start - 0.6 - i * 0.6

        img_key = None
        if k in ["ramp_right", "ramp_left"]:
            img_key = "ramp"
        elif k in COMPONENT_IMAGES:
            img_key = k

        if img_key and img_key in COMPONENT_IMAGES:
            img = COMPONENT_IMAGES[img_key]
            if k == "ramp_right":
                img = np.fliplr(img)
            h, w = img.shape[:2]
            scale = 0.6 / max(h, w) * CELL
            img_x = legend_x - 1.8
            extent = (
                img_x - w * scale / 2,
                img_x + w * scale / 2,
                y_pos - h * scale / 2,
                y_pos + h * scale / 2,
            )
            ax.imshow(img, extent=extent, zorder=zorder, origin="lower")
            name = k.replace('_', ' ')
            ax.text(legend_x, y_pos, f"×{count_str} {name}",
                    ha="right", va="center", fontsize=7, color="#333333", zorder=zorder)
        else:
            sym = PART_SYMBOLS.get(k, "?")
            label = f"×{count_str} {sym}"
            ax.text(legend_x, y_pos, label, ha="right", va="center",
                    fontsize=7.5, color="#333333", zorder=zorder)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_board(task: dict, state: str = "start", output_path: str = None, show_title: bool = True):
    board = task["board"]
    sol = task.get("solution", {})
    hoppers = board.get("ball_hoppers", {})
    tlevs = board.get("trigger_levers", {})

    fixed_comps = board.get("fixed_components", [])
    placed_comps = sol.get("placed_components", []) if state == "solution" else []
    avail = task.get("available_parts", {})

    fig_w_in = 7.0
    fig_h_in = fig_w_in * FIG_H / FIG_W
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(COLOURS["bg"])
    ax.add_patch(plt.Rectangle((0, 0), FIG_W, FIG_H,
                                color=COLOURS["bg"], zorder=0))

    draw_peg_grid(ax)

    task_id = task.get("task_id", "")
    ch_title = task.get("title", "") if show_title else ""
    state_label = ("Starting setup" if state == "start" else "Solution") if show_title else ""
    draw_board_frame(ax,
                     title=ch_title,
                     subtitle=state_label)

    # Hoppers — use actual x from JSON
    blue_h = hoppers.get("blue", {})
    red_h = hoppers.get("red", {})
    if blue_h.get("count", 0) > 0:
        draw_hopper(ax, blue_h["x"], "B", blue_h["count"], "blue", blue_h.get("y", -1))
    if red_h.get("count", 0) > 0:
        draw_hopper(ax, red_h["x"], "R", red_h["count"], "red", red_h.get("y", -1))

    # --- Catcher buckets (bottom) ---
    # Tally ball counts per catcher from final_marble_state (solution view only)
    final_state = sol.get("final_marble_state", []) if state == "solution" else []
    # Determine which catcher each colour feeds into via trigger lever positions.
    # Left lever → blue catcher, right lever → red catcher.
    # final_marble_state lists balls that reached the end; assume they all go
    # to whichever catcher the path terminates at. Count blues vs reds.
    left_count  = sum(1 for b in final_state if b == "blue")
    right_count = sum(1 for b in final_state if b == "red")

    left_t  = tlevs.get("left", {})
    right_t = tlevs.get("right", {})
    if left_t:
        draw_catcher(ax, left_t["x"], "blue",
                     ball_count=left_count,
                     active=(left_count > 0 and state == "solution"))
    if right_t:
        draw_catcher(ax, right_t["x"], "red",
                     ball_count=right_count,
                     active=(right_count > 0 and state == "solution"))

    # Fixed components (dark navy)
    for comp in fixed_comps:
        draw_component(ax, comp, COLOURS["fixed"], zorder=5)

    # Placed / solution components (vivid blue)
    for comp in placed_comps:
        draw_component(ax, comp, COLOURS["placed"], zorder=6)

    # Available parts legend (start view only)
    if state == "start":
        draw_legend(ax, avail, zorder=9)

    # Objective text bottom-right
    obj = task.get("objective", "")
    if obj:
        ax.text(FIG_W - 0.2, 0.12,
                obj, ha="right", va="bottom",
                fontsize=8.5, color="#333333",
                wrap=True, multialignment="right",
                zorder=9, style="italic")

    plt.tight_layout(pad=0.3)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=COLOURS["bg"])
        plt.close(fig)
        return output_path
    return fig


def _parse_sequence_tokens(tokens: list[str]) -> list[Side]:
    """Convert sequence tokens to Side values."""
    sides: list[Side] = []
    for tok in tokens:
        if tok in ("b", "blue"):
            sides.append(Side.BLUE)
        elif tok in ("r", "red"):
            sides.append(Side.RED)
        else:
            raise ValueError(
                f"Invalid marble side '{tok}'. Use blue/red (or b/r)."
            )
    return sides


def _parse_sequence(
    sequence: Optional[str],
    blue_count: int,
    red_count: int,
    task_default: Optional[list[str]] = None,
) -> list[Side]:
    """Parse a user sequence string into Side values.

    If sequence is None, uses task_default first, then falls back to alternating
    blue/red until both hoppers are empty.
    """
    if sequence is None:
        if task_default:
            default_tokens = [
                str(tok).strip().lower() for tok in task_default if str(tok).strip()
            ]
            if default_tokens:
                return _parse_sequence_tokens(default_tokens)

        sides: list[Side] = []
        b, r = blue_count, red_count
        while b > 0 or r > 0:
            if b > 0:
                sides.append(Side.BLUE)
                b -= 1
            if r > 0:
                sides.append(Side.RED)
                r -= 1
        return sides

    tokens = [tok.strip().lower() for tok in sequence.split(",") if tok.strip()]
    return _parse_sequence_tokens(tokens)


def _grid_to_axes(gx: int, gy: int) -> tuple[float, float]:
    """Convert grid coordinates (including y=-1 hopper row) to axis coordinates."""
    vis_y = gy if gy >= 0 else -0.6
    return (
        MARGIN_SIDES + gx * CELL,
        MARGIN_BOTTOM + (BOARD_H - 1 - vis_y) * CELL,
    )


def _build_board_from_task(
    task: dict,
    hopper_entry_mode: str = "column",
) -> tuple[Board, dict[tuple[int, int], str]]:
    """Build a simulator Board from task JSON and return component origin labels."""
    board_data = task.get("board", {})
    hoppers = board_data.get("ball_hoppers", {})
    levers = board_data.get("trigger_levers", {})

    board = Board(
        rows=board_data.get("height", 11),
        cols=board_data.get("width", 11),
        blue_hopper_x=hoppers.get("blue", {}).get("x", 2),
        red_hopper_x=hoppers.get("red", {}).get("x", 8),
        blue_hopper_count=hoppers.get("blue", {}).get("count", 8),
        red_hopper_count=hoppers.get("red", {}).get("count", 8),
        hopper_entry_mode=hopper_entry_mode,
        left_catcher_x=levers.get("left", {}).get("x"),
        right_catcher_x=levers.get("right", {}).get("x"),
    )

    origins: dict[tuple[int, int], str] = {}

    for comp_dict in board_data.get("fixed_components", []):
        comp = Component.from_dict(comp_dict)
        board.place(comp.x, comp.y, comp)
        origins[(comp.x, comp.y)] = "fixed"

    for comp_dict in task.get("solution", {}).get("placed_components", []):
        comp = Component.from_dict(comp_dict)
        board.place(comp.x, comp.y, comp)
        origins[(comp.x, comp.y)] = "placed"

    # Ensure connected gear bits flip together during simulation.
    build_gear_connections(board)

    for pos in board.components:
        origins.setdefault(pos, "derived")

    return board, origins


def _snapshot_frame(
    board: Board,
    marble_side: Optional[Side],
    marble_pos: Optional[tuple[int, int]],
    marble_path: list[tuple[int, int]],
    marble_index: int,
    step_number: Optional[int],
    left_hits: int,
    right_hits: int,
    status: str,
) -> dict:
    """Capture immutable frame data from current simulation state."""
    components = [
        comp.to_dict()
        for _, comp in sorted(board.components.items(), key=lambda item: (item[0][1], item[0][0]))
    ]
    return {
        "components": components,
        "marble_side": marble_side.value if marble_side else None,
        "marble_pos": marble_pos,
        "marble_path": list(marble_path),
        "marble_index": marble_index,
        "step_number": step_number,
        "blue_remaining": board.blue_balls_remaining,
        "red_remaining": board.red_balls_remaining,
        "left_hits": left_hits,
        "right_hits": right_hits,
        "status": status,
    }


def _build_animation_frames(
    task: dict,
    sequence: list[Side],
    hopper_entry_mode: str = "column",
) -> tuple[list[dict], dict[tuple[int, int], str]]:
    """Simulate the task and capture frame-by-frame states."""
    board, origins = _build_board_from_task(task, hopper_entry_mode=hopper_entry_mode)
    frames: list[dict] = []

    left_hits = 0
    right_hits = 0
    queue = list(sequence)
    marble_index = 0

    frames.append(
        _snapshot_frame(
            board,
            marble_side=None,
            marble_pos=None,
            marble_path=[],
            marble_index=0,
            step_number=None,
            left_hits=left_hits,
            right_hits=right_hits,
            status="Ready",
        )
    )

    while queue:
        side = queue.pop(0)

        if side == Side.BLUE and board.blue_balls_remaining <= 0:
            continue
        if side == Side.RED and board.red_balls_remaining <= 0:
            continue

        marble_index += 1
        trail: list[tuple[int, int]] = []

        def on_step(step_board: Board, position: tuple[int, int], step_number: int) -> None:
            trail.append(position)
            frames.append(
                _snapshot_frame(
                    step_board,
                    marble_side=side,
                    marble_pos=position,
                    marble_path=trail,
                    marble_index=marble_index,
                    step_number=step_number,
                    left_hits=left_hits,
                    right_hits=right_hits,
                    status=f"Marble {marble_index} ({side.value}) step {step_number}",
                )
            )

        result = board.release_marble(side, step_callback=on_step)

        if result.caught_by == "left_catcher":
            left_hits += 1
        elif result.caught_by == "right_catcher":
            right_hits += 1

        outcome = result.caught_by or result.termination_reason or "unknown"
        frames.append(
            _snapshot_frame(
                board,
                marble_side=None,
                marble_pos=None,
                marble_path=[],
                marble_index=marble_index,
                step_number=None,
                left_hits=left_hits,
                right_hits=right_hits,
                status=f"Marble {marble_index} ({side.value}) -> {outcome}",
            )
        )

        triggered: list[Side] = []
        while board._pending_trigger_releases:
            paired_side = board._pending_trigger_releases.pop(0)
            if paired_side == Side.BLUE and board.blue_balls_remaining > 0:
                triggered.append(Side.BLUE)
            elif paired_side == Side.RED and board.red_balls_remaining > 0:
                triggered.append(Side.RED)
        queue = triggered + queue

        if result.terminated and result.termination_reason in (
            "infinite_loop",
            "max_steps_exceeded",
            "fell_off_side",
        ):
            break

    return frames, origins


def _draw_animation_frame(
    ax,
    task: dict,
    frame: dict,
    origins: dict[tuple[int, int], str],
) -> None:
    """Draw a single animation frame from captured snapshot data."""
    board_data = task.get("board", {})
    hoppers = board_data.get("ball_hoppers", {})
    tlevs = board_data.get("trigger_levers", {})

    ax.clear()
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(plt.Rectangle((0, 0), FIG_W, FIG_H, color=COLOURS["bg"], zorder=0))
    draw_peg_grid(ax)

    task_id = task.get("task_id", "")
    ch_title = task.get("title", "")
    draw_board_frame(ax, title=ch_title, subtitle="Simulation")

    blue_x = hoppers.get("blue", {}).get("x", 2)
    red_x = hoppers.get("red", {}).get("x", 8)
    blue_y = hoppers.get("blue", {}).get("y", -1)
    red_y = hoppers.get("red", {}).get("y", -1)
    draw_hopper(ax, blue_x, "B", frame["blue_remaining"], "blue", blue_y)
    draw_hopper(ax, red_x, "R", frame["red_remaining"], "red", red_y)

    left_x = tlevs.get("left", {}).get("x", 0)
    right_x = tlevs.get("right", {}).get("x", BOARD_W - 1)
    draw_catcher(ax, left_x, "blue", ball_count=frame["left_hits"], active=False)
    draw_catcher(ax, right_x, "red", ball_count=frame["right_hits"], active=False)

    for comp in frame["components"]:
        pos = (comp["x"], comp["y"])
        origin = origins.get(pos, "derived")
        color = COLOURS["fixed"] if origin == "fixed" else COLOURS["placed"]
        zorder = 5 if origin == "fixed" else 6
        draw_component(ax, comp, color, zorder=zorder)

    marble_side = frame.get("marble_side")
    if frame.get("marble_path") and marble_side in ("blue", "red"):
        path_color = COLOURS["blue_hopper"] if marble_side == "blue" else COLOURS["red_hopper"]
        xs, ys = zip(*[_grid_to_axes(px, py) for px, py in frame["marble_path"]])
        ax.plot(xs, ys, color=path_color, linewidth=1.6, alpha=0.35, zorder=7)

    if frame.get("marble_pos") is not None and marble_side in ("blue", "red"):
        mx, my = frame["marble_pos"]
        cx, cy = _grid_to_axes(mx, my)
        marble_color = COLOURS["blue_hopper"] if marble_side == "blue" else COLOURS["red_hopper"]
        marble = Circle((cx, cy), 0.16, facecolor=marble_color, edgecolor="white", linewidth=1.2, zorder=8)
        marble.set_path_effects([pe.withStroke(linewidth=2.4, foreground=(1, 1, 1, 0.35))])
        ax.add_patch(marble)

    status = frame.get("status", "")
    if status:
        ax.text(
            0.2,
            0.35,
            status,
            ha="left",
            va="bottom",
            fontsize=8,
            color="#333333",
            zorder=10,
        )

    obj = task.get("objective", "")
    if obj:
        ax.text(
            FIG_W - 0.2,
            0.12,
            obj,
            ha="right",
            va="bottom",
            fontsize=6.5,
            color="#555555",
            wrap=True,
            multialignment="right",
            zorder=9,
            style="italic",
        )


def render_simulation_mp4(
    task: dict,
    output_path: str,
    sequence: Optional[str] = None,
    fps: int = 10,
    hold_frames: int = 6,
    hopper_entry_mode: str = "column",
) -> str:
    """Render an MP4 animation of marble simulation for a single task."""
    hoppers = task.get("board", {}).get("ball_hoppers", {})
    blue_count = hoppers.get("blue", {}).get("count", 0)
    red_count = hoppers.get("red", {}).get("count", 0)

    task_sequence = task.get("input_sequence")
    sides = _parse_sequence(
        sequence,
        blue_count,
        red_count,
        task_default=task_sequence if isinstance(task_sequence, list) else None,
    )
    frames, origins = _build_animation_frames(
        task,
        sides,
        hopper_entry_mode=hopper_entry_mode,
    )

    if not frames:
        raise ValueError("No animation frames generated")

    if hold_frames > 0:
        frames = ([frames[0]] * hold_frames) + frames + ([frames[-1]] * hold_frames)

    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError(
            "ffmpeg is required for MP4 export. Install ffmpeg and retry."
        )

    fig_w_in = 7.0
    fig_h_in = fig_w_in * FIG_H / FIG_W
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
    fig.patch.set_facecolor(COLOURS["bg"])

    fps = max(1, int(fps))
    interval_ms = int(1000 / fps)

    def update(frame_idx: int):
        _draw_animation_frame(ax, task, frames[frame_idx], origins)
        return []

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=2800,
        extra_args=["-pix_fmt", "yuv420p"],
    )
    anim.save(output_path, writer=writer, dpi=150)
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------------
# CLI / batch runner
# ---------------------------------------------------------------------------

def render_all(json_dir: str, output_dir: str):
    json_files = sorted(glob.glob(os.path.join(json_dir, "tt-official-*.json")))
    print(f"Rendering {len(json_files)} tasks → {output_dir}")
    ok = err = 0
    for jf in json_files:
        try:
            with open(jf) as fp:
                task = json.load(fp)
            tid = task["task_id"]
            for state in ("start", "solution"):
                out = os.path.join(output_dir, f"{tid}_{state}.png")
                render_board(task, state=state, output_path=out)
            ok += 1
            print(f"  ✓ {tid}")
        except Exception as e:
            err += 1
            print(f"  ✗ {jf}: {e}")
    print(f"\nDone: {ok} tasks rendered ({err} errors). Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render TT board images")
    parser.add_argument("--json-dir", default="tasks/official/challenges/json")
    parser.add_argument("--out-dir", default="tasks/official/challenges/board_images")
    parser.add_argument("--task", default=None)
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Render a real simulation animation and save as MP4 (requires --task)",
    )
    parser.add_argument(
        "--run",
        default=None,
        help="Marble sequence for animation, e.g. 'blue,red,blue' (default: task input_sequence, then alternating fallback)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for MP4 animation (default: 10)",
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=6,
        help="Number of still frames to hold at animation start/end (default: 6)",
    )
    parser.add_argument(
        "--hopper-entry-mode",
        choices=["column", "inward"],
        default="column",
        help="Marble entry semantics for simulation MP4s (default: column to match benchmark tooling)",
    )
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    json_dir = base / args.json_dir
    out_dir = base / args.out_dir

    if args.animate and not args.task:
        parser.error("--animate requires --task")

    if args.task:
        jf = json_dir / f"{args.task}.json"
        with open(jf) as fp:
            task = json.load(fp)
        if args.animate:
            out = out_dir / f"{args.task}_simulation.mp4"
            render_simulation_mp4(
                task,
                output_path=str(out),
                sequence=args.run,
                fps=args.fps,
                hold_frames=args.hold_frames,
                hopper_entry_mode=args.hopper_entry_mode,
            )
            print(f"Saved: {out}")
        else:
            for state in ("start", "solution"):
                out = out_dir / f"{args.task}_{state}.png"
                render_board(task, state=state, output_path=str(out))
                print(f"Saved: {out}")
    else:
        render_all(str(json_dir), str(out_dir))