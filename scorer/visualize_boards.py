#!/usr/bin/env python3
"""
Visualization of benchmark results showing board states.
Renders initial boards with LLM predictions vs expected solutions.
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

# Import the board renderer module
sys.path.insert(0, str(Path(__file__).parent.parent / "simulator"))
from board_renderer import (
    COLOURS, BOARD_W, BOARD_H, MARGIN_TOP, MARGIN_BOTTOM, MARGIN_SIDES, CELL, FIG_W, FIG_H,
    _ax_coord, draw_peg_grid, draw_board_frame, draw_component, draw_hopper,
    draw_ramp_right, draw_ramp_left, draw_catcher,
    render_board as render_board_base,
)


def draw_trigger_lever(ax, x_grid, label, side, zorder=8):
    """Draw a small trigger-lever marker just below the bottom rail.

    Compatibility shim: the upstream `board_renderer` module exposes
    `draw_trigger`/`draw_catcher` rather than a dedicated trigger-lever helper.
    We render a coloured triangle in the appropriate slot so the start board
    still shows where each colour's trigger lives.
    """
    cx = MARGIN_SIDES + (x_grid + 0.5) * CELL
    cy = MARGIN_BOTTOM - 0.50
    col = COLOURS["blue_trigger"] if side == "blue" else COLOURS["red_trigger"]
    ax.plot(cx, cy, "^", color=col, markersize=11, zorder=zorder)
    ax.text(cx, cy - 0.30, label, ha="center", va="top",
            fontsize=8, color=col, fontweight="bold", zorder=zorder)


def draw_board_frame_transparent(ax, title="", subtitle=""):
    """Draw board frame without background fill (transparent background).
    
    Similar to draw_board_frame from board_renderer but with no background color.
    """
    # Board frame (no fill)
    bx = MARGIN_SIDES - 0.5
    by = MARGIN_BOTTOM - 0.5
    bw = (BOARD_W - 1) + 1.0
    bh = (BOARD_H - 1) + 1.0
    rect = FancyBboxPatch(
        (bx, by), bw, bh,
        boxstyle="round,pad=0.2",
        linewidth=2.5,
        edgecolor=COLOURS["frame"],
        facecolor="none",
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
        ax.text(BOARD_W / 2 + MARGIN_SIDES - 0.5, MARGIN_BOTTOM + BOARD_H - 0.2,
                title, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=COLOURS["frame"])
    if subtitle:
        ax.text(BOARD_W / 2 + MARGIN_SIDES - 0.5, MARGIN_BOTTOM + BOARD_H - 0.65,
                subtitle, ha="center", va="bottom",
                fontsize=9, color=COLOURS["frame"], style="italic")


# ============================================================================
# Transcript extraction & formatting
# ============================================================================

# Tool-call display palette (matches the four tools exposed in tool_executor.py)
_TOOL_COLOURS = {
    "place_component": "#2563EB",   # blue
    "remove_component": "#DC2626",  # red
    "run_simulation": "#16A34A",    # green
    "get_board_state": "#7C3AED",   # purple
}
_DEFAULT_TOOL_COLOUR = "#475569"     # slate
_RESPONSE_BG = "#F8FAFC"
_RESPONSE_FG = "#0F172A"
_TOOL_BG = "#FFFFFF"
_TOOL_BORDER = "#CBD5E1"


def extract_transcript(result: dict) -> tuple[str, list[dict]]:
    """Extract (response_text, transcript_entries) from a benchmark result.

    Falls back to the legacy `tool_calls` shape (no results, no per-turn text)
    when `transcript` is absent so older reports still render usefully.
    """
    predicted = result.get("predicted") or {}
    response_text = result.get("llm_response", "") or ""

    transcript = predicted.get("transcript")
    if transcript:
        return response_text, transcript

    # Legacy fallback — synthesise transcript entries from raw tool_calls
    legacy_calls = predicted.get("tool_calls") or []
    transcript = [
        {
            "turn": idx,
            "assistant_text": "",
            "tool_name": tc.get("name", "?"),
            "arguments": tc.get("args", {}),
            "result": None,
            "error": None,
        }
        for idx, tc in enumerate(legacy_calls)
    ]
    return response_text, transcript


def _format_args(args: dict, max_len: int = 80) -> str:
    """Compact one-line arg representation, truncated to max_len."""
    if not isinstance(args, dict):
        s = str(args)
    else:
        parts = []
        for k, v in args.items():
            if isinstance(v, (list, dict)):
                vs = json.dumps(v, separators=(",", ":"))
            else:
                vs = str(v)
            parts.append(f"{k}={vs}")
        s = ", ".join(parts)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _format_result(result, max_len: int = 120) -> str:
    """One-line summary of a tool result for inline display."""
    if result is None:
        return "(no result captured)"
    if isinstance(result, dict):
        # Prefer a few well-known keys
        if "error" in result and result.get("error"):
            return f"error: {result['error']}"
        if "message" in result:
            s = str(result["message"])
        elif "success" in result:
            extras = {k: v for k, v in result.items() if k not in ("success",)}
            s = f"success={result['success']}"
            if extras:
                s += " | " + json.dumps(extras, separators=(",", ":"), default=str)
        else:
            s = json.dumps(result, separators=(",", ":"), default=str)
    else:
        s = str(result)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


# ============================================================================
# Custom rendering functions with different component colors
# ============================================================================


def draw_component_predicted(ax, comp, zorder=6):
    """Draw a predicted component in orange."""
    draw_component(ax, comp, "#F97316", zorder)  # orange


def draw_component_expected(ax, comp, zorder=6):
    """Draw an expected solution component in green."""
    draw_component(ax, comp, COLOURS["fixed"], zorder)  # dark (reuse fixed color)


# ============================================================================
# Board render functions for predictions
# ============================================================================


def render_task_board(task: dict, state: str = "start", output_path: str = None):
    """Render a board with different states."""
    return render_board_base(task, state, output_path)


def render_with_predictions(task: dict, predicted: list, expected: list, 
                        success: bool, output_path: str = None):
    """
    Render a board showing the starting board + predicted + expected placements.
    
    Parameters
    ----------
    task        : parsed JSON task dict
    predicted  : list of predicted placements from LLM
    expected    : list of expected solution placements
    success     : whether the LLM's solution was successful
    output_path : path to save PNG
    """
    board = task["board"]
    hoppers = board.get("ball_hoppers", {})
    tlevs = board.get("trigger_levers", {})
    
    fixed_comps = board.get("fixed_components", [])
    avail = task.get("available_parts", {})
    
    # Figure setup
    fig_w_in = 9.0
    fig_h_in = fig_w_in * FIG_H / FIG_W * 1.3  # taller for status info
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H + 2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0)
    
    # Board background
    ax.add_patch(plt.Rectangle((0, 0), FIG_W, FIG_H, color=COLOURS["bg"], zorder=0))
    
    # Grid
    draw_peg_grid(ax)
    
    # Frame & title
    task_id = task.get("task_id", "")
    ch_title = task.get("title", "")
    draw_board_frame_transparent(ax, title=f"{task_id}  |  {ch_title}", subtitle="")
    
    # Hoppers
    blue_h = hoppers.get("blue", {})
    red_h = hoppers.get("red", {})
    if blue_h.get("count", 0) > 0:
        draw_hopper(ax, blue_h.get("x", 2), "B", blue_h.get("count", 0), "blue")
    if red_h.get("count", 0) > 0:
        draw_hopper(ax, red_h.get("x", 8), "R", red_h.get("count", 0), "red")
    
    # Trigger levers
    left_t = tlevs.get("left", {})
    right_t = tlevs.get("right", {})
    if left_t:
        draw_trigger_lever(ax, left_t.get("x", 2), "b", "blue")
    if right_t:
        draw_trigger_lever(ax, right_t.get("x", 8), "r", "red")
    
    # Fixed components (dark navy)
    for comp in fixed_comps:
        draw_component(ax, comp, COLOURS["fixed"], zorder=5)
    
    # Expected solution components (dark, semi-transparent)
    for comp in expected:
        draw_component(ax, comp, COLOURS["fixed"], zorder=4)
    
    # Predicted components (orange for success, red for failure)
    if success:
        pred_color = "#22C55E"  # green
    else:
        pred_color = "#EF4444"  # red
    
    for comp in predicted:
        draw_component(ax, comp, pred_color, zorder=7)
    
    # Status box
    status_color = "#22C55E" if success else "#EF4444"
    status_text = "PASS" if success else "FAIL"
    status_bg = status_color
    
    # Add status box on the right side
    status_x = FIG_W + 0.3
    status_y = FIG_H / 2
    ax.text(status_x, status_y + 1.0, f"Result: {status_text}", 
           ha="center", va="center", fontsize=14, fontweight="bold",
           color=status_color)
    ax.text(status_x, status_y, f"Predicted: {len(predicted)} parts", 
           ha="center", va="center", fontsize=10)
    ax.text(status_x, status_y - 0.5, f"Expected: {len(expected)} parts", 
           ha="center", va="center", fontsize=10)
    
    # Legend
    legend_y = 1.5
    ax.text(0.3, legend_y, "Legend:", ha="left", va="bottom", fontsize=9, fontweight="bold")
    
    # Fixed components legend
    ax.plot([0.8], [legend_y - 0.15], "o", color=COLOURS["fixed"], markersize=8)
    ax.text(1.2, legend_y - 0.15, "Fixed", ha="left", va="center", fontsize=8)
    
    # Expected (same as fixed but label shows)
    ax.plot([2.5], [legend_y - 0.15], "o", color=COLOURS["fixed"], markersize=8, alpha=0.5)
    ax.text(2.9, legend_y - 0.15, "Expected", ha="left", va="center", fontsize=8)
    
    # Predicted
    pred_legend_color = "#22C55E" if success else "#EF4444"
    ax.plot([4.5], [legend_y - 0.15], "o", color=pred_legend_color, markersize=8)
    ax.text(4.9, legend_y - 0.15, "Predicted", ha="left", va="center", fontsize=8)
    
    # Objective
    obj = task.get("objective", "")
    if obj:
        ax.text(
            FIG_W - 0.2,
            0.15,
            obj,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#555555",
            wrap=True,
            multialignment="right",
            style="italic",
        )
    
    plt.tight_layout(pad=0.3)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="none", transparent=True)
        plt.close(fig)
        return output_path
    return fig


def render_with_transcript(
    task: dict,
    predicted: list,
    expected: list,
    success: bool,
    response_text: str = "",
    transcript: list[dict] | None = None,
    output_path: str | None = None,
):
    """Render the board on the left and the LLM transcript on the right.

    The transcript pane shows:
      1. The model's free-form response text (top section)
      2. An ordered list of tool calls, each with name, arguments, and result

    Parameters
    ----------
    task          : parsed task JSON
    predicted     : list of predicted component placements (type/x/y dicts)
    expected      : list of expected component placements
    success       : whether validation passed
    response_text : the model's narrative response (final or summary text)
    transcript    : ordered list of dicts with keys
                    {turn, assistant_text, tool_name, arguments, result, error}
    output_path   : if given, save PNG and return the path; else return Figure
    """
    transcript = transcript or []

    board = task["board"]
    hoppers = board.get("ball_hoppers", {})
    tlevs = board.get("trigger_levers", {})
    fixed_comps = board.get("fixed_components", [])

    # ------------------------------------------------------------------
    # Figure layout: board pane (left) + transcript pane (right)
    # ------------------------------------------------------------------
    board_w_in = 8.0
    board_h_in = board_w_in * FIG_H / FIG_W
    panel_w_in = 8.5
    fig_h_in = max(board_h_in, 9.0) + 0.6  # leave room for super title
    fig_w_in = board_w_in + panel_w_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0)

    # GridSpec: 1 row, 2 cols (board | transcript)
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[board_w_in, panel_w_in],
        wspace=0.05,
        left=0.02, right=0.98, top=0.94, bottom=0.04,
    )

    # ---- LEFT: board axes ---------------------------------------------
    ax_board = fig.add_subplot(gs[0, 0])
    ax_board.set_xlim(0, FIG_W)
    ax_board.set_ylim(0, FIG_H)
    ax_board.set_aspect("equal")
    ax_board.axis("off")
    ax_board.add_patch(
        plt.Rectangle((0, 0), FIG_W, FIG_H, color=COLOURS["bg"], zorder=0)
    )

    draw_peg_grid(ax_board)
    task_id = task.get("task_id", "")
    ch_title = task.get("title", "")
    draw_board_frame_transparent(
        ax_board,
        title=f"{task_id}  |  {ch_title}",
        subtitle="LLM solution" if predicted else "Starting setup",
    )

    # Hoppers
    blue_h = hoppers.get("blue", {})
    red_h = hoppers.get("red", {})
    if blue_h.get("count", 0) > 0:
        draw_hopper(ax_board, blue_h.get("x", 2), "B", blue_h.get("count", 0), "blue")
    if red_h.get("count", 0) > 0:
        draw_hopper(ax_board, red_h.get("x", 8), "R", red_h.get("count", 0), "red")

    # Trigger levers
    if tlevs.get("left"):
        draw_trigger_lever(ax_board, tlevs["left"].get("x", 2), "b", "blue")
    if tlevs.get("right"):
        draw_trigger_lever(ax_board, tlevs["right"].get("x", 8), "r", "red")

    # Fixed components
    for comp in fixed_comps:
        draw_component(ax_board, comp, COLOURS["fixed"], zorder=5)

    # Expected (dim)
    for comp in expected:
        draw_component(ax_board, comp, COLOURS["fixed"], zorder=4)

    # Predicted (green if success, red if fail)
    pred_color = "#22C55E" if success else "#EF4444"
    for comp in predicted:
        draw_component(ax_board, comp, pred_color, zorder=7)

    # ---- RIGHT: transcript pane ---------------------------------------
    ax_panel = fig.add_subplot(gs[0, 1])
    ax_panel.set_xlim(0, 1)
    ax_panel.set_ylim(0, 1)
    ax_panel.axis("off")
    ax_panel.set_facecolor("none")
    ax_panel.patch.set_alpha(0)

    # Header bar (status badge + summary)
    status_color = "#16A34A" if success else "#DC2626"
    status_text = "PASS" if success else "FAIL"
    ax_panel.add_patch(
        FancyBboxPatch(
            (0.0, 0.955), 1.0, 0.04,
            boxstyle="round,pad=0.005",
            linewidth=0,
            facecolor=status_color,
            transform=ax_panel.transAxes,
            zorder=2,
        )
    )
    ax_panel.text(
        0.015, 0.975,
        f"{status_text}  ·  {len(transcript)} tool call{'s' if len(transcript) != 1 else ''}"
        f"  ·  {len(predicted)} placed",
        transform=ax_panel.transAxes,
        ha="left", va="center",
        fontsize=10, fontweight="bold", color="white",
    )

    # --- Response text section (top) -----------------------------------
    cursor_y = 0.93   # axes-fraction y, growing downward
    section_h = 0.30  # initial allocation for response text

    ax_panel.text(
        0.015, cursor_y,
        "LLM RESPONSE",
        transform=ax_panel.transAxes,
        fontsize=8, fontweight="bold", color="#475569",
        ha="left", va="top",
    )
    cursor_y -= 0.022

    # Response text panel background
    resp_top = cursor_y
    resp_bottom = cursor_y - section_h
    ax_panel.add_patch(
        FancyBboxPatch(
            (0.005, resp_bottom), 0.99, section_h,
            boxstyle="round,pad=0.005",
            linewidth=0.8, edgecolor=_TOOL_BORDER,
            facecolor=_RESPONSE_BG,
            transform=ax_panel.transAxes, zorder=1,
        )
    )

    # Wrap and render the response text
    body = (response_text or "(no response text captured)").strip()
    if len(body) > 1200:
        body = body[:1200].rstrip() + " ..."
    wrapped = "\n".join(textwrap.wrap(body, width=78, break_long_words=False) or [body])
    ax_panel.text(
        0.02, resp_top - 0.012,
        wrapped,
        transform=ax_panel.transAxes,
        fontsize=7.5, color=_RESPONSE_FG,
        ha="left", va="top",
        family="monospace",
        zorder=3,
    )

    cursor_y = resp_bottom - 0.025

    # --- Tool calls section --------------------------------------------
    ax_panel.text(
        0.015, cursor_y,
        f"TOOL CALLS ({len(transcript)})",
        transform=ax_panel.transAxes,
        fontsize=8, fontweight="bold", color="#475569",
        ha="left", va="top",
    )
    cursor_y -= 0.022

    # If there are no tool calls, draw a simple notice
    if not transcript:
        ax_panel.add_patch(
            FancyBboxPatch(
                (0.005, cursor_y - 0.04), 0.99, 0.04,
                boxstyle="round,pad=0.005",
                linewidth=0.8, edgecolor=_TOOL_BORDER,
                facecolor=_TOOL_BG,
                transform=ax_panel.transAxes, zorder=1,
            )
        )
        ax_panel.text(
            0.5, cursor_y - 0.02,
            "(no tool calls)",
            transform=ax_panel.transAxes,
            ha="center", va="center",
            fontsize=8, color="#94A3B8", style="italic",
        )
    else:
        # Compute per-call height that fits remaining vertical space
        remaining = max(0.05, cursor_y - 0.02)
        per_h = min(0.085, remaining / max(1, len(transcript)))
        # Cap at 12 visible calls; collapse the rest into a "..." row
        visible = transcript[:12]
        truncated = len(transcript) - len(visible)

        for idx, entry in enumerate(visible):
            top = cursor_y - idx * per_h
            bot = top - per_h + 0.005
            tname = entry.get("tool_name", "?")
            tcolor = _TOOL_COLOURS.get(tname, _DEFAULT_TOOL_COLOUR)

            # Card background
            ax_panel.add_patch(
                FancyBboxPatch(
                    (0.005, bot), 0.99, per_h - 0.005,
                    boxstyle="round,pad=0.004",
                    linewidth=0.8, edgecolor=_TOOL_BORDER,
                    facecolor=_TOOL_BG,
                    transform=ax_panel.transAxes, zorder=1,
                )
            )
            # Coloured left rule
            ax_panel.add_patch(
                plt.Rectangle(
                    (0.005, bot), 0.012, per_h - 0.005,
                    facecolor=tcolor, edgecolor="none",
                    transform=ax_panel.transAxes, zorder=2,
                )
            )

            # Header line: index + tool name (coloured)
            ax_panel.text(
                0.028, top - 0.012,
                f"#{idx + 1}  {tname}",
                transform=ax_panel.transAxes,
                ha="left", va="top",
                fontsize=8.5, fontweight="bold",
                color=tcolor,
                family="monospace",
            )

            # Args line
            args_str = _format_args(entry.get("arguments") or {})
            ax_panel.text(
                0.028, top - 0.030,
                f"args: {args_str}",
                transform=ax_panel.transAxes,
                ha="left", va="top",
                fontsize=7, color="#1E293B",
                family="monospace",
            )

            # Result line
            err = entry.get("error")
            res_str = _format_result(entry.get("result"))
            res_color = "#DC2626" if err else "#475569"
            res_prefix = "err " if err else "→   "
            ax_panel.text(
                0.028, top - 0.046,
                f"{res_prefix}{res_str}",
                transform=ax_panel.transAxes,
                ha="left", va="top",
                fontsize=7, color=res_color,
                family="monospace",
            )

            # Optional per-turn assistant text (if present and short)
            atext = (entry.get("assistant_text") or "").strip()
            if atext and per_h > 0.06:
                snippet = textwrap.shorten(atext, width=110, placeholder=" ...")
                ax_panel.text(
                    0.028, top - 0.062,
                    f"say: {snippet}",
                    transform=ax_panel.transAxes,
                    ha="left", va="top",
                    fontsize=6.5, color="#64748B",
                    style="italic",
                )

        if truncated > 0:
            ax_panel.text(
                0.5, cursor_y - len(visible) * per_h - 0.012,
                f"... and {truncated} more tool call{'s' if truncated != 1 else ''}",
                transform=ax_panel.transAxes,
                ha="center", va="top",
                fontsize=7.5, color="#94A3B8", style="italic",
            )

    # Super title
    fig.suptitle(
        f"{task_id} — {ch_title}".strip(" —"),
        fontsize=12, fontweight="bold", y=0.985,
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="none", transparent=True)
        plt.close(fig)
        return output_path
    return fig


def load_task_json(task_id: str, tasks_dir: str = None) -> dict:
    """Load a task JSON file."""
    if tasks_dir is None:
        tasks_dir = Path(__file__).parent.parent / "tasks/official/challenges/json"
    else:
        tasks_dir = Path(tasks_dir)
    
    task_path = tasks_dir / f"{task_id}.json"
    with open(task_path) as f:
        return json.load(f)


def _extract_predicted_placements(result: dict) -> list[dict]:
    """Pull predicted placements from agentic or legacy result shape."""
    predicted = result.get("predicted") or {}
    out: list[dict] = []

    # Synthesis shape can be either {row,col,component} or {x,y,component}.
    for p in predicted.get("placements") or []:
        comp_type = p.get("component") or p.get("component_type") or p.get("type")
        x = p.get("x", p.get("col"))
        y = p.get("y", p.get("row"))
        if comp_type is None or x is None or y is None:
            continue
        out.append({"type": comp_type, "x": x, "y": y})

    # Agentic shape: predicted.final_solution = [{"component_type":..,"x":..,"y":..}]
    for p in predicted.get("final_solution") or []:
        comp_type = p.get("component_type") or p.get("component") or p.get("type")
        x = p.get("x", p.get("col"))
        y = p.get("y", p.get("row"))
        if comp_type is None or x is None or y is None:
            continue
        out.append({
            "type": comp_type,
            "x": x, "y": y,
            **({"state": p["state"]} if "state" in p else {}),
        })
    return out


def create_task_visualization(
    result: dict,
    tasks_dir: str = None,
    output_dir: str = None,
    with_transcript: bool = False,
):
    """Create a visualization for a single task result.

    If `with_transcript` is True (or transcript data is present in the result),
    render the side-by-side board + LLM response + tool calls layout. Otherwise
    fall back to the legacy `render_with_predictions` view.
    """
    task_id = result["task_id"]

    # Load the task definition
    task = load_task_json(task_id, tasks_dir)

    predicted = _extract_predicted_placements(result)

    # Get expected solution
    expected = []
    if result.get("expected", {}).get("placed_components"):
        for e in result["expected"]["placed_components"]:
            expected.append({"type": e["type"], "x": e["x"], "y": e["y"]})

    success = result.get("success", False)

    # Determine output path
    if output_dir is None:
        output_dir = Path(__file__).parent / "benchmark_results"
    else:
        output_dir = Path(output_dir)

    response_text, transcript = extract_transcript(result)
    use_transcript_view = with_transcript or bool(transcript) or bool(response_text)

    if use_transcript_view:
        output_path = output_dir / f"{task_id}_transcript.png"
        render_with_transcript(
            task, predicted, expected, success,
            response_text=response_text,
            transcript=transcript,
            output_path=str(output_path),
        )
    else:
        output_path = output_dir / f"{task_id}_visualization.png"
        render_with_predictions(task, predicted, expected, success, str(output_path))

    return output_path


def create_summary_visualization(results: dict, tasks_dir: str = None, output_dir: str = None):
    """Create a summary visualization grid showing all tasks."""
    
    if tasks_dir is None:
        tasks_dir = Path(__file__).parent.parent / "tasks/official/challenges/json"
    else:
        tasks_dir = Path(tasks_dir)
    
    if output_dir is None:
        output_dir = Path(__file__).parent / "benchmark_results"
    else:
        output_dir = Path(output_dir)
    
    num_tasks = len(results["results"])
    
    # Create a figure with subplots for each task
    nrows = 1
    ncols = min(num_tasks, 3)
    if num_tasks > 3:
        nrows = (num_tasks + 2) // 3
        ncols = 3
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 7 * nrows))
    fig.suptitle(f"TuringBench Results: {results['model']} | Success Rate: {results['success_rate']*100:.1f}%", 
                fontsize=16, fontweight='bold')
    
    if num_tasks == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    for idx, result in enumerate(results["results"]):
        ax = axes[idx]
        task_id = result["task_id"]
        
        # Load task
        task = load_task_json(task_id, str(tasks_dir))
        
        # Get placements from any supported result schema.
        predicted = _extract_predicted_placements(result)
        
        expected = []
        if result.get("expected", {}).get("placed_components"):
            for e in result["expected"]["placed_components"]:
                expected.append({
                    "type": e["type"],
                    "x": e["x"],
                    "y": e["y"]
                })
        
        # Simple render on each axis
        _render_task_on_axis(ax, task, predicted, expected, result.get("success", False))
    
    # Hide unused axes
    for idx in range(num_tasks, len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    output_path = output_dir / f"summary_visualization.png"
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor="none", transparent=True)
    plt.close(fig)
    
    return output_path


def _render_task_on_axis(ax, task, predicted, expected, success):
    """Render a simplified board on a single axis."""
    board = task["board"]
    hoppers = board.get("ball_hoppers", {})
    tlevs = board.get("trigger_levers", {})
    fixed_comps = board.get("fixed_components", [])
    
    # Set up axis
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(COLOURS["bg"])
    
    # Grid
    for gx in range(BOARD_W):
        for gy in range(BOARD_H):
            cx, cy = _ax_coord(gx, gy)
            ax.plot(cx, cy, "o", color=COLOURS["grid_dot"], markersize=3, zorder=1)
    
    # Frame
    bx = MARGIN_SIDES - 0.4
    by = MARGIN_BOTTOM - 0.4
    bw = BOARD_W - 1 + 0.8
    bh = BOARD_H - 1 + 0.8
    rect = FancyBboxPatch(
        (bx, by), bw, bh,
        boxstyle="round,pad=0.15",
        linewidth=2,
        edgecolor=COLOURS["frame"],
        facecolor=COLOURS["bg"],
        zorder=0,
    )
    ax.add_patch(rect)
    
    # Title
    task_id = task.get("task_id", "").replace("tt-official-", "")
    ax.set_title(task_id, fontsize=12, fontweight="bold")
    
    # Hoppers (simplified - just draw circles)
    blue_h = hoppers.get("blue", {})
    red_h = hoppers.get("red", {})
    if blue_h:
        cx = MARGIN_SIDES + blue_h.get("x", 2) * CELL
        cy = MARGIN_BOTTOM + BOARD_H * CELL - 0.15
        ax.plot(cx, cy, "s", color=COLOURS["blue_hopper"], markersize=12, zorder=8)
    if red_h:
        cx = MARGIN_SIDES + red_h.get("x", 8) * CELL
        cy = MARGIN_BOTTOM + BOARD_H * CELL - 0.15
        ax.plot(cx, cy, "s", color=COLOURS["red_hopper"], markersize=12, zorder=8)
    
    # Trigger levers (simplified)
    left_t = tlevs.get("left", {})
    right_t = tlevs.get("right", {})
    if left_t:
        cx = MARGIN_SIDES + left_t.get("x", 2) * CELL
        cy = MARGIN_BOTTOM - 0.50
        ax.plot(cx, cy, "^", color=COLOURS["blue_trigger"], markersize=10, zorder=8)
    if right_t:
        cx = MARGIN_SIDES + right_t.get("x", 8) * CELL
        cy = MARGIN_BOTTOM - 0.50
        ax.plot(cx, cy, "^", color=COLOURS["red_trigger"], markersize=10, zorder=8)
    
    # Fixed components
    for comp in fixed_comps:
        draw_component(ax, comp, COLOURS["fixed"], zorder=5)
    
    # Predicted components
    pred_color = "#22C55E" if success else "#EF4444"
    for comp in predicted:
        draw_component(ax, comp, pred_color, zorder=7)
    
    # Status indicator
    status_color = "#22C55E" if success else "#EF4444"
    status_text = "✓" if success else "✗"
    ax.text(FIG_W - 0.5, FIG_H - 0.5, status_text, fontsize=20, 
           fontweight="bold", color=status_color, ha="right", va="top")


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark board results")
    parser.add_argument("results_json", help="Path to benchmark results JSON")
    parser.add_argument("--tasks-dir", default=None, help="Directory with task JSON files")
    parser.add_argument("--output-dir", default=None, help="Output directory for visualizations")
    parser.add_argument("--summary", action="store_true", help="Create summary visualization")
    parser.add_argument(
        "--with-transcript",
        action="store_true",
        help="Force the side-by-side board + LLM response + tool-call view "
             "(default: auto-detect from result data)",
    )
    parser.add_argument(
        "--no-transcript",
        action="store_true",
        help="Force the legacy board-only view even when transcript data is present",
    )
    args = parser.parse_args()

    if args.with_transcript and args.no_transcript:
        parser.error("--with-transcript and --no-transcript are mutually exclusive")

    # Load results
    with open(args.results_json) as f:
        results = json.load(f)

    output_dir = Path(args.output_dir) if args.output_dir else None
    tasks_dir = args.tasks_dir

    if args.summary:
        # Create summary grid
        output_path = create_summary_visualization(results, tasks_dir, output_dir)
        print(f"Summary saved to: {output_path}")
    else:
        # Create individual task visualizations
        for result in results["results"]:
            if args.no_transcript:
                # Strip transcript data so create_task_visualization picks
                # the legacy render path
                pruned = {**result, "predicted": {**(result.get("predicted") or {})}}
                pruned["predicted"].pop("transcript", None)
                pruned["llm_response"] = ""
                output_path = create_task_visualization(
                    pruned, tasks_dir, output_dir, with_transcript=False
                )
            else:
                output_path = create_task_visualization(
                    result, tasks_dir, output_dir,
                    with_transcript=args.with_transcript,
                )
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()