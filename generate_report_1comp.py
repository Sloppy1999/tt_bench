#!/usr/bin/env python3
"""Generate thesis-ready benchmark table for challenges_1comp results.

Reads the latest benchmark report and generates PNG + SVG tables
matching the visual style of generate_table_fig.py exactly.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Paths ──────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "scorer" / "benchmark_results"
CHALLENGES_DIR = Path(__file__).parent / "tasks" / "challenges_1comp"
OUTPUT_DIR = Path(__file__).parent

# ── Academic green palette ─────────────────────────────────────────
COLORS = {
    'bg':              '#ffffff',
    'header_bg':       '#f5f7f4',
    'row_alt':         '#f9fbf9',
    'text_primary':    '#1a2e1a',
    'text_secondary':  '#3d5c3d',
    'text_muted':      '#6b8e6b',
    'border':          '#c8dcc8',
    'border_light':    '#d8e8d8',
    'success':         '#2e7d32',
    'success_bg':      '#e8f5e9',
    'failure':         '#c62828',
    'failure_bg':      '#fce8e8',
    'accent_green':    '#388e3c',
    'accent_teal':     '#00695c',
    'type_under_bg':   '#e8f5e9',
    'type_under_text': '#2e7d32',
    'type_agent_bg':   '#e0f2f1',
    'type_agent_text': '#00695c',
}


def find_latest_report() -> Path:
    json_files = sorted(
        RESULTS_DIR.glob("benchmark_*.json"),
        key=os.path.getmtime, reverse=True
    )
    if not json_files:
        raise FileNotFoundError(f"No benchmark reports in {RESULTS_DIR}")
    return json_files[0]


def load_challenge_title(task_id: str) -> str:
    path = CHALLENGES_DIR / f"{task_id}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data.get("title", task_id)
    return task_id


def extract_table_data(report_path: Path) -> tuple:
    """Return (rows, report).

    Each row is a 7-tuple:
    (board_label, subtitle, type, success, valid, tool_calls, turns)
    """
    with open(report_path) as f:
        report = json.load(f)

    rows = []
    for r in report["results"]:
        task_id = r["task_id"]
        board_label = task_id.replace("tt-official-", "")
        title = load_challenge_title(task_id)
        success = r["success"]
        metrics = r.get("metrics", {})
        valid = metrics.get("valid", None)
        tool_calls = metrics.get("tool_calls_count",
            len(r.get("predicted", {}).get("tool_calls", [])))
        turns = metrics.get("turns", tool_calls)
        rows.append((
            board_label, title, "agentic", success,
            valid, tool_calls, turns
        ))

    return rows, report


def make_table(rows: list, report: dict, figsize=(14.5, 7.0)):
    fig = plt.figure(figsize=figsize, dpi=120)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 7.0)
    ax.axis('off')

    num_rows = len(rows)
    model = report["model"]
    provider = report["provider"]
    total = report["total_tasks"]
    rate = report["success_rate"] * 100
    date = report["timestamp"].split("T")[0]

    # ── Header background ──────────────────────────────────────────
    ax.add_patch(FancyBboxPatch(
        (0.2, 5.9), 14.1, 0.85,
        boxstyle="square,pad=0", facecolor=COLORS['header_bg'],
        edgecolor=COLORS['accent_green'], linewidth=0, zorder=1))

    ax.text(0.4, 6.48, 'Turing Tumble Benchmark: 1-Component Challenges',
            fontsize=13, fontweight='bold', color=COLORS['text_primary'],
            va='center', ha='left', fontfamily='DejaVu Sans')

    ax.text(0.4, 6.12,
            f'Model: {model} ({provider})  •  {date}  •  {total} tasks  •  Success rate: {rate:.0f}%',
            fontsize=9, color=COLORS['text_secondary'], va='center', ha='left')

    badge_color = COLORS['success'] if rate >= 50 else COLORS['failure']
    badge_bg = COLORS['success_bg'] if rate >= 50 else COLORS['failure_bg']
    ax.add_patch(FancyBboxPatch(
        (12.8, 5.98), 1.3, 0.65,
        boxstyle="round,pad=0.04",
        facecolor=badge_bg, edgecolor=badge_color, linewidth=1.2, zorder=2))
    ax.text(13.45, 6.31, f'{rate:.0f}%', fontsize=15, fontweight='bold',
            color=badge_color, va='center', ha='center')

    # ── Table header row ───────────────────────────────────────────
    col_x = [0.2, 3.0, 4.3, 5.2, 8.2, 11.2]
    col_widths = [2.8, 1.3, 0.9, 3.0, 3.0, 3.1]
    headers = ['Board', 'Task Type', 'Success', 'valid', 'tool_calls', 'turns']

    for i, (cx, cw, h) in enumerate(zip(col_x, col_widths, headers)):
        ax.add_patch(FancyBboxPatch(
            (cx, 5.15), cw, 0.62,
            boxstyle="square,pad=0",
            facecolor=COLORS['accent_green'], edgecolor='none', zorder=1))
        ax.text(cx + cw / 2, 5.46, h,
                fontsize=8.5, fontweight='bold', color='white',
                va='center', ha='center')

    # Sub-header for agentic metrics (cols 3-5: 5.2 to 14.3)
    ax.text(9.75, 4.98, 'Agentic Metrics', fontsize=7.5,
            color=COLORS['accent_teal'], va='center', ha='center',
            fontstyle='italic', fontweight='bold')

    # ── Data rows ──────────────────────────────────────────────────
    row_height = 0.42
    start_y = 4.60

    for r_idx, row in enumerate(rows):
        board_label, subtitle, rtype, success, valid, tool_calls, turns = row
        y = start_y - r_idx * row_height
        is_alt = (r_idx % 2 == 1)

        # Row background
        bg = COLORS['row_alt'] if is_alt else COLORS['bg']
        ax.add_patch(FancyBboxPatch(
            (0.2, y - row_height), 14.1, row_height,
            boxstyle="square,pad=0",
            facecolor=bg, edgecolor='none', zorder=0))
        # Bottom border
        ax.plot([0.2, 14.3], [y - row_height, y - row_height],
                color=COLORS['border_light'], linewidth=0.5, zorder=1)

        # Board (two-line)
        bx = col_x[0]
        ax.text(bx + 0.1, y - row_height / 2 - 0.07, board_label,
                fontsize=9.5, fontweight='bold', color=COLORS['accent_teal'],
                va='center', ha='left')
        ax.text(bx + 0.1, y - row_height / 2 + 0.09, subtitle,
                fontsize=7.5, color=COLORS['text_secondary'],
                va='center', ha='left')

        # Task type badge
        if rtype == 'understanding':
            bg_c = COLORS['type_under_bg']
            txt_c = COLORS['type_under_text']
            label = 'understand'
        else:
            bg_c = COLORS['type_agent_bg']
            txt_c = COLORS['type_agent_text']
            label = 'agentic'
        badge_x = col_x[1]
        badge_w = col_widths[1]
        ax.add_patch(FancyBboxPatch(
            (badge_x + 0.05, y - row_height / 2 - 0.15),
            badge_w - 0.1, 0.3,
            boxstyle="round,pad=0.06",
            facecolor=bg_c, edgecolor=txt_c, linewidth=0.8, zorder=2))
        ax.text(badge_x + badge_w / 2, y - row_height / 2, label,
                fontsize=7.5, fontweight='bold', color=txt_c,
                va='center', ha='center')

        # Success
        sc = col_x[2]
        if success:
            ax.text(sc + col_widths[2] / 2, y - row_height / 2, '\u2713',
                    fontsize=14, color=COLORS['success'], fontweight='bold',
                    va='center', ha='center')
        else:
            ax.text(sc + col_widths[2] / 2, y - row_height / 2, '\u2717',
                    fontsize=14, color=COLORS['failure'], fontweight='bold',
                    va='center', ha='center', alpha=0.7)

        # Agentic metrics (cols 3, 4, 5)
        for ci, val in enumerate([valid, tool_calls, turns]):
            v = str(val) if val is not None else '\u2014'
            ax.text(col_x[3 + ci] + 0.1, y - row_height / 2, v,
                    fontsize=8.5, color=COLORS['text_primary'],
                    va='center', ha='left', fontweight='bold')

    # ── Vertical column dividers ──────────────────────────────────
    for cx in col_x[1:]:
        ax.plot([cx, cx],
                [start_y - (num_rows - 1) * row_height - row_height,
                 start_y + 0.62],
                color=COLORS['border_light'], linewidth=0.5, zorder=1)

    # Agentic metrics left border
    ax.plot([col_x[3], col_x[3]],
            [start_y - (num_rows - 1) * row_height - row_height,
             start_y + 0.62],
            color=COLORS['accent_teal'], linewidth=1.8, zorder=2)

    # Outer border
    outer_bottom = start_y - num_rows * row_height
    ax.add_patch(FancyBboxPatch(
        (0.2, outer_bottom), 14.1,
        (start_y + 0.62) - outer_bottom,
        boxstyle="square,pad=0",
        facecolor='none', edgecolor=COLORS['border'], linewidth=1, zorder=3))

    # Footer
    ax.text(7.25, 0.18,
            f'Model: {model} ({provider})  •  {num_rows} agentic synthesis tasks '
            f' •  Total tokens: {sum(r["tokens_used"] for r in report.get("results", [])):,}',
            fontsize=8, color=COLORS['text_muted'], ha='center', va='center',
            fontstyle='italic')

    return fig


def main():
    report_path = find_latest_report()
    print(f"Using report: {report_path}")

    rows, report = extract_table_data(report_path)
    print(f"Extracted {len(rows)} rows")

    # PNG
    fig = make_table(rows, report)
    png_out = OUTPUT_DIR / "benchmark_1comp_table.png"
    fig.savefig(png_out, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig)
    print(f"PNG saved: {png_out}")

    # SVG
    fig2 = make_table(rows, report)
    svg_out = OUTPUT_DIR / "benchmark_1comp_table.svg"
    fig2.savefig(svg_out, format='svg', dpi=150,
                 bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close(fig2)
    print(f"SVG saved: {svg_out}")


if __name__ == "__main__":
    main()
