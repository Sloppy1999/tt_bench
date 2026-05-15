#!/usr/bin/env python3
"""Generate thesis-ready benchmark table as PNG and SVG using matplotlib."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import numpy as np

# Academic green palette
COLORS = {
    'bg': '#ffffff',
    'header_bg': '#f5f7f4',
    'row_alt': '#f9fbf9',
    'text_primary': '#1a2e1a',
    'text_secondary': '#3d5c3d',
    'text_muted': '#6b8e6b',
    'border': '#c8dcc8',
    'border_light': '#d8e8d8',
    'success': '#2e7d32',
    'success_bg': '#e8f5e9',
    'failure': '#c62828',
    'failure_bg': '#fce8e8',
    'accent_green': '#388e3c',
    'accent_teal': '#00695c',
    'type_under_bg': '#e8f5e9',
    'type_under_text': '#2e7d32',
    'type_agent_bg': '#e0f2f1',
    'type_agent_text': '#00695c',
}

data = [
    # Board, Task ID suffix, type, success, latency, trace_acc, state_prec, valid, tool_calls, turns
    ('ch01-pA', 'execution_trace', 'understanding', True,  0, 0.85, 0.80, None, None, None),
    ('ch01-pA', 'component_role',  'understanding', False, 0, 0.30, 0.25, None, None, None),
    ('ch01-pA', 'abstraction',     'understanding', False, 0, 0.10, 0.15, None, None, None),
    ('ch01-pA', 'synthesis',       'agentic',      False, 0, None, None, 0, 0, 0),
    ('ch02-pA', 'execution_trace', 'understanding', False, 0, 0.70, 0.65, None, None, None),
    ('ch02-pA', 'component_role',  'understanding', False, 0, 0.20, 0.30, None, None, None),
    ('ch02-pA', 'abstraction',     'understanding', False, 0, 0.40, 0.35, None, None, None),
    ('ch02-pA', 'synthesis',       'agentic',      False, 0, None, None, 0, 0, 0),
    ('ch03-pA', 'execution_trace', 'understanding', False, 0, 0.55, 0.50, None, None, None),
    ('ch03-pA', 'abstraction',     'understanding', False, 0, 0.15, 0.20, None, None, None),
    ('ch03-pA', 'synthesis',       'agentic',      False, 0, None, None, 0, 0, 0),
]

def make_table(figsize=(14.5, 6.5)):
    fig = plt.figure(figsize=figsize, dpi=120)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    # ── Header background ──────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0.2, 5.6), 14.1, 0.85,
                                boxstyle="square,pad=0", facecolor=COLORS['header_bg'],
                                edgecolor=COLORS['accent_green'], linewidth=0,
                                zorder=1))

    # Title
    ax.text(0.4, 6.18, 'Turing Tumble Benchmark: LLM Evaluation Results',
            fontsize=13, fontweight='bold', color=COLORS['text_primary'],
            va='center', ha='left', fontfamily='DejaVu Sans')

    # Meta line
    ax.text(0.4, 5.82, 'Model: gpt-4 (mock)  •  2026-05-05  •  5 tasks evaluated  •  Success rate: 40%',
            fontsize=9, color=COLORS['text_secondary'], va='center', ha='left')

    # Success rate badge
    ax.add_patch(FancyBboxPatch((12.8, 5.68), 1.3, 0.65,
                                boxstyle="round,pad=0.04",
                                facecolor=COLORS['success_bg'],
                                edgecolor=COLORS['success'], linewidth=1.2, zorder=2))
    ax.text(13.45, 6.01, '40%', fontsize=15, fontweight='bold',
            color=COLORS['success'], va='center', ha='center')

    # ── Table header row ───────────────────────────────────────────
    col_x = [0.2, 1.35, 2.25, 2.9, 3.95, 6.2, 8.5, 11.0, 13.0]
    col_widths = [1.15, 0.9, 0.65, 1.05, 2.25, 2.3, 2.5, 2.0, 1.5]
    headers = ['Board', 'Task Type', 'Success', 'Latency', 'trace_accuracy', 'state_precision', 'valid', 'tool_calls', 'turns']
    header_bg = COLORS['accent_green']

    for i, (cx, cw, h) in enumerate(zip(col_x, col_widths, headers)):
        ax.add_patch(FancyBboxPatch((cx, 4.85), cw, 0.62,
                                    boxstyle="square,pad=0",
                                    facecolor=header_bg, edgecolor='none', zorder=1))
        ax.text(cx + cw/2, 5.16, h,
                fontsize=8.5, fontweight='bold', color='white',
                va='center', ha='center', rotation=0)

    # Sub-headers for metric groups
    ax.text(4.57, 4.68, 'Understanding Metrics', fontsize=7.5,
            color=COLORS['accent_green'], va='center', ha='center',
            fontstyle='italic', fontweight='bold')
    ax.text(7.5, 4.68, 'Agentic Metrics', fontsize=7.5,
            color=COLORS['accent_teal'], va='center', ha='center',
            fontstyle='italic', fontweight='bold')

    # ── Data rows ──────────────────────────────────────────────────
    row_height = 0.48
    start_y = 4.38

    for r_idx, row in enumerate(data):
        y = start_y - r_idx * row_height
        is_alt = (r_idx % 2 == 1)
        is_success = row[3]

        # Row background
        bg = COLORS['row_alt'] if is_alt else COLORS['bg']
        ax.add_patch(FancyBboxPatch((0.2, y - row_height), 14.1, row_height,
                                    boxstyle="square,pad=0",
                                    facecolor=bg, edgecolor='none', zorder=0))
        # Bottom border
        ax.plot([0.2, 14.3], [y - row_height, y - row_height],
                color=COLORS['border_light'], linewidth=0.5, zorder=1)

        # Board
        bx = col_x[0]
        ax.text(bx + 0.1, y - row_height/2 - 0.07, row[0],
                fontsize=9.5, fontweight='bold', color=COLORS['accent_teal'],
                va='center', ha='left')
        ax.text(bx + 0.1, y - row_height/2 + 0.09, row[1],
                fontsize=7.5, color=COLORS['text_secondary'],
                va='center', ha='left')

        # Task type badge
        if row[2] == 'understanding':
            bg_c = COLORS['type_under_bg']
            txt_c = COLORS['type_under_text']
            label = 'understand'
        else:
            bg_c = COLORS['type_agent_bg']
            txt_c = COLORS['type_agent_text']
            label = 'agentic'
        badge_x = col_x[1]
        badge_w = col_widths[1]
        ax.add_patch(FancyBboxPatch((badge_x + 0.05, y - row_height/2 - 0.15),
                                    badge_w - 0.1, 0.3,
                                    boxstyle="round,pad=0.06",
                                    facecolor=bg_c, edgecolor=txt_c, linewidth=0.8, zorder=2))
        ax.text(badge_x + badge_w/2, y - row_height/2, label,
                fontsize=7.5, fontweight='bold', color=txt_c,
                va='center', ha='center')

        # Success
        sc = col_x[2]
        if is_success:
            ax.text(sc + col_widths[2]/2, y - row_height/2, '✓',
                    fontsize=14, color=COLORS['success'], fontweight='bold',
                    va='center', ha='center')
        else:
            ax.text(sc + col_widths[2]/2, y - row_height/2, '✗',
                    fontsize=14, color=COLORS['failure'], fontweight='bold',
                    va='center', ha='center', alpha=0.7)

        # Latency
        lx = col_x[3]
        ax.text(lx + 0.05, y - row_height/2, f'{row[4]} ms',
                fontsize=8.5, color=COLORS['text_primary'],
                va='center', ha='left')
        bar_w = 0.5
        bar_x_start = lx + 0.6
        ax.add_patch(FancyBboxPatch((bar_x_start, y - row_height/2 - 0.06),
                                    bar_w, 0.12,
                                    boxstyle="round,pad=0",
                                    facecolor='#e0e0e0', edgecolor='none'))
        fill_pct = min(row[4] / 2000.0, 1.0) if row[4] > 0 else 0.0
        ax.add_patch(FancyBboxPatch((bar_x_start, y - row_height/2 - 0.06),
                                    bar_w * fill_pct, 0.12,
                                    boxstyle="round,pad=0",
                                    facecolor=COLORS['accent_green'], edgecolor='none', alpha=0.5))

        # Understanding metrics (cols 4,5)
        if row[2] == 'understanding':
            ta = f'{row[5]:.2f}' if row[5] is not None else '—'
            sp = f'{row[6]:.2f}' if row[6] is not None else '—'
            ax.text(col_x[4] + 0.1, y - row_height/2, ta,
                    fontsize=8.5, color=COLORS['text_primary'],
                    va='center', ha='left', fontweight='bold')
            ax.text(col_x[5] + 0.1, y - row_height/2, sp,
                    fontsize=8.5, color=COLORS['text_primary'],
                    va='center', ha='left', fontweight='bold')
        else:
            for cx in [col_x[4], col_x[5]]:
                ax.text(cx + col_widths[4]/2, y - row_height/2, '—',
                        fontsize=8, color=COLORS['text_muted'], va='center', ha='center',
                        fontstyle='italic')

        # Agentic metrics (cols 6,7,8)
        if row[2] == 'agentic':
            for ci, val in enumerate([row[7], row[8], row[9]]):
                v = str(val) if val is not None else '—'
                ax.text(col_x[6+ci] + 0.1, y - row_height/2, v,
                        fontsize=8.5, color=COLORS['text_primary'],
                        va='center', ha='left', fontweight='bold')
        else:
            for ci in range(3):
                ax.text(col_x[6+ci] + col_widths[6+ci]/2, y - row_height/2, '—',
                        fontsize=8, color=COLORS['text_muted'], va='center', ha='center',
                        fontstyle='italic')

    # Vertical column dividers
    for cx in col_x[1:]:
        ax.plot([cx, cx], [start_y - (len(data)-1)*row_height, start_y + 0.62],
                color=COLORS['border_light'], linewidth=0.5, zorder=1)

    # Left border for metric groups
    ax.plot([col_x[4], col_x[4]], [start_y - (len(data)-1)*row_height, start_y + 0.62],
            color=COLORS['accent_green'], linewidth=1.8, zorder=2)
    ax.plot([col_x[6], col_x[6]], [start_y - (len(data)-1)*row_height, start_y + 0.62],
            color=COLORS['accent_teal'], linewidth=1.8, zorder=2)

    # Outer border
    ax.add_patch(FancyBboxPatch((0.2, start_y - (len(data)-1)*row_height),
                                14.1, start_y + 0.62 - (start_y - (len(data)-1)*row_height),
                                boxstyle="square,pad=0",
                                facecolor='none', edgecolor=COLORS['border'], linewidth=1, zorder=3))

    # Footer
    ax.text(7.25, 0.22, 'Model: gpt-4 (mock provider)  •  11 unique tasks across 3 Turing Tumble boards  •  Success criterion: correct marble path + final state match',
            fontsize=8, color=COLORS['text_muted'], ha='center', va='center', fontstyle='italic')

    return fig

# Generate PNG
fig = make_table((14.5, 6.5))
fig.savefig('/home/hacktheduck/Projects/Thesis/LLM_exp/benchmark_table.png',
            dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close(fig)
print("PNG saved.")

# Generate SVG
fig2 = make_table((14.5, 6.5))
fig2.savefig('/home/hacktheduck/Projects/Thesis/LLM_exp/benchmark_table.svg',
             format='svg', bbox_inches='tight', facecolor=COLORS['bg'])
plt.close(fig2)
print("SVG saved.")