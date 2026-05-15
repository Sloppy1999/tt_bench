#!/usr/bin/env python3
"""Regenerate thesis-ready benchmark table from real benchmark output.
Uses: HTML, PNG, SVG output formats. Token count replaces latency column.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

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
    'failure': '#c62828',
    'accent_green': '#2e7d32',
    'accent_teal': '#00695c',
    'type_under_bg': '#e8f5e9',
    'type_under_text': '#2e7d32',
    'type_agent_bg': '#e0f2f1',
    'type_agent_text': '#00695c',
    'badge_border': '#a5d6a7',
    'badge_agent_border': '#80cbc4',
}

BENCHMARK_FILE = 'scorer/benchmark_results/benchmark_2026-05-11T23:23:31.739466.json'
OUT_HTML = 'benchmark_openai_table.html'
OUT_PNG  = 'benchmark_openai_table.png'
OUT_SVG  = 'benchmark_openai_table.svg'

with open(BENCHMARK_FILE) as f:
    d = json.load(f)

model      = d['model']
provider   = d['provider']
timestamp  = d['timestamp'][:10]
total_tasks = d['total_tasks']
successful  = d['successful']
success_rate = d['success_rate']

from collections import defaultdict
by_id = defaultdict(list)
for r in d['results']:
    by_id[r['task_id']].append(r)

var_labels = {
    'execution_trace': 'trace',
    'component_role':  'role',
    'abstraction':     'abstract',
    'synthesis':       'synth',
}

rows = []
for tid in sorted(by_id.keys()):
    runs = by_id[tid]
    r0 = runs[0]
    parts = tid.replace('tt-official-', '').rsplit('_', 1)
    board = parts[0]
    variant_key = parts[1] if len(parts) > 1 else ''
    var_label = var_labels.get(variant_key, variant_key)
    task_type = r0['task_type']
    metrics = r0['metrics']

    tokens = r0.get('tokens_used', 0)
    trace_acc = metrics.get('trace_accuracy', None)
    state_prec = metrics.get('state_precision', None)
    valid = metrics.get('valid', None)
    tool_calls = metrics.get('tool_calls_count', metrics.get('tool_calls', None))
    turns = metrics.get('turns', None)

    rows.append({
        'board': board, 'variant': var_label,
        'type': task_type, 'success': r0['success'],
        'tokens': tokens,
        'trace_acc': trace_acc, 'state_prec': state_prec,
        'valid': valid, 'tool_calls': tool_calls, 'turns': turns,
        'run_label': 'run 1',
    })

    if len(runs) > 1:
        r1 = runs[1]
        tokens2 = r1.get('tokens_used', 0)
        rows.append({
            'board': board, 'variant': var_label,
            'type': task_type, 'success': r1['success'],
            'tokens': tokens2,
            'trace_acc': r1['metrics'].get('trace_accuracy', None),
            'state_prec': r1['metrics'].get('state_precision', None),
            'valid': r1['metrics'].get('valid', None),
            'tool_calls': r1['metrics'].get('tool_calls_count', r1['metrics'].get('tool_calls', None)),
            'turns': r1['metrics'].get('turns', None),
            'run_label': 'run 2',
        })

# ── matplotlib figure ────────────────────────────────────────────────────────

N_ROWS = len(rows)
ROW_H   = 0.42
FIG_H   = max(5.5, 2.0 + N_ROWS * ROW_H + 0.8)
TABLE_TOP = FIG_H - 1.6
TABLE_BOT = 0.6

def make_fig():
    fig = plt.figure(figsize=(13.5, FIG_H), dpi=130)
    ax  = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, FIG_H)
    ax.axis('off')
    fig.patch.set_facecolor(COLORS['bg'])

    # Header band
    ax.add_patch(FancyBboxPatch((0.1, TABLE_TOP + 0.55), 13.3, 0.55,
                                boxstyle="square,pad=0",
                                facecolor=COLORS['header_bg'], edgecolor='none', zorder=1))
    ax.plot([0.1, 13.4], [TABLE_TOP + 0.55, TABLE_TOP + 0.55],
            color=COLORS['accent_green'], linewidth=2.5, zorder=3)
    ax.text(0.25, TABLE_TOP + 0.83,
            'Turing Tumble Benchmark — LLM Evaluation Results',
            fontsize=13, fontweight='bold', color=COLORS['text_primary'], va='center')
    ax.text(0.25, TABLE_TOP + 0.64,
            f'Model: {model}  ·  Provider: {provider}  ·  {timestamp}  ·  '
            f'{total_tasks} tasks  ·  Success rate: {success_rate:.1%}',
            fontsize=8.5, color=COLORS['text_secondary'], va='center')
    # Badge
    ax.add_patch(FancyBboxPatch((12.0, TABLE_TOP + 0.6), 1.2, 0.42,
                                boxstyle="round,pad=0.04",
                                facecolor=COLORS['type_under_bg'],
                                edgecolor=COLORS['success'], linewidth=1.5, zorder=2))
    ax.text(12.6, TABLE_TOP + 0.81, f'{success_rate:.0%}', fontsize=17,
            fontweight='bold', color=COLORS['success'], va='center', ha='center')

    # Stats row
    under_ok   = sum(1 for r in rows if r['type'] == 'understanding' and r['success'] == True)
    under_fail = sum(1 for r in rows if r['type'] == 'understanding' and r['success'] == False)
    agent_ok   = sum(1 for r in rows if r['type'] == 'agentic_synthesis' and r['success'] == True)
    agent_fail = sum(1 for r in rows if r['type'] == 'agentic_synthesis' and r['success'] == False)
    avg_tok = sum(r['tokens'] for r in rows) // len(rows)
    stats = [
        (str(total_tasks), 'total'),
        (str(under_ok),    'understand ✓'),
        (str(under_fail),  'understand ✗'),
        (str(agent_ok),    'agentic ✓'),
        (str(agent_fail),  'agentic ✗'),
        (f'{avg_tok:,}',   'avg tokens'),
    ]
    for si, (val, lbl) in enumerate(stats):
        sx = 0.25 + si * 2.1
        ax.add_patch(FancyBboxPatch((sx, TABLE_TOP + 0.04), 1.9, 0.44,
                                    boxstyle="round,pad=0.03",
                                    facecolor=COLORS['header_bg'],
                                    edgecolor=COLORS['border'], linewidth=0.6, zorder=1))
        ax.text(sx + 0.95, TABLE_TOP + 0.31, val, fontsize=13,
                fontweight='bold', color=COLORS['accent_green'], va='center', ha='center')
        ax.text(sx + 0.95, TABLE_TOP + 0.11, lbl, fontsize=6.5,
                color=COLORS['text_muted'], va='center', ha='center', fontweight='600')

    # Column layout
    col_x = [0.1, 1.45, 2.32, 3.05, 4.25, 5.7, 7.15, 8.6, 10.05, 11.5, 12.95]
    col_w = [1.35, 0.87, 0.73, 1.2, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 0.55]
    headers = ['Board', 'Type', 'Succ.', 'Tokens',
               'trace\naccuracy', 'state\nprecision',
               'valid', 'tool\ncalls', 'turns', 'runs', 'notes']

    hdr_y = TABLE_TOP - 0.05
    for i, (cx, cw, h) in enumerate(zip(col_x, col_w, headers)):
        bg = COLORS['accent_teal'] if i >= 6 else COLORS['accent_green']
        ax.add_patch(FancyBboxPatch((cx, hdr_y - 0.6), cw, 0.62,
                                    boxstyle="square,pad=0",
                                    facecolor=bg, edgecolor='none', zorder=1))
        lines = h.split('\n')
        if len(lines) == 1:
            ax.text(cx + cw/2, hdr_y - 0.01, h, fontsize=8, fontweight='bold',
                    color='white', va='center', ha='center')
        else:
            ax.text(cx + cw/2, hdr_y + 0.08, lines[0], fontsize=7.5, fontweight='bold',
                    color='white', va='center', ha='center')
            ax.text(cx + cw/2, hdr_y - 0.1,  lines[1], fontsize=7.5, fontweight='bold',
                    color='white', va='center', ha='center')

    # Group separators
    ax.plot([col_x[4], col_x[4]], [TABLE_BOT, hdr_y],
            color=COLORS['accent_green'], linewidth=2.2, zorder=5)
    ax.plot([col_x[6], col_x[6]], [TABLE_BOT, hdr_y],
            color=COLORS['accent_teal'], linewidth=2.2, zorder=5)

    # Data rows
    for r_idx, row in enumerate(rows):
        y = hdr_y - 0.62 - r_idx * ROW_H
        is_alt = (r_idx % 2 == 1)
        ax.add_patch(FancyBboxPatch((0.1, y - ROW_H), 13.3, ROW_H,
                                    boxstyle="square,pad=0",
                                    facecolor=COLORS['row_alt'] if is_alt else COLORS['bg'],
                                    edgecolor='none', zorder=0))
        ax.plot([0.1, 13.4], [y - ROW_H, y - ROW_H],
                color=COLORS['border_light'], linewidth=0.4, zorder=1)

        is_under = row['type'] == 'understanding'
        is_success = row['success']

        # Board
        ax.text(col_x[0] + 0.1, y - ROW_H/2 - 0.06, row['board'],
                fontsize=10.5, fontweight='bold', color=COLORS['accent_teal'],
                va='center', ha='left')
        ax.text(col_x[0] + 0.1, y - ROW_H/2 + 0.08, f"{row['variant']} ({row['run_label']})",
                fontsize=7, color=COLORS['text_secondary'], va='center', ha='left')

        # Badge
        if is_under:
            bg_c, bd_c, label = COLORS['type_under_bg'], COLORS['badge_border'], 'understand'
        else:
            bg_c, bd_c, label = COLORS['type_agent_bg'], COLORS['badge_agent_border'], 'agentic'
        ax.add_patch(FancyBboxPatch((col_x[1] + 0.05, y - ROW_H/2 - 0.15),
                                    col_w[1] - 0.1, 0.30,
                                    boxstyle="round,pad=0.07",
                                    facecolor=bg_c, edgecolor=bd_c, linewidth=0.8, zorder=2))
        ax.text(col_x[1] + col_w[1]/2, y - ROW_H/2, label,
                fontsize=7.5, fontweight='bold',
                color=COLORS['text_secondary'] if is_under else COLORS['type_agent_text'],
                va='center', ha='center')

        # Success
        if is_success == True:
            ax.text(col_x[2] + col_w[2]/2, y - ROW_H/2, '✓',
                    fontsize=14, color=COLORS['success'], fontweight='bold', va='center', ha='center')
        elif is_success == False:
            ax.text(col_x[2] + col_w[2]/2, y - ROW_H/2, '✗',
                    fontsize=14, color=COLORS['failure'], fontweight='bold', va='center', ha='center', alpha=0.65)
        else:
            ax.text(col_x[2] + col_w[2]/2, y - ROW_H/2, '—',
                    fontsize=11, color=COLORS['text_muted'], fontstyle='italic', va='center', ha='center')

        # Tokens
        ax.text(col_x[3] + col_w[3]/2, y - ROW_H/2, f'{row["tokens"]:,}',
                fontsize=8.5, color=COLORS['text_primary'], fontweight='600',
                va='center', ha='center')

        # trace_accuracy
        ta = row['trace_acc']
        if ta is not None and isinstance(ta, (int, float)):
            c = COLORS['success'] if ta >= 0.8 else (COLORS['text_primary'] if ta > 0 else COLORS['text_muted'])
            ax.text(col_x[4] + 0.08, y - ROW_H/2, f'{ta:.2f}',
                    fontsize=8.5, color=c, va='center', ha='left', fontweight='600')
            bar_x = col_x[4] + col_w[4] * 0.55
            ax.add_patch(FancyBboxPatch((bar_x, y - ROW_H/2 - 0.07), 0.30, 0.14,
                                        boxstyle="round,pad=0", facecolor='#e8e8e8', edgecolor='none'))
            ax.add_patch(FancyBboxPatch((bar_x, y - ROW_H/2 - 0.07), 0.30 * max(float(ta), 0.01),
                                        0.14, boxstyle="round,pad=0", facecolor=c, edgecolor='none', alpha=0.55))
        else:
            ax.text(col_x[4] + col_w[4]/2, y - ROW_H/2, '—',
                    fontsize=8.5, color=COLORS['text_muted'], fontstyle='italic', va='center', ha='center')

        # state_precision
        sp = row['state_prec']
        if sp is not None and isinstance(sp, (int, float)):
            c = COLORS['success'] if sp >= 0.8 else (COLORS['text_primary'] if sp > 0 else COLORS['text_muted'])
            ax.text(col_x[5] + 0.08, y - ROW_H/2, f'{sp:.2f}',
                    fontsize=8.5, color=c, va='center', ha='left', fontweight='600')
            bar_x = col_x[5] + col_w[5] * 0.55
            ax.add_patch(FancyBboxPatch((bar_x, y - ROW_H/2 - 0.07), 0.30, 0.14,
                                        boxstyle="round,pad=0", facecolor='#e8e8e8', edgecolor='none'))
            ax.add_patch(FancyBboxPatch((bar_x, y - ROW_H/2 - 0.07), 0.30 * max(float(sp), 0.01),
                                        0.14, boxstyle="round,pad=0", facecolor=c, edgecolor='none', alpha=0.55))
        else:
            ax.text(col_x[5] + col_w[5]/2, y - ROW_H/2, '—',
                    fontsize=8.5, color=COLORS['text_muted'], fontstyle='italic', va='center', ha='center')

        # Agentic metrics
        for ci, val in enumerate([row['valid'], row['tool_calls'], row['turns']]):
            if val is not None:
                c = COLORS['success'] if val else COLORS['failure']
                ax.text(col_x[6+ci] + col_w[6+ci]/2, y - ROW_H/2, str(val),
                        fontsize=8.5, color=c, va='center', ha='center', fontweight='600')
            else:
                ax.text(col_x[6+ci] + col_w[6+ci]/2, y - ROW_H/2, '—',
                        fontsize=8.5, color=COLORS['text_muted'], fontstyle='italic', va='center', ha='center')

        # Runs
        ax.text(col_x[9] + col_w[9]/2, y - ROW_H/2, '×1',
                fontsize=8, color=COLORS['text_secondary'], va='center', ha='center')

        # Notes
        note = ''
        if row['success'] == True:
            note = 'path match'
        elif row['success'] == False and ta is not None and ta > 0:
            note = f'trace={ta:.2f}'
        ax.text(col_x[10] + 0.05, y - ROW_H/2, note,
                fontsize=6.5, color=COLORS['text_muted'], va='center', ha='left')

    # Vertical dividers
    for cxi in col_x[1:]:
        ax.plot([cxi, cxi], [TABLE_BOT, hdr_y],
                color=COLORS['border_light'], linewidth=0.4, zorder=1)

    # Outer border
    ax.add_patch(FancyBboxPatch((0.1, TABLE_BOT), 13.3, hdr_y - TABLE_BOT,
                                boxstyle="square,pad=0",
                                facecolor='none', edgecolor=COLORS['border'], linewidth=1, zorder=3))

    # Footer
    ax.text(6.75, 0.2,
            f'Model: {model} ({provider})  ·  {timestamp}  ·  {len(rows)} task rows '
            f'from ch01–ch03 (3 challenges)  ·  Success criterion: correct marble path + final state match',
            fontsize=8, color=COLORS['text_muted'], ha='center', va='center', fontstyle='italic')

    return fig

fig = make_fig()
fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close(fig)
print(f"PNG: {OUT_PNG}")

fig2 = make_fig()
fig2.savefig(OUT_SVG, format='svg', bbox_inches='tight', facecolor=COLORS['bg'])
plt.close(fig2)
print(f"SVG: {OUT_SVG}")


# ── HTML table ────────────────────────────────────────────────────────────────

html_rows = ''
for row in rows:
    s = row['success']
    ta = row['trace_acc']
    sp = row['state_prec']
    tokens = row['tokens']
    is_under = row['type'] == 'understanding'

    status_icon = (
        '<span class="status-success">&#10003;</span>' if s == True else
        '<span class="status-failure">&#10007;</span>' if s == False else
        '<span class="status-pending">&#8212;</span>'
    )
    ta_str  = f'<span class="metric-val">{(ta or 0):.2f}</span>' if ta  is not None else '<span class="dash">&#8212;</span>'
    sp_str  = f'<span class="metric-val">{(sp or 0):.2f}</span>' if sp  is not None else '<span class="dash">&#8212;</span>'
    v_str   = f'<span class="metric-val">{row["valid"]}</span>' if row['valid']      is not None else '<span class="dash">&#8212;</span>'
    tc_str  = f'<span class="metric-val">{row["tool_calls"]}</span>' if row['tool_calls'] is not None else '<span class="dash">&#8212;</span>'
    tr_str  = f'<span class="metric-val">{row["turns"]}</span>' if row['turns'] is not None else '<span class="dash">&#8212;</span>'
    note = 'path match' if s == True else (f'trace={ta:.2f}' if s == False and ta and ta > 0 else '')

    html_rows += f'''<tr>
  <td class="board-cell">
    <span class="board-name">{row['board']}</span>
    <span class="variant-name">{row['variant']} ({row['run_label']})</span>
  </td>
  <td><span class="task-badge {'type-understanding' if is_under else 'type-agentic'}">{'understand' if is_under else 'agentic'}</span></td>
  <td class="status-cell">{status_icon}</td>
  <td class="token-cell">{tokens:,}</td>
  <td class="metric-cell {'under-metric' if is_under else 'inactive-cell'}">{ta_str}</td>
  <td class="metric-cell {'under-metric' if is_under else 'inactive-cell'}">{sp_str}</td>
  <td class="metric-cell {'agent-metric' if not is_under else 'inactive-cell'}">{v_str}</td>
  <td class="metric-cell {'agent-metric' if not is_under else 'inactive-cell'}">{tc_str}</td>
  <td class="metric-cell {'agent-metric' if not is_under else 'inactive-cell'}">{tr_str}</td>
  <td class="notes-cell">{note}</td>
</tr>'''

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Turing Tumble Benchmark — OpenAI Results</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=Source+Serif+4:ital,wght@0,400;0,600&display=swap');
  :root {{
    --bg: {COLORS['bg']}; --header-bg: {COLORS['header_bg']}; --row-alt: {COLORS['row_alt']};
    --text-primary: {COLORS['text_primary']}; --text-secondary: {COLORS['text_secondary']};
    --text-muted: {COLORS['text_muted']}; --border: {COLORS['border']};
    --border-light: {COLORS['border_light']}; --success: {COLORS['success']};
    --success-bg: {COLORS['type_under_bg']}; --failure: {COLORS['failure']};
    --accent-green: {COLORS['accent_green']}; --accent-teal: {COLORS['accent_teal']};
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Source Sans 3', -apple-system, sans-serif; background: var(--bg);
          color: var(--text-primary); padding: 28px 32px;
          max-width: 1600px; margin: 0 auto; font-size: 13px; line-height: 1.4; }}
  .header-block {{ margin-bottom: 18px; padding-bottom: 12px;
                   border-bottom: 2.5px solid var(--accent-green); }}
  .header-block h1 {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 17px;
                       font-weight: 600; color: var(--text-primary); letter-spacing: -0.2px; }}
  .header-meta {{ display: flex; gap: 18px; margin-top: 6px; font-size: 11px;
                  color: var(--text-secondary); flex-wrap: wrap; }}
  .header-meta strong {{ color: var(--text-primary); font-weight: 600; }}
  .summary-row {{ display: flex; gap: 10px; margin-top: 10px; flex-wrap: wrap; align-items: center; }}
  .stat-box {{ background: var(--header-bg); border: 1px solid var(--border-light); border-radius: 4px;
               padding: 5px 12px; display: flex; flex-direction: column; align-items: center; min-width: 85px; }}
  .stat-box .val {{ font-size: 16px; font-weight: 700; color: var(--accent-green); line-height: 1.2; }}
  .stat-box .lbl {{ font-size: 8.5px; text-transform: uppercase; letter-spacing: 0.5px;
                    color: var(--text-muted); font-weight: 600; margin-top: 1px; }}
  .success-badge {{ background: var(--success-bg); border: 1.5px solid var(--success); border-radius: 6px;
                    padding: 7px 16px; text-align: center; margin-left: auto; }}
  .success-badge .val {{ font-size: 18px; font-weight: 700; color: var(--success); }}
  .success-badge .lbl {{ font-size: 7.5px; text-transform: uppercase; letter-spacing: 0.5px;
                         color: var(--success); font-weight: 600; margin-top: 1px; }}
  .table-wrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  thead tr {{ background: var(--header-bg); border-bottom: 2px solid var(--accent-green); }}
  th {{ padding: 7px 9px; text-align: center; font-weight: 700; font-size: 9.5px;
       text-transform: uppercase; letter-spacing: 0.4px; color: var(--text-secondary);
       white-space: nowrap; border-right: 1px solid var(--border-light); vertical-align: bottom; }}
  th:last-child {{ border-right: none; }}
  th:first-child, th:nth-child(2), th:nth-child(3), th:nth-child(4) {{ text-align: left; }}
  tbody tr {{ border-bottom: 1px solid var(--border-light); }}
  tbody tr:nth-child(even) {{ background: var(--row-alt); }}
  tbody tr:hover {{ background: #edf4ed; }}
  td {{ padding: 6px 9px; border-right: 1px solid var(--border-light); vertical-align: middle; }}
  td:last-child {{ border-right: none; }}
  td:nth-child(n+5):nth-child(-n+6) {{ border-left: 1.5px solid var(--accent-green); }}
  td:nth-child(n+7):nth-child(-n+9) {{ border-left: 1.5px solid var(--accent-teal); }}
  .board-cell {{ text-align: left; }}
  .board-name {{ display: block; font-size: 12px; font-weight: 700; color: var(--accent-teal); }}
  .variant-name {{ display: block; font-size: 8.5px; color: var(--text-muted);
                   text-transform: uppercase; letter-spacing: 0.3px; margin-top: 1px; }}
  .task-badge {{ display: inline-block; padding: 2px 6px; border-radius: 10px; font-size: 9px;
                  font-weight: 700; text-transform: uppercase; letter-spacing: 0.3px; }}
  .type-understanding {{ background: var(--success-bg); color: var(--accent-green); border: 1px solid #c8e6c9; }}
  .type-agentic {{ background: #e0f2f1; color: var(--accent-teal); border: 1px solid #b2dfdb; }}
  .status-cell {{ text-align: center; font-size: 14px; font-weight: 700; }}
  .status-success {{ color: var(--success); }}
  .status-failure {{ color: var(--failure); opacity: 0.7; }}
  .status-pending {{ color: var(--text-muted); }}
  .token-cell {{ text-align: right; font-family: monospace; font-weight: 600; font-size: 12px; }}
  .metric-cell {{ text-align: center; }}
  .metric-val {{ font-weight: 600; font-size: 12px; }}
  .dash {{ color: var(--text-muted); font-style: italic; }}
  .inactive-cell {{ color: var(--text-muted); }}
  .under-metric {{ border-left: 1.5px solid var(--accent-green) !important; }}
  .agent-metric {{ border-left: 1.5px solid var(--accent-teal) !important; }}
  .notes-cell {{ font-size: 9px; color: var(--text-muted); font-style: italic; text-align: left; }}
  .footer {{ margin-top: 12px; font-size: 10px; color: var(--text-muted);
             font-style: italic; font-family: 'Source Serif 4', Georgia, serif; text-align: center; }}
</style>
</head>
<body>
<div class="header-block">
  <h1>Turing Tumble Benchmark — LLM Evaluation Results</h1>
  <div class="header-meta">
    <span>Model: <strong>{model}</strong></span>
    <span>Provider: <strong>{provider}</strong></span>
    <span>Date: <strong>{timestamp}</strong></span>
    <span>Boards: <strong>ch01–ch03 (3 challenges, pA variants)</strong></span>
  </div>
  <div class="summary-row">
    <div class="stat-box"><div class="val">{total_tasks}</div><div class="lbl">Total</div></div>
    <div class="stat-box"><div class="val">{successful}</div><div class="lbl">Successful</div></div>
    <div class="stat-box"><div class="val">{total_tasks - successful}</div><div class="lbl">Failed</div></div>
    <div class="stat-box"><div class="val">{sum(r['tokens'] for r in rows)//len(rows):,}</div><div class="lbl">Avg Tokens</div></div>
    <div class="success-badge">
      <div class="val">{success_rate:.1%}</div>
      <div class="lbl">Success Rate</div>
    </div>
  </div>
</div>
<div class="table-wrap">
<table>
  <thead>
    <tr>
      <th style="width:100px">Board</th>
      <th style="width:65px">Type</th>
      <th style="width:38px">Succ.</th>
      <th style="width:85px">Tokens</th>
      <th style="width:100px;border-left:1.5px solid var(--accent-green)">trace_accuracy</th>
      <th style="width:100px">state_precision</th>
      <th style="width:55px;border-left:1.5px solid var(--accent-teal)">valid</th>
      <th style="width:70px">tool_calls</th>
      <th style="width:48px">turns</th>
      <th style="width:70px">Notes</th>
    </tr>
  </thead>
  <tbody>
{html_rows}
  </tbody>
</table>
</div>
<p class="footer">
  Model: {model} ({provider}) &nbsp;&bull;&nbsp; Benchmark: {timestamp} &nbsp;&bull;&nbsp;
  {len(rows)} task rows &nbsp;&bull;&nbsp; Success criterion: correct marble path + final state match
</p>
</body>
</html>'''

with open(OUT_HTML, 'w') as f:
    f.write(html)
print(f"HTML: {OUT_HTML}")

print(f"\n{'='*50}")
print(f"Summary: {successful}/{total_tasks} successful ({success_rate:.1%})")
print(f"Token stats: min={min(r['tokens'] for r in rows):,}, "
      f"max={max(r['tokens'] for r in rows):,}, "
      f"avg={sum(r['tokens'] for r in rows)//len(rows):,}")