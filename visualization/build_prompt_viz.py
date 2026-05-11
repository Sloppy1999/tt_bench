#!/usr/bin/env python3
"""
Generate thesis-ready visualizations for Turing Tumble benchmark prompt anatomy.
Outputs: HTML, SVG, PNG, Markdown
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patheffects as pe

# ── Color Palette (academic grayscale + blue accent) ──────────────────────────
BG = "#fafafa"
TEXT = "#1a1a1a"
GRAY_LIGHT = "#e8e8e8"
GRAY_MID = "#b0b0b0"
GRAY_DARK = "#555555"
BLUE_DEEP = "#1a3a6b"
BLUE_ACCENT = "#2c5aa0"
BLUE_LIGHT = "#d4e4f7"
BLUE_MID = "#4a7ab3"
GRAY_CODE_BG = "#f0f0f0"
LAYER_COLORS = [
    "#2c5aa0",  # System prompt — deep blue (most authority)
    "#3d6b99",  # Board JSON — mid blue
    "#4a7ab3",  # Task objective — medium blue
    "#5a8ac4",  # Component rules — lighter blue
    "#7ba3d4",  # Output format — lightest blue
]

# ── Layer definitions ──────────────────────────────────────────────────────────
LAYERS = [
    {
        "name": "System Prompt",
        "abbr": "SYS",
        "color": LAYER_COLORS[0],
        "description": "Role definition & behavioral constraints",
        "purpose": "Defines the LLM's identity (expert analyst / solver agent) and critical rules like the no-free-fall constraint.",
    },
    {
        "name": "Board (JSON)",
        "abbr": "JSON",
        "color": LAYER_COLORS[1],
        "description": "Full board configuration with solution",
        "purpose": "Complete board geometry: dimensions, hoppers, catchers, fixed components, and for understanding tasks — the reference solution.",
    },
    {
        "name": "Task Objective",
        "abbr": "OBJ",
        "color": LAYER_COLORS[2],
        "description": "Puzzle goal & question",
        "purpose": "What the board should achieve (e.g., 'all blue balls reach the end') and the specific question to answer or behavior to produce.",
    },
    {
        "name": "Component Rules",
        "abbr": "RUL",
        "color": LAYER_COLORS[3],
        "description": "Physics of each component type",
        "purpose": "Canonical rules for Ramps, Bits, GearBits, Gears, Crossover, Interceptor, Trigger — including the free-fall rule and coordinate conventions.",
    },
    {
        "name": "Output Format",
        "abbr": "FMT",
        "color": LAYER_COLORS[4],
        "description": "Response structure specification",
        "purpose": "Exact JSON shape expected back: keys, types, constraints (e.g., final_destination + reasoning for understanding; final_solution + verification for synthesis).",
    },
]

# ── Real ch01 example data ────────────────────────────────────────────────────

SYSTEM_UNDERSTANDING = """You are an expert Turing Tumble analyst.
Given a board configuration, analyze its behavior and answer questions about it.
Respond ONLY with valid JSON in the specified format."""

SYSTEM_AGENTIC = """You are a Turing Tumble solver agent.
You MUST use the provided tools to solve this puzzle. You cannot solve it by just thinking,
you MUST call the tools.

CRITICAL CONSTRAINT: Marbles may NOT fall through empty cells. Every cell a marble visits
between entering the board and reaching a catcher/interceptor MUST contain a component.
Solutions with any empty-cell traversal will be rejected even if the catcher counts are correct.

REQUIRED WORKFLOW (you MUST follow this exactly):
1. First call get_board_state to see what's already placed
2. Call place_component to add components from your available parts
3. Call run_simulation to test if it works
4. If wrong, adjust with more place_component or remove_component calls
5. Repeat steps 3-4 until the solution is correct
6. ONLY when simulation shows correct results, output your final solution

You MUST call run_simulation after EVERY component placement to verify!
Do not just think about the solution - you must USE the tools to build and test it."""

BOARD_JSON_CH01 = {
    "width": 11,
    "height": 11,
    "dimensions": "11x11",
    "components": [
        {"type": "ramp_right", "x": 2, "y": 0},
        {"type": "ramp_left",  "x": 3, "y": 1},
        {"type": "ramp_right", "x": 2, "y": 2},
        {"type": "ramp_left",  "x": 3, "y": 3},
        {"type": "ramp_right", "x": 2, "y": 4},
        {"type": "ramp_left",  "x": 3, "y": 5},
        # solution ramps (understanding task includes solution)
        {"type": "ramp_right", "x": 2, "y": 6, "role": "solution"},
        {"type": "ramp_left",  "x": 3, "y": 7, "role": "solution"},
        {"type": "ramp_right", "x": 2, "y": 8, "role": "solution"},
        {"type": "ramp_left",  "x": 3, "y": 9, "role": "solution"},
    ],
    "ball_hoppers": {"blue": {"x": 2, "count": 8}, "red": {"x": 8, "count": 8}},
    "trigger_levers": {"left": {"x": 2}, "right": {"x": 8}},
    "entry_mode": "inward",
}

OBJECTIVE_CH01 = "Make all of the blue balls (and only the blue balls) reach the end."

RULES_EXCERPT = """Turing Tumble Component Rules:
- CRITICAL: Marbles may NOT fall through empty in-board cells. Every cell a marble
  visits after entering the board must contain a component until reaching a
  catcher or interceptor. Solutions with free-fall gaps are INVALID.
- RAMP_RIGHT: Marble entering from above always exits to the lower-right.
- RAMP_LEFT: Marble entering from above always exits to the lower-left.
- BIT (state 0 pointing right): Marble exits lower-right AND bit flips to state 1.
- BIT (state 1 pointing left):  Marble exits lower-left  AND bit flips to state 0.
- GEAR_BIT: Behaves like BIT on impact. When one flips, every gear_bit in the same
  `gear_groups` entry flips with it (instantly, before the marble exits).
- GEAR: Couples neighbouring gear_bits; does not redirect marbles on its own.
- CROSSOVER: Marble entering from upper-left exits lower-right; upper-right exits lower-left.
- INTERCEPTOR: Marble is caught and the current run ends.
- TRIGGER: Marble passes through AND queues the release of one ball from the OPPOSITE-coloured hopper.
- Ball hoppers: a marble from hopper `side` enters the playfield at column `ball_hoppers.<side>.entry_x`, starting at y=0.
- Trigger levers (catchers): a marble that falls off the bottom is caught only if its column equals
  `trigger_levers.left.x` (left_catcher) or `trigger_levers.right.x` (right_catcher). Any other bottom column is a miss."""

QUESTION_CH01 = "After the 1st blue marble, where does it end up?"
ANSWER_FORMAT_CH01 = '{"final_destination": "left_catcher" or "right_catcher", "reasoning": "step by step..."}'

AVAILABLE_PARTS_CH01 = "ramp_right: 2, ramp_left: 2, crossover: 0, bit: 0, gear_bit: 0, gear: 0, interceptor: 0, trigger: 0"


# ── Helper: syntax-highlight a code block ─────────────────────────────────────
def highlight_json(text: str) -> str:
    """Very lightweight JSON syntax highlighter using HTML spans."""
    import re
    text = re.sub(r'("(?:[^"\\]|\\.)*")\s*:', r'<span class="json-key">\1</span>:', text)
    text = re.sub(r':\s*("(?:[^"\\]|\\.)*")', r': <span class="json-str">\1</span>', text)
    text = re.sub(r':\s*(\d+(?:\.\d+)?)', r': <span class="json-num">\1</span>', text)
    text = re.sub(r':\s*(true|false|null)', r': <span class="json-bool">\1</span>', text)
    return text


# ── Figure A: Full anatomy diagram ───────────────────────────────────────────
def build_layer_diagram(output_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor(BG)

    ax.text(6, 7.7, "Benchmark Prompt Anatomy — Layer Structure",
            ha='center', va='center', fontsize=17, fontweight='bold',
            color=TEXT, fontfamily='DejaVu Serif')
    ax.text(6, 7.35, "Turing Tumble Puzzle-Solving Benchmark",
            ha='center', va='center', fontsize=10, color=GRAY_DARK)

    n = len(LAYERS)
    total_h = 6.2
    box_h = total_h / n - 0.12
    start_y = 1.1
    left_x = 0.3
    box_w = 4.8
    ann_x = 5.4

    for i, layer in enumerate(reversed(LAYERS)):
        y_top = start_y + i * (box_h + 0.12)
        y_bot = y_top + box_h

        rect = FancyBboxPatch((left_x, y_bot), box_w, box_h,
                              boxstyle="round,pad=0.1",
                              facecolor=layer["color"],
                              edgecolor=TEXT, linewidth=1.0, zorder=2)
        ax.add_patch(rect)

        ax.text(left_x + 0.2, y_top + box_h - 0.25, layer["name"],
                ha='left', va='top', fontsize=11, fontweight='bold',
                color='white', zorder=3)
        ax.text(left_x + 0.2, y_top + box_h - 0.55, layer["description"],
                ha='left', va='top', fontsize=8.5, color='#d0e0f0', zorder=3)

        ax.text(left_x + box_w - 0.15, y_top + box_h - 0.2, layer["abbr"],
                ha='right', va='top', fontsize=8, fontweight='bold',
                color='white', fontfamily='monospace', zorder=3)

        ax.text(ann_x, y_top + box_h - 0.3, layer["purpose"],
                ha='left', va='top', fontsize=8, color=TEXT,
                style='italic', zorder=3,
                bbox=dict(boxstyle='round,pad=0.25', facecolor=layer["color"],
                          edgecolor=GRAY_MID, linewidth=0.5, alpha=0.12))

        circle = Circle((ann_x - 0.25, y_top + box_h / 2), 0.1,
                            color=layer["color"], zorder=4)
        ax.add_patch(circle)

        if i > 0:
            ax.annotate('', xy=(ann_x - 0.25, y_bot + 0.02),
                        xytext=(ann_x - 0.25, y_top - 0.02),
                        arrowprops=dict(arrowstyle='-', color=GRAY_MID,
                                        lw=0.8, linestyle='dashed'))

    ax.text(0.15, 4.2, "Prompt\nNesting\nOrder",
            ha='center', va='center', fontsize=8, color=GRAY_DARK, rotation=90)

    ax.annotate('', xy=(2.4, 0.4), xytext=(2.4, 0.9),
                arrowprops=dict(arrowstyle='->', color=BLUE_ACCENT, lw=2))
    ax.text(2.4, 0.25, "-> LLM Input",
            ha='center', va='top', fontsize=10, fontweight='bold',
            color=BLUE_ACCENT)

    ax.annotate('', xy=(11.2, 7.4), xytext=(11.2, 1.2),
                arrowprops=dict(arrowstyle='->', color=GRAY_MID, lw=1.2))
    ax.text(11.35, 4.3, "Data\nFlow",
            ha='left', va='center', fontsize=8, color=GRAY_DARK, rotation=90)

    badges = [
        (8.0, 7.0, "Procedural\nUnderstanding", "#3d6b99"),
        (8.0, 5.8, "Agentic\nSynthesis", "#4a7ab3"),
    ]
    for bx, by, blabel, bcol in badges:
        rect_b = FancyBboxPatch((bx, by), 3.4, 0.9,
                                 boxstyle="round,pad=0.1",
                                 facecolor=bcol, edgecolor=TEXT,
                                 linewidth=0.8, alpha=0.85, zorder=2)
        ax.add_patch(rect_b)
        ax.text(bx + 1.7, by + 0.45, blabel,
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white', zorder=3)
        ax.plot([bx, 6.2], [by + 0.45, by + 0.45],
                color=bcol, lw=1.0, linestyle='--', zorder=1)

    plt.tight_layout(pad=0.5)
    fig.savefig(output_dir / "prompt_anatomy_layer_diagram.png",
                  dpi=300, bbox_inches='tight', facecolor=BG)
    print(f"  -> {output_dir / 'prompt_anatomy_layer_diagram.png'}")
    fig.savefig(output_dir / "prompt_anatomy_layer_diagram.svg",
                  format='svg', bbox_inches='tight', facecolor=BG)
    print(f"  -> {output_dir / 'prompt_anatomy_layer_diagram.svg'}")
    plt.close(fig)


# ── Figure B: Task-type comparison ───────────────────────────────────────────
def build_task_comparison(output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    fig.patch.set_facecolor(BG)

    task_data = [
        ("Procedural Understanding", [
            "System Prompt (expert analyst)",
            "Board JSON + Reference Solution",
            "Task Objective & Question",
            "Component Rules",
            "Output Format (JSON answer)",
        ]),
        ("Agentic Synthesis", [
            "System Prompt (solver agent + workflow)",
            "Board JSON (fixed only, no solution)",
            "Task Objective & Available Parts",
            "Component Rules",
            "Output Format (tool-calling JSON)",
        ]),
    ]

    for ax, (title, items) in zip(axes, task_data):
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(title, fontsize=13, fontweight='bold', color=BLUE_DEEP, pad=14)

        box_h = 1.35
        gap = 0.12
        start_y = 0.5
        for i, item in enumerate(reversed(items)):
            color_idx = i
            y_top = start_y + i * (box_h + gap)
            rect = FancyBboxPatch((0.3, y_top), 5.4, box_h,
                                  boxstyle="round,pad=0.08",
                                  facecolor=LAYER_COLORS[color_idx],
                                  edgecolor='white', linewidth=0.6,
                                  zorder=2, alpha=0.92)
            ax.add_patch(rect)
            ax.text(0.5, y_top + box_h - 0.35, item,
                    ha='left', va='top', fontsize=9, fontweight='bold',
                    color='white', zorder=3)

    plt.tight_layout(pad=0.5)
    fig.savefig(output_dir / "prompt_anatomy_task_comparison.png",
                  dpi=300, bbox_inches='tight', facecolor=BG)
    print(f"  -> {output_dir / 'prompt_anatomy_task_comparison.png'}")
    fig.savefig(output_dir / "prompt_anatomy_task_comparison.svg",
                  format='svg', bbox_inches='tight', facecolor=BG)
    print(f"  -> {output_dir / 'prompt_anatomy_task_comparison.svg'}")
    plt.close(fig)


# ── Figure C: ch01 understanding prompt ──────────────────────────────────────
def build_understanding_fig(output_dir: Path):
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.axis('off')
    fig.patch.set_facecolor(BG)

    sections = [
        ("SYSTEM PROMPT", SYSTEM_UNDERSTANDING, LAYER_COLORS[0]),
        ("BOARD JSON (with solution)", json.dumps(BOARD_JSON_CH01, indent=2), LAYER_COLORS[1]),
        ("COMPONENT RULES", RULES_EXCERPT[:900], LAYER_COLORS[3]),
        ("QUESTION", f"## Question Type: execution_trace\n\n## Question: {QUESTION_CH01}\n\n## Expected Format:\n{ANSWER_FORMAT_CH01}", LAYER_COLORS[2]),
    ]

    y = 11.5
    for label, content, color in sections:
        lines = content.count('\n') + 2
        box_h = max(1.0, min(lines * 0.22 + 0.8, 4.5))
        rect = FancyBboxPatch((0.3, y - box_h), 15.4, box_h,
                              boxstyle="round,pad=0.1",
                              facecolor='white', edgecolor=color,
                              linewidth=2.0, zorder=2)
        ax.add_patch(rect)
        ax.text(0.45, y - 0.25, label,
                ha='left', va='top', fontsize=9, fontweight='bold',
                color=color, zorder=3)
        ax.text(0.45, y - 0.55, content[:700],
                ha='left', va='top', fontsize=7, color=TEXT,
                fontfamily='monospace', zorder=3)
        y -= box_h + 0.25

    ax.text(8, 0.3,
            "Procedural Understanding -- Complete Prompt Example (tt-official-ch01)",
            ha='center', va='bottom', fontsize=11, fontweight='bold', color=BLUE_DEEP)

    plt.tight_layout(pad=0.5)
    fig.savefig(output_dir / "prompt_anatomy_understanding_example.png",
                  dpi=300, bbox_inches='tight', facecolor=BG)
    print(f"  -> {output_dir / 'prompt_anatomy_understanding_example.png'}")
    plt.close(fig)


# ── Figure D: ch01 agentic prompt ─────────────────────────────────────────────
def build_agentic_fig(output_dir: Path):
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('off')
    fig.patch.set_facecolor(BG)

    board_no_sol = {k: v for k, v in BOARD_JSON_CH01.items()
                    if k != "components"}
    board_no_sol["components"] = [c for c in BOARD_JSON_CH01["components"]
                                   if c.get("role") != "solution"]

    sections_d = [
        ("SYSTEM PROMPT", SYSTEM_AGENTIC[:600], LAYER_COLORS[0]),
        ("BOARD JSON (fixed only)", json.dumps(board_no_sol, indent=2), LAYER_COLORS[1]),
        ("AVAILABLE PARTS", AVAILABLE_PARTS_CH01, LAYER_COLORS[2]),
        ("COMPONENT RULES", RULES_EXCERPT[:700], LAYER_COLORS[3]),
        ("OUTPUT FORMAT", '{"final_solution": [...], "success": true, "verification": {...}}', LAYER_COLORS[4]),
    ]

    y = 11.5
    for label, content, color in sections_d:
        lines = content.count('\n') + 2
        box_h = max(1.0, min(lines * 0.22 + 0.8, 4.5))
        rect = FancyBboxPatch((0.3, y - box_h), 15.4, box_h,
                              boxstyle="round,pad=0.1",
                              facecolor='white', edgecolor=color,
                              linewidth=2.0, zorder=2)
        ax.add_patch(rect)
        ax.text(0.45, y - 0.25, label,
                ha='left', va='top', fontsize=9, fontweight='bold',
                color=color, zorder=3)
        ax.text(0.45, y - 0.55, content[:700],
                ha='left', va='top', fontsize=7, color=TEXT,
                fontfamily='monospace', zorder=3)
        y -= box_h + 0.25

    ax.text(8, 0.3,
            "Agentic Synthesis -- Complete Prompt Example (tt-official-ch01)",
            ha='center', va='bottom', fontsize=11, fontweight='bold', color=BLUE_DEEP)

    plt.tight_layout(pad=0.5)
    fig.savefig(output_dir / "prompt_anatomy_agentic_example.png",
                  dpi=300, bbox_inches='tight', facecolor=BG)
    print(f"  -> {output_dir / 'prompt_anatomy_agentic_example.png'}")
    plt.close(fig)


# ── HTML output ────────────────────────────────────────────────────────────────
def build_html(output_path: Path):
    board_json_str = json.dumps(BOARD_JSON_CH01, indent=2)
    board_no_sol = {k: v for k, v in BOARD_JSON_CH01.items() if k != "components"}
    board_no_sol["components"] = [c for c in BOARD_JSON_CH01["components"]
                                   if c.get("role") != "solution"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Turing Tumble Benchmark -- Prompt Anatomy</title>
<style>
  :root {{
    --bg: {BG};
    --text: {TEXT};
    --gray-mid: {GRAY_MID};
    --blue-deep: {BLUE_DEEP};
    --blue-accent: {BLUE_ACCENT};
    --blue-light: {BLUE_LIGHT};
    --blue-mid: {BLUE_MID};
    --code-bg: {GRAY_CODE_BG};
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: Georgia, 'Linux Libertine', serif; background: var(--bg); color: var(--text); padding: 2rem; }}
  h1 {{ font-size: 1.8rem; color: var(--blue-deep); border-bottom: 3px solid var(--blue-accent); padding-bottom: 0.5rem; margin-bottom: 1.5rem; }}
  h2 {{ font-size: 1.3rem; color: var(--blue-deep); margin: 2rem 0 0.8rem; }}
  h3 {{ font-size: 1.1rem; color: var(--blue-mid); margin: 1.2rem 0 0.5rem; }}
  p {{ font-size: 0.95rem; line-height: 1.65; margin-bottom: 0.8rem; }}
  .diagram-wrap {{ background: white; border: 1px solid var(--gray-mid); border-radius: 8px; padding: 1.5rem; margin: 1.5rem 0; }}
  .layer-stack {{ display: flex; flex-direction: column; gap: 4px; margin: 1.2rem 0; }}
  .layer {{ padding: 0.6rem 1rem; border-radius: 6px; color: white; position: relative; }}
  .layer-name {{ font-size: 0.9rem; font-weight: bold; }}
  .layer-desc {{ font-size: 0.78rem; opacity: 0.85; font-style: italic; margin-top: 2px; }}
  .layer-abbr {{ position: absolute; right: 0.8rem; top: 50%; transform: translateY(-50%); font-family: monospace; font-size: 0.7rem; background: rgba(0,0,0,0.2); padding: 2px 6px; border-radius: 4px; }}
  .purpose {{ background: #f5f5f5; border-left: 4px solid var(--blue-accent); padding: 0.5rem 1rem; margin: 0.3rem 0; font-size: 0.88rem; color: #333; border-radius: 0 6px 6px 0; }}
  pre, code {{ font-family: 'Courier New', monospace; }}
  pre {{ background: var(--code-bg); border: 1px solid var(--gray-mid); border-radius: 6px; padding: 1rem; overflow-x: auto; font-size: 0.78rem; line-height: 1.5; margin: 0.8rem 0; }}
  code {{ background: var(--code-bg); padding: 2px 5px; border-radius: 3px; font-size: 0.85em; }}
  .prompt-block {{ border: 1px solid var(--gray-mid); border-radius: 8px; overflow: hidden; margin: 1.2rem 0; }}
  .prompt-header {{ background: var(--blue-accent); color: white; padding: 0.6rem 1rem; font-weight: bold; font-size: 0.9rem; }}
  .prompt-body {{ background: white; padding: 1rem; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0; }}
  @media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  .json-key {{ color: #2c5aa0; font-weight: bold; }}
  .json-str {{ color: #a31515; }}
  .json-num {{ color: #098658; }}
  .json-bool {{ color: #0000ff; }}
  .tag {{ display: inline-block; background: var(--blue-accent); color: white; font-size: 0.7rem; padding: 2px 8px; border-radius: 10px; margin-right: 4px; }}
  .caption {{ text-align: center; font-size: 0.8rem; color: var(--gray-mid); margin-top: 0.3rem; font-style: italic; }}
  .img-wrap {{ text-align: center; margin: 1.5rem 0; }}
  .img-wrap img {{ max-width: 100%; height: auto; border: 1px solid var(--gray-mid); border-radius: 6px; }}
  footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--gray-mid); font-size: 0.8rem; color: var(--gray-mid); }}
</style>
</head>
<body>
<h1>Turing Tumble Benchmark -- Prompt Anatomy</h1>
<p>This visualization shows the anatomical structure of benchmark prompts for the
<strong>Turing Tumble puzzle-solving benchmark</strong>. Two task variants are supported:
<strong>Procedural Understanding</strong> and <strong>Agentic Synthesis</strong>.</p>

<h2>1. Prompt Layer Architecture</h2>
<p>Every benchmark prompt -- regardless of task type -- is composed of five nested layers.
The system prompt wraps everything, followed by the board configuration, task objective,
component rules, and output format specification.</p>

<div class="diagram-wrap">
<div class="layer-stack">
"""

    for layer in LAYERS:
        html += f"""  <div class="layer" style="background: {layer['color']};">
    <div class="layer-name">{layer['name']}</div>
    <div class="layer-desc">{layer['description']}</div>
    <span class="layer-abbr">{layer['abbr']}</span>
  </div>
  <div class="purpose"><strong>Purpose:</strong> {layer['purpose']}</div>
"""

    html += f"""</div>
<p class="caption">Figure 1: The five-layer nesting structure of every benchmark prompt.</p>
</div>

<h2>2. Layer-by-Layer Breakdown</h2>

<h3>Layer 1 -- System Prompt</h3>
<p>The outermost wrapper. Defines the LLM's role identity and critical behavioral constraints.
Two variants exist:</p>
<div class="two-col">
<div>
<div class="prompt-block">
<div class="prompt-header">Procedural Understanding -- System Prompt</div>
<div class="prompt-body">
<pre>{SYSTEM_UNDERSTANDING}</pre>
</div>
</div>
</div>
<div>
<div class="prompt-block">
<div class="prompt-header">Agentic Synthesis -- System Prompt</div>
<div class="prompt-body">
<pre>{SYSTEM_AGENTIC}</pre>
</div>
</div>
</div>
</div>

<h3>Layer 2 -- Board JSON</h3>
<p>The board configuration (board geometry + fixed components). For procedural understanding
tasks the reference solution components are included; for agentic synthesis they are omitted.</p>
<div class="prompt-block">
<div class="prompt-header">Challenge tt-official-ch01 "Gravity" -- Board JSON (with solution)</div>
<div class="prompt-body">
<pre>{highlight_json(board_json_str)}</pre>
</div>
</div>

<h3>Layer 3 -- Task Objective</h3>
<p>The puzzle goal, drawn directly from the challenge JSON <code>objective</code> field.</p>
<div class="prompt-block">
<div class="prompt-header">tt-official-ch01 -- Objective</div>
<div class="prompt-body">
<pre>{OBJECTIVE_CH01}</pre>
</div>
</div>

<h3>Layer 4 -- Component Rules</h3>
<p>The canonical physics reference, shared by both task types. This is the
<code>COMPONENT_RULES</code> constant from <code>scorer/run_benchmark.py</code>.</p>
<div class="prompt-block">
<div class="prompt-header">Component Rules (excerpt)</div>
<div class="prompt-body">
<pre>{RULES_EXCERPT[:600]}</pre>
</div>
</div>

<h3>Layer 5 -- Output Format</h3>
<p>Specifies the JSON structure expected back from the LLM.</p>
<div class="two-col">
<div>
<div class="prompt-block">
<div class="prompt-header">Understanding -- Expected Format</div>
<div class="prompt-body">
<pre>{ANSWER_FORMAT_CH01}</pre>
</div>
</div>
</div>
<div>
<div class="prompt-block">
<div class="prompt-header">Synthesis -- Final Solution Format</div>
<div class="prompt-body">
<pre>{{
  "final_solution": [
    {{"component_type": "ramp_left", "x": 3, "y": 5}},
    {{"component_type": "bit",       "x": 5, "y": 6, "state": 0}}
  ],
  "success": true,
  "verification": {{"left_catcher": 8, "right_catcher": 0}}
}}</pre>
</div>
</div>
</div>

<h2>3. Complete Example Prompts (tt-official-ch01)</h2>

<h3>3a. Procedural Understanding -- Execution Trace Question</h3>
<div class="tag">execution_trace</div><div class="tag">understanding</div><div class="tag">ch01</div>
<div class="prompt-block">
<div class="prompt-header">SYSTEM PROMPT</div>
<div class="prompt-body">
<pre>{SYSTEM_UNDERSTANDING}</pre>
</div>
</div>
<div class="prompt-block">
<div class="prompt-header">USER PROMPT</div>
<div class="prompt-body">
<pre>Analyze this Turing Tumble board configuration.

## Board (JSON)
{board_json_str}

## Component Rules
{RULES_EXCERPT}

## Question Type: execution_trace

## Question: {QUESTION_CH01}

## Expected Answer Format
{ANSWER_FORMAT_CH01}

Respond with JSON containing your answer and reasoning.</pre>
</div>
</div>

<h3>3b. Agentic Synthesis</h3>
<div class="tag">agentic_synthesis</div><div class="tag">tool_calling</div><div class="tag">ch01</div>
<div class="prompt-block">
<div class="prompt-header">SYSTEM PROMPT</div>
<div class="prompt-body">
<pre>{SYSTEM_AGENTIC}</pre>
</div>
</div>
<div class="prompt-block">
<div class="prompt-header">USER PROMPT</div>
<div class="prompt-body">
<pre>Solve this Turing Tumble puzzle using the available tools.

## Board (JSON)
{json.dumps(board_no_sol, indent=2)}

## Available Parts
{AVAILABLE_PARTS_CH01}

## Target Behavior
{OBJECTIVE_CH01}

## Component Rules
{RULES_EXCERPT}

## Your Task
Use the tools to build and verify a solution. Placements must target empty cells.
`get_board_state` returns this same canonical JSON shape after each edit;
`run_simulation` returns catcher counts, execution traces, and final bit states.

When you have a correct solution, output:
{{
  "final_solution": [
    {{"component_type": "ramp_left", "x": 3, "y": 5}},
    {{"component_type": "bit",       "x": 5, "y": 6, "state": 0}}
  ],
  "success": true,
  "verification": {{"left_catcher": 8, "right_catcher": 0}}
}}

Use the tools now. Start by checking the current board state.</pre>
</div>
</div>

<h2>4. Key Differences: Understanding vs. Synthesis</h2>
<div class="prompt-block">
<table style="width:100%; border-collapse: collapse; font-size: 0.9rem;">
<tr style="background: var(--blue-accent); color: white;">
  <th style="padding: 0.6rem; text-align: left;">Aspect</th>
  <th style="padding: 0.6rem; text-align: left;">Procedural Understanding</th>
  <th style="padding: 0.6rem; text-align: left;">Agentic Synthesis</th>
</tr>
<tr style="background: white;">
  <td style="padding: 0.5rem; border-bottom: 1px solid #eee;"><strong>System prompt</strong></td>
  <td style="padding: 0.5rem; border-bottom: 1px solid #eee;">Expert analyst; JSON-only output</td>
  <td style="padding: 0.5rem; border-bottom: 1px solid #eee;">Solver agent; must call tools</td>
</tr>
<tr style="background: #f5f5f5;">
  <td style="padding: 0.5rem; border-bottom: 1px solid #eee;"><strong>Board JSON</strong></td>
  <td style="padding: 0.5rem; border-bottom: 1px solid #eee;">Includes reference solution components</td>
  <td style="padding: 0.5rem; border-bottom: 1px solid #eee;">Fixed components only; no solution</td>
</tr>
<tr style="background: white;">
  <td style="padding: 0.5rem; border-bottom: 1px solid #eee;"><strong>Available parts</strong></td>
  <td style="padding: 0.5rem; border-bottom: 1px solid #eee;">Not shown (analysis only)</td>
  <td style="padding: 0.5rem; border-bottom: 1px solid #eee;">Shown (what the agent can place)</td>
</tr>
<tr style="background: #f5f5f5;">
  <td style="padding: 0.5rem; border-bottom: 1px solid #eee;"><strong>LLM interaction</strong></td>
  <td style="padding: 0.5rem; border-bottom: 1px solid #eee;">Single-shot JSON response</td>
  <td style="padding: 0.5rem; border-bottom: 1px solid #eee;">Multi-turn tool calls (up to 100 turns)</td>
</tr>
<tr style="background: white;">
  <td style="padding: 0.5rem;"><strong>Output validation</strong></td>
  <td style="padding: 0.5rem;">Trace accuracy / state precision metrics</td>
  <td style="padding: 0.5rem;">Simulator replay + free-fall check + inventory check</td>
</tr>
</table>
</div>

<footer>
Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} | Turing Tumble Benchmark | Thesis Visualization
</footer>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"  -> {output_path}")


# ── Markdown output ───────────────────────────────────────────────────────────
def build_markdown(output_path: Path):
    board_json_str = json.dumps(BOARD_JSON_CH01, indent=2)
    board_no_sol = {k: v for k, v in BOARD_JSON_CH01.items() if k != "components"}
    board_no_sol["components"] = [c for c in BOARD_JSON_CH01["components"]
                                   if c.get("role") != "solution"]

    md = f"""# Turing Tumble Benchmark -- Prompt Anatomy

*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}*

---

## 1. Prompt Layer Architecture

Every benchmark prompt -- regardless of task type -- is composed of **five nested layers**:

| Layer | Name | Color | Purpose |
|-------|------|-------|---------|
| 1 | System Prompt | Blue (#2c5aa0) | Role definition + critical behavioral constraints |
| 2 | Board JSON | Mid-blue (#3d6b99) | Board geometry, hoppers, catchers, fixed components |
| 3 | Task Objective | Medium-blue (#4a7ab3) | Puzzle goal and the specific question to answer |
| 4 | Component Rules | Light-blue (#5a8ac4) | Canonical physics reference for all component types |
| 5 | Output Format | Lighter-blue (#7ba3d4) | Exact JSON response structure expected |

### Visual Layer Stack

```
+- Layer 1: System Prompt ------------------------------------------+
|  +- Layer 2: Board JSON ----------------------------------------+ |
|  |  +- Layer 3: Task Objective ------------------------------+ | |
|  |  |  +- Layer 4: Component Rules -----------------------+ | | |
|  |  |  |  +- Layer 5: Output Format --------------------+ | | | |
|  |  |  |  |  [LLM Response Structure]                  | | | | |
|  |  |  |  +---------------------------------------------+ | | |
|  |  |  +---------------------------------------------------+ | |
|  |  +-------------------------------------------------------+ |
|  +-----------------------------------------------------------+ |
+---------------------------------------------------------------+
```

---

## 2. System Prompts

### 2a. Procedural Understanding

```
{SYSTEM_UNDERSTANDING}
```

### 2b. Agentic Synthesis

```
{SYSTEM_AGENTIC.strip()}
```

---

## 3. Complete Example Prompts -- tt-official-ch01 "Gravity"

Challenge: *Make all of the blue balls (and only the blue balls) reach the end.*

### 3a. Procedural Understanding -- Execution Trace Question

#### SYSTEM PROMPT
```
{SYSTEM_UNDERSTANDING}
```

#### USER PROMPT
```
Analyze this Turing Tumble board configuration.

## Board (JSON)
{board_json_str}

## Component Rules
{RULES_EXCERPT}

## Question Type: execution_trace

## Question: {QUESTION_CH01}

## Expected Answer Format
{ANSWER_FORMAT_CH01}

Respond with JSON containing your answer and reasoning.
```

---

### 3b. Agentic Synthesis

#### SYSTEM PROMPT
```
{SYSTEM_AGENTIC.strip()}
```

#### USER PROMPT
```
Solve this Turing Tumble puzzle using the available tools.

## Board (JSON)
{json.dumps(board_no_sol, indent=2)}

## Available Parts
{AVAILABLE_PARTS_CH01}

## Target Behavior
{OBJECTIVE_CH01}

## Component Rules
{RULES_EXCERPT}

## Your Task
Use the tools to build and verify a solution. Placements must target empty cells.
`get_board_state` returns this same canonical JSON shape after each edit;
`run_simulation` returns catcher counts, execution traces, and final bit states.

When you have a correct solution, output:
{{
  "final_solution": [
    {{"component_type": "ramp_left", "x": 3, "y": 5}},
    {{"component_type": "bit",       "x": 5, "y": 6, "state": 0}}
  ],
  "success": true,
  "verification": {{"left_catcher": 8, "right_catcher": 0}}
}}

Use the tools now. Start by checking the current board state.
```

---

## 4. Understanding vs. Synthesis -- Key Differences

| Aspect | Procedural Understanding | Agentic Synthesis |
|--------|-------------------------|-------------------|
| System prompt role | Expert analyst; JSON-only output | Solver agent; must call tools |
| Board JSON | Includes reference solution components | Fixed components only; no solution |
| Available parts | Not shown | Shown (defines agent's inventory) |
| LLM interaction | Single-shot JSON response | Multi-turn tool calls (up to 100 turns) |
| Output validation | Trace accuracy / state precision | Simulator replay + free-fall check + inventory |
| Reference questions | `execution_trace`, `component_role`, `abstraction` | N/A |
| Tool schema | None | `place_component`, `remove_component`, `run_simulation`, `get_board_state` |

---

## 5. Output Files

| File | Format | Purpose |
|------|--------|---------|
| `prompt_anatomy.html` | HTML | Interactive, self-contained (open in browser) |
| `prompt_anatomy_layer_diagram.svg` | Vector SVG | LaTeX Beamer / PowerPoint insertion |
| `prompt_anatomy_task_comparison.svg` | Vector SVG | Side-by-side task-type comparison |
| `prompt_anatomy_understanding_example.png` | Raster PNG @300dpi | Word / LaTeX documents |
| `prompt_anatomy_agentic_example.png` | Raster PNG @300dpi | Word / LaTeX documents |
| `prompt_anatomy.md` | Markdown | This documentation |
"""

    with open(output_path, "w") as f:
        f.write(md)
    print(f"  -> {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────────
def main():
    output_dir = Path(__file__).parent.resolve()
    print(f"Generating visualizations in {output_dir}")

    print("\n[1/3] Generating HTML...")
    build_html(output_dir / "prompt_anatomy.html")

    print("\n[2/3] Generating PNG/SVG (matplotlib)...")
    build_layer_diagram(output_dir)
    build_task_comparison(output_dir)
    build_understanding_fig(output_dir)
    build_agentic_fig(output_dir)

    print("\n[3/3] Generating Markdown...")
    build_markdown(output_dir / "prompt_anatomy.md")

    print("\nAll outputs generated.")


if __name__ == "__main__":
    main()
