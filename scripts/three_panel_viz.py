#!/usr/bin/env python3
"""
three_panel_viz.py
------------------
Generate a presentation-quality three-panel visualization of a Turing Tumble board:

  [ LEFT ]   Python code
  [ MIDDLE ] JSON representation  (truncated for readability)
  [ RIGHT ]  Rendered board       (solution state)

Designed for master-thesis slides: large fonts, high contrast, clean layout.
"""

import json
import sys
import os
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "simulator"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.font_manager as fm
import numpy as np

from board_renderer import render_board
from board_renderer import COLOURS as BR_COLOURS


# ═══════════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════════

CHALLENGE_PATH = (
    Path(__file__).resolve().parent.parent
    / "tasks/official/challenges/json/tt-official-ch01.json"
)
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "board_three_panel.png"

DPI = 250                          # high-DPI for projection / print
FIG_WIDTH_IN  = 20.0               # slide-friendly wide aspect
FIG_HEIGHT_IN = 9.5

# Panel horizontal fractions
CODE_FRAC   = 0.20
JSON_FRAC   = 0.31
BOARD_FRAC  = 0.40
GAP         = 0.015

LEFT_MARGIN   = 0.030
RIGHT_MARGIN  = 0.015
TOP_MARGIN    = 0.055
BOTTOM_MARGIN = 0.040

# ── Colour palette (clean academic) ──
BG            = "none"  # transparent background
PANEL_BG      = "#FCFCFC"
PANEL_BORDER  = "#CCCCCC"
TITLE_COLOUR  = "#222222"
SUBTITLE_CLR  = "#666666"
SEPARATOR_CLR = "#DDDDDD"

CODE_FG       = "#2E3440"
CODE_KEYWORD  = "#3B6EA5"
CODE_STRING   = "#527A3A"
CODE_NUMBER   = "#B85C38"
CODE_COMMENT  = "#909090"
CODE_PUNCT    = "#555555"

JSON_KEY      = "#3B6EA5"
JSON_STR      = "#527A3A"
JSON_NUM      = "#B85C38"
JSON_PUNCT    = "#555555"

PANEL_HEAD_BG = "#F0F0F0"


# ═══════════════════════════════════════════════════════════════════════
#  Load challenge
# ═══════════════════════════════════════════════════════════════════════

def load_challenge(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
#  Code generator — real, runnable Python
# ═══════════════════════════════════════════════════════════════════════

def generate_python_code(task: dict) -> str:
    bd = task["board"]
    sol = task.get("solution", {})
    h = bd["ball_hoppers"]
    tl = bd.get("trigger_levers", {})
    w, H = bd["width"], bd["height"]
    b_count = h["blue"]["count"]
    r_count = h["red"]["count"]

    lines = [
        '"""Build and simulate a Turing Tumble board."""',
        "from tt_sim import (",
        "    Board, Ramp, Bit,",
        "    Crossover, Interceptor,",
        "    GearBit, Gear, Trigger,",
        "    build_gear_connections,",
        ")",
        "",
        f"board = Board(",
        f"    rows={H}, cols={w},",
        f"    blue_hopper_x={h['blue']['x']}, red_hopper_x={h['red']['x']},",
        f"    blue_hopper_count={b_count}, red_hopper_count={r_count},",
        f"    hopper_entry_mode='inward',",
        f"    left_catcher_x={tl.get('left',{}).get('x')},",
        f"    right_catcher_x={tl.get('right',{}).get('x')},",
        ")",
        "",
    ]

    if bd.get("fixed_components"):
        lines.append("# ── fixed components ──")
        for c in bd["fixed_components"]:
            lines.append(_real_place(c))
        lines.append("")

    if sol.get("placed_components"):
        lines.append("# ── solution parts ──")
        for c in sol["placed_components"]:
            lines.append(_real_place(c))
        lines.append("")

    lines.extend([
        "# detect & connect adjacent gears",
        "build_gear_connections(board)",
        "",
        "# release marbles and trace results",
        f"for i in range({b_count}):",
        "    r = board.release_marble("
        "'blue')",
        "    print(f'marble {i+1}: "
        "{r.caught_by}')",
        "",
        "print(board.render())",
    ])
    return "\n".join(lines)


def _real_place(c: dict) -> str:
    t = c["type"]
    x, y = c["x"], c["y"]
    if t == "ramp_right":
        return f"board.place({x}, {y}, Ramp({x}, {y}, 'right'))"
    elif t == "ramp_left":
        return f"board.place({x}, {y}, Ramp({x}, {y}, 'left'))"
    elif t == "crossover":
        return f"board.place({x}, {y}, Crossover({x}, {y}))"
    elif t == "bit":
        return f"board.place({x}, {y}, Bit({x}, {y}, state={c.get('state',0)}))"
    elif t == "gear_bit":
        s = c.get("state", 0)
        g = c.get("gear_group", -1)
        return f"board.place({x}, {y}, GearBit({x}, {y}, state={s}, gear_group={g}))"
    elif t == "gear":
        return f"board.place({x}, {y}, Gear({x}, {y}))"
    elif t == "interceptor":
        return f"board.place({x}, {y}, Interceptor({x}, {y}, side='{c.get('side','left')}'))"
    elif t == "trigger":
        return f"board.place({x}, {y}, Trigger({x}, {y}, side='{c.get('side','blue')}'))"
    return f"board.place({x}, {y}, Component.from_dict({json.dumps(c)}))"


# ═══════════════════════════════════════════════════════════════════════
#  JSON — truncated for presentation readability
# ═══════════════════════════════════════════════════════════════════════

def format_json(task: dict, max_lines: int = 55) -> str:
    """Return a truncated JSON representation suitable for a slide.

    Keeps the structure visible while cutting verbose sections.
    """
    slim = json.loads(json.dumps(task))

    # Truncate long explanation
    if "solution" in slim and "explanation" in slim["solution"]:
        explanation = slim["solution"]["explanation"]
        if len(explanation) > 120:
            slim["solution"]["explanation"] = explanation[:117] + "..."

    text = json.dumps(slim, indent=2)
    lines = text.split("\n")

    if len(lines) <= max_lines:
        return text

    # Keep head + tail with a gap marker
    head = lines[: max_lines - 3]
    tail = lines[-3:]
    return "\n".join(head + ["  ...", f"  # ({len(lines) - max_lines} more lines)", "  ...", ""] + tail)


# ═══════════════════════════════════════════════════════════════════════
#  Tokeniser for syntax highlighting
# ═══════════════════════════════════════════════════════════════════════

PY_KEYWORDS = {
    "from", "import", "def", "class", "return", "if", "elif", "else",
    "for", "while", "in", "not", "and", "or", "True", "False", "None",
    "print", "range", "with", "as", "try", "except", "raise",
}


def _tokenise_line(line: str, rules: dict) -> list[tuple[str, str]]:
    tokens = []
    i, L = 0, len(line)
    while i < L:
        ch = line[i]
        if ch == "#":
            tokens.append((line[i:], rules.get("comment", "#808080")))
            break
        if ch in "\"'":
            end = ch
            j = i + 1
            while j < L and line[j] != end:
                if line[j] == "\\":
                    j += 1
                j += 1
            if j < L:
                j += 1
            tokens.append((line[i:j], rules.get("string", "#527A3A")))
            i = j
            continue
        if ch.isdigit() or (ch == "-" and i + 1 < L and line[i + 1].isdigit()):
            j = i
            if ch == "-":
                j += 1
            while j < L and line[j].isdigit():
                j += 1
            tokens.append((line[i:j], rules.get("number", "#B85C38")))
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < L and (line[j].isalnum() or line[j] == "_"):
                j += 1
            word = line[i:j]
            color = rules.get("keyword", rules["default"]) if word in PY_KEYWORDS else rules["default"]
            tokens.append((word, color))
            i = j
            continue
        tokens.append((ch, rules.get("punct", "#555555")))
        i += 1
    return tokens


# ═══════════════════════════════════════════════════════════════════════
#  Text panel renderer
# ═══════════════════════════════════════════════════════════════════════

def _render_text_panel(
    ax,
    text: str,
    title: str,
    mono_font: str,
    colour_scheme: dict,
    panel_width_in: float,
    font_size: float = 8.0,
    line_h: float | None = None,
    char_ratio: float = 0.62,
):
    """Draw a syntax-highlighted text panel with a header bar.

    Parameters
    ----------
    char_ratio : float
        Monospace char-width / font-size ratio (0.60–0.64 typical).
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Panel background
    rect = FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.025",
        linewidth=0.8,
        edgecolor=colour_scheme["border"],
        facecolor=colour_scheme["bg"],
        transform=ax.transAxes, zorder=0,
    )
    ax.add_patch(rect)

    # Header bar
    bar_h = 0.045
    bar = Rectangle(
        (0.012, 0.945), 0.976, bar_h,
        linewidth=0, edgecolor="none",
        facecolor=colour_scheme["head_bg"],
        transform=ax.transAxes, zorder=1,
    )
    ax.add_patch(bar)
    ax.text(
        0.5, 0.945 + bar_h / 2, title,
        ha="center", va="center",
        fontsize=9.5, fontweight="bold",
        color=colour_scheme["title"],
        fontfamily="sans-serif",
        transform=ax.transAxes, zorder=2,
    )

    # Character advance (monospace)
    char_w_ax = (char_ratio * font_size / 72.0) / panel_width_in

    # Line height
    if line_h is None:
        line_h = font_size / 180.0

    # Render lines
    lines = text.split("\n")
    x0 = 0.038
    y_start = 0.930
    visible_area = 0.930 - 0.025   # roughly usable height in axes coords
    max_visible = int(visible_area / line_h)

    if len(lines) > max_visible:
        lines = lines[:max_visible - 1] + [f"  ... ({len(lines) - max_visible + 1} more lines)"]

    highlight_rules = colour_scheme.get("highlight_rules", {})

    for line_idx, line in enumerate(lines):
        y = y_start - line_idx * line_h
        if not line:
            continue
        tokens = _tokenise_line(line, highlight_rules)
        x = x0
        for tok_text, tok_colour in tokens:
            ax.text(
                x, y, tok_text,
                fontsize=font_size,
                fontfamily=mono_font,
                color=tok_colour,
                va="center",
                transform=ax.transAxes,
            )
            x += len(tok_text) * char_w_ax


# ═══════════════════════════════════════════════════════════════════════
#  Board → image
# ═══════════════════════════════════════════════════════════════════════

def _board_to_image(task: dict) -> np.ndarray:
    buf = io.BytesIO()
    fig = render_board(task, state="solution", show_title=False)
    fig.savefig(buf, format="png", dpi=200,
                pad_inches=0.05,
                facecolor="none", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    task = load_challenge(str(CHALLENGE_PATH))
    task_id = task["task_id"]
    title   = task.get("title", "")

    code_text = generate_python_code(task)
    json_text = format_json(task, max_lines=60)

    c_lines = len(code_text.splitlines())
    j_lines = len(json_text.splitlines())

    print(f"Rendering → {task_id}  ·  {title}")
    print(f"  Code: {c_lines} lines   JSON: {j_lines} lines (truncated from 126)")

    # ── Monospace font ───────────────────────────────────────────────
    mono_font = "monospace"
    for candidate in ["DejaVu Sans Mono", "Liberation Mono",
                       "Source Code Pro", "Courier New"]:
        try:
            fm.findfont(candidate, fallback_to_default=False)
            mono_font = candidate
            break
        except Exception:
            continue

    # ── Figure ───────────────────────────────────────────────────────
    fig = plt.figure(
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        facecolor=BG,
    )

    code_x  = LEFT_MARGIN
    json_x  = LEFT_MARGIN + CODE_FRAC + GAP
    board_x = LEFT_MARGIN + CODE_FRAC + JSON_FRAC + 2 * GAP
    panel_h = 1.0 - TOP_MARGIN - BOTTOM_MARGIN
    panel_y = BOTTOM_MARGIN

    ax_code  = fig.add_axes([code_x,  panel_y, CODE_FRAC,  panel_h])
    ax_json  = fig.add_axes([json_x,  panel_y, JSON_FRAC,  panel_h])
    ax_board = fig.add_axes([board_x, panel_y, BOARD_FRAC, panel_h])

    code_w_in = FIG_WIDTH_IN * CODE_FRAC
    json_w_in = FIG_WIDTH_IN * JSON_FRAC

    # ── Draw code panel ──────────────────────────────────────────────
    code_scheme = {
        "bg": PANEL_BG, "border": PANEL_BORDER,
        "title": TITLE_COLOUR, "head_bg": PANEL_HEAD_BG,
        "highlight_rules": {
            "comment": CODE_COMMENT,
            "string":  CODE_STRING,
            "number":  CODE_NUMBER,
            "keyword": CODE_KEYWORD,
            "punct":   CODE_PUNCT,
            "default": CODE_FG,
        },
    }
    _render_text_panel(
        ax_code, code_text,
        title="Python",
        mono_font=mono_font,
        colour_scheme=code_scheme,
        panel_width_in=code_w_in,
        font_size=9.0,
        line_h=0.0225,
    )

    # ── Draw JSON panel ──────────────────────────────────────────────
    json_scheme = {
        "bg": PANEL_BG, "border": PANEL_BORDER,
        "title": TITLE_COLOUR, "head_bg": PANEL_HEAD_BG,
        "highlight_rules": {
            "comment": CODE_COMMENT,
            "string":  JSON_STR,
            "number":  JSON_NUM,
            "keyword": JSON_KEY,
            "punct":   JSON_PUNCT,
            "default": CODE_FG,
        },
    }
    _render_text_panel(
        ax_json, json_text,
        title="JSON  (task schema)",
        mono_font=mono_font,
        colour_scheme=json_scheme,
        panel_width_in=json_w_in,
        font_size=8.5,
        line_h=0.0156,
    )

    # ── Draw board panel ─────────────────────────────────────────────
    print("  Rendering board …")
    board_img = _board_to_image(task)
    ax_board.imshow(board_img, aspect="equal")
    ax_board.axis("off")

    # Header bar matching the text panels
    bar_h = 0.045
    bar = Rectangle(
        (0.012, 0.945), 0.976, bar_h,
        linewidth=0, edgecolor="none",
        facecolor=PANEL_HEAD_BG,
        transform=ax_board.transAxes, zorder=10,
    )
    ax_board.add_patch(bar)
    ax_board.text(
        0.5, 0.945 + bar_h / 2, "Rendered board  (solution)",
        ha="center", va="center",
        fontsize=9.5, fontweight="bold",
        color=TITLE_COLOUR, fontfamily="sans-serif",
        transform=ax_board.transAxes, zorder=11,
    )

    # ── Save ─────────────────────────────────────────────────────────
    output = str(OUTPUT_PATH)
    fig.savefig(
        output, dpi=DPI,
        facecolor="none", edgecolor="none",
        transparent=True,
    )
    plt.close(fig)

    w_px = int(FIG_WIDTH_IN * DPI)
    h_px = int(FIG_HEIGHT_IN * DPI)
    file_kb = os.path.getsize(output) / 1024
    print(f"\n✔  {output}")
    print(f"   {file_kb:.0f} KB  |  {w_px}×{h_px} px  |  {DPI} dpi")


if __name__ == "__main__":
    main()
