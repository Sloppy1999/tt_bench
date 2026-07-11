#!/usr/bin/env python3
"""
Tier 1 Complete Board Inventory — Visualizations
=================================================
Generates visualizations of ALL board files generated for Tier 1 challenges
(ch01–ch05) across every task category: official, 1comp, 2comp, scaled, mech_sub.

Output: ``data/visualizations/``
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKGROUND = "#faf7f0"
CATEGORY_COLORS: dict[str, str] = {
    "official": "#2C3E50",
    "1comp":    "#3498DB",
    "2comp":    "#E67E22",
    "scaled":   "#27AE60",
}
CATEGORY_LABELS: dict[str, str] = {
    "official": "Official\n(full challenges)",
    "1comp":    "1-Component\nShort",
    "2comp":    "2-Component\nShort",
    "scaled":   "Scaled\n(sz13/sz15/scl)",
}
CHALLENGE_NAMES: dict[int, str] = {
    1: "Gravity", 2: "Re-entry", 3: "Ignition",
    4: "Fusion", 5: "Entropy",
}
VARIANT_COLORS: dict[str, str] = {
    "main":       "#2C3E50",
    "practice_A": "#3498DB",
    "practice_B": "#1ABC9C",
    "practice_C": "#E67E22",
    "base":       "#2C3E50",
    "variants":   "#F39C12",
    "unsolvable": "#E74C3C",
    "insight":    "#9B59B6",
}
SCALE_COLORS: dict[str, str] = {
    "sz13_full":   "#1F618D",
    "sz15_full":   "#2E86C1",
    "sz13_1comp":  "#117A65",
    "sz15_1comp":  "#1ABC9C",
    "sz13_2comp":  "#B9770E",
    "sz15_2comp":  "#F39C12",
    "scl":         "#8E44AD",
}
OUTPUT_DIR_DEFAULT = "data/visualizations"


# ============================================================================
# Tier 1 Hard Data — aggregated from filesystem exploration
# ============================================================================

# Official (from INDEX.json) — challenge JSON files only, Tier 1
OFFICIAL_TIER1: list[dict] = [
    {"ch": 1, "variant": "main",       "task_id": "tt-official-ch01"},
    {"ch": 1, "variant": "practice_A", "task_id": "tt-official-ch01-pA"},
    {"ch": 2, "variant": "practice_A", "task_id": "tt-official-ch02-pA"},
    {"ch": 2, "variant": "main",       "task_id": "tt-official-ch02"},
    {"ch": 3, "variant": "practice_A", "task_id": "tt-official-ch03-pA"},
    {"ch": 3, "variant": "main",       "task_id": "tt-official-ch03"},
    {"ch": 4, "variant": "practice_A", "task_id": "tt-official-ch04-pA"},
    {"ch": 4, "variant": "practice_B", "task_id": "tt-official-ch04-pB"},
    {"ch": 4, "variant": "main",       "task_id": "tt-official-ch04"},
    {"ch": 5, "variant": "practice_A", "task_id": "tt-official-ch05-pA"},
    {"ch": 5, "variant": "main",       "task_id": "tt-official-ch05"},
]

# 1comp: per-challenge breakdown
COMP1: dict[int, dict[str, int]] = {
    1:  {"base": 1, "variants": 9,  "unsolvable": 40, "insight": 10},
    2:  {"base": 1, "variants": 9,  "unsolvable": 40, "insight": 10},
    3:  {"base": 1, "variants": 12, "unsolvable": 52, "insight": 13},
    4:  {"base": 1, "variants": 19, "unsolvable": 80, "insight": 20},
    5:  {"base": 1, "variants": 18, "unsolvable": 76, "insight": 19},
}

# 2comp: per-challenge breakdown
COMP2: dict[int, dict[str, int]] = {
    1:  {"base": 1, "variants": 8,  "unsolvable": 36, "insight": 9},
    2:  {"base": 1, "variants": 8,  "unsolvable": 36, "insight": 9},
    3:  {"base": 1, "variants": 11, "unsolvable": 48, "insight": 12},
    4:  {"base": 1, "variants": 18, "unsolvable": 76, "insight": 19},
    5:  {"base": 1, "variants": 17, "unsolvable": 72, "insight": 18},
}

# Scaled: per-challenge breakdown (updated with generated scl files)
SCALED: dict[int, dict[str, int]] = {
    1:  {"sz13_full": 1, "sz15_full": 1, "sz13_1comp": 10, "sz15_1comp": 10,
         "sz13_2comp": 9, "sz15_2comp": 9, "scl": 114},
    2:  {"sz13_full": 2, "sz15_full": 2, "sz13_1comp": 10, "sz15_1comp": 10,
         "sz13_2comp": 9, "sz15_2comp": 9,  "scl": 57},
    3:  {"sz13_full": 1, "sz15_full": 1, "sz13_1comp": 13, "sz15_1comp": 13,
         "sz13_2comp": 12, "sz15_2comp": 12, "scl": 100},
    4:  {"sz13_full": 0, "sz15_full": 0, "sz13_1comp": 20, "sz15_1comp": 20,
         "sz13_2comp": 19, "sz15_2comp": 19, "scl": 234},
    5:  {"sz13_full": 0, "sz15_full": 0, "sz13_1comp": 19, "sz15_1comp": 19,
         "sz13_2comp": 18, "sz15_2comp": 18, "scl": 222},
}

# mech_sub: 0 for Tier 1
MECH_SUB: dict[int, int] = {ch: 0 for ch in range(1, 6)}

CHALLENGES: list[int] = [1, 2, 3, 4, 5]


def cat_total(ch: int, data: dict[int, dict[str, int]]) -> int:
    return sum(data.get(ch, {}).values())


def _text_color_for_bg(hex_bg: str) -> str:
    """Return white or dark text depending on background luminance."""
    r, g, b = int(hex_bg[1:3], 16), int(hex_bg[3:5], 16), int(hex_bg[5:7], 16)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 140 else "#2C3E50"


def tier1_grand_total() -> int:
    return (
        len(OFFICIAL_TIER1)
        + sum(cat_total(ch, COMP1) for ch in CHALLENGES)
        + sum(cat_total(ch, COMP2) for ch in CHALLENGES)
        + sum(cat_total(ch, SCALED) for ch in CHALLENGES)
    )


# ============================================================================
# Helpers
# ============================================================================


def setup_mpl() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": BACKGROUND,
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": BACKGROUND,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig: plt.Figure, path: Path) -> None:
    fig.patch.set_facecolor(BACKGROUND)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=BACKGROUND)
    plt.close(fig)
    print(f"  ✓ {path.name}")


# ============================================================================
# 00 — Grand Summary Card
# ============================================================================


def viz_grand_summary(out: Path) -> None:
    total = tier1_grand_total()
    o_count = len(OFFICIAL_TIER1)
    c1_count = sum(cat_total(ch, COMP1) for ch in CHALLENGES)
    c2_count = sum(cat_total(ch, COMP2) for ch in CHALLENGES)
    sc_count = sum(cat_total(ch, SCALED) for ch in CHALLENGES)

    # File-type subtotals across 1comp+2comp
    unsolvable_total = sum(
        COMP1[ch]["unsolvable"] + COMP2[ch]["unsolvable"] for ch in CHALLENGES
    )
    insight_total = sum(
        COMP1[ch]["insight"] + COMP2[ch]["insight"] for ch in CHALLENGES
    )
    variant_total = sum(
        COMP1[ch]["variants"] + COMP2[ch]["variants"] for ch in CHALLENGES
    )
    base_total = 5 + 5  # 5 base in 1comp + 5 base in 2comp

    metrics: list[tuple[str, str, str]] = [
        ("Grand Total", f"{total:,}", "#2C3E50"),
        ("Official Challenges", str(o_count), CATEGORY_COLORS["official"]),
        ("1-Component Short", f"{c1_count:,}", CATEGORY_COLORS["1comp"]),
        ("2-Component Short", f"{c2_count:,}", CATEGORY_COLORS["2comp"]),
        ("Scaled Boards", f"{sc_count:,}", CATEGORY_COLORS["scaled"]),
        ("Unsolvable Tests", f"{unsolvable_total:,}", VARIANT_COLORS["unsolvable"]),
        ("Insight Tests", f"{insight_total:,}", VARIANT_COLORS["insight"]),
        ("Variant Boards", f"{variant_total:,}", VARIANT_COLORS["variants"]),
        ("Board Sizes", "11 / 13 / 15 / scl", "#7F8C8D"),
        ("Challenges", "ch01–ch05", "#2C3E50"),
        ("Avg. per Challenge", f"{total // 5:,} files", "#7F8C8D"),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    cols, rows = 4, 3
    for i, (label, value, color) in enumerate(metrics):
        col, row = divmod(i, cols)
        x = col * 3 + 1.5
        y = 7.5 - row * 2.5
        ax.text(x, y, value, ha="center", va="center",
                fontsize=20, fontweight="bold", color=color)
        ax.text(x, y - 0.55, label, ha="center", va="center",
                fontsize=10, color="#566573")

    ax.set_title("Tier 1 Complete Board Inventory — Grand Summary",
                 fontsize=16, fontweight="bold", pad=20, color="#2C3E50")
    fig.text(0.5, 0.005, "Source: data/tasks/ (official + 1comp + 2comp + scaled + mech_sub)  ·  ch01–ch05 only",
             ha="center", fontsize=9, color="#7F8C8D")

    save(fig, out)


# ============================================================================
# 01 — Category Donut
# ============================================================================


def viz_category_donut(out: Path) -> None:
    o_count = len(OFFICIAL_TIER1)
    c1_count = sum(cat_total(ch, COMP1) for ch in CHALLENGES)
    c2_count = sum(cat_total(ch, COMP2) for ch in CHALLENGES)
    sc_count = sum(cat_total(ch, SCALED) for ch in CHALLENGES)

    categories = ["official", "1comp", "2comp", "scaled"]
    sizes = [o_count, c1_count, c2_count, sc_count]
    colors = [CATEGORY_COLORS[c] for c in categories]
    explode = (0, 0.02, 0.02, 0.05)

    fig, ax = plt.subplots(figsize=(9, 7))
    result = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct/100*sum(sizes))):,})",
        pctdistance=0.55,
        startangle=140,
        explode=explode,
        wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 10, "fontweight": "bold"},
    )
    autotexts: list = result[-1]
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")

    labels = [
        f"{CATEGORY_LABELS[c].split(chr(10))[0]:20s}  {sizes[i]:,} files"
        for i, c in enumerate(categories)
    ]
    ax.legend(result[0], labels, title="Task Category", loc="center left",
              bbox_to_anchor=(1, 0.5), frameon=False, fontsize=10, title_fontsize=11)

    ax.set_title("Board Distribution by Task Category",
                 fontsize=14, fontweight="bold", pad=16, color="#2C3E50")
    save(fig, out)


# ============================================================================
# 02 — Per-Challenge Stacked Bar
# ============================================================================


def viz_per_challenge_stacked(out: Path) -> None:
    categories = ["official", "1comp", "2comp", "scaled"]
    cat_colors = [CATEGORY_COLORS[c] for c in categories]

    data: dict[int, list[int]] = {}
    for ch in CHALLENGES:
        data[ch] = [
            sum(1 for o in OFFICIAL_TIER1 if o["ch"] == ch),
            cat_total(ch, COMP1),
            cat_total(ch, COMP2),
            cat_total(ch, SCALED),
        ]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(CHALLENGES))
    width = 0.55
    bottom = np.zeros(len(CHALLENGES))

    for i, (cat, color) in enumerate(zip(categories, cat_colors)):
        values = [data[ch][i] for ch in CHALLENGES]
        bars = ax.bar(x, values, width, bottom=bottom, label=cat, color=color,
                      edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        str(val) if val < 100 else f"{val}",
                        ha="center", va="center", fontsize=9, fontweight="bold",
                        color=_text_color_for_bg(color))
        bottom += np.array(values)

    for ch, total_h in zip(CHALLENGES, bottom):
        ax.text(CHALLENGES.index(ch), total_h + 4, str(int(total_h)),
                ha="center", fontsize=10, fontweight="bold", color="#2C3E50")

    ax.set_xticks(x)
    ax.set_xticklabels([f"ch{ch:02d}\n{CHALLENGE_NAMES[ch]}" for ch in CHALLENGES], fontsize=9)
    ax.set_ylabel("Number of Board Files", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=4)
    ax.set_title("Per-Challenge Board Inventory (All Categories)",
                 fontsize=14, fontweight="bold", pad=16, color="#2C3E50")
    ax.set_ylim(0, max(bottom) * 1.12)
    save(fig, out)


# ============================================================================
# 03 — File Type Breakdown (1comp + 2comp)
# ============================================================================


def viz_file_type_breakdown(out: Path) -> None:
    """Compare the internal structure: base, variants, unsolvable, insight across 1comp and 2comp."""
    file_types = ["base", "variants", "unsolvable", "insight"]
    type_labels = ["Base", "Variants", "Unsolvable", "Insight"]
    type_colors = ["#2C3E50", "#F39C12", "#E74C3C", "#9B59B6"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax_idx, (comp_data, comp_label) in enumerate([(COMP1, "1-Component Short"), (COMP2, "2-Component Short")]):
        ax = axes[ax_idx]
        x = np.arange(len(CHALLENGES))
        width = 0.55
        bottom = np.zeros(len(CHALLENGES))

        for i, (ft, ft_color) in enumerate(zip(file_types, type_colors)):
            values = [comp_data[ch].get(ft, 0) for ch in CHALLENGES]
            bars = ax.bar(x, values, width, bottom=bottom, label=type_labels[i],
                          color=ft_color, edgecolor="white", linewidth=0.6)
            for bar, val in zip(bars, values):
                if val > 0 and val >= 5:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_y() + bar.get_height() / 2,
                            str(val), ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color=_text_color_for_bg(ft_color))
            bottom += np.array(values)

        ax.set_xticks(x)
        ax.set_xticklabels([f"ch{ch:02d}" for ch in CHALLENGES], fontsize=9)
        ax.set_title(comp_label, fontsize=12, fontweight="bold", color="#2C3E50")
        if ax_idx == 0:
            ax.set_ylabel("Number of Files", fontsize=11)

        # Total above each bar
        totals = [sum(comp_data[ch].values()) for ch in CHALLENGES]
        for ch_idx, total_val in zip(x, totals):
            ax.text(ch_idx, total_val + 2, str(total_val), ha="center",
                    fontsize=9, fontweight="bold", color="#2C3E50")

        ax.set_ylim(0, max(totals) * 1.1)

    axes[1].legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    fig.suptitle("File Type Composition — 1comp vs 2comp",
                 fontsize=14, fontweight="bold", color="#2C3E50", y=1.01)
    fig.tight_layout()
    save(fig, out)


# ============================================================================
# 04 — Scaled Board Breakdown
# ============================================================================


def viz_scaled_breakdown(out: Path) -> None:
    scale_types = ["sz13_full", "sz15_full", "sz13_1comp", "sz15_1comp",
                   "sz13_2comp", "sz15_2comp", "scl"]
    scale_labels = ["Full sz13", "Full sz15", "1comp sz13", "1comp sz15",
                    "2comp sz13", "2comp sz15", "scl"]
    scale_colors = [SCALE_COLORS[k] for k in scale_types]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(CHALLENGES))
    width = 0.55
    bottom = np.zeros(len(CHALLENGES))

    # Hide scl for non-ch01 since it's 0
    active_types = [(st, sl, sc) for st, sl, sc in zip(scale_types, scale_labels, scale_colors)
                    if any(SCALED[ch].get(st, 0) > 0 for ch in CHALLENGES)]

    for st, sl, sc in active_types:
        values = [SCALED[ch].get(st, 0) for ch in CHALLENGES]
        bars = ax.bar(x, values, width, bottom=bottom, label=sl, color=sc,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            if val > 2:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        str(val), ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color=_text_color_for_bg(sc))
        bottom += np.array(values)

    totals = bottom.copy()
    for ch_idx, total_val in zip(x, totals):
        ax.text(ch_idx, total_val + 3, str(int(total_val)), ha="center",
                fontsize=9, fontweight="bold", color="#2C3E50")

    ax.set_xticks(x)
    ax.set_xticklabels([f"ch{ch:02d}\n{CHALLENGE_NAMES[ch]}" for ch in CHALLENGES], fontsize=9)
    ax.set_ylabel("Number of Scaled Files", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=3)
    ax.set_title("Scaled Board Variants by Challenge and Board Size",
                 fontsize=14, fontweight="bold", pad=16, color="#2C3E50")

    # Annotation for ch01 scl outlier
    ax.annotate("ch01: 114 scl files\n(scale-factor 4–16)",
                xy=(0, SCALED[1]["scl"]), xytext=(1.5, SCALED[1]["scl"] + 60),
                arrowprops={"arrowstyle": "->", "color": "#8E44AD", "lw": 1.5},
                fontsize=9, color="#8E44AD", fontweight="bold",
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#8E44AD", "alpha": 0.9})
    ax.set_ylim(0, max(totals) * 1.2)
    save(fig, out)


# ============================================================================
# 05 — Detailed Table
# ============================================================================


def viz_detailed_table(out: Path) -> None:
    rows: list[list[str]] = []

    for ch in CHALLENGES:
        o_main = sum(1 for o in OFFICIAL_TIER1 if o["ch"] == ch and o["variant"] == "main")
        o_var = sum(1 for o in OFFICIAL_TIER1 if o["ch"] == ch and o["variant"] != "main")
        c1 = COMP1[ch]
        c2 = COMP2[ch]
        sc = SCALED[ch]
        sc_total = sum(sc.values())

        rows.append([
            f"ch{ch:02d}",
            CHALLENGE_NAMES[ch],
            str(o_main),
            str(o_var),
            str(c1["base"]), str(c1["variants"]),
            f"{c1['unsolvable']}/{c1['insight']}",
            str(sum(c1.values())),
            str(c2["base"]), str(c2["variants"]),
            f"{c2['unsolvable']}/{c2['insight']}",
            str(sum(c2.values())),
            str(sc_total),
            str(o_main + o_var + sum(c1.values()) + sum(c2.values()) + sc_total),
        ])

    col_labels = [
        "ID", "Title", "O-Main", "O-Var",
        "1C-Base", "1C-Var", "1C-Uns/Ins", "1C-Total",
        "2C-Base", "2C-Var", "2C-Uns/Ins", "2C-Total",
        "Scaled", "Total",
    ]
    col_widths = [0.55, 1.6, 0.5, 0.5, 0.55, 0.55, 0.7, 0.58, 0.55, 0.55, 0.7, 0.58, 0.55, 0.55]

    # Total row
    total_row = ["", "TOTAL", "", "", "", "", "", "", "", "", "", "", "", ""]
    for j in range(2, len(col_labels)):
        col_label = col_labels[j]
        # Skip combined columns (like "1C-Uns/Ins") that contain non-numeric strings
        if "/" in col_label:
            total_row[j] = "—"
            continue
        col_total = sum(int(rows[i][j]) for i in range(len(rows)))
        total_row[j] = str(col_total)
    rows.append(total_row)

    fig, ax = plt.subplots(figsize=(16, 5 + len(rows) * 0.35))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(rows) + 1.5)

    table = ax.table(cellText=rows, colLabels=col_labels, colWidths=col_widths,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.15)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)

    # Style rows
    tier_bg = ["#EBF2FA", "#EDF7ED", "#FDEBD0", "#F5EEEA", "#F5EEEA"]
    for i in range(len(rows) - 1):
        bg = tier_bg[i % 5] if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(bg)

    # Style total row
    for j in range(len(col_labels)):
        cell = table[len(rows), j]
        cell.set_facecolor("#D5D8DC")
        cell.set_text_props(fontweight="bold", fontsize=9)

    ax.set_title("Tier 1 Complete Inventory — Detailed Breakdown",
                 fontsize=14, fontweight="bold", pad=16, color="#2C3E50")
    fig.text(0.5, 0.005, "O = Official  ·  1C = 1-Component Short  ·  2C = 2-Component Short  ·  Uns = Unsolvable  ·  Ins = Insight",
             ha="center", fontsize=9, color="#7F8C8D")
    save(fig, out)


# ============================================================================
# 06 — 1comp vs 2comp Comparison Grid
# ============================================================================


def viz_comp_grid_comparison(out: Path) -> None:
    """Side-by-side comparison of variant counts and unsolvable counts per challenge."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(CHALLENGES))
    width = 0.35

    # Variant comparison
    c1_vars = [COMP1[ch]["variants"] for ch in CHALLENGES]
    c2_vars = [COMP2[ch]["variants"] for ch in CHALLENGES]
    bars1 = ax1.bar(x - width / 2, c1_vars, width, label="1-Component",
                    color=CATEGORY_COLORS["1comp"], edgecolor="white", linewidth=0.8)
    bars2 = ax1.bar(x + width / 2, c2_vars, width, label="2-Component",
                    color=CATEGORY_COLORS["2comp"], edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars1, c1_vars):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(val), ha="center", fontsize=9, fontweight="bold", color=CATEGORY_COLORS["1comp"])
    for bar, val in zip(bars2, c2_vars):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(val), ha="center", fontsize=9, fontweight="bold", color=CATEGORY_COLORS["2comp"])

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"ch{ch:02d}" for ch in CHALLENGES], fontsize=9)
    ax1.set_ylabel("Variant Board Files", fontsize=11)
    ax1.legend(fontsize=9, frameon=False)
    ax1.set_title("Variant Counts", fontsize=12, fontweight="bold", color="#2C3E50")
    ax1.set_ylim(0, max(max(c1_vars), max(c2_vars)) * 1.18)

    # Unsolvable comparison
    c1_uns = [COMP1[ch]["unsolvable"] for ch in CHALLENGES]
    c2_uns = [COMP2[ch]["unsolvable"] for ch in CHALLENGES]
    bars1 = ax2.bar(x - width / 2, c1_uns, width, label="1-Component",
                    color=CATEGORY_COLORS["1comp"], edgecolor="white", linewidth=0.8, alpha=0.8)
    bars2 = ax2.bar(x + width / 2, c2_uns, width, label="2-Component",
                    color=CATEGORY_COLORS["2comp"], edgecolor="white", linewidth=0.8, alpha=0.8)

    for bar, val in zip(bars1, c1_uns):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(val), ha="center", fontsize=9, fontweight="bold", color=CATEGORY_COLORS["1comp"])
    for bar, val in zip(bars2, c2_uns):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(val), ha="center", fontsize=9, fontweight="bold", color=CATEGORY_COLORS["2comp"])

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"ch{ch:02d}" for ch in CHALLENGES], fontsize=9)
    ax2.set_ylabel("Unsolvable Test Files", fontsize=11)
    ax2.legend(fontsize=9, frameon=False)
    ax2.set_title("Unsolvable Counts", fontsize=12, fontweight="bold", color="#2C3E50")
    ax2.set_ylim(0, max(max(c1_uns), max(c2_uns)) * 1.18)

    fig.suptitle("1-Component vs 2-Component Short — Variant & Unsolvable Distribution",
                 fontsize=14, fontweight="bold", color="#2C3E50", y=1.02)
    fig.tight_layout()
    save(fig, out)


# ============================================================================
# 07 — File Type Treemap-style (sunburst alternative)
# ============================================================================


def viz_file_type_sunburst(out: Path) -> None:
    """Multi-level donut showing the hierarchy: Category → File Type."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Outer ring: per-category total
    o_total = len(OFFICIAL_TIER1)
    c1_total = sum(cat_total(ch, COMP1) for ch in CHALLENGES)
    c2_total = sum(cat_total(ch, COMP2) for ch in CHALLENGES)
    sc_total = sum(cat_total(ch, SCALED) for ch in CHALLENGES)

    outer_sizes = [o_total, c1_total, c2_total, sc_total]
    outer_colors = [CATEGORY_COLORS[c] for c in ["official", "1comp", "2comp", "scaled"]]
    outer_labels = ["Official", "1-Component", "2-Component", "Scaled"]

    # Inner ring: file-type subtotals (1comp + 2comp combined, scaled detail)
    c1_base = sum(COMP1[ch]["base"] for ch in CHALLENGES)
    c1_var = sum(COMP1[ch]["variants"] for ch in CHALLENGES)
    c1_uns = sum(COMP1[ch]["unsolvable"] for ch in CHALLENGES)
    c1_ins = sum(COMP1[ch]["insight"] for ch in CHALLENGES)
    c2_base = sum(COMP2[ch]["base"] for ch in CHALLENGES)
    c2_var = sum(COMP2[ch]["variants"] for ch in CHALLENGES)
    c2_uns = sum(COMP2[ch]["unsolvable"] for ch in CHALLENGES)
    c2_ins = sum(COMP2[ch]["insight"] for ch in CHALLENGES)

    sc_full = sum(SCALED[ch].get("sz13_full", 0) + SCALED[ch].get("sz15_full", 0) for ch in CHALLENGES)
    sc_1c = sum(SCALED[ch].get("sz13_1comp", 0) + SCALED[ch].get("sz15_1comp", 0) for ch in CHALLENGES)
    sc_2c = sum(SCALED[ch].get("sz13_2comp", 0) + SCALED[ch].get("sz15_2comp", 0) for ch in CHALLENGES)
    sc_scl = sum(SCALED[ch].get("scl", 0) for ch in CHALLENGES)

    inner_sizes = [
        o_total,
        c1_base, c1_var, c1_uns, c1_ins,
        c2_base, c2_var, c2_uns, c2_ins,
        sc_full, sc_1c, sc_2c, sc_scl,
    ]
    inner_colors = [
        CATEGORY_COLORS["official"],
        *["#85C1E9", "#AED6F1", "#D6EAF8", "#EBF5FB"],  # 1comp light palette
        *["#F0B27A", "#F5CBA7", "#FADBD8", "#FDEDEC"],  # 2comp light palette
        *["#82E0AA", "#A9DFBF", "#D5F5E3", "#EAFAF1"],  # scaled light palette
    ]
    inner_labels = [
        "Official",
        "Base", "Variants", "Unsolvable", "Insight",
        "Base", "Variants", "Unsolvable", "Insight",
        "Full", "1comp", "2comp", "scl",
    ]

    # Outer ring
    outer_result = ax.pie(
        outer_sizes, radius=1, labels=None, colors=outer_colors,
        autopct="%1.1f%%", pctdistance=0.85,
        startangle=90,
        wedgeprops={"width": 0.3, "edgecolor": "white", "linewidth": 2},
    )
    outer_autotexts: list = outer_result[-1]
    for at in outer_autotexts:
        at.set_fontsize(9); at.set_fontweight("bold")

    # Inner ring
    inner_result = ax.pie(
        inner_sizes, radius=0.7, labels=None, colors=inner_colors,
        startangle=90,
        wedgeprops={"width": 0.35, "edgecolor": "white", "linewidth": 1},
    )

    # Center text
    ax.text(0, 0, f"{tier1_grand_total():,}\nfiles", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#2C3E50")

    # Legend for categories
    legend_elements = [
        Patch(facecolor=CATEGORY_COLORS["official"], label="Official"),
        Patch(facecolor="#85C1E9", label="1comp — Base"),
        Patch(facecolor="#AED6F1", label="1comp — Variants"),
        Patch(facecolor="#D6EAF8", label="1comp — Unsolvable"),
        Patch(facecolor="#EBF5FB", label="1comp — Insight"),
        Patch(facecolor="#F0B27A", label="2comp — Base"),
        Patch(facecolor="#F5CBA7", label="2comp — Variants"),
        Patch(facecolor="#FADBD8", label="2comp — Unsolvable"),
        Patch(facecolor="#FDEDEC", label="2comp — Insight"),
        Patch(facecolor="#82E0AA", label="Scaled — Full"),
        Patch(facecolor="#A9DFBF", label="Scaled — 1comp"),
        Patch(facecolor="#D5F5E3", label="Scaled — 2comp"),
        Patch(facecolor="#EAFAF1", label="Scaled — scl"),
    ]
    ax.legend(handles=legend_elements, loc="center left",
              bbox_to_anchor=(1.15, 0.5), frameon=False, fontsize=8.5,
              ncol=1, title="File Categories", title_fontsize=10)

    ax.set_title("Tier 1 Board Inventory — Hierarchical Breakdown",
                 fontsize=14, fontweight="bold", pad=16, color="#2C3E50")
    save(fig, out)


# ============================================================================
# Orchestrator
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier 1 complete board inventory visualization.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / args.output_dir

    setup_mpl()
    print(f"Tier 1 total boards: {tier1_grand_total():,}")
    print(f"Generating visualizations → {out_dir}/\n")

    visualizations: list[tuple[str, Callable]] = [
        ("tier1_00_grand_summary",       viz_grand_summary),
        ("tier1_01_category_donut",      viz_category_donut),
        ("tier1_02_per_challenge",       viz_per_challenge_stacked),
        ("tier1_03_file_type_breakdown", viz_file_type_breakdown),
        ("tier1_04_scaled_breakdown",    viz_scaled_breakdown),
        ("tier1_05_detailed_table",      viz_detailed_table),
        ("tier1_06_1comp_vs_2comp",      viz_comp_grid_comparison),
        ("tier1_07_hierarchical_donut",  viz_file_type_sunburst),
    ]

    for name, func in visualizations:
        path = out_dir / f"{name}.png"
        try:
            func(path)
        except Exception as exc:
            print(f"  ✗ {name}.png FAILED: {exc}")

    print(f"\nDone. {len(visualizations)} files in {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
