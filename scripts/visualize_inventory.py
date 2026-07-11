#!/usr/bin/env python3
"""
Benchmark Inventory — Individual Visualizations
================================================
Generates standalone publication-quality PNG files — one per visualization
type — from ``data/tasks/official/INDEX.json``.

Output: ``data/visualizations/``
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIER_COLORS: dict[int, str] = {
    1: "#4C72B0", 2: "#55A868", 3: "#C44E52", 4: "#937860",
}
TIER_LABELS: dict[int, str] = {
    1: "Tier 1 — Beginner",
    2: "Tier 2 — Intermediate",
    3: "Tier 3 — Advanced",
    4: "Tier 4 — Expert",
}
TIER_SHORT: dict[int, str] = {1: "T1", 2: "T2", 3: "T3", 4: "T4"}

# Colour pool for individual tags (qualitative, colourblind-friendly)
TAG_PALETTE: list[str] = [
    "#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
    "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9",
    "#F0E442", "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000", "#E66101",
    "#5E3C99", "#FDB863",
]

OUTPUT_DIR_DEFAULT = "data/visualizations"


# ============================================================================
# Helpers
# ============================================================================


def load_index(index_path: str | Path) -> list[dict]:
    with open(index_path) as f:
        return json.load(f)["tasks"]


BACKGROUND = "#faf7f0"


def _text_color_for_bg(hex_bg: str) -> str:
    """Return white or dark text depending on background luminance."""
    r, g, b = int(hex_bg[1:3], 16), int(hex_bg[3:5], 16), int(hex_bg[5:7], 16)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 140 else "#2C3E50"


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
# 01 — Key Metrics Summary Card
# ============================================================================


def viz_key_metrics(tasks: list[dict], out: Path) -> None:
    total = len(tasks)
    unique = len({t["challenge_number"] for t in tasks})
    variants = total - unique
    q_total = sum(t["question_count"] for t in tasks)
    all_tags = {tag for t in tasks for tag in t["tags"]}
    tiers = len({t["tier"] for t in tasks})
    max_page = max(t["page_number"] for t in tasks)

    metrics = [
        ("Unique Puzzles", f"{unique}", "#2C3E50"),
        ("Total Variants", f"{variants}", "#7F8C8D"),
        ("Total Tasks", f"{total}", "#2C3E50"),
        ("Questions", f"{q_total}", "#2C3E50"),
        ("Tiers", f"{tiers}", "#2C3E50"),
        ("Unique Tags", f"{len(all_tags)}", "#2C3E50"),
        ("Pages in Guide", f"{max_page}", "#7F8C8D"),
        ("Tier 1 Challenges", f"{sum(1 for t in tasks if t['tier']==1)}", TIER_COLORS[1]),
        ("Tier 2 Challenges", f"{sum(1 for t in tasks if t['tier']==2)}", TIER_COLORS[2]),
        ("Tier 3 Challenges", f"{sum(1 for t in tasks if t['tier']==3)}", TIER_COLORS[3]),
        ("Tier 4 Challenges", f"{sum(1 for t in tasks if t['tier']==4)}", TIER_COLORS[4]),
        ("Questions / Tier (avg)", f"{q_total / tiers:.1f}", "#7F8C8D"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    cols, rows = 4, 3
    for i, (label, value, color) in enumerate(metrics):
        col, row = divmod(i, cols)
        x = col * 3 + 1.5
        y = 7.5 - row * 2.5
        ax.text(x, y, value, ha="center", va="center",
                fontsize=22, fontweight="bold", color=color)
        ax.text(x, y - 0.55, label, ha="center", va="center",
                fontsize=8.5, color="#566573")

    ax.set_title("Turing Tumble Benchmark — Key Metrics",
                 fontsize=16, fontweight="bold", pad=20, color="#2C3E50")
    fig.text(0.5, 0.02, f"Source: data/tasks/official/INDEX.json · {total} tasks, 30 unique challenges",
             ha="center", fontsize=7, color="#7F8C8D")

    save(fig, out)


# ============================================================================
# 02 — Tier Distribution (Donut)
# ============================================================================


def viz_tier_distribution(tasks: list[dict], out: Path) -> None:
    tier_counts = Counter(t["tier"] for t in tasks)
    tiers = sorted(tier_counts.keys())
    sizes = [tier_counts[t] for t in tiers]
    colors = [TIER_COLORS[t] for t in tiers]

    fig, ax = plt.subplots(figsize=(8, 6))
    result = ax.pie(sizes, labels=None, colors=colors, autopct="%1.1f%%",
                    pctdistance=0.82, startangle=90,
                    wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 2})
    autotexts: list = result[-1]
    for at in autotexts:
        at.set_fontsize(11); at.set_fontweight("bold")

    labels = [f"{TIER_LABELS[t].split('—')[1].strip():20s}  ({tier_counts[t]} tasks)"
              for t in tiers]
    ax.legend(result[0], labels, title="Tiers", loc="center left",
              bbox_to_anchor=(1, 0.5), frameon=False, fontsize=10, title_fontsize=11)

    ax.set_title("Challenge Distribution by Tier", fontsize=14, fontweight="bold", pad=16, color="#2C3E50")
    save(fig, out)


# ============================================================================
# 03 — Variant Breakdown (Stacked Horizontal Bar)
# ============================================================================


def viz_variant_breakdown(tasks: list[dict], out: Path) -> None:
    tv: dict[int, dict[str, int]] = defaultdict(
        lambda: {"Main": 0, "Practice A": 0, "Practice B": 0, "Practice C": 0, "Variant A": 0}
    )
    for t in tasks:
        tier = t["tier"]
        v = t.get("variant")
        if v is None:       tv[tier]["Main"] += 1
        elif v == "practice_A": tv[tier]["Practice A"] += 1
        elif v == "practice_B": tv[tier]["Practice B"] += 1
        elif v == "practice_C": tv[tier]["Practice C"] += 1
        elif v == "variant_A":  tv[tier]["Variant A"] += 1

    tiers = sorted(tv.keys())
    categories = ["Main", "Practice A", "Practice B", "Practice C", "Variant A"]
    cat_colors = ["#2C3E50", "#3498DB", "#1ABC9C", "#E67E22", "#9B59B6"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    y = np.arange(len(tiers))
    left = np.zeros(len(tiers))

    for cat, ccolor in zip(categories, cat_colors):
        values = [tv[t].get(cat, 0) for t in tiers]
        bars = ax.barh(y, values, 0.6, left=left, label=cat, color=ccolor,
                       edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                        str(val), ha="center", va="center", fontsize=9, fontweight="bold",
                        color=_text_color_for_bg(ccolor))
        left += np.array(values)

    ax.set_yticks(y)
    ax.set_yticklabels([TIER_LABELS[t] for t in tiers], fontsize=10)
    ax.set_xlabel("Number of Tasks", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=5)
    ax.set_title("Task Composition: Main vs Variant by Tier",
                 fontsize=14, fontweight="bold", pad=14, color="#2C3E50")
    ax.set_xlim(0, left.max() + 2)
    save(fig, out)


# ============================================================================
# 04 — Tag Distribution (Horizontal Bar)
# ============================================================================


def viz_tag_distribution(tasks: list[dict], out: Path) -> None:
    tag_counter: Counter[str] = Counter()
    for t in tasks:
        for tag in t["tags"]:
            tag_counter[tag] += 1

    top = tag_counter.most_common()
    top.reverse()
    labels = [f"{tag}" for tag, _ in top]
    values = [count for _, count in top]
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.15, 0.9, len(labels)))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.6, height=0.7)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=9, fontweight="bold", color="#2C3E50")
    ax.set_xlabel("Frequency (# challenges)", fontsize=11)
    ax.set_title("Concept Tag Distribution Across All 58 Tasks",
                 fontsize=14, fontweight="bold", pad=14, color="#2C3E50")
    ax.set_xlim(0, max(values) * 1.16)
    ax.tick_params(axis="y", labelsize=9)
    save(fig, out)


# ============================================================================
# 05 — Questions per Tier
# ============================================================================


def viz_questions_per_tier(tasks: list[dict], out: Path) -> None:
    tier_q = Counter()
    tier_count = Counter()
    for t in tasks:
        tier_q[t["tier"]] += t["question_count"]
        tier_count[t["tier"]] += 1

    tiers = sorted(tier_q.keys())
    values = [tier_q[t] for t in tiers]
    avg_q = [tier_q[t] / tier_count[t] for t in tiers]
    colors = [TIER_COLORS[t] for t in tiers]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([TIER_SHORT[t] for t in tiers], values, color=colors,
                  edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val, avg in zip(bars, values, avg_q):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 4,
                f"{val} total\n(avg {avg:.1f}/ch)",
                ha="center", va="top", fontsize=11, fontweight="bold", color="white")

    ax.set_ylabel("Total Questions", fontsize=11)
    ax.set_title("Procedural Understanding Questions per Tier",
                 fontsize=14, fontweight="bold", pad=14, color="#2C3E50")
    ax.set_ylim(0, max(values) * 1.2)
    save(fig, out)


# ============================================================================
# 06 — Tag × Tier Heatmap
# ============================================================================


def viz_tag_tier_heatmap(tasks: list[dict], out: Path) -> None:
    tag_counter: Counter[str] = Counter()
    tt: Counter[tuple[int, str]] = Counter()
    for t in tasks:
        for tag in t["tags"]:
            tag_counter[tag] += 1
            tt[(t["tier"], tag)] += 1

    top_tags = [tag for tag, _ in tag_counter.most_common(18)]
    tiers = [1, 2, 3, 4]
    matrix = np.zeros((len(top_tags), len(tiers)))
    for i, tag in enumerate(top_tags):
        for j, tier in enumerate(tiers):
            matrix[i, j] = tt.get((tier, tag), 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels([f"T{t}\n({sum(matrix[:,j]):.0f})" for j, t in enumerate(tiers)],
                        fontsize=9)
    ax.set_yticks(range(len(top_tags)))
    ax.set_yticklabels(top_tags, fontsize=9)

    for i in range(len(top_tags)):
        for j in range(len(tiers)):
            val = int(matrix[i, j])
            if val > 0:
                color = "white" if matrix[i, j] > matrix.max() / 2 else "#2C3E50"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

    ax.set_title("Concept Tag × Tier Co-occurrence Matrix",
                 fontsize=14, fontweight="bold", pad=16, color="#2C3E50")
    cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("Challenges", fontsize=9)
    save(fig, out)


# ============================================================================
# 07 — Challenge Family Table
# ============================================================================


def viz_challenge_table(tasks: list[dict], out: Path) -> None:
    families: dict[int, list[dict]] = defaultdict(list)
    for t in tasks:
        families[t["challenge_number"]].append(t)

    sorted_ch = sorted(families.keys())
    rows: list[list[str]] = []
    for ch_num in sorted_ch:
        entries = families[ch_num]
        main = next((e for e in entries if e.get("variant") is None), entries[0])
        variants = [e for e in entries if e.get("variant") is not None]
        var_str = ", ".join(
            e["variant"].replace("practice_", "P").replace("variant_", "V")
            for e in variants
        ) if variants else "—"
        tags = ", ".join(main["tags"][:5])
        if len(main["tags"]) > 5:
            tags += f" +{len(main['tags'])-5}"
        rows.append([
            f"ch{ch_num:02d}", main["title"], str(main["tier"]),
            var_str, str(len(variants)), tags,
            str(main["question_count"]),
        ])

    col_labels = ["ID", "Title", "Tier", "Variants", "#", "Tags", "Q"]
    col_widths = [0.7, 2.6, 0.4, 1.5, 0.3, 4.0, 0.3]

    fig, ax = plt.subplots(figsize=(14, 9.5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(rows) + 2)

    table = ax.table(cellText=rows, colLabels=col_labels, colWidths=col_widths,
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 0.88)

    # Header style
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)

    # Alternating rows + tier highlight
    tier_bg = {1: "#EBF2FA", 2: "#EDF7ED", 3: "#FDEDEC", 4: "#F5EEEA"}
    for i in range(len(rows)):
        tier = int(rows[i][2])
        bg = tier_bg.get(tier, "white") if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(bg)

    ax.set_title("Challenge Family Overview — All 30 Puzzles with Variants",
                 fontsize=14, fontweight="bold", pad=16, color="#2C3E50")
    fig.text(0.5, 0.005, "Q = question count from INDEX.json  ·  Tier 1-4 colour bands",
             ha="center", fontsize=8, color="#7F8C8D")
    save(fig, out)


# ============================================================================
# 08 — Tag Co-occurrence Network
# ============================================================================


def viz_tag_network(tasks: list[dict], out: Path) -> None:
    """Build a simple co-occurrence network: nodes = tags, edges = shared challenges."""
    tag_counter: Counter[str] = Counter()
    tag_pairs: Counter[tuple[str, str]] = Counter()
    tag_tiers: dict[str, set[int]] = defaultdict(set)

    for t in tasks:
        for tag in t["tags"]:
            tag_counter[tag] += 1
            tag_tiers[tag].add(t["tier"])
        for t1, t2 in combinations(sorted(t["tags"]), 2):
            tag_pairs[(t1, t2)] += 1

    # Keep top tags by frequency
    top_n = 20
    top_set = {tag for tag, _ in tag_counter.most_common(top_n)}

    # Radial layout: tier 1-2 on left semicircle, tier 3-4 on right
    nodes = sorted(top_set, key=lambda x: -tag_counter[x])
    n = len(nodes)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 1.0
    positions: dict[str, tuple[float, float]] = {}
    for i, node in enumerate(nodes):
        x = radius * np.cos(angles[i])
        y = radius * np.sin(angles[i])
        positions[node] = (x, y)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.axis("off")

    # Draw edges (only between top tags)
    edge_list = []
    for (t1, t2), weight in tag_pairs.items():
        if t1 in top_set and t2 in top_set:
            edge_list.append((t1, t2, weight))

    # Sort by weight so thicker edges draw last
    edge_list.sort(key=lambda x: x[2])
    max_weight = max(e[2] for e in edge_list) if edge_list else 1

    for t1, t2, weight in edge_list:
        if weight < 2:
            continue  # skip single-shared edges to reduce clutter
        x1, y1 = positions[t1]
        x2, y2 = positions[t2]
        alpha = 0.2 + 0.6 * (weight / max_weight)
        lw = 0.8 + 4.0 * (weight / max_weight)
        ax.plot([x1, x2], [y1, y2], color="#BDC3C7", lw=lw, alpha=alpha,
                zorder=1, solid_capstyle="round")

    # Draw nodes
    for node in nodes:
        x, y = positions[node]
        tier_set = tag_tiers.get(node, {1})
        avg_tier = sum(tier_set) / len(tier_set)
        color = TIER_COLORS.get(round(avg_tier), "#7F8C8D")
        size = 200 + tag_counter[node] * 30
        ax.scatter(x, y, s=size, c=color, edgecolors="white",
                   linewidth=2, zorder=5, alpha=0.9)

        # Label offset
        offset_x = 0.18 * x / (abs(x) + abs(y) + 0.01)
        offset_y = 0.18 * y / (abs(x) + abs(y) + 0.01)
        ax.annotate(node, (x, y),
                    textcoords="offset points", xytext=(offset_x * 40, offset_y * 40),
                    fontsize=max(8, 10 - len(node) * 0.4),
                    fontweight="bold", color="#2C3E50",
                    ha="center", va="center",
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85})

    # Legend: tier colors
    legend_elements = [
        Patch(facecolor=TIER_COLORS[1], label="Mainly Tier 1-2"),
        Patch(facecolor=TIER_COLORS[3], label="Mainly Tier 3-4"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=False, fontsize=9)

    ax.set_title("Concept Tag Co-occurrence Network",
                 fontsize=14, fontweight="bold", pad=12, color="#2C3E50")
    fig.text(0.5, 0.02, "Node size = frequency  ·  Edge thickness = co-occurrence strength  ·  Node colour = predominant tier",
             ha="center", fontsize=8, color="#7F8C8D")
    save(fig, out)


# ============================================================================
# 09 — Tier Tag Radar
# ============================================================================


def viz_tier_tag_radar(tasks: list[dict], out: Path) -> None:
    """Spider/radar chart showing normalized tag prevalence per tier."""
    tag_counter: Counter[str] = Counter()
    tier_tag: dict[int, Counter[str]] = defaultdict(Counter)
    for t in tasks:
        for tag in t["tags"]:
            tag_counter[tag] += 1
            tier_tag[t["tier"]][tag] += 1

    # Top tags that appear across multiple tiers
    multi_tier_tags = sorted(
        [tag for tag, _ in tag_counter.most_common(12)],
        key=lambda tag: -tag_counter[tag]
    )

    n_tags = len(multi_tier_tags)
    angles = np.linspace(0, 2 * np.pi, n_tags, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(9, 8), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for tier in [1, 2, 3, 4]:
        n_tasks_in_tier = sum(1 for t in tasks if t["tier"] == tier)
        values = [tier_tag[tier].get(tag, 0) / n_tasks_in_tier * 100 for tag in multi_tier_tags]
        values += values[:1]
        ax.fill(angles, values, alpha=0.15, color=TIER_COLORS[tier])
        ax.plot(angles, values, "o-", color=TIER_COLORS[tier], linewidth=2,
                markersize=5, label=f"{TIER_LABELS[tier].split('—')[1].strip()}")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(multi_tier_tags, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="#566573")
    ax.set_title("Tag Prevalence by Tier (Radar Chart)",
                 fontsize=14, fontweight="bold", pad=22, color="#2C3E50")
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05), frameon=False, fontsize=9)
    save(fig, out)


# ============================================================================
# 10 — Challenge Count Distribution (violin/box)
# ============================================================================


def viz_question_boxplot(tasks: list[dict], out: Path) -> None:
    """Questions-per-challenge distribution by tier."""
    tier_data: dict[int, list[int]] = defaultdict(list)
    for t in tasks:
        tier_data[t["tier"]].append(t["question_count"])

    tiers = sorted(tier_data.keys())
    data = [tier_data[t] for t in tiers]
    colors = [TIER_COLORS[t] for t in tiers]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bp = ax.boxplot(data, tick_labels=[TIER_SHORT[t] for t in tiers],
                    patch_artist=True, widths=0.5, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "#E74C3C",
                               "markersize": 7, "markeredgecolor": "white"})

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Jittered strip plot overlay
    for i, (tier, vals) in enumerate(tier_data.items()):
        jitter = np.random.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full_like(vals, i + 1, dtype=float) + jitter, vals,
                   alpha=0.7, s=30, c=TIER_COLORS[tier], edgecolors="white",
                   linewidth=0.5, zorder=5)

    ax.set_ylabel("Questions per Challenge", fontsize=11)
    ax.set_title("Question Count Distribution by Tier",
                 fontsize=14, fontweight="bold", pad=14, color="#2C3E50")
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks([2, 3, 4])
    ax.grid(axis="y", alpha=0.3)

    # Legend for mean marker
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#E74C3C",
               markersize=8, label="Mean")
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=False, fontsize=8)
    save(fig, out)


# ============================================================================
# Orchestrator
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate individual benchmark inventory visuals.")
    parser.add_argument("--index", default="data/tasks/official/INDEX.json")
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    index_path = project_root / args.index
    out_dir = project_root / args.output_dir

    if not index_path.exists():
        raise FileNotFoundError(f"INDEX.json not found: {index_path}")

    tasks = load_index(index_path)
    print(f"Loaded {len(tasks)} tasks from {index_path}")
    setup_mpl()

    # Map: number -> (function, filename)
    visualizations: list[tuple[str, callable]] = [
        ("01_key_metrics",           viz_key_metrics),
        ("02_tier_distribution",     viz_tier_distribution),
        ("03_variant_breakdown",     viz_variant_breakdown),
        ("04_tag_distribution",      viz_tag_distribution),
        ("05_questions_per_tier",    viz_questions_per_tier),
        ("06_tag_tier_heatmap",      viz_tag_tier_heatmap),
        ("07_challenge_table",       viz_challenge_table),
        ("08_tag_network",           viz_tag_network),
        ("09_tier_tag_radar",        viz_tier_tag_radar),
        ("10_question_boxplot",      viz_question_boxplot),
    ]

    print(f"\nGenerating {len(visualizations)} visualizations → {out_dir}/\n")
    for i, (name, func) in enumerate(visualizations, 1):
        path = out_dir / f"{name}.png"
        try:
            func(tasks, path)
        except Exception as exc:
            print(f"  ✗ {name}.png FAILED: {exc}")

    print(f"\nDone. {len(visualizations)} files in {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
