#!/usr/bin/env python3
"""Comprehensive benchmark visualization — includes official, 1comp, 2comp, and scaled tasks.

Generates publication-quality plots for thesis supervisor meetings.
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

REPO = Path(__file__).resolve().parent.parent
TASKS = REPO / "data" / "tasks"
OUTDIR = REPO / "data" / "visualizations"

# ── Palettes ─────────────────────────────────────────────────────────────────
CAT_COLORS = {
    "official": "#1f77b4",
    "1comp":    "#2ca02c",
    "2comp":    "#ff7f0e",
    "scaled":   "#9467bd",
    "mech_sub": "#e377c2",
}
SOLV_COLORS = {"solvable": "#2ca02c", "unsolvable": "#d62728"}
TIER_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}
VT_COLORS = {
    "base":             "#7f7f7f",
    "position_variant": "#17becf",
    "insight":          "#9467bd",
    "unsolvable_t1":    "#d62728",
    "unsolvable_t2":    "#e377c2",
    "unsolvable_g1":    "#ff7f0e",
    "unsolvable_g2":    "#8c564b",
    "scaled_base":      "#bcbd22",
    "scaled_var":       "#2ca02c",
    "mechsub_gearbit":  "#e377c2",
    "mechsub_independent": "#e377c2",
    "mechsub_coupled":  "#d62728",
    "mechsub_dirflip":  "#ff7f0e",
    "mechsub_other":    "#17becf",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Classification
# ═══════════════════════════════════════════════════════════════════════════════

def classify_challenge(fpath: Path) -> dict | None:
    """Extract metadata from a challenge JSON. Returns None on failure."""
    try:
        with open(fpath) as f:
            d = json.load(f)
    except Exception:
        return None

    tid = d.get("task_id", fpath.stem)
    tier = d.get("tier", 0)
    title = d.get("title", "")

    fixed = d.get("board", {}).get("fixed_components", [])
    placed = d.get("solution", {}).get("placed_components", [])
    total_comp = len(fixed) + len(placed)

    types = Counter()
    for c in fixed + placed:
        types[c["type"]] += 1

    board_w = d.get("board", {}).get("width", 11)
    board_h = d.get("board", {}).get("height", 11)
    board_size = f"{board_w}x{board_h}"

    # Category from path
    rel_str = str(fpath.relative_to(TASKS))
    parts = rel_str.split("/")

    category = "unknown"
    for p in parts:
        pl = p.lower()
        if pl == "official":
            category = "official"; break
        elif "1comp" in pl and "2comp" not in pl:
            category = "1comp"; break
        elif "2comp" in pl:
            category = "2comp"; break
        elif pl == "scaled":
            category = "scaled"; break
        elif pl == "mech_sub":
            category = "mech_sub"; break

    meta = d.get("_meta", {})
    vt = meta.get("variant_type", "")
    subtype = meta.get("unsolvable_subtype", "")

    # Variant type
    if category == "mech_sub":
        msub = meta.get("mechanism_substitution", "")
        if msub == "bit_to_gearbit":
            variant_type = f"mechsub_{meta.get('substitution_subtype', 'gearbit')}"
        elif msub == "counter_direction_flip":
            variant_type = "mechsub_dirflip"
        else:
            variant_type = "mechsub_other"
    elif category == "scaled":
        if "_var_" in tid:
            variant_type = "scaled_var"
        else:
            variant_type = "scaled_base"
    elif vt == "insight":
        variant_type = "insight"
    elif vt == "unsolvable":
        if "extra_gap" in subtype:
            num = subtype.split("_")[-1]
            variant_type = f"unsolvable_g{num}"
        elif "T1" in subtype or "same category" in subtype:
            variant_type = "unsolvable_t1"
        elif "T2" in subtype or "different category" in subtype:
            variant_type = "unsolvable_t2"
        else:
            variant_type = "unsolvable_t1"
    elif "_var_" in tid or "_position" in tid:
        variant_type = "position_variant"
    else:
        variant_type = "base"

    is_unsolvable = variant_type.startswith("unsolvable")

    # Scaled-specific fields
    scale_level = None
    sz = None
    comp_count = None
    var_num = None
    origin_ch = None

    if category == "scaled":
        # Extract scale level
        sm = re.search(r"scl(\d+)", tid)
        if sm: scale_level = int(sm.group(1))
        # Extract board size
        sm = re.search(r"sz(\d+)", tid)
        if sm: sz = int(sm.group(1))
        # Extract comp count
        sm = re.search(r"(\d)comp", tid)
        if sm: comp_count = int(sm.group(1))
        # Extract variant number
        sm = re.search(r"_var_(\d+)", tid)
        if sm: var_num = int(sm.group(1))
        # Extract origin challenge
        sm = re.search(r"ch(\d+)", tid)
        if sm: origin_ch = int(sm.group(1))
        elif re.search(r"offici[ai]l[_-](\d+)", tid):
            sm = re.search(r"offici[ai]l[_-](\d+)", tid)
            origin_ch = int(sm.group(1))

    # All placed component coordinates for zone analysis
    placed_coords = [(c["x"], c["y"]) for c in placed]

    # All fixed component coordinates
    fixed_coords = [(c["x"], c["y"]) for c in fixed]

    # Determine solvability
    solvable = not is_unsolvable

    return {
        "task_id": tid,
        "tier": tier,
        "category": category,
        "variant_type": variant_type,
        "solvable": solvable,
        "total_components": total_comp,
        "placed_count": len(placed),
        "fixed_count": len(fixed),
        "component_types": dict(types),
        "available_parts": d.get("available_parts", {}),
        "board_size": board_size,
        "board_w": board_w,
        "board_h": board_h,
        # Scaled-specific
        "scale_level": scale_level,
        "sz": sz,
        "comp_count": comp_count,
        "var_num": var_num,
        "origin_ch": origin_ch,
        # Zone analysis
        "placed_coords": placed_coords,
        "fixed_coords": fixed_coords,
    }


def collect_all(max_tier: int | None = None) -> list[dict]:
    """Scan all challenge JSONs recursively under data/tasks/."""
    results = []
    # Exclude questions directory
    for f in sorted(TASKS.rglob("*.json")):
        rel = str(f.relative_to(TASKS))
        if "questions" in rel:
            continue
        if "INDEX.json" in f.name:
            continue
        c = classify_challenge(f)
        if c:
            if max_tier is not None and c["tier"] > max_tier:
                continue
            results.append(c)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_dataset_growth(data: list[dict], outdir: Path):
    """Before (53 Tier 1-2) vs After (filtered) growth bar."""
    total = len(data)
    before = 53  # 33 official T1-T2 + 20 synthetic T1-T2

    fig, ax = plt.subplots(figsize=(8, 4.5))

    stages = ["Original T1-T2\n(Jul 2025)", "Current T1-T2\n(Jul 2026)"]
    counts = [before, total]
    colors = ["#7f7f7f", "#2ca02c"]

    bars = ax.bar(stages, counts, color=colors, edgecolor="white", width=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                str(count), ha="center", fontsize=16, fontweight="bold")

    ax.set_ylabel("Number of Tasks", fontsize=13)
    ax.set_title("Dataset Growth (Tier 1-2 Only)", fontsize=15, fontweight="bold")
    ax.set_ylim(0, total * 1.2)
    ax.spines[["top", "right"]].set_visible(False)

    # Growth annotation
    ax.annotate(f"+{total - before} tasks\n({(total/before - 1)*100:.0f}% increase)",
                xy=(1, total), xytext=(1.3, total * 0.7),
                fontsize=12, fontweight="bold", color="#2ca02c",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fff0", edgecolor="#2ca02c"),
                arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2))

    fig.tight_layout()
    fig.savefig(outdir / "00_dataset_growth.png", dpi=200)
    plt.close(fig)


def plot_category_overview(data: list[dict], outdir: Path):
    """Sunburst-like nested donut: category → solvability."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left: main category donut ──
    cat_counts = Counter(d["category"] for d in data)
    cat_order = ["official", "1comp", "2comp", "scaled", "mech_sub"]
    sizes = [cat_counts.get(c, 0) for c in cat_order]
    colors = [CAT_COLORS[c] for c in cat_order]

    wedges, texts, autotexts = axes[0].pie(
        sizes, labels=[f"{c.upper()}\n({s})" for c, s in zip(cat_order, sizes)],
        colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.6,
        textprops={"fontsize": 10}, wedgeprops={"edgecolor": "white", "linewidth": 2})
    for at in autotexts:
        at.set_fontsize(9)
    axes[0].set_title(f"Category Distribution\n(n={len(data)})", fontsize=14)

    # ── Right: category + variant type stacked bar ──
    vt_order = ["base", "position_variant", "insight",
                "unsolvable_t1", "unsolvable_t2", "unsolvable_g1", "unsolvable_g2",
                "scaled_base", "scaled_var"]
    vt_labels = ["Base", "Pos.Var", "Insight", "U-T1", "U-T2", "U-G1", "U-G2",
                 "Scale-Base", "Scale-Var"]

    matrix = {}
    for cat in cat_order:
        cat_data = [d for d in data if d["category"] == cat]
        matrix[cat] = [sum(1 for d in cat_data if d["variant_type"] == v) for v in vt_order]

    x = np.arange(len(cat_order))
    width = 0.7
    bottom = np.zeros(len(cat_order))

    for i, vt in enumerate(vt_order):
        counts = [matrix[cat][i] for cat in cat_order]
        if sum(counts) == 0:
            bottom += np.array(counts)
            continue
        axes[1].bar(x, counts, width, bottom=bottom, label=vt_labels[i],
                    color=VT_COLORS.get(vt, "#999"), edgecolor="white", linewidth=0.5)
        bottom += np.array(counts)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c.upper() for c in cat_order], fontsize=10)
    axes[1].set_ylabel("Number of Tasks", fontsize=12)
    axes[1].set_title("Variant Type by Category", fontsize=14)
    axes[1].legend(fontsize=7, ncol=2, loc="upper left")

    fig.tight_layout()
    fig.savefig(outdir / "01_category_overview.png", dpi=200)
    plt.close(fig)


def plot_scaled_breakdown(data: list[dict], outdir: Path):
    """Detailed breakdown of scaled tasks."""
    scaled = [d for d in data if d["category"] == "scaled"]
    if not scaled:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ── Top-left: by comp count ──
    comp_counts = Counter(d.get("comp_count") for d in scaled)
    labels = ["Base (no comp)", "1 Component", "2 Components"]
    vals = [comp_counts.get(None, 0), comp_counts.get(1, 0), comp_counts.get(2, 0)]
    colors = ["#7f7f7f", "#2ca02c", "#ff7f0e"]
    bars = axes[0, 0].bar(labels, vals, color=colors, edgecolor="white")
    for bar, v in zip(bars, vals):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        str(v), ha="center", fontweight="bold")
    axes[0, 0].set_title("Scaled: By Component Requirement", fontsize=13)
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_ylim(0, max(vals) * 1.15)
    axes[0, 0].spines[["top", "right"]].set_visible(False)

    # ── Top-right: by board size ──
    sz_counts = Counter(d.get("sz") for d in scaled)
    sz_labels = ["11×11 (scl only)", "13×13", "15×15"]
    sz_vals = [sz_counts.get(None, 0), sz_counts.get(13, 0), sz_counts.get(15, 0)]
    sz_colors = ["#7f7f7f", "#1f77b4", "#d62728"]
    bars = axes[0, 1].bar(sz_labels, sz_vals, color=sz_colors, edgecolor="white")
    for bar, v in zip(bars, sz_vals):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        str(v), ha="center", fontweight="bold")
    axes[0, 1].set_title("Scaled: By Board Size", fontsize=13)
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_ylim(0, max(sz_vals) * 1.15)
    axes[0, 1].spines[["top", "right"]].set_visible(False)

    # ── Bottom-left: scale level histogram ──
    scl_with = [d["scale_level"] for d in scaled if d.get("scale_level") is not None]
    axes[1, 0].hist(scl_with, bins=range(3, 18), color="#9467bd", edgecolor="white", alpha=0.85)
    axes[1, 0].axvline(np.mean(scl_with) if scl_with else 0, color="purple",
                       linestyle="--", linewidth=2, label=f"Mean: {np.mean(scl_with):.1f}" if scl_with else "")
    axes[1, 0].set_xlabel("Scale Level (sclN)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title(f"Scale Level Distribution\n(n={len(scl_with)} with scale)", fontsize=13)
    axes[1, 0].legend()
    axes[1, 0].spines[["top", "right"]].set_visible(False)

    # ── Bottom-right: base vs variant ──
    base_count = sum(1 for d in scaled if d["variant_type"] == "scaled_base")
    var_count = sum(1 for d in scaled if d["variant_type"] == "scaled_var")
    axes[1, 1].pie([base_count, var_count], labels=[f"Base\n({base_count})", f"Variant\n({var_count})"],
                   autopct="%1.1f%%", colors=["#bcbd22", "#2ca02c"],
                   startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1, 1].set_title("Scaled: Base vs Position Variants", fontsize=13)

    fig.suptitle("Scaled Tasks Breakdown", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "02_scaled_breakdown.png", dpi=200)
    plt.close(fig)


def plot_variant_decomposition(data: list[dict], outdir: Path):
    """Deep dive into variant types across 1comp and 2comp."""
    synth = [d for d in data if d["category"] in ("1comp", "2comp")]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ── Top-left: 1comp variants ──
    c1 = [d for d in synth if d["category"] == "1comp"]
    vt1 = Counter(d["variant_type"] for d in c1)
    vt_order = ["base", "insight", "unsolvable_t1", "unsolvable_t2", "unsolvable_g1", "unsolvable_g2"]
    vt_labels = ["Base\nInsight", "Position\nInsight", "U. T1\n(same-cat)", "U. T2\n(diff-cat)",
                 "U. G1\n(N+1 gaps)", "U. G2\n(N+2 gaps)"]
    vt_colors = ["#7f7f7f", "#9467bd", "#d62728", "#e377c2", "#ff7f0e", "#8c564b"]

    # Actually, let me separate base insight and base unsolvable
    base_c1 = [d for d in c1 if d["variant_type"] in ("base",) and d["solvable"]]
    base_uns_c1 = [d for d in c1 if d["variant_type"] in ("base",) and not d["solvable"]]
    # Position variants
    pos_insight_c1 = [d for d in c1 if d["variant_type"] == "insight"]
    pos_uns_c1 = [d for d in c1 if d["variant_type"].startswith("unsolvable")]

    # Better: group by base_type first
    def get_base_type(d):
        vt = d["variant_type"]
        if vt == "base" and d["solvable"]:
            return "Base\n(Solvable)"
        elif vt == "base" and not d["solvable"]:
            return "Base\n(Unsolvable)"
        elif vt == "insight":
            return "Insight\n(Pos.Var)"
        elif vt.startswith("unsolvable"):
            return "Unsolvable\n(Pos.Var)"
        return "Other"

    bt1 = Counter(get_base_type(d) for d in c1)
    bt_order = ["Base\n(Solvable)", "Base\n(Unsolvable)", "Insight\n(Pos.Var)", "Unsolvable\n(Pos.Var)"]
    bt_vals = [bt1.get(b, 0) for b in bt_order]
    bt_colors = ["#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]

    bars = axes[0, 0].bar(bt_order, bt_vals, color=bt_colors, edgecolor="white")
    for bar, v in zip(bars, bt_vals):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                        str(v), ha="center", fontweight="bold")
    axes[0, 0].set_title(f"1-Component Variants\n(n={len(c1)})", fontsize=13)
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_ylim(0, max(bt_vals) * 1.2)
    axes[0, 0].spines[["top", "right"]].set_visible(False)

    # ── Top-right: 2comp variants ──
    c2 = [d for d in synth if d["category"] == "2comp"]
    bt2 = Counter(get_base_type(d) for d in c2)
    bt2_vals = [bt2.get(b, 0) for b in bt_order]
    bars = axes[0, 1].bar(bt_order, bt2_vals, color=bt_colors, edgecolor="white")
    for bar, v in zip(bars, bt2_vals):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                        str(v), ha="center", fontweight="bold")
    axes[0, 1].set_title(f"2-Component Variants\n(n={len(c2)})", fontsize=13)
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_ylim(0, max(bt2_vals) * 1.2)
    axes[0, 1].spines[["top", "right"]].set_visible(False)

    # ── Bottom-left: unsolvable subtypes breakdown (1comp) ──
    uns_c1 = [d for d in c1 if not d["solvable"]]
    usub_c1 = Counter(d["variant_type"] for d in uns_c1)
    u_order = ["unsolvable_t1", "unsolvable_t2", "unsolvable_g1", "unsolvable_g2"]
    u_labels = ["T1\n(same-cat)", "T2\n(diff-cat)", "G1\n(N+1 gaps)", "G2\n(N+2 gaps)"]
    u_vals = [usub_c1.get(u, 0) for u in u_order]
    u_colors = [VT_COLORS[u] for u in u_order]
    bars = axes[1, 0].bar(u_labels, u_vals, color=u_colors, edgecolor="white")
    for bar, v in zip(bars, u_vals):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        str(v), ha="center", fontweight="bold")
    axes[1, 0].set_title(f"1-Component Unsolvable Subtypes\n(n={len(uns_c1)})", fontsize=13)
    axes[1, 0].set_ylim(0, max(u_vals) * 1.15)
    axes[1, 0].spines[["top", "right"]].set_visible(False)

    # ── Bottom-right: unsolvable subtypes breakdown (2comp) ──
    uns_c2 = [d for d in c2 if not d["solvable"]]
    usub_c2 = Counter(d["variant_type"] for d in uns_c2)
    u2_vals = [usub_c2.get(u, 0) for u in u_order]
    bars = axes[1, 1].bar(u_labels, u2_vals, color=u_colors, edgecolor="white")
    for bar, v in zip(bars, u2_vals):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        str(v), ha="center", fontweight="bold")
    axes[1, 1].set_title(f"2-Component Unsolvable Subtypes\n(n={len(uns_c2)})", fontsize=13)
    axes[1, 1].set_ylim(0, max(u2_vals) * 1.15)
    axes[1, 1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Synthetic Variant Decomposition", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "03_variant_decomposition.png", dpi=200)
    plt.close(fig)


def plot_tier_heatmap(data: list[dict], outdir: Path):
    """Tier distribution as a grouped bar chart across all categories."""
    fig, ax = plt.subplots(figsize=(12, 5))

    cats = ["official", "1comp", "2comp", "scaled", "mech_sub"]
    tiers = sorted(set(d["tier"] for d in data if d["tier"] > 0))

    matrix = {}
    for cat in cats:
        cat_data = [d for d in data if d["category"] == cat]
        matrix[cat] = [sum(1 for d in cat_data if d["tier"] == t) for t in tiers]

    x = np.arange(len(cats))
    width = 0.8 / len(tiers)
    offsets = np.linspace(-0.35, 0.35, len(tiers))

    for i, tier in enumerate(tiers):
        counts = [matrix[cat][i] for cat in cats]
        bars = ax.bar(x + offsets[i], counts, width, label=f"Tier {tier}",
                      color=TIER_COLORS.get(tier, "#999"), edgecolor="white")
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                        str(count), ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in cats], fontsize=11)
    ax.set_ylabel("Number of Tasks", fontsize=12)
    ax.set_title("Tier Distribution by Category", fontsize=14)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(outdir / "04_tier_heatmap.png", dpi=200)
    plt.close(fig)


def plot_solvability_overview(data: list[dict], outdir: Path):
    """Solvability breakdown across all categories."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: stacked bar solvable/unsolvable per category ──
    cats = ["official", "1comp", "2comp", "scaled", "mech_sub"]
    solv_per_cat = {}
    uns_per_cat = {}
    for cat in cats:
        cat_data = [d for d in data if d["category"] == cat]
        solv_per_cat[cat] = sum(1 for d in cat_data if d["solvable"])
        uns_per_cat[cat] = sum(1 for d in cat_data if not d["solvable"])

    x = np.arange(len(cats))
    width = 0.5
    s_vals = [solv_per_cat[c] for c in cats]
    u_vals = [uns_per_cat[c] for c in cats]

    axes[0].bar(x, s_vals, width, label="Solvable", color="#2ca02c", edgecolor="white")
    axes[0].bar(x, u_vals, width, bottom=s_vals, label="Unsolvable", color="#d62728", edgecolor="white")
    for i, (s, u) in enumerate(zip(s_vals, u_vals)):
        if s > 0:
            axes[0].text(i, s / 2, str(s), ha="center", va="center", fontweight="bold", color="white")
        if u > 0:
            axes[0].text(i, s + u / 2, str(u), ha="center", va="center", fontweight="bold", color="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c.upper() for c in cats])
    axes[0].set_ylabel("Count")
    axes[0].set_title("Solvability by Category", fontsize=13)
    axes[0].legend()
    axes[0].spines[["top", "right"]].set_visible(False)

    # ── Right: overall pie ──
    total_solv = sum(1 for d in data if d["solvable"])
    total_uns = sum(1 for d in data if not d["solvable"])
    axes[1].pie([total_solv, total_uns], labels=["Solvable", "Unsolvable"],
                autopct="%1.1f%%", colors=["#2ca02c", "#d62728"],
                startangle=90, explode=(0, 0.05), textprops={"fontsize": 12},
                wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title(f"Overall Solvability\n(n={len(data)})", fontsize=13)

    fig.tight_layout()
    fig.savefig(outdir / "05_solvability_overview.png", dpi=200)
    plt.close(fig)


def plot_zone_heatmap(data: list[dict], outdir: Path):
    """Heatmap showing where components are placed on the board."""
    # Collect all placed + fixed component coordinates
    all_coords = []
    solvable_coords = []
    unsolvable_coords = []

    for d in data:
        for x, y in d["placed_coords"] + d["fixed_coords"]:
            all_coords.append((x, y))
            if d["solvable"]:
                solvable_coords.append((x, y))
            else:
                unsolvable_coords.append((x, y))

    if not all_coords:
        return

    # Determine board bounds
    max_x = max(c[0] for c in all_coords) + 1
    max_y = max(c[1] for c in all_coords) + 1

    def make_heatmap(coords, title):
        grid = np.zeros((max_y, max_x))
        for x, y in coords:
            if 0 <= y < max_y and 0 <= x < max_x:
                grid[y, x] += 1
        return grid

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, coords, title, cmap in [
        (axes[0], all_coords, "All Components", "YlOrRd"),
        (axes[1], solvable_coords, "Solvable Tasks", "Greens"),
        (axes[2], unsolvable_coords, "Unsolvable Tasks", "Reds"),
    ]:
        grid = make_heatmap(coords, title)
        # Log scale for better contrast
        grid_log = np.log1p(grid)
        im = ax.imshow(grid_log, cmap=cmap, aspect="auto", origin="upper",
                       interpolation="bilinear")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Column (x)")
        ax.set_ylabel("Row (y)")
        # Annotate cells with actual counts if not too many
        if max_y <= 15 and max_x <= 15:
            for i in range(max_y):
                for j in range(max_x):
                    v = grid[i, j]
                    if v > 0:
                        ax.text(j, i, f"{int(v)}", ha="center", va="center",
                                fontsize=6, color="black" if grid_log[i, j] < grid_log.max() * 0.6 else "white")
        plt.colorbar(im, ax=ax, shrink=0.8, label="log(count+1)")

    fig.suptitle("Component Placement Zone Analysis", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "06_zone_heatmap.png", dpi=200)
    plt.close(fig)


def plot_component_complexity(data: list[dict], outdir: Path):
    """Histograms of component counts per category."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # By solvability
    solv = [d["total_components"] for d in data if d["solvable"]]
    uns = [d["total_components"] for d in data if not d["solvable"]]

    bins_all = range(0, max(max(solv) if solv else 0, max(uns) if uns else 0) + 3)

    axes[0].hist(solv, bins=bins_all, color="#2ca02c", edgecolor="white", alpha=0.7, label=f"Solvable (n={len(solv)})")
    axes[0].hist(uns, bins=bins_all, color="#d62728", edgecolor="white", alpha=0.7, label=f"Unsolvable (n={len(uns)})")
    axes[0].axvline(np.mean(solv) if solv else 0, color="darkgreen", linestyle="--", lw=2)
    axes[0].axvline(np.mean(uns) if uns else 0, color="darkred", linestyle="--", lw=2)
    axes[0].set_xlabel("Total Components")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Component Distribution (All)", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    # By category
    cats = ["official", "1comp", "2comp", "mech_sub"]
    cat_colors = [CAT_COLORS[c] for c in cats]
    for i, (cat, color) in enumerate(zip(cats, cat_colors)):
        comps = [d["total_components"] for d in data if d["category"] == cat]
        if not comps:
            continue
        axes[1].hist(comps, bins=bins_all, color=color, edgecolor="white", alpha=0.5,
                     label=f"{cat.upper()} (μ={np.mean(comps):.1f})")
    axes[1].set_xlabel("Total Components")
    axes[1].set_ylabel("Count")
    axes[1].set_title("By Source Category", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)

    # Scaled component distribution
    scaled = [d["total_components"] for d in data if d["category"] == "scaled"]
    if scaled:
        axes[2].hist(scaled, bins=range(min(scaled), max(scaled) + 3),
                     color="#9467bd", edgecolor="white", alpha=0.7)
        axes[2].axvline(np.mean(scaled), color="purple", linestyle="--", lw=2,
                        label=f"Mean: {np.mean(scaled):.1f}")
        axes[2].set_xlabel("Total Components")
        axes[2].set_ylabel("Count")
        axes[2].set_title(f"Scaled Tasks (n={len(scaled)})", fontsize=12, fontweight="bold")
        axes[2].legend(fontsize=9)
        axes[2].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(outdir / "07_component_complexity.png", dpi=200)
    plt.close(fig)


def plot_challenge_origin(data: list[dict], outdir: Path):
    """Which official challenges have the most synthetic variants."""
    # Map origin challenge number → count
    origin_counts = Counter()
    for d in data:
        ch = d.get("origin_ch")
        if ch is not None:
            origin_counts[ch] += 1

    if not origin_counts:
        return

    fig, ax = plt.subplots(figsize=(16, 5))

    sorted_chs = sorted(origin_counts.items())
    chs = [f"ch{c}" for c, _ in sorted_chs]
    counts = [cnt for _, cnt in sorted_chs]

    colors = []
    for c, _ in sorted_chs:
        if c <= 10:
            colors.append("#2ca02c")  # Tier 1
        elif c <= 20:
            colors.append("#ff7f0e")  # Tier 2
        else:
            colors.append("#d62728")  # Tier 3+

    bars = ax.bar(chs, counts, color=colors, edgecolor="white")
    for bar, count in zip(bars, counts):
        if count > 20:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(count), ha="center", fontsize=7, fontweight="bold", rotation=90)

    ax.set_xlabel("Official Challenge")
    ax.set_ylabel("Number of Synthetic Variants")
    ax.set_title("Synthetic Variant Generation by Source Challenge", fontsize=14)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", label="Tier 1 (ch1-10)"),
        Patch(facecolor="#ff7f0e", label="Tier 2 (ch11-20)"),
        Patch(facecolor="#d62728", label="Tier 3+ (ch21+)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(outdir / "08_challenge_origin.png", dpi=200)
    plt.close(fig)


def plot_summary_table(data: list[dict], outdir: Path):
    """Generate a clean summary table as a figure with delta vs baseline."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis("off")

    # Previous standard baseline (pre-dataset-extension)
    BASELINE: dict[str, dict[int, int]] = {
        "official": {1: 11, 2: 22},
        "1comp":    {1: 5,  2: 5},
        "2comp":    {1: 5,  2: 5},
        "scaled":   {1: 0,  2: 0},
    }

    # Compute stats
    cats = ["official", "1comp", "2comp", "scaled"]
    rows = []

    for cat in cats:
        cd = [d for d in data if d["category"] == cat]
        if not cd:
            continue
        solv = sum(1 for d in cd if d["solvable"])
        uns = sum(1 for d in cd if not d["solvable"])
        t1 = sum(1 for d in cd if d["tier"] == 1)
        t2 = sum(1 for d in cd if d["tier"] == 2)
        t3 = sum(1 for d in cd if d["tier"] == 3)
        t4 = sum(1 for d in cd if d["tier"] == 4)
        bl = BASELINE.get(cat, {1: 0, 2: 0})
        dt1 = t1 - bl.get(1, 0)
        dt2 = t2 - bl.get(2, 0)
        delta_t1 = f"+{dt1}" if dt1 > 0 else str(dt1) if dt1 < 0 else "—"
        delta_t2 = f"+{dt2}" if dt2 > 0 else str(dt2) if dt2 < 0 else "—"
        rows.append([cat.upper(), len(cd), solv, uns, t1, delta_t1, t2, delta_t2, t3, t4])

    # Totals
    total = len(data)
    total_s = sum(1 for d in data if d["solvable"])
    total_u = sum(1 for d in data if not d["solvable"])
    t1_all = sum(1 for d in data if d["tier"] == 1)
    t2_all = sum(1 for d in data if d["tier"] == 2)
    bl_t1 = sum(BASELINE[c].get(1, 0) for c in cats)
    bl_t2 = sum(BASELINE[c].get(2, 0) for c in cats)
    dt1_all = t1_all - bl_t1
    dt2_all = t2_all - bl_t2
    rows.append(["TOTAL", total, total_s, total_u, t1_all,
                 f"+{dt1_all}" if dt1_all > 0 else str(dt1_all),
                 t2_all,
                 f"+{dt2_all}" if dt2_all > 0 else str(dt2_all),
                 sum(1 for d in data if d["tier"] == 3),
                 sum(1 for d in data if d["tier"] == 4)])

    col_labels = ["Category", "Tasks", "Solvable", "Unsolvable",
                  "Tier 1", "Δ T1", "Tier 2", "Δ T2", "Tier 3", "Tier 4"]
    ncols = 10
    cell_colors = []
    for row in rows:
        if row[0] == "TOTAL":
            cell_colors.append(["#e0e0e0"] * ncols)
        elif row[0] == "OFFICIAL":
            cell_colors.append(["#d4e6f1"] * ncols)
        elif row[0] == "1COMP":
            cell_colors.append(["#d5f5e3"] * ncols)
        elif row[0] == "2COMP":
            cell_colors.append(["#fdebd0"] * ncols)
        elif row[0] == "SCALED":
            cell_colors.append(["#e8daef"] * ncols)
        elif row[0] == "MECH_SUB":
            cell_colors.append(["#fdedec"] * ncols)
        else:
            cell_colors.append(["white"] * ncols)

    # Highlight delta columns with a subtle tint
    for i in range(len(rows)):
        for j in [5, 7]:  # ΔT1 and ΔT2 columns
            cell_colors[i][j] = "#f0f3f4"

    table = ax.table(cellText=rows, colLabels=col_labels, cellColours=cell_colors,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Bold header and total row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white")
        if row == len(rows):
            cell.set_text_props(fontweight="bold")

    ax.set_title("Benchmark Composition Summary", fontsize=16, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(outdir / "09_summary_table.png", dpi=200)
    plt.close(fig)


def print_console_summary(data: list[dict]):
    """Print detailed console summary."""
    print(f"\n{'='*70}")
    print(f"  TURING TUMBLE BENCHMARK — COMPOSITION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total tasks:         {len(data)}")
    print(f"  Solvable:            {sum(1 for d in data if d['solvable'])}")
    print(f"  Unsolvable:          {sum(1 for d in data if not d['solvable'])}")
    print()

    # By category
    print(f"  {'Category':<12} {'Total':>7} {'Solv':>7} {'Unsolv':>7} {'T1':>5} {'T2':>5} {'T3':>5} {'T4':>5}")
    print(f"  {'-'*59}")
    for cat in ["official", "1comp", "2comp", "scaled", "mech_sub"]:
        cd = [d for d in data if d["category"] == cat]
        solv = sum(1 for d in cd if d["solvable"])
        uns = sum(1 for d in cd if not d["solvable"])
        t1 = sum(1 for d in cd if d["tier"] == 1)
        t2 = sum(1 for d in cd if d["tier"] == 2)
        t3 = sum(1 for d in cd if d["tier"] == 3)
        t4 = sum(1 for d in cd if d["tier"] == 4)
        print(f"  {cat:<12} {len(cd):>7} {solv:>7} {uns:>7} {t1:>5} {t2:>5} {t3:>5} {t4:>5}")

    # By variant type
    print(f"\n  {'Variant Type':<30} {'Count':>7}")
    print(f"  {'-'*39}")
    for vt in ["base", "position_variant", "insight", "unsolvable_t1",
               "unsolvable_t2", "unsolvable_g1", "unsolvable_g2",
               "scaled_base", "scaled_var"]:
        cnt = sum(1 for d in data if d["variant_type"] == vt)
        if cnt > 0:
            print(f"  {vt:<30} {cnt:>7}")

    # Board sizes
    print(f"\n  {'Board Size':<15} {'Count':>7}")
    print(f"  {'-'*24}")
    for sz, cnt in sorted(Counter(d["board_size"] for d in data).items()):
        print(f"  {sz:<15} {cnt:>7}")

    # Comps stats
    comps = [d["total_components"] for d in data]
    print(f"\n  Component count — min: {min(comps)}, max: {max(comps)}, "
          f"mean: {np.mean(comps):.1f}, median: {np.median(comps):.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    outdir = OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    print("Collecting all challenge metadata (Tier 1-2 only)...")
    data = collect_all(max_tier=2)
    print(f"  → {len(data)} challenges found across {len(set(d['category'] for d in data))} categories")

    # Filter out anything that looks like a question file (by path)
    data = [d for d in data if "questions" not in d.get("task_id", "")]

    print(f"  → {len(data)} after filtering questions")

    print("\nGenerating plots...")

    plots = [
        ("Dataset Growth", plot_dataset_growth, [data, outdir]),
        ("Category Overview", plot_category_overview, [data, outdir]),
        ("Scaled Breakdown", plot_scaled_breakdown, [data, outdir]),
        ("Variant Decomposition", plot_variant_decomposition, [data, outdir]),
        ("Tier Heatmap", plot_tier_heatmap, [data, outdir]),
        ("Solvability Overview", plot_solvability_overview, [data, outdir]),
        ("Zone Heatmap", plot_zone_heatmap, [data, outdir]),
        ("Component Complexity", plot_component_complexity, [data, outdir]),
        ("Challenge Origin", plot_challenge_origin, [data, outdir]),
        ("Summary Table", plot_summary_table, [data, outdir]),
    ]

    for name, fn, args in plots:
        try:
            fn(*args)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}", file=sys.stderr)

    print(f"\n  → Plots saved to {outdir}/")
    print_console_summary(data)


if __name__ == "__main__":
    main()
