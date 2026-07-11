#!/usr/bin/env python3
"""
Dataset decomposition + experiment results comparison.
"""
import json, sys
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from visualize_benchmark_v2 import collect_all, CAT_COLORS

OUTDIR = REPO / "data" / "visualizations"

# Experiment results (hardcoded from clean run)
EXP_RESULTS = {
    "official": {"tasks": 11, "success": 0, "rate": 0.0, "calls": 20.8},
    "1comp":    {"tasks": 72, "success": 57, "rate": 79.2, "calls": 9.2},
    "2comp":    {"tasks": 67, "success": 33, "rate": 49.3, "calls": 11.7},
    "scaled":   {"tasks": 0, "success": 0, "rate": 0, "calls": 0},
}

def plot_dataset_vs_experiment(outdir: Path):
    """Side-by-side: dataset composition vs experiment success rate."""
    data = collect_all(max_tier=2)
    
    cats = ["official", "1comp", "2comp", "scaled", "mech_sub"]
    cat_data = {}
    for cat in cats:
        cat_data[cat] = [d for d in data if d["category"] == cat]
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor="white")
    
    # ── LEFT: Task count by category ──
    counts = [len(cat_data.get(c, [])) for c in cats]
    colors = [CAT_COLORS.get(c, "#999") for c in cats]
    labels = [c.upper() for c in cats]
    
    bars = axes[0].bar(labels, counts, color=colors, edgecolor="white")
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                    str(count), ha="center", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Number of Tasks", fontsize=12)
    axes[0].set_title("Dataset Composition\n(Tier 1-2)", fontsize=13, fontweight="bold")
    axes[0].set_ylim(0, max(counts) * 1.15)
    axes[0].spines[["top", "right"]].set_visible(False)
    
    # ── CENTER: Solvable vs unsolvable ──
    solv_counts = {}
    uns_counts = {}
    for cat in cats:
        cd = cat_data.get(cat, [])
        solv_counts[cat] = sum(1 for d in cd if d["solvable"])
        uns_counts[cat] = sum(1 for d in cd if not d["solvable"])
    
    x = np.arange(len(cats))
    width = 0.5
    s_vals = [solv_counts.get(c, 0) for c in cats]
    u_vals = [uns_counts.get(c, 0) for c in cats]
    
    axes[1].bar(x, s_vals, width, label="Solvable", color="#2ca02c", edgecolor="white")
    axes[1].bar(x, u_vals, width, bottom=s_vals, label="Unsolvable", color="#d62728", edgecolor="white")
    for i, (s, u) in enumerate(zip(s_vals, u_vals)):
        if s > 0:
            axes[1].text(i, s/2, str(s), ha="center", va="center", fontweight="bold", color="white", fontsize=10)
        if u > 0:
            axes[1].text(i, s + u/2, str(u), ha="center", va="center", fontweight="bold", color="white", fontsize=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Solvability by Category\n(Tier 1-2)", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)
    
    # ── RIGHT: Experiment success rate ──
    exp_cats = ["official", "1comp", "2comp"]
    exp_rates = [EXP_RESULTS[c]["rate"] / 100 for c in exp_cats]
    exp_tasks = [EXP_RESULTS[c]["tasks"] for c in exp_cats]
    exp_colors = [CAT_COLORS.get(c, "#999") for c in exp_cats]
    exp_labels = [c.upper() for c in exp_cats]
    
    bars = axes[2].bar(exp_labels, exp_rates, color=exp_colors, edgecolor="white")
    for bar, rate, n in zip(bars, exp_rates, exp_tasks):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{rate*100:.0f}%\n(n={n})", ha="center", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("Success Rate", fontsize=12)
    axes[2].set_title("Experiment Results\nqwen3-coder-30b-a3b", fontsize=13, fontweight="bold")
    axes[2].set_ylim(0, 1.1)
    axes[2].spines[["top", "right"]].set_visible(False)
    
    fig.suptitle("Benchmark Decomposition & Experiment Results — Tier 1-2",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "decomposition_experiment.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ decomposition_experiment.png")


def plot_variant_type_pie(outdir: Path):
    """Pie chart of variant types across all categories."""
    data = collect_all(max_tier=2)
    
    vt_counts = Counter()
    for d in data:
        vt = d["variant_type"]
        # Simplify labels
        label_map = {
            "base": "Base",
            "position_variant": "Position Variant",
            "insight": "Insight",
            "unsolvable_t1": "U. T1 (same-cat)",
            "unsolvable_t2": "U. T2 (diff-cat)",
            "unsolvable_g1": "U. G1 (N+1)",
            "unsolvable_g2": "U. G2 (N+2)",
            "scaled_base": "Scaled Base",
            "scaled_var": "Scaled Variant",
        }
        vt_counts[label_map.get(vt, vt)] += 1
    
    # Remove zero-count types
    vt_counts = {k: v for k, v in vt_counts.items() if v > 0}
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
    
    vt_colors_map = {
        "Base": "#7f7f7f", "Position Variant": "#17becf",
        "Insight": "#9467bd",
        "U. T1 (same-cat)": "#d62728", "U. T2 (diff-cat)": "#e377c2",
        "U. G1 (N+1)": "#ff7f0e", "U. G2 (N+2)": "#8c564b",
        "Scaled Base": "#bcbd22", "Scaled Variant": "#2ca02c",
    }
    
    labels = list(vt_counts.keys())
    sizes = list(vt_counts.values())
    colors = [vt_colors_map.get(l, "#999") for l in labels]
    
    wedges, texts, autotexts = ax.pie(
        sizes, labels=[f"{l}\n({s})" for l, s in zip(labels, sizes)],
        colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.6,
        textprops={"fontsize": 9},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(8)
    
    ax.set_title(f"Variant Type Distribution\n(n={sum(sizes)} Tier 1-2 tasks)", fontsize=14, fontweight="bold")
    
    fig.tight_layout()
    fig.savefig(outdir / "decomposition_variant_types.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ decomposition_variant_types.png")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    
    print("Generating task decomposition visualizations...")
    plot_dataset_vs_experiment(OUTDIR)
    plot_variant_type_pie(OUTDIR)


if __name__ == "__main__":
    main()
