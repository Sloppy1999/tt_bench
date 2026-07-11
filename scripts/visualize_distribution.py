#!/usr/bin/env python3
"""Visualize challenge distribution across categories, tiers, and variant types."""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO = Path(__file__).resolve().parent.parent
TASKS = REPO / "data" / "tasks"

# ── Colour palette ─────────────────────────────────────────────────────────
CAT_COLORS = {
    "official": "#1f77b4",
    "1comp": "#2ca02c",
    "2comp": "#ff7f0e",
}

VARIANT_COLORS = {
    "solvable": "#2ca02c",
    "insight": "#9467bd",
    "unsolvable_t1": "#d62728",
    "unsolvable_t2": "#e377c2",
    "unsolvable_g1": "#ff7f0e",
    "unsolvable_g2": "#8c564b",
}

TIER_COLORS = {1: "#1f77b4", 2: "#d62728"}


def classify_challenge(fpath: Path) -> dict:
    """Extract classification metadata from a challenge JSON."""
    with open(fpath) as f:
        d = json.load(f)

    tid = d.get("task_id", fpath.stem)
    tier = d.get("tier", 0)
    title = d.get("title", "")

    # Determine source set from path
    rel = fpath.relative_to(TASKS)
    parts = rel.parts

    # Fixed components (what the board starts with)
    fixed = d.get("board", {}).get("fixed_components", [])
    # Solution components (what needs placing)
    placed = d.get("solution", {}).get("placed_components", [])
    total_comp = len(fixed) + len(placed)

    # Count component types
    types = Counter()
    for c in fixed + placed:
        types[c["type"]] += 1

    # Determine category from directory structure only (not filename)
    category = "unknown"
    for part in parts[:-1]:  # skip filename
        pl = part.lower()
        if pl == "official":
            category = "official"
            break
        elif "1comp" in pl:
            category = "1comp"
            break
        elif "2comp" in pl:
            category = "2comp"
            break

    # Determine variant type
    meta = d.get("_meta", {})
    vt = meta.get("variant_type", "")
    subtype = meta.get("unsolvable_subtype", "")
    if vt == "insight":
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
    elif "_var_" in tid:
        variant_type = "position_variant"
    else:
        variant_type = "base"

    # Classify as solvable/unsolvable
    is_unsolvable = variant_type.startswith("unsolvable")
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
    }


def collect_all(roots: list[Path]) -> list[dict]:
    """Scan all challenge JSONs and return classification dicts."""
    results = []
    for root in roots:
        for f in sorted(root.rglob("tt-official-*.json")):
            # Skip understanding question files
            if "_questions" in f.stem:
                continue
            try:
                results.append(classify_challenge(f))
            except Exception as e:
                print(f"  WARN: {f.name}: {e}", file=sys.stderr)
    return results


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_category_distribution(data: list[dict], outdir: Path):
    """Bar chart: challenges per category + variant type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: by source category ──
    cat_counts = Counter(d["category"] for d in data)
    cats = ["official", "1comp", "2comp"]
    counts = [cat_counts.get(c, 0) for c in cats]
    colors = [CAT_COLORS[c] for c in cats]
    bars = axes[0].bar(cats, counts, color=colors, edgecolor="white")
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     str(count), ha="center", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Number of challenges", fontsize=12)
    axes[0].set_title("By Source Category", fontsize=13, fontweight="bold")
    axes[0].set_ylim(0, max(counts) * 1.15)

    # ── Right: by variant type ──
    vt_counts = Counter(d["variant_type"] for d in data)
    order = ["base", "position_variant", "insight",
             "unsolvable_t1", "unsolvable_t2", "unsolvable_g1", "unsolvable_g2"]
    labels = ["Base", "Position\nVariant", "Insight",
              "Unsolv.\nT1", "Unsolv.\nT2", "Unsolv.\nG1", "Unsolv.\nG2"]
    counts = [vt_counts.get(o, 0) for o in order]
    colors = [VARIANT_COLORS.get(o, "#999") for o in order]
    bars = axes[1].bar(range(len(order)), counts, color=colors, edgecolor="white")
    for bar, count in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     str(count), ha="center", fontsize=9, fontweight="bold")
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("Number of challenges", fontsize=12)
    axes[1].set_title("By Variant Type", fontsize=13, fontweight="bold")
    axes[1].set_ylim(0, max(counts) * 1.15)

    fig.tight_layout()
    fig.savefig(outdir / "01_category_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_solvability(data: list[dict], outdir: Path):
    """Pie + bar: solvable vs unsolvable breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: solvable vs unsolvable pie ──
    solvable = sum(1 for d in data if d["solvable"])
    unsolvable = sum(1 for d in data if not d["solvable"])
    axes[0].pie([solvable, unsolvable], labels=["Solvable", "Unsolvable"],
                autopct="%1.1f%%", colors=["#2ca02c", "#d62728"],
                startangle=90, explode=(0, 0.05), textprops={"fontsize": 12})
    axes[0].set_title(f"Solvability (n={len(data)})", fontsize=13, fontweight="bold")

    # ── Right: unsolvable subtypes ──
    unsolv = [d for d in data if not d["solvable"]]
    subtype_counts = Counter(d["variant_type"] for d in unsolv)
    order = ["unsolvable_t1", "unsolvable_t2", "unsolvable_g1", "unsolvable_g2"]
    labels = ["T1\n(same-cat\ntype swap)", "T2\n(diff-cat\ntype swap)",
              "G1\n(N+1 gaps)", "G2\n(N+2 gaps)"]
    counts = [subtype_counts.get(o, 0) for o in order]
    colors = [VARIANT_COLORS[o] for o in order]
    bars = axes[1].bar(range(4), counts, color=colors, edgecolor="white")
    for bar, count in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     str(count), ha="center", fontsize=10, fontweight="bold")
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title("Unsolvable Subtypes", fontsize=13, fontweight="bold")
    axes[1].set_ylim(0, max(counts) * 1.15)

    fig.tight_layout()
    fig.savefig(outdir / "02_solvability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tier_distribution(data: list[dict], outdir: Path):
    """Stacked bar: tier distribution by category."""
    fig, ax = plt.subplots(figsize=(10, 5))

    categories = ["official", "1comp", "2comp"]
    tiers = [1, 2]

    matrix = {}
    for cat in categories:
        cat_data = [d for d in data if d["category"] == cat]
        matrix[cat] = [sum(1 for d in cat_data if d["tier"] == t) for t in tiers]

    x = np.arange(len(categories))
    width = 0.35
    bottom = np.zeros(len(categories))

    for i, tier in enumerate(tiers):
        counts = [matrix[cat][i] for cat in categories]
        bars = ax.bar(x, counts, width, bottom=bottom,
                      label=f"Tier {tier}", color=TIER_COLORS[tier])
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        str(count), ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        bottom += counts

    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in categories], fontsize=11)
    ax.set_ylabel("Number of challenges", fontsize=12)
    ax.set_title("Tier Distribution by Category", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(outdir / "03_tier_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_component_counts(data: list[dict], outdir: Path):
    """Histogram: distribution of total component count."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: solvable component distribution ──
    solvable = [d for d in data if d["solvable"]]
    counts_s = [d["total_components"] for d in solvable]
    axes[0].hist(counts_s, bins=range(5, 26), color="#2ca02c", edgecolor="white",
                 alpha=0.8)
    axes[0].axvline(np.mean(counts_s), color="darkgreen", linestyle="--", linewidth=2,
                    label=f"Mean: {np.mean(counts_s):.1f}")
    axes[0].set_xlabel("Total components (fixed + placed)", fontsize=11)
    axes[0].set_ylabel("Number of challenges", fontsize=11)
    axes[0].set_title(f"Solvable (n={len(solvable)})", fontsize=12, fontweight="bold")
    axes[0].legend()

    # ── Right: unsolvable component distribution ──
    unsolv = [d for d in data if not d["solvable"]]
    counts_u = [d["total_components"] for d in unsolv]
    axes[1].hist(counts_u, bins=range(5, 26), color="#d62728", edgecolor="white",
                 alpha=0.8)
    axes[1].axvline(np.mean(counts_u), color="darkred", linestyle="--", linewidth=2,
                    label=f"Mean: {np.mean(counts_u):.1f}")
    axes[1].set_xlabel("Total components (fixed + placed)", fontsize=11)
    axes[1].set_ylabel("Number of challenges", fontsize=11)
    axes[1].set_title(f"Unsolvable (n={len(unsolv)})", fontsize=12, fontweight="bold")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(outdir / "04_component_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_type_distribution(data: list[dict], outdir: Path):
    """Stacked bar: component types used across categories."""
    fig, ax = plt.subplots(figsize=(12, 5))

    categories = ["official", "1comp", "2comp"]
    type_list = ["ramp_right", "ramp_left", "crossover", "bit", "gear_bit", "gear",
                 "interceptor", "trigger"]

    matrix = defaultdict(lambda: defaultdict(int))
    for d in data:
        cat = d["category"]
        for t, cnt in d["component_types"].items():
            matrix[cat][t] += cnt

    x = np.arange(len(type_list))
    width = 0.25
    colors = {"official": "#1f77b4", "1comp": "#2ca02c", "2comp": "#ff7f0e"}

    for i, cat in enumerate(categories):
        counts = [matrix[cat].get(t, 0) for t in type_list]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, counts, width, label=cat.upper(),
                      color=colors[cat], edgecolor="white")
        for bar, count in zip(bars, counts):
            if count > 50:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                        str(count), ha="center", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in type_list], fontsize=9)
    ax.set_ylabel("Total component instances", fontsize=12)
    ax.set_title("Component Type Distribution by Category", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)

    fig.tight_layout()
    fig.savefig(outdir / "05_type_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_placed_count(data: list[dict], outdir: Path):
    """Bar: how many components the solver must place (1, 2, 3+)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    placed_counts = Counter(d["placed_count"] for d in data)
    x = sorted(placed_counts.keys())
    counts = [placed_counts[k] for k in x]
    bars = ax.bar([str(k) for k in x], counts, color="#1f77b4", edgecolor="white")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Components to place", fontsize=12)
    ax.set_ylabel("Number of challenges", fontsize=12)
    ax.set_title("Components to Place Distribution", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.15)

    fig.tight_layout()
    fig.savefig(outdir / "06_placed_count.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(data: list[dict]):
    """Print text summary."""
    print(f"\n{'='*60}")
    print(f"  CHALLENGE DISTRIBUTION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total challenges:   {len(data)}")
    print(f"  Solvable:           {sum(1 for d in data if d['solvable'])}")
    print(f"  Unsolvable:         {sum(1 for d in data if not d['solvable'])}")
    print()

    print(f"  {'Category':<20} {'Count':>6} {'Tier1':>6} {'Tier2':>6}")
    print(f"  {'-'*42}")
    for cat in ["official", "1comp", "2comp"]:
        cat_data = [d for d in data if d["category"] == cat]
        t1 = sum(1 for d in cat_data if d["tier"] == 1)
        t2 = sum(1 for d in cat_data if d["tier"] == 2)
        print(f"  {cat:<20} {len(cat_data):>6} {t1:>6} {t2:>6}")

    print()
    print(f"  {'Variant Type':<25} {'Count':>6}")
    print(f"  {'-'*33}")
    for vt in ["base", "position_variant", "insight",
               "unsolvable_t1", "unsolvable_t2", "unsolvable_g1", "unsolvable_g2"]:
        cnt = sum(1 for d in data if d["variant_type"] == vt)
        print(f"  {vt:<25} {cnt:>6}")

    comps = [d["total_components"] for d in data]
    print()
    print(f"  Component count — min: {min(comps)}, max: {max(comps)}, "
          f"mean: {np.mean(comps):.1f}, median: {np.median(comps):.0f}")


def main():
    roots = [
        TASKS / "official" / "challenges" / "json",
        TASKS / "challenges_1comp",
        TASKS / "challenges_2comp",
    ]

    print("Collecting challenge metadata...")
    data = collect_all(roots)
    print(f"  → {len(data)} challenges found")

    outdir = REPO / "data" / "visualizations"
    outdir.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    plot_category_distribution(data, outdir)
    plot_solvability(data, outdir)
    plot_tier_distribution(data, outdir)
    plot_component_counts(data, outdir)
    plot_type_distribution(data, outdir)
    plot_placed_count(data, outdir)

    print(f"  → 6 plots saved to {outdir}/")
    print_summary(data)


if __name__ == "__main__":
    main()
