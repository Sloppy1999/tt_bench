#!/usr/bin/env python3
"""Benchmark framing: program synthesis vs unsolvability detection.

Generates a plot showing the benchmark as measuring two orthogonal
capabilities, not as a monolithic task set.

Saves to data/visualizations/10_capability_framing.png
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
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

REPO = Path(__file__).resolve().parent.parent
TASKS = REPO / "data" / "tasks"
OUTDIR = REPO / "data" / "visualizations"

# ── Colour palette ─────────────────────────────────────────────────────────
SYNTH_COLORS = {
    "official":       "#1f77b4",
    "insight":        "#2ca02c",
    "scaled_base":    "#bcbd22",
    "scaled_var":     "#9467bd",
    "position_var":   "#17becf",
}
DETECT_COLORS = {
    "T1 (same-cat swap)":  "#d62728",
    "T2 (diff-cat swap)":  "#e377c2",
    "G1 (N+1 gaps)":       "#ff7f0e",
    "G2 (N+2 gaps)":       "#8c564b",
}


# ═══════════════════════════════════════════════════════════════════════════
# Classification
# ═══════════════════════════════════════════════════════════════════════════

def classify(fpath: Path) -> dict | None:
    try:
        with open(fpath) as f:
            d = json.load(f)
    except Exception:
        return None

    tid = d.get("task_id", fpath.stem)
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

    meta = d.get("_meta", {})
    vt = meta.get("variant_type", "")
    subtype = meta.get("unsolvable_subtype", "")

    # Determine solvable / unsolvable
    if category == "scaled":
        is_unsolvable = False
    elif vt == "insight":
        is_unsolvable = False
    elif vt == "unsolvable":
        is_unsolvable = True
    elif category == "official":
        is_unsolvable = False
    else:
        # base synthetic — check if it's in an unsolvable directory
        is_unsolvable = "unsolvable" in rel_str.lower()

    # Unsolvable subtype
    usub = "unknown"
    if is_unsolvable:
        if "extra_gap" in subtype:
            usub = f"G{subtype.split('_')[-1]}"
        elif "T1" in subtype or "same category" in subtype:
            usub = "T1"
        elif "T2" in subtype or "different category" in subtype:
            usub = "T2"
        else:
            # Infer from directory
            if "t1" in rel_str.lower():
                usub = "T1"
            elif "t2" in rel_str.lower():
                usub = "T2"
            elif "g1" in rel_str.lower():
                usub = "G1"
            elif "g2" in rel_str.lower():
                usub = "G2"
            elif "unsolvable" in rel_str.lower():
                usub = "T1"  # default

    # Solvable sub-category
    solv_sub = "unknown"
    if not is_unsolvable:
        if category == "official":
            solv_sub = "official"
        elif category == "scaled":
            if "_var_" in tid:
                solv_sub = "scaled_var"
            else:
                solv_sub = "scaled_base"
        elif vt == "insight":
            solv_sub = "insight"
        elif "_var_" in tid:
            solv_sub = "position_var"
        else:
            solv_sub = "insight"  # base insight

    return {
        "solvable": not is_unsolvable,
        "unsolvable_subtype": usub if is_unsolvable else None,
        "solvable_subtype": solv_sub if not is_unsolvable else None,
        "tier": d.get("tier", 0),
    }


def collect_all(max_tier: int | None = None):
    results = []
    for f in sorted(TASKS.rglob("*.json")):
        rel = str(f.relative_to(TASKS))
        if "questions" in rel or "INDEX.json" in f.name:
            continue
        c = classify(f)
        if c:
            if max_tier is not None and c.get("tier", 0) > max_tier:
                continue
            results.append(c)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_capability_framing(data: list[dict], outdir: Path):
    solvable = [d for d in data if d["solvable"]]
    unsolvable = [d for d in data if not d["solvable"]]

    n_solv = len(solvable)
    n_uns = len(unsolvable)
    total = n_solv + n_uns

    fig = plt.figure(figsize=(20, 8))

    # ── Grid layout ──
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.35,
                          left=0.06, right=0.94, top=0.88, bottom=0.15)

    # ═══════════════════════════════════════════════════════════════════
    # LEFT: Program Synthesis
    # ═══════════════════════════════════════════════════════════════════
    ax_synth = fig.add_subplot(gs[0, 0])

    synth_counter = Counter(d["solvable_subtype"] for d in solvable)
    synth_order = ["official", "insight", "position_var", "scaled_base", "scaled_var"]
    synth_labels = [
        "Official\nPuzzles",
        "Insight\nVariants",
        "Position\nVariants",
        "Scaled\nBase",
        "Scaled\nVariants",
    ]
    synth_vals = [synth_counter.get(k, 0) for k in synth_order]
    synth_colors = [SYNTH_COLORS[k] for k in synth_order]

    # Filter out zeros
    nonzero = [(l, v, c) for l, v, c in zip(synth_labels, synth_vals, synth_colors) if v > 0]
    synth_labels, synth_vals, synth_colors = zip(*nonzero) if nonzero else ([], [], [])

    wedges, texts, autotexts = ax_synth.pie(
        synth_vals, labels=synth_labels, colors=synth_colors,
        autopct=lambda pct: f"{int(pct * n_solv / 100)}" if pct > 3 else "",
        startangle=90, pctdistance=0.75,
        textprops={"fontsize": 9},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_fontweight("bold")

    # Center text
    ax_synth.text(0, 0, f"SYNTHESIS\n{n_solv} tasks\n({n_solv/total*100:.0f}%)",
                  ha="center", va="center", fontsize=14, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0fff0",
                            edgecolor="#2ca02c", linewidth=2))

    ax_synth.set_title("Program Synthesis", fontsize=16, fontweight="bold",
                        color="#2ca02c", pad=20)

    # ═══════════════════════════════════════════════════════════════════
    # RIGHT: Unsolvability Detection
    # ═══════════════════════════════════════════════════════════════════
    ax_detect = fig.add_subplot(gs[0, 1])

    detect_counter = Counter()
    for d in unsolvable:
        usub = d["unsolvable_subtype"]
        if usub in ("T1", "T2"):
            detect_counter[f"{usub} (same-cat swap)" if usub == "T1" else f"{usub} (diff-cat swap)"] += 1
        elif usub in ("G1", "G2"):
            detect_counter[f"{usub} (N+1 gaps)" if usub == "G1" else f"{usub} (N+2 gaps)"] += 1
        else:
            detect_counter[usub] += 1

    detect_order = [
        "T1 (same-cat swap)",
        "T2 (diff-cat swap)",
        "G1 (N+1 gaps)",
        "G2 (N+2 gaps)",
    ]
    detect_labels = [
        "T1\nSame-category\ntype swap",
        "T2\nCross-category\ntype swap",
        "G1\nN+1 slots\navailable",
        "G2\nN+2 slots\navailable",
    ]
    detect_vals = [detect_counter.get(k, 0) for k in detect_order]
    detect_colors = [DETECT_COLORS[k] for k in detect_order]

    wedges2, texts2, autotexts2 = ax_detect.pie(
        detect_vals, labels=detect_labels, colors=detect_colors,
        autopct=lambda pct: f"{int(pct * n_uns / 100)}" if pct > 3 else "",
        startangle=90, pctdistance=0.7,
        textprops={"fontsize": 8.5},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts2:
        at.set_fontsize(8)
        at.set_fontweight("bold")

    ax_detect.text(0, 0, f"DETECTION\n{n_uns} tasks\n({n_uns/total*100:.0f}%)",
                   ha="center", va="center", fontsize=14, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff0f0",
                             edgecolor="#d62728", linewidth=2))

    ax_detect.set_title("Unsolvability Detection", fontsize=16, fontweight="bold",
                         color="#d62728", pad=20)

    # ═══════════════════════════════════════════════════════════════════
    # Footer annotation with interpretation
    # ═══════════════════════════════════════════════════════════════════
    footer = (
        "Benchmark Framing: two orthogonal capabilities, not a monolithic task set.\n"
        "Synthesis measures program-generation ability. Detection measures the model's "
        "capacity to identify ill-posed configurations\n"
        "(type mismatches and resource-constraint violations) — a prerequisite for "
        "safe deployment in open-ended reasoning tasks."
    )
    fig.text(0.5, 0.03, footer, ha="center", va="bottom", fontsize=9,
             style="italic", color="#555555")

    # Title
    fig.suptitle("Turing Tumble Benchmark — Capability Decomposition",
                 fontsize=18, fontweight="bold", y=0.96)

    fig.savefig(outdir / "10_capability_framing.png", dpi=200, facecolor="white")
    plt.close(fig)


def plot_unsolvable_detail(data: list[dict], outdir: Path):
    """Compact detail of what each unsolvable subtype measures, with examples."""
    unsolvable = [d for d in data if not d["solvable"]]

    detect_counter = Counter()
    for d in unsolvable:
        usub = d["unsolvable_subtype"]
        if usub == "T1":
            detect_counter["T1"] += 1
        elif usub == "T2":
            detect_counter["T2"] += 1
        elif usub == "G1":
            detect_counter["G1"] += 1
        elif usub == "G2":
            detect_counter["G2"] += 1
        else:
            detect_counter["other"] += 1

    fig, ax = plt.subplots(figsize=(14, 4.5))

    categories = [
        ("T1", "Same-category\ntype swap", "Ramp right → Ramp left\n(within 'ramp' family)"),
        ("T2", "Cross-category\ntype swap", "Ramp → Crossover\n(different semantics)"),
        ("G1", "N+1 slots\navailable", "1 component needed,\n2 slots provided"),
        ("G2", "N+2 slots\navailable", "1 component needed,\n3 slots provided"),
    ]
    sub_labels = [c[1] for c in categories]
    examples = [c[2] for c in categories]
    vals = [detect_counter.get(c[0], 0) for c in categories]
    colors = ["#d62728", "#e377c2", "#ff7f0e", "#8c564b"]

    x = np.arange(len(categories))
    bars = ax.bar(x, vals, color=colors, edgecolor="white", width=0.55)

    for bar, v, ex in zip(bars, vals, examples):
        # Count on top
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                str(v), ha="center", fontsize=13, fontweight="bold")
        # Example inside bar
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                ex, ha="center", va="center", fontsize=7.5, color="white",
                fontweight="bold", linespacing=1.3)

    ax.set_xticks(x)
    ax.set_xticklabels(sub_labels, fontsize=9)
    ax.set_ylabel("Number of Tasks", fontsize=11)
    ax.set_title("Unsolvable Subtype Breakdown — What Each Category Tests",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "11_unsolvable_detail.png", dpi=200, facecolor="white")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("Collecting challenge metadata (Tier 1-2 only)...")
    data = collect_all(max_tier=2)
    print(f"  → {len(data)} tasks")

    solv = sum(1 for d in data if d["solvable"])
    uns = sum(1 for d in data if not d["solvable"])
    print(f"  → {solv} synthesis + {uns} detection")

    print("\nGenerating plots...")
    plot_capability_framing(data, OUTDIR)
    print("  ✓ 10_capability_framing.png")
    plot_unsolvable_detail(data, OUTDIR)
    print("  ✓ 11_unsolvable_detail.png")

    print(f"\n  → Saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
