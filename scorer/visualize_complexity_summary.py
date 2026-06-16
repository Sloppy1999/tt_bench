#!/usr/bin/env python3
"""Publication-quality summary figures for the complexity metrics report.

Generates seven figures suitable for thesis presentations:

  fig1_tier_distributions.png  — Violin + strip plots of 4 core metrics by tier
  fig2_correlation_heatmap.png — Pairwise Spearman correlation of all 13 metrics
  fig3_tier_progression.png    — Mean ± SEM of core metrics across tiers 1–4
  fig4_boxplots.png            — Box plots + jittered points of core metrics by tier
  fig5_radar.png               — Radar/spider chart of tier metric profiles
  fig6_beeswarm.png            — Beeswarm strip plots with individual points
  fig7_tier_counts.png         — Stacked bar chart of challenge inventory by tier

Usage:
    PYTHONPATH=simulator uv run python scorer/visualize_complexity_summary.py \
        --csv scorer/complexity_metrics.csv \
        --output-dir experiments/complexity_summary

Output:
    experiments/complexity_summary/fig1_tier_distributions.{png,pdf}
    experiments/complexity_summary/fig2_correlation_heatmap.{png,pdf}
    experiments/complexity_summary/fig3_tier_progression.{png,pdf}

Dependencies: matplotlib, numpy (stdlib csv + math, no pandas/seaborn/scipy).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Must set before importing matplotlib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Suppress numpy divide-by-zero in corrcoef (handled via NaN guards)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# ---------------------------------------------------------------------------
# Styling — publication-quality defaults
# ---------------------------------------------------------------------------

# Okabe-Ito colorblind-safe palette (8 colors)
OKABE_ITO = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

# Tier colour mapping
TIER_COLORS = {
    1: OKABE_ITO[1],  # sky blue
    2: OKABE_ITO[0],  # orange
    3: OKABE_ITO[2],  # green
    4: OKABE_ITO[5],  # vermillion
}

TIER_LABELS = {
    1: "Tier 1\n(Routing)",
    2: "Tier 2\n(State & Control)",
    3: "Tier 3\n(Counting)",
    4: "Tier 4\n(Gears & Latching)",
}

CORE_METRICS = ["scr", "bici", "psde", "k_approx"]
CORE_LABELS = {
    "scr": "SCR",
    "bici": "BICI",
    "psde": "PSDE",
    "k_approx": "K̃",
}

ALL_METRICS = [
    "scr", "ctd", "dependency_depth", "gcc", "rpcc", "ibr", "hic",
    "bici", "sac_norm", "synthesis_load", "psde", "oss", "k_approx",
]
ALL_LABELS = {
    "scr": "SCR",
    "ctd": "CTD",
    "dependency_depth": "Dep. Depth",
    "gcc": "GCC",
    "rpcc": "RPCC",
    "ibr": "IBR",
    "hic": "HIC",
    "bici": "BICI",
    "sac_norm": "SAC (norm)",
    "synthesis_load": "Synth. Load",
    "psde": "PSDE",
    "oss": "OSS",
    "k_approx": "K̃",
}


def _setup_style():
    """Apply publication-quality defaults."""
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _label_panel(ax, letter: str, x: float = -0.08, y: float = 1.02):
    """Add bold panel label."""
    ax.text(
        x, y, letter, transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="bottom", ha="left",
    )


# ---------------------------------------------------------------------------
# Data loading (stdlib csv)
# ---------------------------------------------------------------------------

def _load_data(csv_path: Path) -> dict:
    """Load CSV into dict of lists.

    Returns {"task_id": [...], "tier": [...], "scr": [...], ...}
    with only rows that have a valid tier.
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        raw = list(reader)

    columns = raw[0].keys() if raw else []
    data: Dict[str, list] = {col: [] for col in columns}

    for row in raw:
        tier_str = row.get("tier", "").strip()
        if not tier_str:
            continue
        for col in columns:
            val = row.get(col, "").strip()
            if col == "task_id":
                data[col].append(val)
            elif col == "tier":
                data[col].append(int(val))
            else:
                try:
                    data[col].append(float(val))
                except (ValueError, TypeError):
                    data[col].append(np.nan)

    return data


def _tier_values(data: dict, metric: str, tier: int) -> np.ndarray:
    """Extract metric values for a specific tier, dropping NaN."""
    vals = []
    for i, t in enumerate(data["tier"]):
        if t == tier:
            v = data[metric][i]
            if not math.isnan(v):
                vals.append(v)
    return np.array(vals)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _mean(vals: np.ndarray) -> float:
    return float(np.mean(vals)) if len(vals) > 0 else 0.0


def _sem(vals: np.ndarray) -> float:
    if len(vals) <= 1:
        return 0.0
    return float(np.std(vals, ddof=1) / math.sqrt(len(vals)))


def _spearman_r(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation between two arrays."""
    mask = ~(np.isnan(x) | np.isnan(y))
    xc, yc = x[mask], y[mask]
    if len(xc) < 3:
        return float("nan")
    # If either array is constant, correlation is undefined
    if np.std(xc) == 0 or np.std(yc) == 0:
        return float("nan")
    rx = _rank(xc)
    ry = _rank(yc)
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank(vals: np.ndarray) -> np.ndarray:
    """Return ranks with average tie-breaking (like scipy rankdata)."""
    n = len(vals)
    order = np.argsort(vals)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and vals[order[j]] == vals[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


# ---------------------------------------------------------------------------
# Figure 1 — Core metric distributions by tier (violin + strip)
# ---------------------------------------------------------------------------

def fig_tier_distributions(data: dict, output_dir: Path):
    """2×2 grid of violin + strip plots for 4 core metrics."""
    _setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    axes = axes.flatten()

    for i, metric in enumerate(CORE_METRICS):
        ax = axes[i]
        data_by_tier = [_tier_values(data, metric, t) for t in [1, 2, 3, 4]]
        positions = [1, 2, 3, 4]
        colors = [TIER_COLORS[t] for t in [1, 2, 3, 4]]

        # Violins
        vp = ax.violinplot(
            data_by_tier, positions=positions,
            showmeans=False, showmedians=True, showextrema=True,
            widths=0.5,
        )
        for j, body in enumerate(vp["bodies"]):
            body.set_facecolor(colors[j])
            body.set_alpha(0.45)
            body.set_edgecolor(colors[j])
            body.set_linewidth(0.8)
        for part in ["cmedians", "cmins", "cmaxes"]:
            if part in vp:
                vp[part].set_color("#333333")
                vp[part].set_linewidth(1.0)

        # Strip plot (jittered points)
        rng = np.random.default_rng(42)
        for j, vals in enumerate(data_by_tier):
            n = len(vals)
            if n == 0:
                continue
            jitter = rng.uniform(-0.15, 0.15, n)
            ax.scatter(
                np.full(n, positions[j]) + jitter, vals,
                s=18, c=colors[j], alpha=0.6, edgecolors="white",
                linewidths=0.3, zorder=3,
            )

        # Build x-tick labels with n= counts
        ns = [len(d) for d in data_by_tier]
        ax.set_xticks(positions)
        ax.set_xticklabels([f"T1\n(n={ns[0]})", f"T2\n(n={ns[1]})",
                            f"T3\n(n={ns[2]})", f"T4\n(n={ns[3]})"])
        ax.set_ylabel(CORE_LABELS[metric])
        ax.set_xlabel("Challenge tier")
        ax.set_ylim(-0.05, 1.08)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        # Tier mean annotation — capped within plot bounds
        means = [_mean(d) for d in data_by_tier]
        for j, m in enumerate(means):
            ann_y = min(m + 0.07, 1.02)
            ax.annotate(
                f"{m:.3f}", (positions[j], ann_y),
                ha="center", fontsize=7, color=colors[j],
                fontweight="bold",
            )

        _label_panel(ax, chr(65 + i))  # A, B, C, D

    fig.suptitle(
        "Core complexity metrics by challenge tier",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(output_dir / f"fig1_tier_distributions.{fmt}")
    plt.close(fig)
    print(f"  OK  fig1_tier_distributions.png/pdf")


# ---------------------------------------------------------------------------
# Figure 2 — Correlation heatmap
# ---------------------------------------------------------------------------

def fig_correlation_heatmap(data: dict, output_dir: Path):
    """Pairwise Spearman correlation heatmap of all metrics."""
    _setup_style()

    available = [m for m in ALL_METRICS if m in data]
    n = len(available)
    corr = np.zeros((n, n))

    for i, mi in enumerate(available):
        xi = np.array(data[mi], dtype=float)
        for j, mj in enumerate(available):
            xj = np.array(data[mj], dtype=float)
            corr[i, j] = _spearman_r(xi, xj)

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(9, 7.5))

    # Manual heatmap (no seaborn dependency)
    cmap = plt.cm.PuOr
    vmin, vmax = -1.0, 1.0

    for i in range(n):
        for j in range(n):
            val = corr[i, j]
            if mask[i, j]:
                color = "#f0f0f0"
            elif np.isnan(val):
                color = "#eeeeee"
            else:
                color = cmap((val - vmin) / (vmax - vmin))
            rect = plt.Rectangle(
                (j, i), 1, 1,
                facecolor=color, edgecolor="white", linewidth=0.5,
            )
            ax.add_patch(rect)
            if mask[i, j]:
                continue
            elif np.isnan(val):
                ax.text(
                    j + 0.5, i + 0.5, "—",
                    ha="center", va="center", fontsize=7,
                    color="#999999",
                )
            else:
                ax.text(
                    j + 0.5, i + 0.5, f"{val:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(val) > 0.6 else "black",
                )

    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(
        [ALL_LABELS.get(m, m) for m in available],
        rotation=45, ha="right", fontsize=7.5,
    )
    ax.set_yticklabels(
        [ALL_LABELS.get(m, m) for m in available],
        rotation=0, fontsize=7.5,
    )
    ax.set_aspect("equal")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Spearman ρ", fontsize=9)

    # Highlight core 4-metric block
    core_idx = [available.index(m) for m in CORE_METRICS if m in available]
    if len(core_idx) == 4:
        for ci in core_idx:
            for cj in core_idx:
                if ci != cj:
                    rect = plt.Rectangle(
                        (cj, ci), 1, 1,
                        fill=False, edgecolor=OKABE_ITO[4],
                        linewidth=2.0, zorder=10,
                    )
                    ax.add_patch(rect)

    ax.set_title(
        "Spearman correlation of complexity metrics",
        fontsize=12, fontweight="bold", pad=12,
    )

    fig.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(output_dir / f"fig2_correlation_heatmap.{fmt}")
    plt.close(fig)
    print(f"  OK  fig2_correlation_heatmap.png/pdf")


# ---------------------------------------------------------------------------
# Figure 3 — Tier progression (mean ± SEM)
# ---------------------------------------------------------------------------

def fig_tier_progression(data: dict, output_dir: Path):
    """Mean ± SEM of 4 core metrics across tiers."""
    _setup_style()
    tiers = [1, 2, 3, 4]

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    axes = axes.flatten()

    for i, metric in enumerate(CORE_METRICS):
        ax = axes[i]
        means_arr, sems_arr, ns = [], [], []
        for t in tiers:
            vals = _tier_values(data, metric, t)
            means_arr.append(_mean(vals))
            sems_arr.append(_sem(vals))
            ns.append(len(vals))

        # Plot each point in its tier colour, connected by grey segments
        color = OKABE_ITO[4]  # deep blue line connecting
        ax.plot(
            tiers, means_arr, "-", color="#aaaaaa", linewidth=1.5, zorder=2,
        )
        for j, t in enumerate(tiers):
            tc = TIER_COLORS[t]
            ax.plot(
                [tiers[j]], [means_arr[j]], "o",
                color=tc, markersize=10, markerfacecolor=tc,
                markeredgecolor="white", markeredgewidth=1.2, zorder=4,
            )
            # SEM whisker
            ax.errorbar(
                tiers[j], means_arr[j], yerr=sems_arr[j],
                color=tc, linewidth=1.5, capsize=5, capthick=1.5, zorder=3,
            )

        ax.set_ylabel(CORE_LABELS[metric])
        ax.set_xticks(tiers)
        ax.set_xticklabels([f"T{t}\n(n={ns[j]})" for j, t in enumerate(tiers)])
        ax.set_ylim(-0.05, 1.08)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        for j, m in enumerate(means_arr):
            ax.annotate(
                f"  {m:.3f}", (tiers[j], m),
                textcoords="offset points", xytext=(8, -6),
                fontsize=7.5, color="#444444",
            )

        _label_panel(ax, chr(65 + i))

    fig.suptitle(
        "Tier progression of core metrics (mean ± SEM)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(output_dir / f"fig3_tier_progression.{fmt}")
    plt.close(fig)
    print(f"  OK  fig3_tier_progression.png/pdf")


# ---------------------------------------------------------------------------
# Figure 4 — Box plots by tier
# ---------------------------------------------------------------------------

def fig_boxplots(data: dict, output_dir: Path):
    """2×2 grid of box plots for 4 core metrics — standard presentation format."""
    _setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    axes = axes.flatten()

    for i, metric in enumerate(CORE_METRICS):
        ax = axes[i]
        data_by_tier = [_tier_values(data, metric, t) for t in [1, 2, 3, 4]]
        positions = [1, 2, 3, 4]
        colors = [TIER_COLORS[t] for t in [1, 2, 3, 4]]

        bp = ax.boxplot(
            data_by_tier, positions=positions, widths=0.45,
            patch_artist=True, showfliers=True, showmeans=False,
            medianprops={"color": "#333333", "linewidth": 1.2},
            flierprops={"marker": "o", "markersize": 4,
                        "markerfacecolor": "#555555",
                        "markeredgecolor": "#555555", "alpha": 0.5},
        )
        for j, box in enumerate(bp["boxes"]):
            box.set_facecolor(colors[j])
            box.set_alpha(0.55)
            box.set_edgecolor(colors[j])
            box.set_linewidth(1.2)
        for whisker in bp["whiskers"]:
            whisker.set_color("#555555")
            whisker.set_linewidth(0.9)
        for cap in bp["caps"]:
            cap.set_color("#555555")
            cap.set_linewidth(0.9)

        # Overlay individual points
        rng = np.random.default_rng(42)
        for j, vals in enumerate(data_by_tier):
            n = len(vals)
            if n == 0:
                continue
            jitter = rng.uniform(-0.10, 0.10, n)
            ax.scatter(
                np.full(n, positions[j]) + jitter, vals,
                s=14, c=colors[j], alpha=0.35, edgecolors="white",
                linewidths=0.2, zorder=3,
            )

        # n= in x-tick labels
        ns = [len(d) for d in data_by_tier]
        ax.set_xticks(positions)
        ax.set_xticklabels([f"T1\n(n={ns[0]})", f"T2\n(n={ns[1]})",
                            f"T3\n(n={ns[2]})", f"T4\n(n={ns[3]})"])
        ax.set_ylabel(CORE_LABELS[metric])
        ax.set_xlabel("Challenge tier")
        ax.set_ylim(-0.05, 1.08)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        _label_panel(ax, chr(65 + i))

    fig.suptitle(
        "Core metrics — box plots by tier",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(output_dir / f"fig4_boxplots.{fmt}")
    plt.close(fig)
    print(f"  OK  fig4_boxplots.png/pdf")


# ---------------------------------------------------------------------------
# Figure 5 — Radar / spider chart (tier profiles)
# ---------------------------------------------------------------------------

def fig_radar(data: dict, output_dir: Path):
    """Radar chart of mean metric profiles across the 4 tiers.

    Each tier is a polygon spanning the 4 core metrics (normalized to [0,1]).
    Shows tier separation and shape differences at a glance.
    """
    _setup_style()

    tiers = [1, 2, 3, 4]
    metrics = CORE_METRICS
    n_metrics = len(metrics)

    # Compute tier means
    tier_means = {}
    for t in tiers:
        tier_means[t] = [_mean(_tier_values(data, m, t)) for m in metrics]

    # Radar setup
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    line_styles = ["-", "--", "-.", ":"]
    for t in reversed(tiers):  # Tier 4 on top
        values = tier_means[t] + tier_means[t][:1]
        color = TIER_COLORS[t]
        ls = line_styles[(t - 1) % len(line_styles)]
        ax.fill(
            angles, values, alpha=0.08, color=color,
        )
        ax.plot(
            angles, values, linestyle=ls, marker="o", color=color,
            linewidth=2.2, markersize=8, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=1.2,
            label=f"Tier {t}",
        )
        # Value labels at each vertex
        for k, (angle, val) in enumerate(zip(angles[:-1], values[:-1])):
            ax.annotate(
                f"{val:.3f}",
                xy=(angle, val), xytext=(4, 4), textcoords="offset points",
                fontsize=6.5, color=color, fontweight="bold",
                ha="left", va="bottom",
            )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([CORE_LABELS[m] for m in metrics], fontsize=10)
    ax.set_ylim(0, 0.60)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticklabels(["0.1", "0.2", "0.3", "0.4", "0.5"], fontsize=7)
    ax.grid(True, alpha=0.3, linestyle="--")

    ax.legend(
        loc="upper right", fontsize=8.5, frameon=True,
        facecolor="white", edgecolor="#dddddd", framealpha=0.9,
    )
    ax.set_title(
        "Mean metric profiles by tier", fontsize=12, fontweight="bold", pad=20,
    )

    fig.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(output_dir / f"fig5_radar.{fmt}")
    plt.close(fig)
    print(f"  OK  fig5_radar.png/pdf")


# ---------------------------------------------------------------------------
# Figure 6 — Beeswarm / strip plot (every point visible)
# ---------------------------------------------------------------------------

def _beeswarm_positions(vals: np.ndarray, spread: float = 0.35) -> np.ndarray:
    """Compute beeswarm x-offsets: pack points to avoid overlap by pushing
    them outward from the centre line, nearest first."""
    n = len(vals)
    if n == 0:
        return np.array([])
    # Sort by value, then assign alternating sides
    order = np.argsort(vals)
    offsets = np.zeros(n)
    for i, idx in enumerate(order):
        side = 1 if i % 2 == 0 else -1
        layer = (i // 2) + 1
        offsets[idx] = side * layer * (spread / max(4, n // 2))
    return np.clip(offsets, -spread, spread)


def fig_beeswarm(data: dict, output_dir: Path):
    """2×2 grid of beeswarm-style strip plots for 4 core metrics.

    Every data point is individually plotted with collision-avoiding
    jitter — no distribution shape distortion from violin bandwidth.
    """
    _setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    axes = axes.flatten()

    rng = np.random.default_rng(42)

    for i, metric in enumerate(CORE_METRICS):
        ax = axes[i]
        positions = [1, 2, 3, 4]

        ns = []
        for j, t in enumerate([1, 2, 3, 4]):
            vals = _tier_values(data, metric, t)
            n = len(vals)
            ns.append(n)
            if n == 0:
                continue
            offsets = _beeswarm_positions(vals, spread=0.32)
            ax.scatter(
                np.full(n, positions[j]) + offsets, vals,
                s=22, c=TIER_COLORS[t], alpha=0.70,
                edgecolors="white", linewidths=0.3, zorder=3,
            )

            # Diamond mean marker
            mean_v = _mean(vals)
            ax.scatter(
                [positions[j]], [mean_v],
                s=36, c="white", edgecolors="#333333",
                linewidths=1.5, zorder=6, marker="D",
            )

        ax.set_xticks(positions)
        ax.set_xticklabels([f"T1\n(n={ns[0]})", f"T2\n(n={ns[1]})",
                            f"T3\n(n={ns[2]})", f"T4\n(n={ns[3]})"])
        ax.set_ylabel(CORE_LABELS[metric])
        ax.set_xlabel("Challenge tier")
        ax.set_ylim(-0.05, 1.08)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(axis="y", alpha=0.2, linestyle="--")

        _label_panel(ax, chr(65 + i))

    fig.suptitle(
        "Core metrics — individual challenge values by tier",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    for fmt in ["png", "pdf"]:
        fig.savefig(output_dir / f"fig6_beeswarm.{fmt}")
    plt.close(fig)
    print(f"  OK  fig6_beeswarm.png/pdf")


# ---------------------------------------------------------------------------
# Figure 7 — Challenge count distribution by tier (from tasks/ inventory)
# ---------------------------------------------------------------------------

def fig_tier_counts(data: dict, output_dir: Path, index_path: Optional[Path] = None):
    """Stacked bar chart of challenge counts per tier.

    Sources official challenges from INDEX.json and also counts
    1-component and 2-component variant boards, mapping them to
    tiers based on their parent challenge number (ch01–05 → T1, ch06–10 → T2).
    """
    _setup_style()

    tier_main = {1: 0, 2: 0, 3: 0, 4: 0}
    tier_practice = {1: 0, 2: 0, 3: 0, 4: 0}
    tier_bonus = {1: 0, 2: 0, 3: 0, 4: 0}
    tier_synth = {1: 0, 2: 0, 3: 0, 4: 0}  # 1-comp and 2-comp variants

    # --- Official challenges from INDEX.json ---
    index_loaded = False
    if index_path is not None and index_path.exists():
        import json
        with open(index_path) as f:
            idx = json.load(f)
        for entry in idx.get("tasks", []):
            t = entry.get("tier")
            if t not in (1, 2, 3, 4):
                continue
            variant = entry.get("variant") or ""
            if variant.startswith("practice"):
                tier_practice[t] += 1
            elif variant.startswith("bonus"):
                tier_bonus[t] += 1
            else:
                tier_main[t] += 1
        index_loaded = True

    if not index_loaded:
        # Fallback to CSV-based counting
        for i, tid in enumerate(data["task_id"]):
            t = data["tier"][i]
            if t not in (1, 2, 3, 4):
                continue
            if "-pA" in tid or "-pB" in tid or "-pC" in tid:
                tier_practice[t] += 1
            elif "-bA" in tid:
                tier_bonus[t] += 1
            else:
                tier_main[t] += 1

    # --- 1-comp and 2-comp variants — map to tier by challenge number ---
    here = Path(__file__).resolve().parent.parent
    for vdir_name in ("challenges_1comp", "challenges_2comp"):
        vpath = here / "tasks" / vdir_name
        if not vpath.is_dir():
            continue
        for fpath in sorted(vpath.iterdir()):
            if fpath.suffix != ".json":
                continue
            # Parse challenge number from filename: tt-official-ch<NN>-...json
            import re
            m = re.match(r"tt-official-ch(\d+)", fpath.stem)
            if not m:
                continue
            ch_num = int(m.group(1))
            if 1 <= ch_num <= 5:
                tier_synth[1] += 1
            elif 6 <= ch_num <= 10:
                tier_synth[2] += 1
            elif 11 <= ch_num <= 16:
                tier_synth[2] += 1
            elif 17 <= ch_num <= 22:
                tier_synth[3] += 1
            elif 23 <= ch_num <= 30:
                tier_synth[4] += 1

    # --- Build the figure ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    tiers = [1, 2, 3, 4]
    x = np.arange(len(tiers))
    width = 0.55

    bottom = np.zeros(len(tiers), dtype=float)

    segments = [
        ("Main", [tier_main[t] for t in tiers], OKABE_ITO[4]),         # blue
        ("Practice", [tier_practice[t] for t in tiers], OKABE_ITO[2]), # green
        ("Bonus", [tier_bonus[t] for t in tiers], OKABE_ITO[0]),       # orange
        ("Synth. variant", [tier_synth[t] for t in tiers], OKABE_ITO[5]),  # vermillion
    ]

    for label, values, color in segments:
        bars = ax.bar(x, values, width, bottom=bottom, color=color,
                     alpha=0.85, edgecolor="none",
                     label=label, zorder=3)
        for j, v in enumerate(values):
            if v > 0:
                ax.text(
                    x[j], bottom[j] + v / 2, str(v),
                    ha="center", va="center", fontsize=8.5,
                    fontweight="bold", color="white",
                )
        bottom += np.array(values, dtype=float)

    # Total annotations on top
    for j, t in enumerate(tiers):
        total = tier_main[t] + tier_practice[t] + tier_bonus[t] + tier_synth[t]
        ax.text(
            x[j], total + 1.2, str(total),
            ha="center", va="bottom", fontsize=11,
            fontweight="bold", color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABELS[t] for t in tiers], fontsize=7.5)
    ax.set_ylabel("Number of challenges")
    max_y = max(
        tier_main[t] + tier_practice[t] + tier_bonus[t] + tier_synth[t]
        for t in tiers
    )
    ax.set_ylim(0, max_y + 5)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    ax.set_title(
        "Challenge inventory by tier",
        fontsize=12, fontweight="bold", pad=10,
    )

    fig.tight_layout()

    total_all = sum(
        tier_main[t] + tier_practice[t] + tier_bonus[t] + tier_synth[t]
        for t in tiers
    )
    n_synth = sum(tier_synth.values())
    print(f"  OK  fig7_tier_counts.png/pdf  ({total_all} total: {total_all - n_synth} official + {n_synth} synth variants)")

    for fmt in ["png", "pdf"]:
        fig.savefig(output_dir / f"fig7_tier_counts.{fmt}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality summary figures for complexity metrics"
    )
    parser.add_argument(
        "--csv", type=Path,
        default=Path("scorer/complexity_metrics.csv"),
        help="Path to complexity_metrics.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("experiments/complexity_summary"),
        help="Output directory for generated figures",
    )
    parser.add_argument(
        "--index", type=Path,
        default=Path("tasks/official/INDEX.json"),
        help="Path to INDEX.json for official challenge metadata",
    )
    parser.add_argument(
        "--figures", type=str, nargs="+",
        default=["all"],
        choices=[
            "all",
            "violins", "correlation", "progression",
            "boxplots", "radar", "beeswarm", "counts",
        ],
        help="Which figures to generate (default: all)",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Error: CSV not found at {args.csv}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = _load_data(args.csv)

    # Tier counts
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for t in data["tier"]:
        tier_counts[t] = tier_counts.get(t, 0) + 1

    print(f"Loaded {len(data['task_id'])} challenges from {args.csv}")
    for t in [1, 2, 3, 4]:
        print(f"  Tier {t}: {tier_counts[t]} challenges")
    print()

    figs = set(args.figures)

    if "all" in figs or "violins" in figs:
        print("Generating Figure 1: Violin distributions...")
        fig_tier_distributions(data, args.output_dir)

    if "all" in figs or "correlation" in figs:
        print("Generating Figure 2: Correlation heatmap...")
        fig_correlation_heatmap(data, args.output_dir)

    if "all" in figs or "progression" in figs:
        print("Generating Figure 3: Tier progression...")
        fig_tier_progression(data, args.output_dir)

    if "all" in figs or "boxplots" in figs:
        print("Generating Figure 4: Box plots...")
        fig_boxplots(data, args.output_dir)

    if "all" in figs or "radar" in figs:
        print("Generating Figure 5: Radar chart...")
        fig_radar(data, args.output_dir)

    if "all" in figs or "beeswarm" in figs:
        print("Generating Figure 6: Beeswarm plot...")
        fig_beeswarm(data, args.output_dir)

    if "all" in figs or "counts" in figs:
        print("Generating Figure 7: Tier counts...")
        fig_tier_counts(data, args.output_dir, args.index)

    print(f"\nDone — figures saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
