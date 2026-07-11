#!/usr/bin/env python3
"""
Success rate vs tool calls: binned analysis + distribution per model.
Shows diminishing returns — success rate plateaus after ~12 tool calls.
"""
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUTDIR = REPO / "data" / "visualizations"

MODELS = {
    "qwen3-coder-30b": REPO / "benchmark_results/tier1_retry/2026-07-06T162736",
    "gpt-oss-120b": REPO / "benchmark_results/tier1_models/gpt-oss-120b/2026-07-07T044614",
    "llama-3.1-8b": REPO / "benchmark_results/tier1_models/llama-3.1-8b/2026-07-07T063324",
}

MODEL_COLORS = {
    "qwen3-coder-30b": "#1f77b4",
    "gpt-oss-120b": "#ff7f0e",
    "llama-3.1-8b": "#2ca02c",
}


def load_tasks(path: Path) -> list:
    tasks = []
    for rp in sorted(path.rglob("benchmark_*.json")):
        set_name = rp.parent.name
        with open(rp) as f:
            data = json.load(f)
        for r in data.get("results", []):
            cat_map = {"official": "official", "1comp": "1comp", "1comp_var": "1comp",
                      "2comp": "2comp", "2comp_var": "2comp"}
            tasks.append({
                "success": r.get("success", False),
                "tool_calls": r.get("metrics", {}).get("tool_calls_count", 0),
                "category": cat_map.get(set_name, set_name),
                "component_score": r.get("component_score", 0) or 0,
            })
    return tasks


def plot_binned_combined(all_data: dict, outdir: Path):
    """Combined plot: binned success rate + swarm per model."""
    models = list(all_data.keys())
    
    # ── Bins: 0-5, 6-10, 11-15, 16-20, 21+ ──
    bins = [(0, 5), (6, 10), (11, 15), (16, 20), (21, 50)]
    bin_labels = ["0-5", "6-10", "11-15", "16-20", "21+"]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), facecolor="white")
    
    # ── LEFT: Binned success rate per model ──
    ax_bin = axes[0]
    x = np.arange(len(bin_labels))
    width = 0.25
    
    for im, model in enumerate(models):
        mdata = all_data[model]
        rates = []
        counts = []
        for lo, hi in bins:
            bt = [t for t in mdata if lo <= t["tool_calls"] <= hi]
            rates.append(sum(1 for t in bt if t["success"]) / len(bt) if bt else 0)
            counts.append(len(bt))
        offset = (im - 1) * width
        bars = ax_bin.bar(x + offset, rates, width, label=model,
                         color=MODEL_COLORS[model], edgecolor="white")
        for bar, rate, n in zip(bars, rates, counts):
            if n > 0:
                ax_bin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f"n={n}", ha="center", fontsize=7, fontweight="bold",
                           color=MODEL_COLORS[model])
    
    ax_bin.set_xticks(x)
    ax_bin.set_xticklabels(bin_labels, fontsize=11)
    ax_bin.set_xlabel("Tool Calls", fontsize=12)
    ax_bin.set_ylabel("Success Rate", fontsize=12)
    ax_bin.set_title("Success Rate by Tool Call Range", fontsize=13, fontweight="bold")
    ax_bin.set_ylim(0, 1.15)
    ax_bin.legend(fontsize=9, loc="upper left")
    ax_bin.grid(axis="y", alpha=0.3)
    
    # ── RIGHT: Strip plot — individual tool calls colored by success ──
    ax_strip = axes[1]
    
    for im, model in enumerate(models):
        mdata = all_data[model]
        succ = [t for t in mdata if t["success"]]
        fail = [t for t in mdata if not t["success"]]
        
        y_base = im * 1.0
        # Failed — red dots
        if fail:
            xs_fail = [t["tool_calls"] for t in fail]
            ys_fail = [y_base + 0.25 + np.random.uniform(-0.12, 0.12) for _ in fail]
            ax_strip.scatter(xs_fail, ys_fail, alpha=0.4, s=20, color="#d62728",
                           edgecolors="none", label="Fail" if im == 0 else "")
        # Success — green dots
        if succ:
            xs_succ = [t["tool_calls"] for t in succ]
            ys_succ = [y_base - 0.25 + np.random.uniform(-0.12, 0.12) for _ in succ]
            ax_strip.scatter(xs_succ, ys_succ, alpha=0.5, s=20, color="#2ca02c",
                           edgecolors="none", label="Success" if im == 0 else "")
        
        # Model label
        ax_strip.text(-1.5, y_base, model, ha="right", va="center", fontsize=10,
                     fontweight="bold", color=MODEL_COLORS[model])
        
        # Mean line
        mean_calls = np.mean([t["tool_calls"] for t in mdata])
        ax_strip.axhline(y=y_base, color="#999", linewidth=0.5, linestyle=":")
        ax_strip.scatter([mean_calls], [y_base], marker="|", s=200, color=MODEL_COLORS[model],
                        zorder=10, linewidths=2)
    
    ax_strip.set_xlabel("Tool Calls", fontsize=12)
    ax_strip.set_yticks([])
    ax_strip.set_title("Tool Calls Distribution — Success (green) vs Fail (red)",
                       fontsize=13, fontweight="bold")
    ax_strip.legend(fontsize=9, loc="upper right")
    ax_strip.set_xlim(-0.5, max(max(t["tool_calls"] for t in all_data[m]) for m in models) + 3)
    
    fig.suptitle("Success Rate vs Tool Calls — Tier 1 Agentic Synthesis",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "calls_vs_success.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ calls_vs_success.png")


def plot_per_model(all_data: dict, outdir: Path):
    """Per-model detailed plot: binned bars + histogram of tool calls."""
    bins = [(0,5),(6,10),(11,15),(16,20),(21,25),(26,30),(31,50)]
    bin_labels = ["0-5","6-10","11-15","16-20","21-25","26-30","31+"]
    
    for model, mdata in all_data.items():
        model_dir = outdir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")
        
        # LEFT: success rate per bin
        rates = []; counts = []; sizes = []
        for lo, hi in bins:
            bt = [t for t in mdata if lo <= t["tool_calls"] <= hi]
            rates.append(sum(1 for t in bt if t["success"]) / len(bt) if bt else 0)
            counts.append(len(bt))
            sizes.append(len(bt))
        
        x = np.arange(len(bin_labels))
        colors_bin = [plt.colormaps["RdYlGn"](r) for r in rates]
        bars = ax1.bar(x, rates, color=colors_bin, edgecolor="white")
        for bar, rate, n in zip(bars, rates, counts):
            if n > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{rate*100:.0f}%\nn={n}", ha="center", fontsize=8, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(bin_labels, fontsize=9)
        ax1.set_xlabel("Tool Calls", fontsize=11)
        ax1.set_ylabel("Success Rate", fontsize=11)
        ax1.set_title(f"{model} — Success Rate by Tool Calls", fontsize=12, fontweight="bold")
        ax1.set_ylim(0, 1.2)
        ax1.grid(axis="y", alpha=0.3)
        
        # RIGHT: stacked histogram success/fail
        succ_calls = [t["tool_calls"] for t in mdata if t["success"]]
        fail_calls = [t["tool_calls"] for t in mdata if not t["success"]]
        
        all_calls = succ_calls + fail_calls
        bin_edges = np.arange(0, max(all_calls) + 4, 3) if all_calls else np.arange(0, 30, 3)
        
        ax2.hist([succ_calls, fail_calls], bins=bin_edges, stacked=True,
                color=["#2ca02c", "#d62728"], edgecolor="white",
                label=["Success", "Fail"])
        ax2.set_xlabel("Tool Calls", fontsize=11)
        ax2.set_ylabel("Number of Tasks", fontsize=11)
        ax2.set_title(f"{model} — Tool Calls Histogram", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.3)
        
        n_succ = len(succ_calls)
        n_total = len(mdata)
        fig.suptitle(f"{model} — {n_succ}/{n_total} success ({n_succ/n_total*100:.0f}%) | "
                    f"μ calls={np.mean(all_calls):.1f}",
                    fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(model_dir / "calls_analysis.png", dpi=200, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {model}/calls_analysis.png")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading results...")
    all_data = {}
    for model, path in MODELS.items():
        tasks = load_tasks(path)
        all_data[model] = tasks
        print(f"  {model}: {len(tasks)} tasks")
    
    plot_binned_combined(all_data, OUTDIR)
    plot_per_model(all_data, OUTDIR)


if __name__ == "__main__":
    main()
