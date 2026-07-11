#!/usr/bin/env python3
"""Multi-model Tier 1 comparison: qwen3-coder vs gpt-oss-120b vs llama-3.1-8b."""
import json, sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from tt_bench.simulator.renderer import (
    COLOURS, BOARD_W, BOARD_H, MARGIN_BOTTOM, MARGIN_SIDES, FIG_W, FIG_H,
    draw_peg_grid, draw_board_frame, draw_hopper, draw_catcher,
)

OUTDIR = REPO / "data" / "visualizations"

MODEL_DIRS = {
    "qwen3-coder-30b": "benchmark_results/tier1_retry/2026-07-06T162736",
    "gpt-oss-120b": "benchmark_results/tier1_models/gpt-oss-120b/2026-07-07T044614",
    "llama-3.1-8b": "benchmark_results/tier1_models/llama-3.1-8b/2026-07-07T063324",
}

MODEL_COLORS = {
    "qwen3-coder-30b": "#1f77b4",
    "gpt-oss-120b": "#ff7f0e",
    "llama-3.1-8b": "#2ca02c",
}


def load_model_results(model_name: str, results_dir: str) -> list:
    tasks = []
    base = REPO / results_dir
    for rp in sorted(base.rglob("benchmark_*.json")):
        set_name = rp.parent.name
        with open(rp) as f:
            data = json.load(f)
        for r in data.get("results", []):
            sol = r.get("expected", {}).get("solution", {})
            placed = sol.get("placed_components", [])
            cat_map = {"official": "official", "1comp": "1comp", "1comp_var": "1comp",
                      "2comp": "2comp", "2comp_var": "2comp"}
            cat = cat_map.get(set_name, set_name)
            tasks.append({
                "task_id": r.get("task_id"),
                "success": r.get("success", False),
                "component_score": r.get("component_score", 0),
                "tool_calls": r.get("metrics", {}).get("tool_calls_count", 0),
                "category": cat, "model": model_name,
                "positions": [(c["x"], c["y"]) for c in placed],
            })
    return tasks


def plot_comparison(all_data: dict, outdir: Path):
    models = list(all_data.keys())
    cats = ["official", "1comp", "2comp"]
    
    # ═════════════════════════════════════════════════
    # IMAGE 1: Success rate by model & category
    # ═════════════════════════════════════════════════
    fig1, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    
    x = np.arange(len(cats))
    width = 0.25
    
    for i, model in enumerate(models):
        mdata = all_data[model]
        rates = []
        counts = []
        for cat in cats:
            ct = [t for t in mdata if t["category"] == cat]
            rates.append(sum(1 for t in ct if t["success"]) / len(ct) if ct else 0)
            counts.append(len(ct))
        offset = (i - 1) * width
        bars = ax.bar(x + offset, rates, width, label=model, color=MODEL_COLORS[model],
                     edgecolor="white")
        for bar, rate, n in zip(bars, rates, counts):
            if n > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f"{rate*100:.0f}%", ha="center", fontsize=9, fontweight="bold",
                       color=MODEL_COLORS[model])
    
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in cats], fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Tier 1 Agentic Synthesis — Model Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig1.savefig(outdir / "models_01_comparison.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig1)
    print("  ✓ models_01_comparison.png")

    # ═════════════════════════════════════════════════
    # IMAGE 2: Zone analysis per model
    # ═════════════════════════════════════════════════
    zones_order = ["Top (y=0-3)", "Mid (y=4-6)", "Bot (y=7-9)"]
    zone_colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    
    fig2, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor="white")
    
    # Left: grouped bars by model×zone for 1comp
    ax1 = axes[0]
    x1 = np.arange(len(models))
    w1 = 0.25
    for iz, (zn, zc) in enumerate(zip(zones_order, zone_colors)):
        ylo, yhi = [(0,3), (4,6), (7,9)][iz]
        rates = []
        for model in models:
            ct = [t for t in all_data[model] if t["category"] == "1comp"
                 and any(ylo <= y <= yhi for (_, y) in t["positions"])]
            rates.append(sum(1 for t in ct if t["success"]) / len(ct) if ct else 0)
        offset = (iz - 1) * w1
        ax1.bar(x1 + offset, rates, w1, label=zn, color=zc, edgecolor="white")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.set_ylabel("Success Rate", fontsize=11)
    ax1.set_title("1-Component by Zone", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    
    # Right: 2comp by zone
    ax2 = axes[1]
    for iz, (zn, zc) in enumerate(zip(zones_order, zone_colors)):
        ylo, yhi = [(0,3), (4,6), (7,9)][iz]
        rates = []
        for model in models:
            ct = [t for t in all_data[model] if t["category"] == "2comp"
                 and any(ylo <= y <= yhi for (_, y) in t["positions"])]
            rates.append(sum(1 for t in ct if t["success"]) / len(ct) if ct else 0)
        offset = (iz - 1) * w1
        ax2.bar(x1 + offset, rates, w1, label=zn, color=zc, edgecolor="white")
    ax2.set_xticks(x1)
    ax2.set_xticklabels(models, fontsize=10)
    ax2.set_ylabel("Success Rate", fontsize=11)
    ax2.set_title("2-Component by Zone", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    
    fig2.suptitle("Zone Sensitivity by Model", fontsize=14, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(outdir / "models_02_zones.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig2)
    print("  ✓ models_02_zones.png")

    # ═════════════════════════════════════════════════
    # IMAGE 3: Board heatmaps per model (3 columns)
    # ═════════════════════════════════════════════════
    fig3, axes3 = plt.subplots(1, 3, figsize=(28, 10), facecolor="white")
    
    for im, (model, ax_heat) in enumerate(zip(models, axes3)):
        mdata = all_data[model]
        ax_heat.set_xlim(0, FIG_W)
        ax_heat.set_ylim(0, FIG_H)
        ax_heat.set_aspect("equal")
        ax_heat.axis("off")
        ax_heat.set_facecolor(COLOURS["bg"])
        
        draw_peg_grid(ax_heat)
        n_succ = sum(1 for t in mdata if t["success"])
        draw_board_frame(ax_heat, title=model, subtitle=f"{len(mdata)} tasks, {n_succ/len(mdata)*100:.0f}% success")
        draw_hopper(ax_heat, 2, "B", 8, "blue", -1)
        draw_hopper(ax_heat, 8, "R", 8, "red", -1)
        draw_catcher(ax_heat, 2, "blue")
        draw_catcher(ax_heat, 8, "red")
        
        cell_data = defaultdict(lambda: {"success": 0, "total": 0})
        for t in mdata:
            for (x, y) in t["positions"]:
                if 0 <= x <= 10 and 0 <= y <= 10:
                    cell_data[(x, y)]["total"] += 1
                    if t["success"]: cell_data[(x, y)]["success"] += 1
        
        for (x, y), cd in cell_data.items():
            rate = cd["success"] / cd["total"] if cd["total"] > 0 else 0
            n = cd["total"]
            cx = MARGIN_SIDES + x
            cy = MARGIN_BOTTOM + (BOARD_H - 1 - y)
            color = plt.colormaps["RdYlGn"](rate)
            rect = mpatches.Rectangle((cx - 0.48, cy - 0.48), 0.96, 0.96,
                                      facecolor=color, edgecolor="#333",
                                      linewidth=1, alpha=0.75, zorder=10)
            ax_heat.add_patch(rect)
            ax_heat.text(cx, cy + 0.12, f"{rate*100:.0f}%", ha="center", va="center",
                        fontsize=6.5, fontweight="bold", color="black", zorder=11)
            ax_heat.text(cx, cy - 0.15, f"n={n}", ha="center", va="center",
                        fontsize=5, color="#333", zorder=11)
    
    sm = plt.cm.ScalarMappable(cmap=plt.colormaps["RdYlGn"], norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig3.colorbar(sm, ax=axes3, shrink=0.4, aspect=30, location="bottom", pad=0.06)
    cbar.set_label("Success Rate", fontsize=10)
    
    fig3.suptitle("Success Rate Heatmap by Component Position — Tier 1 (1comp + 2comp)",
                  fontsize=15, fontweight="bold", y=0.98)
    fig3.savefig(outdir / "models_03_heatmaps.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig3)
    print("  ✓ models_03_heatmaps.png")

    # ═════════════════════════════════════════════════
    # IMAGE 4: Summary table
    # ═════════════════════════════════════════════════
    fig4, ax_table = plt.subplots(figsize=(16, 6), facecolor="white")
    ax_table.axis("off")
    
    rows = []
    for model in models:
        mdata = all_data[model]
        for cat in cats:
            ct = [t for t in mdata if t["category"] == cat]
            if not ct: continue
            succ = sum(1 for t in ct if t["success"])
            avg_calls = np.mean([t["tool_calls"] for t in ct]) if ct else 0
            rows.append([model, cat.upper(), f"{len(ct)}", f"{succ}/{len(ct)}",
                        f"{succ/len(ct)*100:.0f}%", f"{avg_calls:.1f}"])
        total_s = sum(1 for t in mdata if t["success"])
        total_n = len(mdata)
        rows.append([model, "TOTAL", f"{total_n}", f"{total_s}/{total_n}",
                    f"{total_s/total_n*100:.0f}%",
                    f"{np.mean([t['tool_calls'] for t in mdata]):.1f}"])
    
    col_labels = ["Model", "Category", "Tasks", "Success", "Rate", "Avg Calls"]
    cell_colors = []
    for row in rows:
        if "TOTAL" in row[1]:
            cell_colors.append(["#e0e0e0"] * 6)
        elif row[0] == "qwen3-coder-30b":
            cell_colors.append(["#d4e6f1"] * 6)
        elif row[0] == "gpt-oss-120b":
            cell_colors.append(["#fdebd0"] * 6)
        else:
            cell_colors.append(["#d5f5e3"] * 6)
    
    table = ax_table.table(cellText=rows, colLabels=col_labels, cellColours=cell_colors,
                          loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white")
        if row > 0 and "TOTAL" in rows[row-1][1]:
            cell.set_text_props(fontweight="bold")
    
    ax_table.set_title("Tier 1 Model Comparison Summary", fontsize=14, fontweight="bold", pad=15)
    fig4.savefig(outdir / "models_04_table.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig4)
    print("  ✓ models_04_table.png")


def print_summary(all_data: dict):
    print(f"\n{'='*80}")
    print(f"  TIER 1 MODEL COMPARISON")
    print(f"{'='*80}")
    cats = ["official", "1comp", "2comp"]
    for model in all_data:
        mdata = all_data[model]
        total_s = sum(1 for t in mdata if t["success"])
        total_n = len(mdata)
        avg_calls = np.mean([t["tool_calls"] for t in mdata])
        print(f"\n  {model} ({total_n} tasks, {total_s/total_n*100:.1f}% success, {avg_calls:.1f} avg calls)")
        for cat in cats:
            ct = [t for t in mdata if t["category"] == cat]
            if not ct: continue
            succ = sum(1 for t in ct if t["success"])
            print(f"    {cat:10s}: {succ}/{len(ct):3d} ({succ/len(ct)*100:5.1f}%)")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    
    all_data = {}
    for model, path in MODEL_DIRS.items():
        print(f"Loading {model}...")
        tasks = load_model_results(model, path)
        print(f"  → {len(tasks)} tasks")
        all_data[model] = tasks
    
    plot_comparison(all_data, OUTDIR)
    print_summary(all_data)


if __name__ == "__main__":
    main()
