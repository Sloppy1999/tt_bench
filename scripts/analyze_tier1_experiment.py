#!/usr/bin/env python3
"""
Tier 1 full experiment analysis: success/failure heatmap by component position.
"""
import json, sys, re
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
    COLOURS, BOARD_W, BOARD_H, MARGIN_TOP, MARGIN_BOTTOM, MARGIN_SIDES, FIG_W, FIG_H,
    draw_peg_grid, draw_board_frame, draw_component, draw_hopper, draw_catcher,
)

OUTDIR = REPO / "data" / "visualizations"

def load_all_results(results_path: str | None = None):
    if results_path:
        latest = Path(results_path)
    else:
        results_dir = REPO / "benchmark_results/tier1_retry"
        runs = sorted(results_dir.glob("*"))
        if not runs:
            results_dir = REPO / "benchmark_results/tier1_full"
            runs = sorted(results_dir.glob("*"))
        if not runs:
            print("No results found")
            return []
        latest = runs[-1]
    print(f"Loading from: {latest}")
    
    tasks = []
    for rp in sorted(latest.rglob("benchmark_*.json")):
        # Determine set from parent directory name
        set_name = rp.parent.name  # official, 1comp, 1comp_var, 2comp, 2comp_var, scaled
        with open(rp) as f:
            data = json.load(f)
        for r in data.get("results", []):
            tid = r.get("task_id", "")
            # Map set_name to category
            cat_map = {
                "official": "official",
                "1comp": "1comp", "1comp_var": "1comp",
                "2comp": "2comp", "2comp_var": "2comp",
                "scaled": "scaled",
            }
            cat = cat_map.get(set_name, set_name)
            
            # Get expected solution positions
            sol = r.get("expected", {}).get("solution", {})
            placed = sol.get("placed_components", [])
            
            # Get board dimensions
            exp = r.get("expected", {})
            bd = exp.get("board", {}) if "board" in exp else {}
            bw = bd.get("width", 11)
            bh = bd.get("height", 11)
            
            tasks.append({
                "task_id": tid,
                "success": r.get("success", False),
                "component_score": r.get("component_score", 0),
                "tool_calls": r.get("metrics", {}).get("tool_calls_count", 0),
                "turns": r.get("metrics", {}).get("turns", 0),
                "category": cat,
                "board_w": bw,
                "board_h": bh,
                "positions": [(c["x"], c["y"]) for c in placed],
                "position_types": [c.get("type", "?") for c in placed],
            })
    return tasks


def plot_tier1_analysis(tasks: list, outdir: Path):
    if not tasks:
        return
    
    n_total = len(tasks)
    n_success = sum(1 for t in tasks if t["success"])
    
    cats = ["official", "1comp", "2comp"]
    cat_colors_scatter = {"official": "#1f77b4", "1comp": "#2ca02c", "2comp": "#ff7f0e"}
    
    # ═══════════════════════════════════════════════════════════
    # IMAGE 1: Board heatmap — success rate per cell
    # ═══════════════════════════════════════════════════════════
    fig1, ax_heat = plt.subplots(figsize=(12, 12), facecolor="white")
    ax_heat.set_xlim(0, FIG_W)
    ax_heat.set_ylim(0, FIG_H)
    ax_heat.set_aspect("equal")
    ax_heat.axis("off")
    ax_heat.set_facecolor(COLOURS["bg"])
    
    draw_peg_grid(ax_heat)
    draw_board_frame(ax_heat, title="Success Rate by Component Position", subtitle=f"qwen3-coder-30b-a3b | {n_total} tasks | {n_success/n_total*100:.0f}% overall")
    draw_hopper(ax_heat, 2, "B", 8, "blue", -1)
    draw_hopper(ax_heat, 8, "R", 8, "red", -1)
    draw_catcher(ax_heat, 2, "blue")
    draw_catcher(ax_heat, 8, "red")
    
    cell_data = defaultdict(lambda: {"success": 0, "total": 0, "calls": [], "cats": set()})
    for t in tasks:
        for (x, y) in t["positions"]:
            if 0 <= x <= 10 and 0 <= y <= 10:
                cell_data[(x, y)]["total"] += 1
                if t["success"]:
                    cell_data[(x, y)]["success"] += 1
                cell_data[(x, y)]["calls"].append(t["tool_calls"])
                cell_data[(x, y)]["cats"].add(t["category"])
    
    _CELL = 1.0
    for (x, y), cd in cell_data.items():
        rate = cd["success"] / cd["total"] if cd["total"] > 0 else 0
        n = cd["total"]
        cx = MARGIN_SIDES + x * _CELL
        cy = MARGIN_BOTTOM + (BOARD_H - 1 - y) * _CELL
        color = plt.colormaps["RdYlGn"](rate)
        rect = mpatches.Rectangle((cx - 0.48, cy - 0.48), 0.96, 0.96,
                                  facecolor=color, edgecolor="#333",
                                  linewidth=1, alpha=0.75, zorder=10)
        ax_heat.add_patch(rect)
        ax_heat.text(cx, cy + 0.12, f"{rate*100:.0f}%", ha="center", va="center",
                    fontsize=7, fontweight="bold", color="black", zorder=11)
        ax_heat.text(cx, cy - 0.15, f"n={n}", ha="center", va="center",
                    fontsize=5.5, color="#333", zorder=11)
    
    sm = plt.cm.ScalarMappable(cmap=plt.colormaps["RdYlGn"], norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig1.colorbar(sm, ax=ax_heat, shrink=0.5, aspect=20, location="bottom", pad=0.06)
    cbar.set_label("Success Rate", fontsize=9)
    fig1.savefig(outdir / "tier1_01_heatmap.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig1)
    print("  ✓ tier1_01_heatmap.png")
    
    # ═══════════════════════════════════════════════════════════
    # IMAGE 2: Bar chart — success rate by category & zone
    # ═══════════════════════════════════════════════════════════
    fig2, ax_bar = plt.subplots(figsize=(12, 6), facecolor="white")
    
    zones_order = ["Top (y=0-3)", "Mid (y=4-6)", "Bot (y=7-9)"]
    zone_colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    
    matrix = {}
    for cat in cats:
        cat_tasks = [t for t in tasks if t["category"] == cat]
        matrix[cat] = {}
        for zone_name, y_lo, y_hi in [("Top (y=0-3)", 0, 3), ("Mid (y=4-6)", 4, 6), ("Bot (y=7-9)", 7, 9)]:
            zone_tasks = [t for t in cat_tasks if any(y_lo <= y <= y_hi for (_, y) in t["positions"])]
            if zone_tasks:
                matrix[cat][zone_name] = {
                    "rate": sum(1 for t in zone_tasks if t["success"]) / len(zone_tasks),
                    "n": len(zone_tasks),
                    "avg_calls": np.mean([t["tool_calls"] for t in zone_tasks]),
                }
    
    x = np.arange(len(cats))
    width = 0.25
    for i, (zone_name, zcolor) in enumerate(zip(zones_order, zone_colors)):
        rates = [matrix[cat].get(zone_name, {}).get("rate", 0) for cat in cats]
        counts = [matrix[cat].get(zone_name, {}).get("n", 0) for cat in cats]
        offset = (i - 1) * width
        bars = ax_bar.bar(x + offset, rates, width, label=zone_name, color=zcolor, edgecolor="white")
        for bar, rate, n in zip(bars, rates, counts):
            if n > 0:
                ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f"n={n}", ha="center", fontsize=8, fontweight="bold", color=zcolor)
    
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([c.upper() for c in cats], fontsize=12)
    ax_bar.set_ylabel("Success Rate", fontsize=12)
    ax_bar.set_title("Success Rate by Category & Zone — qwen3-coder-30b-a3b", fontsize=14, fontweight="bold")
    ax_bar.set_ylim(0, 1.1)
    ax_bar.legend(fontsize=10, ncol=3, loc="upper right")
    ax_bar.grid(axis="y", alpha=0.3)
    fig2.savefig(outdir / "tier1_02_zone_bars.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig2)
    print("  ✓ tier1_02_zone_bars.png")
    
    # ═══════════════════════════════════════════════════════════
    # IMAGE 3: Tool calls vs success scatter
    # ═══════════════════════════════════════════════════════════
    fig3, ax_scatter = plt.subplots(figsize=(12, 6), facecolor="white")
    
    for cat in cats:
        cat_tasks = [t for t in tasks if t["category"] == cat]
        if not cat_tasks: continue
        xs = [t["tool_calls"] for t in cat_tasks]
        ys = [1 if t["success"] else 0 for t in cat_tasks]
        ys_j = [y + np.random.uniform(-0.06, 0.06) for y in ys]
        ax_scatter.scatter(xs, ys_j, alpha=0.5, s=30, color=cat_colors_scatter[cat], label=f"{cat.upper()} (n={len(cat_tasks)})")
    
    ax_scatter.set_xlabel("Tool Calls", fontsize=12)
    ax_scatter.set_ylabel("Success (1=yes, 0=no)", fontsize=12)
    ax_scatter.set_title("Tool Calls vs Success — qwen3-coder-30b-a3b", fontsize=14, fontweight="bold")
    ax_scatter.set_yticks([0, 1])
    ax_scatter.set_yticklabels(["Fail", "Success"])
    ax_scatter.legend(fontsize=10, loc="upper left")
    ax_scatter.grid(alpha=0.3)
    fig3.savefig(outdir / "tier1_03_scatter.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig3)
    print("  ✓ tier1_03_scatter.png")
    
    # ═══════════════════════════════════════════════════════════
    # IMAGE 4: Summary table
    # ═══════════════════════════════════════════════════════════
    fig4, ax_table = plt.subplots(figsize=(14, 5), facecolor="white")
    ax_table.axis("off")
    
    rows = []
    for cat in cats:
        ct = [t for t in tasks if t["category"] == cat]
        if not ct: continue
        succ = sum(1 for t in ct if t["success"])
        avg_score = np.mean([t["component_score"] or 0 for t in ct])
        avg_calls = np.mean([t["tool_calls"] for t in ct])
        # By zone
        for zone_name, ylo, yhi in [("Top (y=0-3)", 0, 3), ("Mid (y=4-6)", 4, 6), ("Bot (y=7-9)", 7, 9)]:
            zt = [t for t in ct if any(ylo <= y <= yhi for (_, y) in t["positions"])]
            if zt:
                zsucc = sum(1 for t in zt if t["success"])
                zcalls = np.mean([t["tool_calls"] for t in zt]) if zt else 0
                rows.append([f"  └ {zone_name}", f"{len(zt)}", f"{zsucc}/{len(zt)}",
                            f"{zsucc/len(zt)*100:.0f}%", f"{zcalls:.1f}"])
        rows.append([cat.upper(), f"{len(ct)}", f"{succ}/{len(ct)}",
                    f"{succ/len(ct)*100:.0f}%", f"{avg_calls:.1f}"])
    
    total_succ = sum(1 for t in tasks if t["success"])
    rows.append(["TOTAL", f"{n_total}", f"{total_succ}/{n_total}",
                f"{total_succ/n_total*100:.0f}%",
                f"{np.mean([t['tool_calls'] for t in tasks]):.1f}"])
    
    col_labels = ["Category / Zone", "Tasks", "Success", "Rate", "Avg Calls"]
    cell_colors = []
    for row in rows:
        if row[0] == "TOTAL":
            cell_colors.append(["#e0e0e0"] * 5)
        elif row[0].startswith("  └"):
            cell_colors.append(["#fafafa"] * 5)
        elif row[0] == "OFFICIAL":
            cell_colors.append(["#d4e6f1"] * 5)
        elif row[0] == "1COMP":
            cell_colors.append(["#d5f5e3"] * 5)
        elif row[0] == "2COMP":
            cell_colors.append(["#fdebd0"] * 5)
    
    table = ax_table.table(cellText=rows, colLabels=col_labels, cellColours=cell_colors,
                          loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white")
        if row == len(rows):
            cell.set_text_props(fontweight="bold")
    
    ax_table.set_title("Tier 1 Experiment — qwen3-coder-30b-a3b-instruct", fontsize=14, fontweight="bold", pad=15)
    fig4.savefig(outdir / "tier1_04_table.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig4)
    print("  ✓ tier1_04_table.png")
    
    return outdir / "tier1_01_heatmap.png"


def print_summary(tasks: list):
    cats = ["official", "1comp", "2comp", "scaled"]
    print(f"\n{'='*70}")
    print(f"  TIER 1 EXPERIMENT SUMMARY — qwen3-coder-30b-a3b-instruct")
    print(f"{'='*70}")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  Successful:  {sum(1 for t in tasks if t['success'])} ({sum(1 for t in tasks if t['success'])/len(tasks)*100:.1f}%)")
    print()
    
    for cat in cats:
        ct = [t for t in tasks if t["category"] == cat]
        if not ct: continue
        succ = sum(1 for t in ct if t["success"])
        avg_calls = np.mean([t["tool_calls"] for t in ct])
        avg_score = np.mean([t["component_score"] or 0 for t in ct])
        
        # By zone
        for zone_name, ylo, yhi in [("Top y=0-3", 0, 3), ("Mid y=4-6", 4, 6),
                                       ("Bot y=7-9", 7, 9), ("Deep y≥10", 10, 99)]:
            zt = [t for t in ct if any(ylo <= y <= yhi for (_, y) in t["positions"])]
            if zt:
                zsucc = sum(1 for t in zt if t["success"])
                zcalls = np.mean([t["tool_calls"] for t in zt])
                print(f"  {cat:10s} {zone_name:14s} {zsucc}/{len(zt):3d} ({zsucc/len(zt)*100:5.1f}%)  calls={zcalls:.1f}")
        print(f"  {cat:10s} {'OVERALL':14s} {succ}/{len(ct):3d} ({succ/len(ct)*100:5.1f}%)  calls={avg_calls:.1f}  score={avg_score:.2f}")
        print()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=None,
                       help="Path to benchmark results directory")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for visualizations (default: data/visualizations/)")
    parser.add_argument("--model-label", type=str, default=None,
                       help="Model label for output filenames")
    args = parser.parse_args()
    
    outdir = Path(args.output_dir) if args.output_dir else OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("Loading experiment results...")
    tasks = load_all_results(args.results_dir)
    print(f"  → {len(tasks)} tasks loaded")
    
    plot_tier1_analysis(tasks, outdir)
    print_summary(tasks)


if __name__ == "__main__":
    main()
