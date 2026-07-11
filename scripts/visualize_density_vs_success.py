#!/usr/bin/env python3
"""Success rate vs component density visualization for the latest experiment.

Aggregates benchmark results across qwen3-coder-30b, gpt-oss-120b, and
llama-3.1-8b, computes per-task component density (total components / board area),
and plots success rate against density.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent

# ── Model result paths ──
MODELS = {
    "qwen3-coder-30b": REPO / "benchmark_results/tier1_retry/2026-07-06T162736",
    "gpt-oss-120b": REPO / "benchmark_results/tier1_models/gpt-oss-120b/2026-07-07T044614",
    "llama-3.1-8b": REPO / "benchmark_results/tier1_models/llama-3.1-8b/2026-07-07T063324",
}

SETS = ["official", "1comp", "1comp_var", "2comp", "2comp_var"]

MODEL_COLORS = {
    "qwen3-coder-30b": "#2ca02c",
    "gpt-oss-120b": "#1f77b4",
    "llama-3.1-8b": "#d62728",
}

SET_MARKERS = {
    "official": "s",
    "1comp": "o",
    "1comp_var": "o",
    "2comp": "^",
    "2comp_var": "^",
}

SET_NAMES = {
    "official": "Official",
    "1comp": "1-comp base",
    "1comp_var": "1-comp variants",
    "2comp": "2-comp base",
    "2comp_var": "2-comp variants",
}


def load_density(task_id: str) -> float | None:
    """Compute component density for a task ID by locating its JSON file."""
    tasks_root = REPO / "data/tasks"

    # Synthetic tasks contain "1comp" or "2comp" in the ID
    if "1comp" in task_id:
        comp_dir = tasks_root / "challenges_1comp"
    elif "2comp" in task_id:
        comp_dir = tasks_root / "challenges_2comp"
    else:
        # Official challenge
        path = tasks_root / "official/challenges/json" / f"{task_id}.json"
        if path.exists():
            data = json.loads(path.read_text())
            fixed = len(data["board"].get("fixed_components", []))
            sol = len(data.get("solution", {}).get("placed_components", []))
            w = data["board"].get("width", 11)
            h = data["board"].get("height", 11)
            return (fixed + sol) / (w * h)
        return None

    # Try subdirectories for synthetic tasks
    subdirs = [
        "",
        "variants",
        "variants/insight",
        "variants/unsolvable",
        "insight",
        "unsolvable",
    ]
    for sub in subdirs:
        path = comp_dir / sub / f"{task_id}.json"
        if path.exists():
            data = json.loads(path.read_text())
            fixed = len(data["board"].get("fixed_components", []))
            sol = len(data.get("solution", {}).get("placed_components", []))
            w = data["board"].get("width", 11)
            h = data["board"].get("height", 11)
            return (fixed + sol) / (w * h)

    return None


def main():
    OUTDIR = REPO / "data/visualizations"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Collect per-task data: model -> [(density, success, set_name, task_id)]
    per_model: dict[str, list] = defaultdict(list)

    for model_name, model_dir in MODELS.items():
        for s in SETS:
            files = sorted(model_dir.glob(f"{s}/benchmark_*.json"))
            if not files:
                continue
            data = json.loads(files[0].read_text())
            for r in data["results"]:
                density = load_density(r["task_id"])
                if density is None:
                    continue
                per_model[model_name].append(
                    (density, 1.0 if r["success"] else 0.0, s, r["task_id"])
                )

    # ── Build the figure ──
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor="white")
    fig.suptitle(
        "Success Rate vs Component Density — Tier 1 Agentic Synthesis",
        fontsize=16, fontweight="bold", y=0.98,
    )

    for ax, (model_name, points) in zip(axes, per_model.items()):
        color = MODEL_COLORS[model_name]
        densities = [p[0] for p in points]
        successes = [p[1] for p in points]
        sets_list = [p[2] for p in points]

        # Density buckets for trend line
        buckets = defaultdict(list)
        for d, s in zip(densities, successes):
            bucket = round(d, 2)  # 0.01 granularity
            buckets[bucket].append(s)

        bucket_x = sorted(buckets.keys())
        bucket_y = [np.mean(buckets[b]) for b in bucket_x]
        bucket_n = [len(buckets[b]) for b in bucket_x]

        # Filter buckets with >= 3 tasks
        bucket_x_f = [x for x, n in zip(bucket_x, bucket_n) if n >= 3]
        bucket_y_f = [y for x, y, n in zip(bucket_x, bucket_y, bucket_n) if n >= 3]

        # Scatter: individual tasks (with jitter + transparency)
        np.random.seed(42)
        jittered_x = [d + np.random.uniform(-0.003, 0.003) for d in densities]
        jittered_y = [s + np.random.uniform(-0.02, 0.02) for s in successes]

        for s_name, marker in SET_MARKERS.items():
            idxs = [i for i, sn in enumerate(sets_list) if sn == s_name]
            if idxs:
                ax.scatter(
                    [jittered_x[i] for i in idxs],
                    [jittered_y[i] for i in idxs],
                    c=color, marker=marker, alpha=0.25, s=30,
                    edgecolors="none",
                    label=SET_NAMES[s_name] if model_name == "qwen3-coder-30b" else "",
                )

        # Trend line: aggregated by density bucket
        ax.plot(
            bucket_x_f, bucket_y_f,
            color=color, linewidth=2.5, marker="D", markersize=6,
            label="_nolegend_", zorder=5,
        )

        # Fitted trend line
        if len(bucket_x_f) >= 3:
            try:
                slope, intercept, r_val, p_val, _ = stats.linregress(
                    bucket_x_f, bucket_y_f
                )
                x_range = np.linspace(min(bucket_x_f), max(bucket_x_f), 50)
                ax.plot(
                    x_range, slope * x_range + intercept,
                    color=color, linewidth=1, linestyle="--", alpha=0.5,
                )
                r_label = f"r={r_val:.2f}" if r_val < 0 else f"r=+{r_val:.2f}"
            except Exception:
                r_label = ""

        # Styling
        ax.set_xlabel("Component Density\n(total components / board area)", fontsize=10)
        ax.set_ylabel("Success Rate", fontsize=10)
        ax.set_title(model_name, fontsize=13, fontweight="bold", color=color, pad=10)
        ax.set_xlim(0.02, 0.28)
        ax.set_ylim(-0.08, 1.08)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.axhline(y=0.5, color="#ccc", linewidth=0.5, linestyle=":", zorder=1)
        ax.grid(axis="y", color="#eee", linewidth=0.5, zorder=1)
        ax.set_facecolor("#fafafa")

        # Annotate overall success rate
        overall = np.mean(successes)
        n = len(points)
        ax.text(
            0.95, 0.05, f"n={n}  μ={overall:.1%}",
            transform=ax.transAxes, fontsize=9, color=color,
            ha="right", va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=color, linewidth=1, alpha=0.9),
        )

    # Legend on first subplot only
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(
        by_label.values(), by_label.keys(),
        loc="upper right", fontsize=7, framealpha=0.9,
        ncol=2, columnspacing=0.5,
    )

    # Footer
    fig.text(
        0.5, 0.01,
        "Each point = one task (jittered for visibility). "
        "Diamond markers = binned means (≥3 tasks per bucket). "
        "Dashed line = linear fit. Official tasks span higher densities; "
        "1-comp/2-comp variants cluster at low density (0.08).",
        ha="center", fontsize=8.5, color="#888", style="italic",
    )

    outpath = OUTDIR / "density_vs_success.png"
    fig.savefig(outpath, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved: {outpath}")
    print(f"  qwen3-coder-30b: {sum(1 for p in per_model['qwen3-coder-30b'] if p[1])}/{len(per_model['qwen3-coder-30b'])} tasks")
    print(f"  gpt-oss-120b:    {sum(1 for p in per_model['gpt-oss-120b'] if p[1])}/{len(per_model['gpt-oss-120b'])} tasks")
    print(f"  llama-3.1-8b:    {sum(1 for p in per_model['llama-3.1-8b'] if p[1])}/{len(per_model['llama-3.1-8b'])} tasks")


if __name__ == "__main__":
    main()
