#!/usr/bin/env python3
"""Render the tier-1/tier-2 sweep figures from results_summary.json.

Companion to plot_results.py, which plots from the raw benchmark_results/ tree.
That tree is ~9 GB and lives only on JURECA; results_summary.json is the portable
artifact (one row per model x set x tier) and carries everything these figures
need. Plotting from it means the figures and the tables in docs/results_analysis.md
are computed from the same file and cannot drift apart.

Two things this script must get right that plot_results.py does not:

* **Tier.** A tier-1 and a tier-2 run of the same set share one results
  directory, so anything keyed on the set name alone silently keeps whichever
  tier was written last. Every row here is keyed (set, tier).
* **Unsolvable variants.** scaled_1comp and scaled_2comp are ~2/3 unsolvable
  tasks, where success means declaring the task unsolvable rather than building
  a board. Their aggregate success rate is therefore capped near the solvable
  fraction for any model that never declares one, and reads as a collapse in
  compositional skill that has not happened. Those two sets are plotted only as
  a solvable/unsolvable decomposition (figure 3), never as a bare rate.

    uv run python jureca/plot_summary.py

Requires unsolvable_breakdown.json for figure 3 (per-task solvable/unsolvable
counts, built on the cluster from the same reports results_summary.json cites).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from plot_results import THEMES, apply_theme, strip_chrome  # noqa: E402

from tt_bench.simulator.renderer import (  # noqa: E402
    BOARD_H,
    MARGIN_BOTTOM,
    MARGIN_SIDES,
    draw_board_frame,
    draw_catcher,
    draw_hopper,
    draw_peg_grid,
)

# Sets with no unsolvable variants — the only ones whose bare success rate is a
# like-for-like measure. Ordered small-to-large so the eye moves from the
# 5-task smoke sets to the ~1400/2170-task ones that carry the claims.
CLEAN_SETS = ["1comp", "2comp", "official", "scaled"]
SPLIT_SETS = ["scaled_1comp", "scaled_2comp"]

# Column order for the success matrix only — CLEAN_SETS stays the membership
# list that rank() and the failure-family figure test against, where order is
# irrelevant. official leads: it is the headline rate every other column is read
# against. The three generalisation sets follow as one group, held off from
# official by a gap, so a 5-task smoke cell is not read as a like-for-like
# neighbour of the rate that carries the claim.
MATRIX_SETS = ["official", "1comp", "2comp", "scaled"]

# Display names matching docs/results_analysis.md, so a figure and the table
# beside it name the same model the same way.
DISPLAY = {
    "deepseek-coder-v2-lite": "DeepSeek-Coder-V2-Lite",
    "gemma-4-26b-a4b": "gemma-4-26B-A4B-it",
    "gemma-4-31b-it": "gemma-4-31B-it",
    "gpt-oss-120b": "gpt-oss-120b",
    "qwen2.5-coder-7b": "Qwen2.5-Coder-7B",
    "qwen3.6-35b-a3b": "qwen3.6-35B-A3B",
    "qwen3.8-flash-next-awq4": "qwen3.8-flash-next-awq4",
}

TIER_LABEL = {"1": "Tier 1", "2": "Tier 2"}


def load(summary_path: Path) -> dict:
    """{model: {(set, tier): row}} — keyed on tier, never on set alone."""
    raw = json.loads(summary_path.read_text())
    out: dict[str, dict[tuple[str, str], dict]] = {}
    for model, rows in raw.items():
        out[model] = {}
        for r in rows:
            # Only the default run of a set: a _tN/_T/_a/_sN row shares its
            # (base_set, tier) key and would silently replace it.
            if r["set"] != r["base_set"] or r.get("temperature", "0.0") != "0.0" or r.get("sample"):
                continue
            key = (r["base_set"], r["tier"])
            if key in out[model]:
                raise ValueError(f"{model}: two default rows for {key}")
            out[model][key] = r
    return out


def rank(data: dict) -> list[str]:
    """Models best-to-worst by pooled rate over the unconfounded cells, both tiers.

    Pooled (successes / tasks), so a 5-task cell does not weigh as much as a
    2170-task one, and only over cells every model has, so a model missing a
    hard cell is not ranked higher for it. One order, reused by every figure:
    a model that moves between figures reads as a different model.
    """
    cells = set.intersection(
        *({k for k in data[m] if k[0] in CLEAN_SETS} for m in data)
    ) if data else set()

    def pooled(model: str) -> float:
        rows = [data[model][k] for k in cells]
        total = sum(r["total"] for r in rows)
        return sum(r["successful"] for r in rows) / total if total else 0.0

    return sorted(data, key=pooled, reverse=True)


# Figure widths. A figure drawn 11 in wide and placed at \textwidth in the
# thesis (~6.3 in) is scaled to ~0.55, taking an 8 pt label down to 4.4 pt.
# The print sizes are close to the final width, so explicit font sizes survive
# at roughly the size they were chosen at.
SIZES = {
    "screen": {
        "success_matrix": (11.2, 4.6),
        "tier_gap": (10.6, 4.0),
        "solvable_split": (11.4, 7.2),
        "budget_exhaustion": (9.0, 4.4),
        "turn_distribution": (9.6, 4.4),
        "failure_families": (9.6, 4.8),
    },
    "print": {
        "success_matrix": (6.6, 4.4),
        "tier_gap": (6.6, 4.2),
        "solvable_split": (6.6, 7.8),
        "budget_exhaustion": (6.6, 4.0),
        "turn_distribution": (6.6, 4.0),
        "failure_families": (6.6, 4.4),
    },
}
MODE = "screen"


def size(name: str) -> tuple[float, float]:
    return SIZES[MODE][name]


def footer(fig, text: str, t: dict) -> None:
    fig.text(0.008, 0.012, text, fontsize=7 if MODE == "screen" else 5.5,
             color=t["muted"], ha="left", va="bottom")


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / name}.png / .svg")


# ── Figure 1: success-rate matrix ────────────────────────────────────────────
def fig_success_matrix(data: dict, t: dict, out: Path, models: list[str]) -> None:
    """Model x set success rate, one panel per tier.

    A heatmap rather than 7 bar groups: the categorical palette has six slots,
    and a 7th model would be dropped or forced into an invented hue. Magnitude
    is the whole message here, so one sequential hue carries it and every cell
    is labelled — identity never rests on colour alone.
    """
    # One hue, light to dark, lightness strictly decreasing: magnitude reads off
    # the ramp without a second hue implying a second meaning.
    ramp = LinearSegmentedColormap.from_list("tt_blue", ["#f4f7fb", "#8fb8e8", "#2a78d6", "#123f74"])
    fig, axes = plt.subplots(1, 2, figsize=size("success_matrix"))

    for ax, tier in zip(axes, ("1", "2")):
        grid = []
        for m in models:
            grid.append(
                [
                    data[m][(s, tier)]["success_rate_pct"] if (s, tier) in data[m] else float("nan")
                    for s in MATRIX_SETS
                ]
            )
        ax.imshow(grid, cmap=ramp, vmin=0, vmax=100, aspect="auto")
        # 1comp / 2comp carry 5 and 13 tasks. At n=5 a single task moves a cell by
        # 20 points, so a cell there is not evidence of the same weight as the
        # 1397-task scaled cell drawn the same size and shade. The gap after
        # official sets the smoke sets apart from the two rates that carry claims.
        ax.axvline(0.5, color=t["surface"], linewidth=2.6, zorder=3)

        for yi, m in enumerate(models):
            for xi, s in enumerate(MATRIX_SETS):
                row = data[m].get((s, tier))
                if not row:
                    continue
                v = row["success_rate_pct"]
                ax.text(
                    xi, yi, f"{v:.0f}",
                    ha="center", va="center", fontsize=9,
                    color="#ffffff" if v >= 55 else t["ink"],
                )

        n = {s: (data[models[0]].get((s, tier)) or {}).get("total", "?") for s in MATRIX_SETS}
        ax.set_xticks(range(len(MATRIX_SETS)))
        ax.set_xticklabels([f"{s}\nn={n[s]}" for s in MATRIX_SETS], fontsize=8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([DISPLAY.get(m, m) for m in models] if tier == "1" else [], fontsize=8.5)
        ax.set_title(TIER_LABEL[tier], fontsize=10.5, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    footer(
        fig,
        "Unconfounded sets: every task is solvable. Single run, temperature 0, 25-turn budget. "
        "1comp / 2comp are smoke sets (n=5-13): 8-20 points per task. "
        "scaled_1comp / scaled_2comp appear in the split figure.",
        t,
    )
    save(fig, out, "success_matrix")


# ── Figure 2: tier-1 → tier-2 gap ────────────────────────────────────────────
def fig_tier_gap(data: dict, t: dict, out: Path, models: list[str]) -> None:
    """Every model's tier-1 and tier-2 rate on one row, joined.

    The claim is directional — tier 2 is harder for everyone — so the mark that
    matters is the connector, not either endpoint.
    """
    sets = ["official", "scaled"]
    fig, axes = plt.subplots(1, 2, figsize=size("tier_gap"), sharey=True)
    order = list(reversed(models))  # best at the top

    for ax, cset in zip(axes, sets):
        for yi, m in enumerate(order):
            a = data[m].get((cset, "1"))
            b = data[m].get((cset, "2"))
            if not a or not b:
                continue
            x1, x2 = a["success_rate_pct"], b["success_rate_pct"]
            ax.plot([x1, x2], [yi, yi], color=t["baseline"], linewidth=2, zorder=1,
                    solid_capstyle="round")
            ax.scatter([x1], [yi], s=58, marker="o", color=t["series"][0], zorder=3,
                       edgecolor=t["surface"], linewidth=1.4)
            ax.scatter([x2], [yi], s=52, marker="s", color=t["series"][1], zorder=3,
                       edgecolor=t["surface"], linewidth=1.4)

        n1 = (data[models[0]].get((cset, "1")) or {}).get("total", "?")
        n2 = (data[models[0]].get((cset, "2")) or {}).get("total", "?")
        ax.set_title(f"{cset}   (tier 1 n={n1}, tier 2 n={n2})", fontsize=10)
        ax.set_xlim(-4, 104)
        ax.set_xlabel("success rate (%)", fontsize=9)
        strip_chrome(ax, t, xgrid=True)

    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([DISPLAY.get(m, m) for m in order], fontsize=8.5)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=8,
                   color=t["series"][0], label="Tier 1"),
        plt.Line2D([], [], marker="s", linestyle="none", markersize=8,
                   color=t["series"][1], label="Tier 2"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.08), fontsize=9)
    # Not "every model": Qwen2.5-Coder-7B rises 17.0 -> 20.9 on scaled, and both
    # floor models sit at 0.0 on official in both tiers. A headline that
    # overstates the trend is exactly what the figure is supposed to catch.
    fig.suptitle("Tier 1 → Tier 2: success falls for every model except Qwen2.5-Coder-7B on scaled",
                 fontsize=12, y=1.02)
    save(fig, out, "tier_gap")


# ── Figure 3: solvable vs unsolvable ─────────────────────────────────────────
def fig_solvable_split(breakdown: list[dict], t: dict, out: Path, models: list[str]) -> None:
    """Two abilities the aggregate rate conflates on scaled_1comp/scaled_2comp.

    These sets are ~2/3 unsolvable variants. A model that never declares a task
    unsolvable is capped at the solvable fraction no matter how good its board
    building is, which is why the aggregate rate looks like a collapse.
    """
    idx = {(b["model"], b["set"], b["tier"]): b for b in breakdown}
    order = list(reversed(models))
    fig, axes = plt.subplots(2, 2, figsize=size("solvable_split"), sharex=True, sharey=True)

    for ri, cset in enumerate(SPLIT_SETS):
        for ci, tier in enumerate(("1", "2")):
            ax = axes[ri][ci]
            h = 0.36
            for yi, m in enumerate(order):
                b = idx.get((m, cset, tier))
                if not b:
                    continue
                sol = 100 * b["solvable_ok"] / b["solvable_total"]
                uns = 100 * b["unsolvable_ok"] / b["unsolvable_total"]
                ax.barh(yi + h / 2, sol, height=h * 0.86, color=t["series"][0], zorder=2)
                ax.barh(yi - h / 2, uns, height=h * 0.86, color=t["series"][1],
                        hatch="///", edgecolor=t["ink_secondary"], linewidth=0.4, zorder=2)
                ax.text(sol + 1.5, yi + h / 2, f"{sol:.0f}", va="center", fontsize=7.5,
                        color=t["ink_secondary"])
                ax.text(uns + 1.5, yi - h / 2, f"{uns:.0f}", va="center", fontsize=7.5,
                        color=t["ink_secondary"])

            any_b = next((idx[(m, cset, tier)] for m in models if (m, cset, tier) in idx), None)
            if any_b:
                # Labelled in the title, not on the plot: an in-plot annotation
                # lands on top of the longest bar in exactly the panels where
                # that bar reaches 100.
                cap = 100 * any_b["solvable_total"] / any_b["total"]
                ax.axvline(cap, color=t["muted"], linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
                # At print width the composition does not fit beside the
                # neighbouring panel's title, and the LaTeX caption states it
                # anyway; screen keeps it on the figure, where it must stand alone.
                if MODE == "screen":
                    ax.set_title(
                        f"{cset} · {TIER_LABEL[tier]}   "
                        f"({any_b['solvable_total']} solvable + "
                        f"{any_b['unsolvable_total']} unsolvable → {cap:.0f}% ceiling)",
                        fontsize=9.5,
                    )
                else:
                    ax.set_title(
                        f"{cset} · {TIER_LABEL[tier]}  ({cap:.0f}% ceiling)", fontsize=8.5
                    )
            ax.set_xlim(0, 112)
            strip_chrome(ax, t, xgrid=True)

    for ax in axes[1]:
        ax.set_xlabel("rate within that half of the set (%)", fontsize=9)
    for row in axes:
        row[0].set_yticks(range(len(order)))
        row[0].set_yticklabels([DISPLAY.get(m, m) for m in order], fontsize=8)

    handles = [
        plt.Line2D([], [], marker="s", linestyle="none", markersize=9,
                   color=t["series"][0], label="Solvable tasks solved"),
        mpatches.Patch(facecolor=t["series"][1], edgecolor=t["ink_secondary"],
                       linewidth=0.4, hatch="///",
                       label="Unsolvable tasks correctly declared unsolvable"),
        plt.Line2D([], [], linestyle=(0, (4, 3)), linewidth=1.2, color=t["muted"],
                   label="Aggregate ceiling if no unsolvable task is detected"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.05), fontsize=8.5)
    fig.suptitle("scaled_1comp / scaled_2comp decompose into two different abilities",
                 fontsize=12.5, y=1.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    save(fig, out, "solvable_split")


# ── Figure 4: budget exhaustion ──────────────────────────────────────────────
def fig_budget_exhaustion(data: dict, t: dict, out: Path, models: list[str]) -> None:
    """Share of failures that used the whole 25-turn budget, tier 2.

    Separates running out of room from getting it wrong early: two failure modes
    that the success rate alone cannot tell apart.
    """
    sets = ["official", "scaled"]
    order = list(reversed(models))
    fig, ax = plt.subplots(figsize=size("budget_exhaustion"))
    h = 0.36

    for yi, m in enumerate(order):
        for si, cset in enumerate(sets):
            row = data[m].get((cset, "2"))
            if not row:
                continue
            failed = row["total"] - row["successful"]
            if failed <= 0:
                continue
            share = 100 * row["fail_at_ceiling"] / failed
            off = h / 2 if si == 0 else -h / 2
            ax.barh(yi + off, share, height=h * 0.86, color=t["series"][si],
                    hatch=("" if si == 0 else "///"),
                    edgecolor=t["ink_secondary"], linewidth=0.4, zorder=2)
            ax.text(share + 1.2, yi + off, f"{share:.0f}%  ({row['fail_at_ceiling']}/{failed})",
                    va="center", fontsize=7.5, color=t["ink_secondary"])

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DISPLAY.get(m, m) for m in order], fontsize=8.5)
    ax.set_xlim(0, 125)
    ax.set_xlabel("failures that reached the 25-turn ceiling (%)", fontsize=9)
    strip_chrome(ax, t, xgrid=True)

    handles = [
        mpatches.Patch(facecolor=t["series"][i], edgecolor=t["ink_secondary"],
                       linewidth=0.4, hatch=("" if i == 0 else "///"),
                       label=f"Tier 2 {s}")
        for i, s in enumerate(sets)
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    ax.set_title("Budget exhaustion: which models fail by running out of turns",
                 fontsize=12, pad=10)
    save(fig, out, "budget_exhaustion")


# ── Figure 5: turn distribution ──────────────────────────────────────────────
# Raw scorer strings are unreadable on an axis and several are rare enough that
# a slot of their own would be noise. Anything unlisted folds into "Other".
FAMILY_LABEL = [
    ("Illegal free fall", "Incomplete path"),
    ("Model did not recognize task as unsolvable", "Missed infeasible"),
    ("solution.final_marble_state", "Wrong output sequence"),
    ("Max turns", "Budget exhausted"),
    ("No solution found", "No solution submitted"),
]

# Ordered to match the `fams` list in fig_failure_families. All five reach 3:1
# against the surface and every adjacent pair in the stack separates by at least
# 8 CIE L*; the old series palette mixed green #1baf7a with green #008300 and
# paired #1baf7a with pink #e87ba4 at dL* 1.4, i.e. indistinguishable in grayscale.
FAMILY_COLOURS = ["#0072B2", "#D55E00", "#7B3294", "#B8860B", "#555555"]


def family_label(raw: str) -> str:
    for needle, label in FAMILY_LABEL:
        if raw.startswith(needle):
            return label
    return "Other"


def fig_turn_distribution(data: dict, t: dict, out: Path, models: list[str]) -> None:
    """Where each outcome's episodes end, against the 25-turn budget.

    Marker area is proportional to how many tasks ended on that turn, so the
    pile-up at the ceiling is visible as mass rather than inferred from a range.
    """
    order = list(reversed(models))
    fig, ax = plt.subplots(figsize=size("turn_distribution"))
    budget = 25
    off = 0.19

    for yi, m in enumerate(order):
        rows = list(data[m].values())
        series = {}
        for key in ("turns_ok", "turns_fail"):
            counts: dict[int, int] = {}
            for r in rows:
                for v in r[key]:
                    counts[v] = counts.get(v, 0) + 1
            series[key] = counts
        # One scale per model, shared by both outcomes: scaling each series to
        # its own maximum would make a solved dot and a failed dot of equal area
        # mean different counts inside a single row.
        top = max((n for c in series.values() for n in c.values()), default=1)
        for key, dy, colour in (("turns_ok", off, t["ok"]), ("turns_fail", -off, t["fail"])):
            counts = series[key]
            if not counts:
                continue
            xs = sorted(counts)
            ax.scatter(xs, [yi + dy] * len(xs),
                       s=[8 + 190 * (counts[x] / top) ** 0.5 for x in xs],
                       color=colour, alpha=0.75, linewidths=0, zorder=2)

    ax.axvline(budget, color=t["muted"], linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
    ax.text(budget - 0.3, len(order) - 0.4, "25-turn budget", fontsize=7.5,
            color=t["muted"], ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DISPLAY.get(m, m) for m in order], fontsize=8.5)
    ax.set_xlabel(
        "turns used   ·   marker area ∝ tasks ending on that turn, "
        "scaled within each model row (not comparable across rows)",
        fontsize=7.5,
    )
    ax.set_xlim(0, budget + 1.5)
    strip_chrome(ax, t, xgrid=True)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=8, color=t["ok"], label="Solved"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=8, color=t["fail"], label="Failed"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.30),
              ncol=2, frameon=False, fontsize=8.5)
    ax.set_title("Where episodes end, by outcome", fontsize=12, pad=10)
    save(fig, out, "turn_distribution")


# ── Figure 6: failure families ───────────────────────────────────────────────
def fig_failure_families(data: dict, t: dict, out: Path, models: list[str]) -> None:
    """What the failures actually were, as a share of each model's failures.

    Shares rather than counts: the models differ by an order of magnitude in how
    often they fail, and raw counts would compare the wrong thing.

    Fully-solvable sets only. On scaled_1comp / scaled_2comp about two thirds of
    tasks are unsolvable variants, and failing one means not declaring it
    unsolvable — a refusal trait, not a construction failure. Pooling them made
    "missed infeasible" the largest band for four models (95 % for qwen3.8) and
    contradicted the incomplete-path profile those same models show on solvable
    tasks. Those sets are decomposed in fig_solvable_split, not pooled here.
    """
    order = list(reversed(models))
    fams = ["Incomplete path", "Wrong output sequence", "Budget exhausted",
            "No solution submitted", "Other"]
    fig, ax = plt.subplots(figsize=size("failure_families"))

    for yi, m in enumerate(order):
        tot: dict[str, int] = {}
        for (cset, _tier), r in data[m].items():
            if cset not in CLEAN_SETS:
                continue
            for raw, n in (r.get("error_families") or {}).items():
                lab = family_label(raw)
                tot[lab] = tot.get(lab, 0) + n
        n_all = sum(tot.values())
        if not n_all:
            continue
        left = 0.0
        for fi, f in enumerate(fams):
            v = 100 * tot.get(f, 0) / n_all
            if v <= 0:
                continue
            ax.barh(yi, v, left=left, height=0.62, color=FAMILY_COLOURS[fi],
                    edgecolor=t["surface"], linewidth=0.8, zorder=2)
            if v >= 7:
                ax.text(left + v / 2, yi, f"{v:.0f}", ha="center", va="center",
                        fontsize=7.5, color="#ffffff")
            left += v
        ax.text(101, yi, f"n={n_all:,}", va="center", fontsize=7.5, color=t["ink_secondary"])

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DISPLAY.get(m, m) for m in order], fontsize=8.5)
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(
        "share of that model's failures (%)\n"
        "fully-solvable sets only: 1comp, 2comp, official, scaled",
        fontsize=8.5,
    )
    strip_chrome(ax, t, xgrid=True)
    handles = [plt.Line2D([], [], marker="s", linestyle="none", markersize=8,
                          color=FAMILY_COLOURS[i], label=f) for i, f in enumerate(fams)]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.42),
              ncol=3, frameon=False, fontsize=8)
    ax.set_title("Terminal failure family, as a share of each model's failures",
                 fontsize=12, pad=10)
    save(fig, out, "failure_families")


# ── Figure 7: board-position heatmaps ────────────────────────────────────────
SETDIR = {"1comp": "1comp", "scaled_1comp": "1comp", "2comp": "2comp",
          "scaled_2comp": "2comp", "official": "official", "scaled": "scaled"}


def _aggregate_board(pertask: dict, index: dict, model: str):
    ok = [[0] * 11 for _ in range(11)]
    tot = [[0] * 11 for _ in range(11)]
    for key, recs in pertask.items():
        mm, base, _tier = key.split("|")
        if mm != model:
            continue
        want = SETDIR[base]
        for tid, (succ, _cs) in recs.items():
            if "unsolvable" in tid:
                continue
            cand = index.get(tid)
            if not cand:
                continue
            rec = cand.get(want) or next(iter(cand.values()))
            if rec["w"] != 11 or rec["h"] != 11:
                continue
            for x, y in rec["cells"]:
                if 0 <= x < 11 and 0 <= y < 11:
                    tot[y][x] += 1
                    ok[y][x] += succ
    return ok, tot


# Three models chosen for the website's board figure: the leader, the budget-
# exhaustion regime, and the only model that detects unsolvability. Those are
# the three behaviour clusters the write-up names. fig_board_heatmaps_web adds
# the lowest scorer as a fourth panel, so the figure spans leader to floor.
WEB_MODELS = ["qwen3.8-flash-next-awq4", "gemma-4-31b-it", "gpt-oss-120b"]


def _board_heatmap_figure(pertask, index, t, out, models, *, ncol, nrow, figsize,
                          cell_fs, n_fs, name) -> None:
    """Boards on the simulator's light plate, rate and n printed in every cell."""
    ramp = plt.colormaps["viridis"]
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.96, bottom=0.075, hspace=0.06)
    fig.patch.set_facecolor(t["surface"])
    axes = axes.ravel()

    for ax, m in zip(axes, models):
        ok, tot = _aggregate_board(pertask, index, m)
        # Cropped to the board content: the renderer's canvas reserves a title
        # band at the top and leaves a strip below the catchers, either of which
        # would otherwise letterbox every panel or clip the hopper/catcher labels.
        ax.set_xlim(0.5, 12.5)
        ax.set_ylim(0.2, 16.0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor(t["surface"])
        draw_peg_grid(ax)
        draw_board_frame(ax)
        ax.set_title(DISPLAY.get(m, m), fontsize=cell_fs * 1.55, color=t["ink"], pad=2)
        draw_hopper(ax, 2, "B", 8, "blue", -1)
        draw_hopper(ax, 8, "R", 8, "red", -1)
        draw_catcher(ax, 2, "blue")
        draw_catcher(ax, 8, "red")
        for y in range(11):
            for x in range(11):
                n = tot[y][x]
                if not n:
                    continue
                rate = ok[y][x] / n
                cx = MARGIN_SIDES + x
                cy = MARGIN_BOTTOM + (BOARD_H - 1 - y)
                rgba = ramp(rate)
                # Ink from the rendered fill's luminance: a fixed colour is
                # unreadable at one end of the map.
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ink = "#0b0b0b" if lum > 0.45 else "#ffffff"
                ax.add_patch(mpatches.Rectangle(
                    (cx - 0.46, cy - 0.46), 0.92, 0.92, facecolor=rgba,
                    edgecolor=t["surface"], linewidth=0.6, zorder=10))
                ax.text(cx, cy + 0.12, f"{rate * 100:.0f}%", ha="center",
                        va="center", fontsize=cell_fs, color=ink, zorder=11)
                ax.text(cx, cy - 0.16, f"n={n}", ha="center", va="center",
                        fontsize=n_fs, color=ink, zorder=11)

    for ax in axes[len(models):]:
        ax.axis("off")
        ax.text(0.06, 0.92, "READING THIS FIGURE", transform=ax.transAxes,
                ha="left", va="top", fontsize=cell_fs * 1.55, color=t["ink"],
                fontweight="bold")
        ax.text(0.06, 0.80,
                "square    a board cell the reference solution fills\n"
                "number    success rate on the tasks that fill it\n"
                "n         how many tasks those are\n"
                "dot       a cell no reference solution fills\n"
                "\n"
                "Each hopper holds 8 marbles; the left and right\n"
                "catchers receive the output. Solvable tasks on\n"
                "native 11\u00d711 boards.",
                transform=ax.transAxes, ha="left", va="top", fontsize=cell_fs * 1.3,
                color=t["ink_secondary"], linespacing=1.6)

    cax = fig.add_axes([0.40, 0.066, 0.20, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=ramp, norm=plt.Normalize(0, 1)),
                      cax=cax, orientation="horizontal")
    cb.set_label("success rate on tasks filling the cell", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    footer(fig, "Solvable tasks, native 11\u00d711 boards. Each cell gives the "
                "success rate and n = the number of tasks whose reference "
                "solution fills that cell.", t)
    save(fig, out, name)


def fig_board_heatmaps(pertask: dict, index: dict, t: dict, out: Path,
                       models: list[str]) -> None:
    """Every model's board in one image, for the thesis figure."""
    _board_heatmap_figure(pertask, index, t, out, models, ncol=4, nrow=2,
                          figsize=(14.6, 11.4), cell_fs=5.4, n_fs=4.1,
                          name="board_heatmaps")


def fig_board_heatmaps_web(pertask: dict, index: dict, t: dict, out: Path,
                           models: list[str]) -> None:
    """The three representative models plus the lowest scorer, for the website.

    Seven boards in a single image is not readable at page width. This keeps the
    three behaviour clusters, gives each board several times the area, and fills
    the last panel with the worst model rather than a legend block, so the range
    from leader to floor is visible without a separate figure. `models` arrives
    best-to-worst, so the lowest scorer is the last entry.
    """
    subset = [m for m in models if m in WEB_MODELS]
    if models and models[-1] not in subset:
        subset.append(models[-1])
    _board_heatmap_figure(pertask, index, t, out, subset, ncol=2, nrow=2,
                          figsize=(8.8, 13.0), cell_fs=8.0, n_fs=6.0,
                          name="board_heatmaps_web")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default="results_summary.json")
    ap.add_argument("--breakdown", default="unsolvable_breakdown.json")
    ap.add_argument("--pertask", default="per_task_outcomes.json")
    ap.add_argument("--index", default="task_index.json")
    ap.add_argument("--out", default="docs/assets/figures")
    ap.add_argument("--dark", action="store_true")
    ap.add_argument("--print", dest="print_mode", action="store_true",
                    help="size figures for placement at \\textwidth in the thesis")
    args = ap.parse_args()

    global MODE
    MODE = "print" if args.print_mode else "screen"
    t = THEMES["dark" if args.dark else "light"]
    apply_theme(t)

    data = load(Path(args.summary))
    models = rank(data)
    out = Path(args.out)

    fig_success_matrix(data, t, out, models)
    fig_tier_gap(data, t, out, models)
    fig_budget_exhaustion(data, t, out, models)
    fig_turn_distribution(data, t, out, models)
    fig_failure_families(data, t, out, models)

    pt, ix = Path(args.pertask), Path(args.index)
    if pt.exists() and ix.exists():
        pertask, tindex = json.loads(pt.read_text()), json.loads(ix.read_text())
        fig_board_heatmaps(pertask, tindex, t, out, models)
        fig_board_heatmaps_web(pertask, tindex, t, out, models)
    else:
        print("  ! per-task data missing — skipping the board heatmaps", file=sys.stderr)

    bpath = Path(args.breakdown)
    if bpath.exists():
        fig_solvable_split(json.loads(bpath.read_text()), t, out, models)
    else:
        print(f"  ! {bpath} missing — skipping the solvable/unsolvable figure", file=sys.stderr)


if __name__ == "__main__":
    main()
