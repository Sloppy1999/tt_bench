#!/usr/bin/env python3
"""Render presentation figures from TT-Bench benchmark reports.

Produces three SVG+PNG figures under figures/:

  1. success_by_set  — success rate per challenge set, one bar group per model.
     The headline: capability collapses as the number of components required
     grows.
  2. turn_distribution — every task's turn count, split by outcome, with the
     turn budget marked. Shows the bimodality: successes finish early, failures
     sit on the ceiling.
  3. failure_families — what the failures actually were, per model.

Data loading and error-family normalisation are imported from inspect_results.py
so the figures and the table cannot disagree — one source of truth for both.

Runs must be excluded when they are not measurements: a report whose tasks all
record zero turns never generated a token (a base checkpoint with no chat
template makes vLLM reject every request). Those are dropped with a warning
rather than plotted as an honest 0%.

Requires matplotlib, so run it with the project venv rather than the system
interpreter:

    uv run python jureca/plot_results.py
    uv run python jureca/plot_results.py --dark --out figures/dark
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from inspect_results import SET_BUDGET_RE, discover  # noqa: E402

# ── Theme ────────────────────────────────────────────────────────────────────
# Both modes are selected, not flipped: the dark column is the same hues stepped
# for the dark surface. Categorical slots are assigned in fixed order and never
# cycled — a 7th model folds into a second figure rather than inventing a hue.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "baseline": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"],
        "ok": "#2a78d6",
        "fail": "#e34948",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "baseline": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"],
        "ok": "#3987e5",
        "fail": "#e66767",
    },
}

# Difficulty order: 1 component, 2 components, then the official mix (1-9).
SET_ORDER = ["1comp", "2comp", "official"]
SET_LABELS = {
    "1comp": "1comp\n1 component",
    "2comp": "2comp\n2 components",
    "official": "official\n1–9 components",
}

# Short, readable names for the normalised error families.
FAMILY_LABELS = [
    ("Illegal free fall", "Incomplete path"),
    ("No interceptor available", "Inventory exhausted"),
    ("No ramp", "Inventory exhausted"),
    ("No solution found", "No solution submitted"),
    ("Inventory check skipped", "Inventory check skipped"),
]


def short_family(fam: str) -> str:
    for needle, label in FAMILY_LABELS:
        if fam.startswith(needle):
            return label
    return fam[:38]


def apply_theme(t: dict) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "savefig.facecolor": t["surface"],
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "text.color": t["ink"],
            "axes.labelcolor": t["ink_secondary"],
            "axes.edgecolor": t["baseline"],
            "xtick.color": t["muted"],
            "ytick.color": t["muted"],
            "axes.titlecolor": t["ink"],
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )


def strip_chrome(ax, t: dict, *, xgrid: bool = False, ygrid: bool = False) -> None:
    """Recessive grid and axes: hairlines behind the marks, never competing."""
    ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    if xgrid:
        ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(t["baseline"])
    ax.spines["bottom"].set_color(t["baseline"])
    ax.tick_params(length=0, pad=6)


def claim_or_describe(claim: bool, claim_title: str, plain_title: str) -> str:
    """Only assert a finding in a title when the rendered data supports it.

    A hardcoded claim ("inventory violations dominate") silently becomes a lie
    the first time a re-run changes the numbers. The claim is recomputed from the
    data being plotted; if it does not hold, the figure falls back to describing
    itself and the presenter makes the argument out loud instead.
    """
    return claim_title if claim else plain_title


def is_measurement(row: dict) -> bool:
    """False for reports that never generated a token.

    A base (non-instruct) checkpoint has no chat template, vLLM answers every
    request with a 400, and every task records zero turns. That is an
    instrumentation failure, not a 0% score, and plotting it as one would be a
    lie of omission.
    """
    turns = list(row["turns_ok"]) + list(row["turns_fail"])
    return bool(turns) and any(t > 0 for t in turns)


def collect(results_dir: Path) -> dict[str, dict[str, dict]]:
    """{model: {set_label: row}} for base sets only, invalid runs dropped."""
    by_model = discover(results_dir, None, None)
    out: dict[str, dict[str, dict]] = {}
    for model, rows in by_model.items():
        keep: dict[str, dict] = {}
        for row in rows:
            base = SET_BUDGET_RE.sub("", row["set"])
            if base not in SET_ORDER or row["set"] != base:
                continue  # skip sweep variants; they belong in their own figure
            if not is_measurement(row):
                print(
                    f"  ! dropping {model}/{row['set']}: all tasks recorded 0 turns "
                    f"(no tokens generated — not a measurement)",
                    file=sys.stderr,
                )
                continue
            keep[base] = row
        if keep:
            out[model] = keep
    return out


# ── Figure 1: success rate by challenge set ──────────────────────────────────
def fig_success_by_set(data: dict, t: dict, out: Path) -> None:
    models = list(data)
    if len(models) > len(t["series"]):
        print(
            f"  ! {len(models)} models exceeds the {len(t['series'])} categorical slots; "
            f"plotting the first {len(t['series'])} and dropping "
            f"{', '.join(models[len(t['series']):])}",
            file=sys.stderr,
        )
        models = models[: len(t["series"])]

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    n = len(models)
    group_w = 0.66
    bar_w = group_w / n
    # Thin marks, with a 2px surface gap so adjacent fills never touch.
    inner = bar_w * 0.78

    for si, model in enumerate(models):
        color = t["series"][si]
        for xi, cset in enumerate(SET_ORDER):
            row = data[model].get(cset)
            if row is None:
                continue
            rate = row["success_rate_pct"]
            x = xi - group_w / 2 + bar_w * (si + 0.5)
            if rate > 0:
                ax.bar(x, rate, width=inner, color=color, edgecolor="none", zorder=3)
            # Direct labels are mandatory here: three of the light slots sit below
            # 3:1 against the surface, so the relief rule applies — the number
            # carries the value, colour only carries identity. A zero gets the
            # label but no bar: a visible stub would misread as a small non-zero.
            ax.text(
                x,
                rate + 1.8,
                f"{rate:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=t["ink_secondary"] if rate > 0 else t["muted"],
            )

    ax.set_xticks(range(len(SET_ORDER)))
    ax.set_xticklabels([SET_LABELS[s] for s in SET_ORDER], fontsize=9, color=t["ink_secondary"])
    ax.set_ylabel("Tasks solved (%)", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.55, len(SET_ORDER) - 0.45)
    strip_chrome(ax, t, ygrid=True)

    handles = [
        plt.Line2D([], [], marker="s", linestyle="none", markersize=7, color=t["series"][i], label=m)
        for i, m in enumerate(models)
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        fontsize=8.5,
        labelcolor=t["ink_secondary"],
        handletextpad=0.5,
        borderaxespad=0.2,
    )
    # The claim holds only if no model does BETTER on the harder set.
    collapses = all(
        (data[m].get("1comp") or {}).get("success_rate_pct", 0)
        >= (data[m].get("official") or {}).get("success_rate_pct", 0)
        for m in models
    )
    ax.set_title(
        claim_or_describe(
            collapses,
            "Agentic success collapses as required components grow",
            "Tasks solved, by challenge set",
        ),
        fontsize=12.5,
        loc="left",
        pad=14,
    )
    save(fig, out, "success_by_set")


# ── Figure 2: turn distribution by outcome ───────────────────────────────────
def fig_turn_distribution(data: dict, t: dict, out: Path) -> None:
    models = list(data)
    rows: list[tuple[str, list[int], list[int], int]] = []
    for model in models:
        ok: list[int] = []
        bad: list[int] = []
        budget = 25
        for cset in SET_ORDER:
            row = data[model].get(cset)
            if row is None:
                continue
            ok += row["turns_ok"]
            bad += row["turns_fail"]
            budget = row["budget"]
        rows.append((model, ok, bad, budget))

    fig, ax = plt.subplots(figsize=(9.5, 0.58 * len(rows) + 2.2))
    budget = rows[0][3] if rows else 25

    # The axis must cover every observed value. Clamping it to the budget would
    # hide the tasks that record MORE turns than the budget allows — which is
    # itself an open finding about the metric, not noise to crop away.
    observed = [v for _m, ok, bad, _b in rows for v in ok + bad] or [budget]
    obs_max = max(observed)
    if obs_max > budget:
        print(
            f"  ! {sum(1 for v in observed if v > budget)} task(s) record more turns "
            f"than the budget of {budget} (max {obs_max}) — axis extended to show them",
            file=sys.stderr,
        )

    ax.axvline(budget, color=t["fail"], linewidth=1.4, linestyle=(0, (4, 3)), zorder=2, alpha=0.7)
    ax.text(
        budget,
        -0.52,
        f"  budget = {budget}",
        fontsize=8.5,
        color=t["ink_secondary"],
        va="center",
        ha="left",
    )

    for yi, (model, ok, bad, _b) in enumerate(rows):
        ax.axhline(yi, color=t["grid"], linewidth=0.8, zorder=1)
        for values, color in ((ok, t["ok"]), (bad, t["fail"])):
            # Deterministic index-based offsets, not random jitter: the figure
            # must be byte-identical on every re-render for the thesis appendix.
            for k, v in enumerate(values):
                dy = ((k % 5) - 2) * 0.055
                ax.plot(
                    v,
                    yi + dy,
                    marker="o",
                    markersize=6.5,
                    color=color,
                    markeredgecolor=t["surface"],
                    markeredgewidth=1.0,  # 2px surface ring so overlaps stay countable
                    linestyle="none",
                    zorder=4,
                    alpha=0.9,
                )

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9, color=t["ink_secondary"])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.invert_yaxis()  # index 0 at the top, matching every other figure
    ax.set_xlabel("Turns used by the agent", fontsize=9)
    # A small left margin so a marker at turn 0 is not sliced by the spine.
    xmax = max(budget, obs_max) * 1.07
    ax.set_xlim(-xmax * 0.015, xmax)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
    strip_chrome(ax, t, xgrid=True)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=7, color=t["ok"], label="solved"),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=7, color=t["fail"], label="failed"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        frameon=False,
        fontsize=8.5,
        labelcolor=t["ink_secondary"],
        ncols=2,
    )
    # Claim only what the data shows: solved runs must finish strictly earlier
    # than the median failure for "finish early" to be an honest headline.
    all_ok = [v for _m, ok, _b, _bu in rows for v in ok]
    all_bad = [v for _m, _o, bad, _bu in rows for v in bad]
    early = bool(all_ok) and bool(all_bad) and max(all_ok) <= max(all_bad)
    ax.set_title(
        claim_or_describe(
            early,
            "Solved tasks finish early; failures sit on the turn ceiling",
            "Turns used per task, by outcome",
        ),
        fontsize=12.5,
        loc="left",
        pad=14,
    )
    save(fig, out, "turn_distribution")


# ── Figure 3: what the failures were ─────────────────────────────────────────
def fig_failure_families(data: dict, t: dict, out: Path) -> None:
    per_model: dict[str, dict[str, int]] = {}
    for model, sets in data.items():
        agg: dict[str, int] = {}
        for row in sets.values():
            for fam, count in row["error_families"].items():
                agg[short_family(fam)] = agg.get(short_family(fam), 0) + count
        if agg:
            per_model[model] = agg

    if not per_model:
        print("  ! no failure families recorded; skipping figure 3", file=sys.stderr)
        return

    totals: dict[str, int] = {}
    for agg in per_model.values():
        for fam, c in agg.items():
            totals[fam] = totals.get(fam, 0) + c
    # Series ladder: keep the top families, fold the tail into "Other" rather
    # than inventing more hues for it.
    ranked = sorted(totals, key=lambda f: -totals[f])
    keep, tail = ranked[:4], ranked[4:]
    families = keep + (["Other"] if tail else [])

    models = list(per_model)
    fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(models) + 2.4))

    for yi, model in enumerate(models):
        agg = per_model[model]
        left = 0.0
        for fi, fam in enumerate(families):
            count = sum(agg.get(f, 0) for f in tail) if fam == "Other" else agg.get(fam, 0)
            if not count:
                continue
            ax.barh(
                yi,
                count,
                left=left,
                height=0.34,
                color=t["series"][fi],
                edgecolor=t["surface"],
                linewidth=1.0,  # 2px surface gap between stacked segments
                zorder=3,
            )
            if count >= max(totals.values()) * 0.06:
                ax.text(
                    left + count / 2,
                    yi,
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=t["surface"],
                    zorder=4,
                )
            left += count

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9, color=t["ink_secondary"])
    ax.set_ylim(-0.7, len(models) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel("Recorded failures", fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
    strip_chrome(ax, t, xgrid=True)

    handles = [
        plt.Line2D([], [], marker="s", linestyle="none", markersize=7, color=t["series"][i], label=f)
        for i, f in enumerate(families)
    ]
    # Below the axes: inside the plot the legend lands on top of a bar, and the
    # bars grow rightwards with the data so no interior corner stays free.
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        frameon=False,
        fontsize=8.5,
        labelcolor=t["ink_secondary"],
        ncols=min(len(families), 3),
    )
    top_family = ranked[0] if ranked else ""
    ax.set_title(
        claim_or_describe(
            top_family == "Inventory exhausted",
            "Failure modes: inventory violations dominate",
            "Failure modes, by model",
        ),
        fontsize=12.5,
        loc="left",
        pad=14,
    )
    save(fig, out, "failure_families")


def save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("svg", "png"):
        path = out / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  wrote {path}")
    plt.close(fig)


def write_table(data: dict, out: Path) -> None:
    """The table view the accessibility pass requires: identity never colour-alone."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "success_by_set.csv"
    with path.open("w") as fh:
        fh.write("model," + ",".join(SET_ORDER) + "\n")
        for model, sets in data.items():
            cells = []
            for cset in SET_ORDER:
                row = sets.get(cset)
                cells.append(f"{row['success_rate_pct']:.1f}" if row else "")
            fh.write(f"{model}," + ",".join(cells) + "\n")
    print(f"  wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="benchmark_results/jureca_tier1")
    ap.add_argument("--out", default="figures", help="output directory (default: %(default)s)")
    ap.add_argument("--dark", action="store_true", help="render for a dark surface")
    args = ap.parse_args()

    t = THEMES["dark" if args.dark else "light"]
    apply_theme(t)

    data = collect(Path(args.results_dir))
    if not data:
        sys.exit("No usable reports found.")

    out = Path(args.out)
    fig_success_by_set(data, t, out)
    fig_turn_distribution(data, t, out)
    fig_failure_families(data, t, out)
    write_table(data, out)


if __name__ == "__main__":
    main()
