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

# Difficulty order: 1 component, 2 components, then the mixed sets. The scaled_*
# variants are the same difficulty tiers at ~1000 tasks instead of 5 — at n=5 a
# single task moves the rate by 20 points, which is the noise floor that made two
# identical runs of the same model differ by 20 points. Prefer them for anything
# quantitative; the small sets are useful for a fast smoke check, not for a claim.
SET_ORDER = ["1comp", "2comp", "official", "scaled_1comp", "scaled_2comp", "scaled"]
SET_LABELS = {
    "1comp": "1comp\n1 component",
    "2comp": "2comp\n2 components",
    "official": "official\n1–9 components",
    "scaled_1comp": "scaled_1comp\n1 component",
    "scaled_2comp": "scaled_2comp\n2 components",
    # NOT "variants + unsolvable" as the job script's comment claimed: all 2121
    # task files carry a solution with placed_components, and the filenames
    # (..._sz13, ..._sz15) identify them as the official challenges re-rendered on
    # larger boards. There is no unsolvable subset and therefore no ceiling below
    # 100% — an axis label asserting otherwise would misstate the result.
    "scaled": "scaled\nlarger boards (13×13, 15×15)",
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


def collect(results_dir: Path, wanted: list[str]) -> dict[str, dict[str, dict]]:
    """{model: {set_label: row}} for base sets only, invalid runs dropped."""
    by_model = discover(results_dir, None, None)
    out: dict[str, dict[str, dict]] = {}
    for model, rows in by_model.items():
        keep: dict[str, dict] = {}
        for row in rows:
            base = SET_BUDGET_RE.sub("", row["set"])
            if base not in wanted or row["set"] != base:
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
        elif rows:
            # The model HAS results, just not for the requested sets — usually a
            # job that has not reached the scaled sets yet. Say so: a model that
            # vanishes from a comparison chart without a word is indistinguishable
            # from one that was never run.
            have = ", ".join(sorted(SET_BUDGET_RE.sub("", r["set"]) for r in rows))
            print(
                f"  ! {model} has no data for the requested sets — omitted from the "
                f"figures entirely. It does have: {have}",
                file=sys.stderr,
            )
    return out


# ── Figure 1: success rate by challenge set ──────────────────────────────────
def fig_success_by_set(data: dict, t: dict, out: Path, sets: list[str]) -> None:
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
        for xi, cset in enumerate(sets):
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

    # Put n on the axis. A rate from 5 tasks and a rate from 1000 are not the
    # same measurement, and the reader cannot tell them apart otherwise.
    ticks = []
    for cs in sets:
        n = max((data[m][cs]["total"] for m in models if cs in data[m]), default=0)
        ticks.append(f"{SET_LABELS.get(cs, cs)}\nn={n}")
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels(ticks, fontsize=8.5, color=t["ink_secondary"])
    ax.set_ylabel("Tasks solved (%)", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.55, len(sets) - 0.45)
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
    first, last = (sets[0], sets[-1]) if len(sets) > 1 else ("", "")
    collapses = bool(first) and all(
        (data[m].get(first) or {}).get("success_rate_pct", 0)
        >= (data[m].get(last) or {}).get("success_rate_pct", 0)
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
def fig_turn_distribution(data: dict, t: dict, out: Path, sets: list[str]) -> None:
    models = list(data)
    rows: list[tuple[str, list[int], list[int], int]] = []
    for model in models:
        ok: list[int] = []
        bad: list[int] = []
        budget = 25
        for cset in sets:
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
def fig_failure_families(data: dict, t: dict, out: Path, sets: list[str]) -> None:
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


def write_table(data: dict, out: Path, sets: list[str]) -> None:
    """The table view the accessibility pass requires: identity never colour-alone."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "success_by_set.csv"
    with path.open("w") as fh:
        fh.write("model," + ",".join(sets) + "\n")
        for model, per_set in data.items():
            cells = []
            for cset in sets:
                row = per_set.get(cset)
                cells.append(f"{row['success_rate_pct']:.1f}" if row else "")
            fh.write(f"{model}," + ",".join(cells) + "\n")
    print(f"  wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="benchmark_results/jureca_tier1")
    ap.add_argument("--out", default="figures", help="output directory (default: %(default)s)")
    ap.add_argument("--dark", action="store_true", help="render for a dark surface")
    ap.add_argument(
        "--models",
        default="",
        help="comma-separated models to include (default: all). Use it to keep a "
        "comparison chart consistent while a job is still running — a model with "
        "only some of the plotted sets leaves a gap in its bar group.",
    )
    ap.add_argument(
        "--exclude",
        default="",
        help="comma-separated models to leave out. Use for runs that are not "
        "comparable rather than not present — e.g. a model evaluated with a "
        "different effective context, whose rate does not belong in the same chart.",
    )
    ap.add_argument(
        "--sets",
        default="",
        help="comma-separated sets to plot, in order (default: whichever of "
        f"{','.join(SET_ORDER)} are present). Prefer the scaled_* variants for "
        "anything quantitative — they carry ~1000 tasks against 5.",
    )
    args = ap.parse_args()

    t = THEMES["dark" if args.dark else "light"]
    apply_theme(t)

    if args.sets:
        wanted = [s.strip() for s in args.sets.split(",") if s.strip()]
        if unknown := [s for s in wanted if s not in SET_ORDER]:
            sys.exit(f"Unknown set(s): {', '.join(unknown)}. Known: {', '.join(SET_ORDER)}")
    else:
        wanted = list(SET_ORDER)

    data = collect(Path(args.results_dir), wanted)
    if not data:
        sys.exit("No usable reports found.")

    if args.exclude:
        drop = [m.strip() for m in args.exclude.split(",") if m.strip()]
        if unknown := [m for m in drop if m not in data]:
            print(
                f"  ! --exclude names {', '.join(unknown)}, which is not in the data anyway",
                file=sys.stderr,
            )
        for m in drop:
            if data.pop(m, None) is not None:
                print(f"  ! excluding {m} by request", file=sys.stderr)
        if not data:
            sys.exit("Every model was excluded.")

    if args.models:
        want_models = [m.strip() for m in args.models.split(",") if m.strip()]
        if unknown := [m for m in want_models if m not in data]:
            sys.exit(
                f"Unknown or unusable model(s): {', '.join(unknown)}.\n"
                f"Available for these sets: {', '.join(sorted(data))}"
            )
        data = {m: data[m] for m in want_models}

    # Name the models that are missing some of the plotted sets. Their bar group
    # will have a hole in it, and a hole is easy to misread as a very low score.
    for model, per_set in data.items():
        if gaps := [s for s in wanted if s not in per_set]:
            print(
                f"  ! {model} has no data for {', '.join(gaps)} — that slot will be "
                f"blank, not zero (a job still running, most likely)",
                file=sys.stderr,
            )

    # Plot only the sets that actually have data, keeping the requested order —
    # an empty bar group for a set nobody ran reads as a zero score.
    sets = [s for s in wanted if any(s in per_set for per_set in data.values())]
    if not sets:
        sys.exit("No reports for the requested sets.")
    if missing := [s for s in wanted if s not in sets]:
        print(f"  ! no data for {', '.join(missing)}; omitting from the figures", file=sys.stderr)
    if len(sets) > 4:
        print(
            f"  ! plotting {len(sets)} sets x {len(data)} models — that is a busy chart. "
            f"Consider --sets scaled_1comp,scaled_2comp,scaled for the presentation.",
            file=sys.stderr,
        )

    out = Path(args.out)
    fig_success_by_set(data, t, out, sets)
    fig_turn_distribution(data, t, out, sets)
    fig_failure_families(data, t, out, sets)
    write_table(data, out, sets)


if __name__ == "__main__":
    main()
