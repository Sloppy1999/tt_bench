#!/usr/bin/env python3
"""Render the Turing Tumble leaderboard: score (%) vs total parameters (B).

Self-contained: all data inline, no network access, no repo imports.

Data source
-----------
``docs/results_analysis.md`` section 14.3 ("Success by tier"), table at
lines 1059-1067. That table is the only complete per-tier source for the
official challenge sweep: single run per cell, temperature 0, 25-turn
budget, vLLM 0.29.0. Tier board counts: T1=11, T2=22, T3=8, T4=14 (55 total).

Outputs (written to ``figures/leaderboard_by_tier/``)
-----------------------------------------------------
* ``leaderboard_by_tier.png`` (200 dpi, opaque background)
* ``leaderboard_by_tier.pdf``
* ``leaderboard_by_tier.svg``
* ``data.csv``       -- the exact plotted values
* ``provenance.json`` -- source lines, run config, disclosures, file hashes

Run::

    uv run python scripts/plot_leaderboard_by_tier.py

Publishing to the docs site
---------------------------
The PNG is not version-controlled. To publish, copy it into the docs figure
directory::

    cp figures/leaderboard_by_tier/leaderboard_by_tier.png docs/assets/figures/

``website/assets/figures/`` is intentionally NOT updated: the research site
replaced this PNG with a dynamic component and no longer uses the file.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.legend import Legend  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, LogLocator, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "leaderboard_by_tier"

SOURCE_FILE = "docs/results_analysis.md"
SOURCE_SECTION = "14.3 Success by tier"
SOURCE_LINES = "1059-1067"

# ---------------------------------------------------------------------------
# DATA -- success rate (%) by tier [T1, T2, T3, T4]
# Source: docs/results_analysis.md lines 1059-1067 (verbatim; do not alter).
# Total parameter counts in billions (B), ascending. 177 B for
# qwen3.8-flash-next-awq4 was supplied by the user and is absent from repo
# metadata; all other counts come from the model names/metadata.
# ---------------------------------------------------------------------------
MODELS: list[dict] = [
    {"name": "Qwen2.5-Coder-7B", "params_b": 7.0, "scores": [9.1, 0.0, 25.0, 0.0]},
    {"name": "DeepSeek-Coder-V2-Lite", "params_b": 16.0, "scores": [0.0, 0.0, 0.0, 0.0]},
    {"name": "gemma-4-26B-A4B-it", "params_b": 26.0, "scores": [18.2, 4.5, 25.0, 7.1]},
    {"name": "gemma-4-31B-it", "params_b": 31.0, "scores": [45.5, 27.3, 37.5, 7.1]},
    {"name": "qwen3.6-35B-A3B", "params_b": 35.0, "scores": [45.5, 22.7, 12.5, 7.1]},
    {"name": "gpt-oss-120b", "params_b": 120.0, "scores": [9.1, 22.7, 0.0, 7.1]},
    {"name": "qwen3.8-flash-next-awq4", "params_b": 177.0, "scores": [100.0, 72.7, 62.5, 57.1]},
]

# ---------------------------------------------------------------------------
# MODEL FAMILY GROUPING -- inferred from the model names/IDs.
# The repo metadata records only the serving engine (vLLM 0.29.0), not the
# model publisher, so the grouping below is an inference and is disclosed as
# such in CAPTION.md and provenance.json.
# COLOUR ENCODES THE MODEL FAMILY (one Okabe-Ito colour per family; the same
# hexes are used by the website so the two artifacts match). The difficulty
# tier is carried by marker shape + dash pattern ONLY, so no visual channel
# carries two meanings.
# Dashed connectors are drawn per tier ONLY inside a family group with
# >= 2 models (Qwen x3, Google/Gemma x2). Single-model families (DeepSeek,
# OpenAI) keep lone markers and no connecting line.
# ---------------------------------------------------------------------------
PROVIDERS: list[dict] = [
    {
        "id": "qwen",
        "label": "Qwen (Alibaba)",
        "color": "#0072B2",  # Okabe-Ito blue
        "models": ["Qwen2.5-Coder-7B", "qwen3.6-35B-A3B", "qwen3.8-flash-next-awq4"],
    },
    {
        "id": "google",
        "label": "Google (Gemma)",
        "color": "#D55E00",  # Okabe-Ito vermillion
        "models": ["gemma-4-26B-A4B-it", "gemma-4-31B-it"],
    },
    {
        "id": "deepseek",
        "label": "DeepSeek",
        "color": "#009E73",  # Okabe-Ito bluish green
        "models": ["DeepSeek-Coder-V2-Lite"],
    },
    {"id": "openai", "label": "OpenAI", "color": "#CC79A7", "models": ["gpt-oss-120b"]},
]
MODEL_PROVIDER: dict[str, str] = {
    name: provider["id"] for provider in PROVIDERS for name in provider["models"]
}

FAMILY_COLOR: dict[str, str] = {provider["id"]: provider["color"] for provider in PROVIDERS}

# Every model maps to exactly one family, every family has exactly one colour,
# and no two families share a colour -- asserted so neither the mapping nor the
# colour channel can silently drift.
assert set(MODEL_PROVIDER) == {m["name"] for m in MODELS}, "provider mapping must cover MODELS exactly"
assert len(MODEL_PROVIDER) == sum(len(p["models"]) for p in PROVIDERS), "model listed under two providers"
assert set(FAMILY_COLOR) == {p["id"] for p in PROVIDERS}, "every family must define a colour"
assert len(set(FAMILY_COLOR.values())) == len(FAMILY_COLOR), "family colours must be distinct"
assert all(
    MODEL_PROVIDER[m["name"]] in FAMILY_COLOR for m in MODELS
), "every model must resolve to a family colour"


def provider_members(provider: dict) -> list[dict]:
    """MODELS belonging to ``provider``, ordered by ascending total parameters."""
    return sorted(
        (m for m in MODELS if MODEL_PROVIDER[m["name"]] == provider["id"]),
        key=lambda m: m["params_b"],
    )


# Provider id for models that get a connector (group has >= 2 models).
CONNECTOR_GROUP: dict[str, str] = {}
for _provider in PROVIDERS:
    if len(provider_members(_provider)) >= 2:
        for _model in provider_members(_provider):
            CONNECTOR_GROUP[_model["name"]] = _provider["id"]


def build_connectors() -> list[dict]:
    """Provider-scoped connectors: one per provider group with >= 2 models.

    The per-tier horizontal dodge is applied identically to every model in a
    tier, so it can never mix providers; the assertions below make that
    invariant explicit and fail loudly if the grouping is ever edited wrongly.
    """
    connectors: list[dict] = []
    for provider in PROVIDERS:
        members = provider_members(provider)
        if len(members) < 2:
            continue
        joined_providers = {MODEL_PROVIDER[m["name"]] for m in members}
        assert joined_providers == {provider["id"]}, (
            f"connector for {provider['id']!r} would join providers {sorted(joined_providers)!r}"
        )
        params = [m["params_b"] for m in members]
        assert params == sorted(params), f"connector for {provider['id']!r} is not ascending by parameters"
        connectors.append({"provider": provider, "models": members})
    return connectors


def connector_inventory() -> list[dict]:
    """Provider -> points per tier and whether a dashed connector is drawn."""
    inventory: list[dict] = []
    for provider in PROVIDERS:
        members = provider_members(provider)
        inventory.append(
            {
                "provider": provider["id"],
                "label": provider["label"],
                "models": [m["name"] for m in members],
                "points_per_tier": len(members),
                "connector": len(members) >= 2,
            }
        )
    return inventory


# Tier identity is marker shape + dash pattern ONLY. Colour is reserved for
# the model family (PROVIDERS), so no visual channel carries two meanings.
TIERS: list[dict] = [
    {"tier": 1, "n_boards": 11, "marker": "o", "dash": (0, (7.0, 3.0))},
    {"tier": 2, "n_boards": 22, "marker": "s", "dash": (0, (4.0, 2.6))},
    {"tier": 3, "n_boards": 8, "marker": "^", "dash": (0, (3.2, 2.0))},
    {"tier": 4, "n_boards": 14, "marker": "D", "dash": (0, (1.0, 2.2))},
]

assert all("color" not in tier for tier in TIERS), (
    "tier colour retired: colour encodes the model family only"
)

# Tiny per-tier horizontal dodge (log10 decades) so that the exactly
# coincident 0 % markers (DeepSeek-Coder-V2-Lite in all four tiers,
# Qwen2.5-Coder-7B in T2/T4) stay individually visible. Total spread is
# 0.050 decades, i.e. <= 5.9 % in parameter value.
DODGE_LOG10: dict[int, float] = {1: -0.025, 2: -0.0083, 3: 0.0083, 4: 0.025}

# Direct model labels: anchor = (model, tier) point, offsets in points.
# The 26/31/35 B cluster is fanned out with staggered offsets; 35 B also gets
# a thin leader line. DeepSeek is a two-line label because all four of its
# points sit on 0 %.
LABELS: dict[str, dict] = {
    "Qwen2.5-Coder-7B": {"tier": 3, "dx": 0, "dy": 9, "ha": "center", "va": "bottom", "leader": False},
    "DeepSeek-Coder-V2-Lite": {"tier": 1, "dx": 0, "dy": 44, "ha": "center", "va": "bottom", "leader": True},
    "gemma-4-26B-A4B-it": {"tier": 3, "dx": -8, "dy": 8, "ha": "right", "va": "bottom", "leader": False},
    "gemma-4-31B-it": {"tier": 1, "dx": 0, "dy": 8, "ha": "center", "va": "bottom", "leader": False},
    "qwen3.6-35B-A3B": {"tier": 1, "dx": 9, "dy": 30, "ha": "left", "va": "bottom", "leader": True},
    "gpt-oss-120b": {"tier": 2, "dx": 0, "dy": 22, "ha": "right", "va": "bottom", "leader": False},
    "qwen3.8-flash-next-awq4": {"tier": 1, "dx": -12, "dy": 4, "ha": "right", "va": "bottom", "leader": False},
}

TWO_LINE_LABELS: dict[str, str] = {
    "DeepSeek-Coder-V2-Lite": "DeepSeek-Coder-V2-Lite\n(0 % in all four tiers)",
}

# ---------------------------------------------------------------------------
# Style -- dark leaderboard theme (ARC-AGI-like)
# ---------------------------------------------------------------------------
BG = "#0d1117"          # figure background
AXES_BG = "#161b22"     # slightly lighter plot area
TEXT = "#e6edf3"        # primary text
MUTED = "#8b949e"       # secondary text
GRID = "#30363d"        # grid / rules
GRID_MINOR = "#21262d"  # minor grid
LEADER = "#484f58"      # leader lines / zero rule
NEUTRAL_LEGEND = "#c9ced6"  # neutral swatch colour for the tier legend only

FIG_SIZE = (13.5, 7.5)
RENDER_DPI = 200
AXES_RECT = (0.065, 0.095, 0.915, 0.755)

FONT_SANS = ["Noto Sans", "DejaVu Sans"]
FONT_MONO = ["JetBrainsMono NF", "DejaVu Sans Mono"]

# Recorded palette screening (palette_audit.py, background #0d1117, role
# graphical, threshold 3.0) for the four Okabe-Ito model-family colours.
# Re-run to reproduce:
#   uv run python .agents/skills/scientific-visualization/scripts/palette_audit.py \
#     --color 0072B2 --color D55E00 --color 009E73 --color CC79A7 \
#     --background 0d1117 --role graphical
PALETTE_AUDIT = {
    "tool": ".agents/skills/scientific-visualization/scripts/palette_audit.py",
    "palette": "Okabe-Ito (colourblind-safe) - one colour per model family",
    "background": "#0D1117",
    "role": "graphical",
    "contrast_threshold": 3.0,
    "contrast_ratios": {"#0072B2": 3.650, "#D55E00": 4.894, "#009E73": 5.532, "#CC79A7": 6.182},
    "contrast_review_findings": 0,
    "grayscale_delta_lstar_min": 3.304,
    "grayscale_min_threshold": 10.0,
    "grayscale_review_findings": 4,
    "verdict": "PASS (contrast) - all four family colours >= 3:1 against the figure background. "
               "The grayscale heuristic flags 4 of 6 pairs (< 10 CIE L* delta), so colour "
               "alone does not separate families in grayscale; family identity also rests on the "
               "MODEL FAMILY legend and the direct model labels, while tier identity is carried "
               "by marker shape and dash pattern.",
}


def build_rows() -> list[dict]:
    """One row per (model, tier) with the exact plotted coordinates."""
    rows: list[dict] = []
    for model in MODELS:
        for tier in TIERS:
            idx = tier["tier"] - 1
            family = MODEL_PROVIDER[model["name"]]
            rows.append(
                {
                    "model": model["name"],
                    "provider": family,
                    "family": family,
                    "family_color": FAMILY_COLOR[family],
                    "provider_connector": CONNECTOR_GROUP.get(model["name"], "none"),
                    "total_params_b": model["params_b"],
                    "tier": tier["tier"],
                    "tier_n_boards": tier["n_boards"],
                    "score_pct": model["scores"][idx],
                    "x_plotted_b": round(model["params_b"] * 10 ** DODGE_LOG10[tier["tier"]], 3),
                }
            )
    return rows


def write_data_csv(rows: list[dict]) -> Path:
    path = OUT_DIR / "data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "provider",
                "family",
                "family_color",
                "provider_connector",
                "total_params_b",
                "tier",
                "tier_n_boards",
                "score_pct",
                "x_plotted_b",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def render() -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": FONT_SANS,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=FIG_SIZE, dpi=RENDER_DPI, facecolor=BG)
    ax = fig.add_axes(AXES_RECT)
    ax.set_facecolor(AXES_BG)

    ax.set_axisbelow(True)
    ax.grid(which="major", color=GRID, linewidth=0.6, alpha=0.9)
    ax.grid(which="minor", color=GRID_MINOR, linewidth=0.45, alpha=0.55)
    ax.axhline(0.0, color=LEADER, linewidth=0.8, zorder=1)

    # --- series: colour = model family, marker shape + dash = tier -----------
    # All seven markers are drawn for every tier, coloured by family. Dashed
    # connectors join only models from the same family (>= 2 models: Qwen x3,
    # Google/Gemma x2); single-model families (DeepSeek, OpenAI) keep lone
    # markers. Connectors are coloured by family and dashed by tier.
    # build_connectors() asserts no connector mixes families, so the per-tier
    # dodge can never join two publishers.
    connectors = build_connectors()
    family_handles: list[Line2D] = []
    tier_handles: list[Line2D] = []
    for tier in TIERS:
        tier_idx = tier["tier"] - 1
        dodge = 10 ** DODGE_LOG10[tier["tier"]]
        for provider in PROVIDERS:
            members = provider_members(provider)
            ax.plot(
                [m["params_b"] * dodge for m in members],
                [m["scores"][tier_idx] for m in members],
                linestyle="none",
                color=provider["color"],
                marker=tier["marker"],
                markersize=5.4,
                markeredgewidth=0.0,
                alpha=0.96,
                zorder=4,
            )
        for connector in connectors:
            ax.plot(
                [m["params_b"] * dodge for m in connector["models"]],
                [m["scores"][tier_idx] for m in connector["models"]],
                linestyle=tier["dash"],
                color=connector["provider"]["color"],
                linewidth=1.7,
                alpha=0.96,
                zorder=3,
            )
        tier_handles.append(
            Line2D(
                [],
                [],
                linestyle=tier["dash"],
                color=NEUTRAL_LEGEND,
                linewidth=1.7,
                marker=tier["marker"],
                markersize=5.4,
                markeredgewidth=0.0,
                label=f"Tier {tier['tier']} (n={tier['n_boards']})",
            )
        )
    for provider in PROVIDERS:
        family_handles.append(
            Line2D(
                [],
                [],
                linestyle="-",
                color=provider["color"],
                linewidth=2.2,
                marker="o",
                markersize=6.0,
                markeredgewidth=0.0,
                label=provider["label"],
            )
        )

    # --- direct labels: every model exactly once -----------------------------
    for model in MODELS:
        spec = LABELS[model["name"]]
        anchor_tier = spec["tier"] - 1
        xy = (model["params_b"] * 10 ** DODGE_LOG10[spec["tier"]], model["scores"][anchor_tier])
        text = TWO_LINE_LABELS.get(model["name"], model["name"])
        arrowprops = None
        if spec["leader"]:
            arrowprops = {"arrowstyle": "-", "color": LEADER, "linewidth": 0.7, "shrinkA": 2, "shrinkB": 5}
        ax.annotate(
            text,
            xy=xy,
            xytext=(spec["dx"], spec["dy"]),
            textcoords="offset points",
            ha=spec["ha"],
            va=spec["va"],
            fontsize=7.8,
            color=TEXT,
            linespacing=1.25,
            zorder=6,
            arrowprops=arrowprops,
            path_effects=[path_effects.withStroke(linewidth=2.6, foreground=AXES_BG)],
        )

    # --- legends: colour = family; marker shape + dash = tier ----------------
    legend_kwargs = dict(
        loc="upper left",
        frameon=True,
        framealpha=0.94,
        facecolor=AXES_BG,
        edgecolor=GRID,
        labelcolor=TEXT,
        fontsize=9,
        borderpad=0.7,
        labelspacing=0.55,
        handlelength=2.7,
        handletextpad=0.8,
    )

    def style_legend(legend: Legend) -> None:
        legend.get_title().set_color(MUTED)
        legend.get_title().set_fontsize(8.5)
        legend.get_frame().set_linewidth(0.8)

    family_legend = ax.legend(
        handles=family_handles,
        bbox_to_anchor=(0.012, 0.988),
        title="MODEL FAMILY",
        **legend_kwargs,
    )
    style_legend(family_legend)
    ax.add_artist(family_legend)

    # The tier legend is anchored to the drawn family-legend bbox so the two
    # legends are stacked and can never overlap each other.
    fig.canvas.draw()
    family_bbox_axes = family_legend.get_window_extent(fig.canvas.get_renderer()).transformed(
        ax.transAxes.inverted()
    )
    tier_legend = ax.legend(
        handles=tier_handles,
        bbox_to_anchor=(0.012, family_bbox_axes.y0 - 0.014),
        title="DIFFICULTY TIER — MARKER + DASH",
        **legend_kwargs,
    )
    style_legend(tier_legend)

    # --- run-config badge ----------------------------------------------------
    ax.text(
        0.5,
        0.968,
        "vLLM 0.29.0 - T=0 - 25-turn budget",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color=MUTED,
        fontfamily=FONT_MONO,
        zorder=7,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": AXES_BG, "edgecolor": GRID, "linewidth": 0.8},
    )

    # --- axes ----------------------------------------------------------------
    ax.set_yscale("linear")
    ax.set_xscale("log")
    ax.set_xlim(6.0, 230.0)
    ax.set_ylim(-5.0, 105.0)

    ax.set_yticks(np.arange(0, 101, 10))
    ax.xaxis.set_major_locator(FixedLocator([10.0, 30.0, 100.0]))
    ax.xaxis.set_major_formatter(FixedFormatter(["10", "30", "100"]))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0), numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.tick_params(axis="both", which="major", colors=LEADER, labelcolor=TEXT, labelsize=9.5,
                   length=4.5, width=0.8, direction="out")
    ax.tick_params(axis="x", which="minor", colors=LEADER, length=2.8, width=0.7)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT_MONO)

    ax.set_xlabel("TOTAL PARAMETERS (B, log scale)", fontsize=11, color=TEXT, labelpad=9)
    ax.set_ylabel("SCORE (%)", fontsize=11, color=TEXT, labelpad=9)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)

    # --- header: title, subtitle, accent rule --------------------------------
    fig.text(0.055, 0.968, "TURING TUMBLE LEADERBOARD", fontsize=20, fontweight="bold",
             color=TEXT, ha="left", va="top")
    fig.text(0.056, 0.916, "Official sweep - 7 models - single run per cell - Tier 1-4",
             fontsize=10, color=MUTED, ha="left", va="top")
    fig.add_artist(Line2D([0.055, 0.945], [0.877, 0.877], color=GRID, linewidth=1.0))
    fig.add_artist(Line2D([0.055, 0.135], [0.877, 0.877], color=MUTED, linewidth=2.4))

    # --- export (explicit page size, opaque background, no tight bbox) --------
    paths = []
    for ext, extra in (("png", {"dpi": RENDER_DPI}), ("pdf", {}), ("svg", {})):
        path = OUT_DIR / f"leaderboard_by_tier.{ext}"
        fig.savefig(path, facecolor=BG, edgecolor="none", **extra)
        paths.append(path)
    plt.close(fig)

    flatten_png_alpha(paths[0])
    return paths


def flatten_png_alpha(path: Path) -> None:
    """Drop the fully opaque alpha channel so the delivered PNG has no transparency."""
    from PIL import Image

    with Image.open(path) as image:
        if image.mode == "RGBA":
            image.convert("RGB").save(path, dpi=(RENDER_DPI, RENDER_DPI))


def write_provenance(rows: list[dict], figure_paths: list[Path], csv_path: Path) -> Path:
    files = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in [*figure_paths, csv_path]}
    provenance = {
        "figure": "Turing Tumble leaderboard - score (%) vs total parameters (B, log scale)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "scripts/plot_leaderboard_by_tier.py",
        "source": {
            "file": SOURCE_FILE,
            "section": SOURCE_SECTION,
            "lines": SOURCE_LINES,
            "table": "Success by tier (official challenge sweep)",
        },
        "run_config": {
            "engine": "vLLM 0.29.0",
            "temperature": 0,
            "max_turns": 25,
            "replicates": "single run per cell",
        },
        "tier_n_boards": {"tier_1": 11, "tier_2": 22, "tier_3": 8, "tier_4": 14, "total": 55},
        "axes": {
            "y": "SCORE (%)",
            "y_limits": [-5.0, 105.0],
            "y_axis_note": "axis extends 5 percentage points below zero so the 0 % points stay visible",
            "x": "TOTAL PARAMETERS (B)",
            "x_scale": "log10",
            "x_limits": [6.0, 230.0],
            "x_ticks": [10.0, 30.0, 100.0],
        },
        "x_dodge": {
            "log10_decades_per_tier": {str(k): v for k, v in DODGE_LOG10.items()},
            "reason": "exactly coincident 0 % markers (DeepSeek-Coder-V2-Lite all tiers; "
                      "Qwen2.5-Coder-7B T2/T4) would otherwise overlap completely",
            "max_parameter_distortion_pct": round((10 ** max(abs(v) for v in DODGE_LOG10.values()) - 1) * 100, 2),
        },
        "provider_connectors": {
            "rule": "One dashed connector per family group with >= 2 models, ordered by "
                    "ascending total parameters; single-model families are drawn as lone "
                    "markers with no connecting line.",
            "colour_rule": "Connectors are coloured by model family and dashed by difficulty tier.",
            "attribution": "Provider/publisher inferred from the model names/IDs: repo "
                           "metadata records only the serving engine (vLLM 0.29.0), not "
                           "the model publisher.",
            "assertion": "build_connectors() asserts every connector joins exactly one family and "
                         "that its members are ordered by ascending parameters.",
            "groups": connector_inventory(),
        },
        "palette": {
            "scheme": "Okabe-Ito (colourblind-safe) - one colour per model family",
            "family_colors": FAMILY_COLOR,
            "legend_encoding": "Colour encodes the model family (MODEL FAMILY legend). Marker "
                               "shape and dash pattern encode the difficulty tier (DIFFICULTY "
                               "TIER - MARKER + DASH legend, neutral #c9ced6 swatches). No "
                               "visual channel carries two meanings.",
            "tier_markers": {f"tier_{t['tier']}": t["marker"] for t in TIERS},
            "tier_dashes": {f"tier_{t['tier']}": list(t["dash"][1]) for t in TIERS},
            "audit": PALETTE_AUDIT,
        },
        "disclosures": [
            "Source: docs/results_analysis.md section 14.3, table at lines 1059-1067.",
            "Colour encodes the model family (Qwen #0072B2, Google/Gemma #D55E00, "
            "DeepSeek #009E73, OpenAI #CC79A7); marker shape and dash pattern encode the "
            "difficulty tier. No visual channel carries two meanings.",
            "Single run per cell - no confidence intervals; the figure shows point estimates only.",
            "Dashed connectors join models from the same family only: Qwen x3 (7/35/177 B) and "
            "Google/Gemma x2 (26/31 B) per tier. DeepSeek (16 B) and OpenAI (120 B) are "
            "single-model families and appear as lone markers with no connector.",
            "Provider attribution is inferred from the model names/IDs; repo metadata records "
            "only the serving engine (vLLM), not the model publisher.",
            "Connect-the-dots connectors are visual guides within one family, not fitted trends.",
            "All four family colours clear 3:1 contrast against #0d1117, but the grayscale "
            "CIE L* heuristic flags 4 of 6 family pairs below 10: family identity also rests on "
            "the MODEL FAMILY legend and the direct model labels, not on colour alone.",
            "Mixture-of-experts models are plotted at TOTAL parameter count, not active parameter count.",
            "qwen3.8-flash-next-awq4 = 177 B supplied by the user, absent from repo metadata.",
            "The y-axis extends 5 percentage points below zero; the 0 % points sit on the zero rule.",
            "Per-tier horizontal dodge (<= {:.3f} decades, <= {:.1f} % in parameter value) keeps coincident markers visible.".format(
                max(abs(v) for v in DODGE_LOG10.values()),
                (10 ** max(abs(v) for v in DODGE_LOG10.values()) - 1) * 100,
            ),
        ],
        "no_invented_values": "No scores, parameter counts, confidence intervals, error bars, or trend fits "
                              "were generated for this figure.",
        "files": files,
    }
    path = OUT_DIR / "provenance.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    csv_path = write_data_csv(rows)
    figure_paths = render()
    provenance_path = write_provenance(rows, figure_paths, csv_path)

    print(f"wrote {len(rows)} plotted rows to {csv_path.relative_to(ROOT)}")
    for path in [*figure_paths, provenance_path]:
        print(f"wrote {path.relative_to(ROOT)}")
    print()
    print("model                          params(B)  T1     T2     T3     T4")
    for model in MODELS:
        scores = "  ".join(f"{value:5.1f}" for value in model["scores"])
        print(f"{model['name']:<30} {model['params_b']:>7.1f}  {scores}")
    print()
    print("family connectors (per tier):")
    for connector in build_connectors():
        names = ", ".join(f"{m['name']} ({m['params_b']:.0f} B)" for m in connector["models"])
        print(
            f"  {connector['provider']['label']:<16} {connector['provider']['color']}  "
            f"{len(connector['models'])} points -> {names}"
        )
    lone = [m["name"] for m in MODELS if m["name"] not in CONNECTOR_GROUP]
    print(f"  lone markers (single-model families): {', '.join(lone)}")


if __name__ == "__main__":
    main()
