#!/usr/bin/env python3
"""Tidy completed episodes and paired puzzle-cluster bootstrap estimates.

Refuses incomplete pairing; no silent intersection or exclusion of failures.
Reads only controlled-ladder episode.json files, not historical benchmark data.
"""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def bootstrap(matrix, rng, draws=10000):
    """Resample puzzle IDs; retain all paired seeds within each puzzle."""
    puzzle_means = np.asarray(matrix, dtype=float).mean(axis=1)
    samples = rng.choice(puzzle_means, (draws, len(puzzle_means)), replace=True).mean(axis=1)
    return [float(v) for v in np.quantile(samples, [.025, .975])]


def paired_randomization(matrix, rng, draws=10000):
    """Exchange arm labels per puzzle, retaining all seed pairs in each block."""
    means = np.asarray(matrix, dtype=float).mean(axis=1)
    if len(means) < 20:
        return None
    observed = abs(means.mean())
    null = np.abs((rng.choice([-1, 1], (draws, len(means))) * means).mean(axis=1))
    return float((1 + np.count_nonzero(null >= observed - 1e-12)) / (draws + 1))


def holm(values):
    result = [None] * len(values)
    ordered = sorted((value, index) for index, value in enumerate(values) if value is not None)
    previous = 0.0
    for rank, (value, index) in enumerate(ordered):
        previous = max(previous, min(1.0, value * (len(ordered) - rank)))
        result[index] = previous
    return result


def attach_accounting(frame, accounting):
    """Keep allocated job cost authoritative; prorate it for episode summaries."""
    jobs = pd.read_csv(accounting, sep="|", dtype={"JobIDRaw": str})
    jobs = jobs[~jobs.JobIDRaw.str.contains(r"\.")].copy()
    if jobs.JobIDRaw.duplicated().any():
        raise ValueError("Duplicate job accounting rows")
    costs = []
    frame["gpu_hours"] = 0.0
    covered = set()
    for _, job in jobs.iterrows():
        match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", str(job.AllocTRES))
        gpu_hours = int(job.ElapsedRaw) * int(match[1]) / 3600 if match else None
        costs.append(gpu_hours)
        indices = frame.index[frame.slurm_job_ids.map(lambda ids: job.JobIDRaw in ids)]
        if gpu_hours is None:
            continue
        weights = frame.loc[indices, "wall_seconds"].clip(lower=0.001)
        if len(indices):
            frame.loc[indices, "gpu_hours"] += gpu_hours * weights / weights.sum()
            covered.add(job.JobIDRaw)
    for index, row in frame.iterrows():
        if not row.slurm_job_ids or set(row.slurm_job_ids) - covered:
            frame.loc[index, "gpu_hours"] = np.nan
    jobs["allocated_gpu_hours"] = costs
    return jobs


def analyze(root, output, expected_arms=None, accounting=None):
    paths = sorted(root.rglob("episode.json"))
    if not paths:
        raise ValueError("No completed episodes; cannot report intervention effects")
    rows = [json.loads(p.read_text()) for p in paths]
    frame = pd.DataFrame(rows)
    synthetic = (root / "smoke_summary.json").exists()
    if not synthetic:
        manifests = {}
        for path, row in zip(paths, rows):
            identity = json.loads((path.parents[1] / "manifest.json").read_text())["identity"]
            key = (row["model"], row["tier"], row["seed"], identity["config"].get("shard_id"))
            if key in manifests and manifests[key] != identity:
                raise ValueError(f"Code, task or configuration mismatch across arms for {key}")
            manifests[key] = identity
    keys = ["model", "tier", "arm", "task_id", "seed"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate episodes: select a single experiment namespace")
    jobs = attach_accounting(frame, accounting) if accounting else None
    rng = np.random.default_rng(20260919)
    aggregates = []
    contrasts = []
    for (model, tier), group in frame.groupby(["model", "tier"]):
        baseline = group[group.arm == "A0_baseline"].set_index(["task_id", "seed"])
        if baseline.empty:
            raise ValueError(f"Missing baseline for {model}, tier {tier}")
        if expected_arms and set(expected_arms) - set(group.arm):
            raise ValueError(f"Missing requested arms for {model}/{tier}")
        for arm, cell in group.groupby("arm"):
            if set(cell.set_index(["task_id", "seed"]).index) != set(baseline.index):
                raise ValueError(f"Incomplete paired roster for {model}/{tier}/{arm}")
        for arm, cell in group.groupby("arm"):
            indexed = cell.set_index(["task_id", "seed"])
            if set(indexed.index) != set(baseline.index):
                raise ValueError(f"Incomplete paired roster for {model}/{tier}/{arm}")
            indexed = indexed.reindex(baseline.index)
            success = indexed.success.astype(float).unstack("seed")
            difference = (indexed.success.astype(float) - baseline.success.astype(float)).unstack("seed")
            if success.isna().any().any():
                raise ValueError("Every puzzle must have every seed")
            lo, hi = bootstrap(success.to_numpy(), rng)
            dlo, dhi = bootstrap(difference.to_numpy(), rng)
            aggregates.append({"model": model, "tier": tier, "arm": arm,
                "episodes": len(cell), "puzzles": len(success), "success_rate": cell.success.mean(),
                "ci_low": lo, "ci_high": hi, "paired_difference": difference.to_numpy().mean(),
                "difference_ci_low": dlo, "difference_ci_high": dhi,
                "mean_tokens": cell.tokens.mean(), "mean_seconds": cell.wall_seconds.mean(),
                "gpu_hours": cell.gpu_hours.sum(min_count=1),
                "success_after_first_failed_simulation": cell.success_after_first_failed_simulation.mean(),
                "infrastructure_review_required": cell.infrastructure_review_required.sum()})
        arms = sorted(group.arm.unique())
        comparisons = [(arm, "A0_baseline") for arm in arms if arm != "A0_baseline"]
        comparisons += [(arms[i], arms[i - 1]) for i in range(2, len(arms))]
        family = []
        for treatment, comparator in comparisons:
            treated = group[group.arm == treatment].set_index(["task_id", "seed"])
            control = group[group.arm == comparator].set_index(["task_id", "seed"])
            delta = (treated.success.astype(float) - control.success.astype(float)).unstack("seed")
            low, high = bootstrap(delta.to_numpy(), rng)
            family.append({"model": model, "tier": tier, "treatment": treatment,
                "comparator": comparator, "puzzles": len(delta),
                "paired_difference": float(delta.to_numpy().mean()),
                "ci_low": low, "ci_high": high,
                "paired_randomization_p": paired_randomization(delta.to_numpy(), rng)})
        for row, adjusted in zip(family, holm([r["paired_randomization_p"] for r in family])):
            row["holm_p_within_model_tier"] = adjusted
        contrasts.extend(family)
    # Refuse overwrites, including partial output directories from prior reports.
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "episodes.csv", index=False)
    if jobs is not None:
        jobs.to_csv(output / "job_costs.csv", index=False)
    summary = pd.DataFrame(aggregates)
    summary.to_csv(output / "aggregate.csv", index=False)
    pd.DataFrame(contrasts).to_csv(output / "paired_contrasts.csv", index=False)
    frame.groupby(["model", "tier", "arm", "seed"], dropna=False).agg(
        episodes=("success", "size"), success_rate=("success", "mean"),
        mean_tokens=("tokens", "mean"), mean_seconds=("wall_seconds", "mean"),
    ).to_csv(output / "per_seed.csv")
    frame.groupby(["model", "tier", "arm", "failure_category"]).size().rename("episodes").to_csv(output / "failures.csv")
    manifest = [{"episode_path": str(p), "job_ids": r["slurm_job_ids"]} for p, r in zip(paths, rows)]
    (output / "jobs.json").write_text(json.dumps(manifest, indent=2))
    lines = ["# Controlled ladder results", "",
             "SCRIPTED SMOKE ONLY — not model-performance evidence." if synthetic else "Completed model episodes.",
             "95% percentile bootstrap intervals resample puzzle IDs and retain paired seeds.",
             "Official failures remain in denominators. Infrastructure flags require log review.",
             ("Episode GPU-hours prorate allocated job cost by recorded latency, including startup; job_costs.csv is authoritative."
              if jobs is not None else "GPU-hours are unavailable until Slurm accounting is joined; they are never inferred as zero."),
             "Search ranking uses observable routing, not official success or hidden targets.", "",
             "| Model | Tier | Arm | Success | 95% CI | Paired difference | 95% CI |",
             "|---|---|---|---:|---|---:|---|"]
    for r in aggregates:
        lines.append(f"| {r['model']} | {r['tier']} | {r['arm']} | {r['success_rate']:.3f} | "
                     f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {r['paired_difference']:+.3f} | "
                     f"[{r['difference_ci_low']:+.3f}, {r['difference_ci_high']:+.3f}] |")
    lines += ["", "These intervals describe completed comparable runs only. Small puzzle counts yield unstable or degenerate intervals.",
              "paired_contrasts.csv includes both baseline and incremental comparisons. For at least 20 puzzle IDs, it reports a two-sided paired label-randomization test (10,000 draws), retaining seeds within puzzles, with Holm adjustment within each model/tier family. Effect sizes and confidence intervals are primary."]
    (output / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-arms", nargs="+", default=[
        "A0_baseline", "A1_step_sim", "A2_structured_feedback", "A3_legality_masking", "A4_best_of_4"])
    parser.add_argument("--accounting", type=Path, help="sacct -P --format=JobIDRaw,State,ElapsedRaw,AllocTRES,ExitCode output")
    args = parser.parse_args()
    analyze(args.root, args.output, args.expected_arms, args.accounting)
