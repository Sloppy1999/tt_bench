#!/usr/bin/env python3
"""Estimate full-study resources from completed, comparable pilot invocations.

No submission occurs here. The safety factor is a planning margin, not a
statistical confidence bound. Both actual and requested GPU-hour totals are saved.
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd


def estimate(pilot_root, accounting_path, full_matrix, available_gpu_hours, factor=1.5,
             allow_reviewed_failures=False):
    accounting = pd.read_csv(accounting_path, sep="|", dtype={"JobIDRaw": str})
    accounting = accounting[~accounting.JobIDRaw.str.contains(r"\.")]
    times = {r.JobIDRaw: float(r.ElapsedRaw) for _, r in accounting.iterrows()
             if r.State == "COMPLETED" and r.ExitCode == "0:0"}
    rates, jobs, storage = {}, {}, {}
    reviewed_episode_count = 0
    reference = {}
    for manifest_path in pilot_root.rglob("manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        identity = manifest["identity"]
        config = identity["config"]
        directory = manifest_path.parent
        episodes = [json.loads(p.read_text()) for p in directory.glob("*/episode.json")]
        if len(episodes) != len(config["tasks"]):
            raise ValueError(f"Incomplete pilot or infrastructure flags: {directory}")
        reviewed = sum(bool(r["infrastructure_review_required"]) for r in episodes)
        reviewed_episode_count += reviewed
        if reviewed and not allow_reviewed_failures:
            raise ValueError(f"Infrastructure review required: {directory}")
        invocations = [json.loads(p.read_text()) for p in directory.glob("invocation_*.json")]
        if len(invocations) != 1 or invocations[0]["status"] != "complete":
            raise ValueError(f"Pilot needs clean invocation accounting: {directory}")
        invocation = invocations[0]
        job_id = invocation["slurm_job_id"]
        if job_id not in times:
            raise ValueError(f"Missing successful final accounting for {job_id}")
        key = (config["model"], config["tier"], manifest["arm"])
        if key in rates:
            raise ValueError(f"Duplicate pilot cell: {key}")
        rates[key] = invocation["wall_seconds"] / len(episodes)
        storage[key] = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()) / len(episodes)
        jobs.setdefault(job_id, {"model": config["model"], "tier": config["tier"], "seconds": 0})["seconds"] += invocation["wall_seconds"]
        comparison = {k: v for k, v in config.items() if k not in ("tasks", "seed", "shard_id")}
        reference[(config["model"], config["tier"])] = comparison
    startup = {(r["model"], r["tier"]): max(0, times[job_id] - r["seconds"]) for job_id, r in jobs.items()}
    estimates = []
    for index, cell in enumerate(json.loads(full_matrix.read_text())):
        config = json.loads(Path(cell["config"]).read_text())
        key = (config["model"], config["tier"])
        comparison = {k: v for k, v in config.items() if k not in ("tasks", "seed", "shard_id")}
        if reference.get(key) != comparison:
            raise ValueError(f"Pilot/full configuration mismatch or absent pilot for {key}")
        model_seconds = sum(rates[(*key, arm)] for arm in cell["arms"]) * len(config["tasks"])
        wall_seconds = factor * (startup[key] + model_seconds)
        requested = max(1800, math.ceil(wall_seconds / 1800) * 1800)
        if requested > 86400:
            raise ValueError(f"Shard {index} exceeds the 24h QOS limit; use smaller shards")
        estimates.append({"matrix_index": index, "model": key[0], "tier": key[1],
            "predicted_wall_seconds_with_margin": wall_seconds,
            "requested_wall_seconds": requested,
            "predicted_gpu_hours": wall_seconds * 4 / 3600,
            "requested_gpu_hours": requested * 4 / 3600,
            "predicted_storage_bytes_with_margin": factor * len(config["tasks"]) * sum(storage[(*key, a)] for a in cell["arms"])})
    predicted = sum(r["predicted_gpu_hours"] for r in estimates)
    requested = sum(r["requested_gpu_hours"] for r in estimates)
    return {"planning_safety_factor": factor, "pilot_accounting": str(accounting_path),
            "reviewed_failure_policy": "included_in_cost_estimate" if allow_reviewed_failures else "refuse",
            "reviewed_episode_count": reviewed_episode_count,
            "predicted_gpu_hours": predicted, "requested_gpu_hours_upper_bound": requested,
            "available_gpu_hours": available_gpu_hours,
            "fits_allocation": requested <= available_gpu_hours,
            "predicted_storage_bytes_with_margin": sum(r["predicted_storage_bytes_with_margin"] for r in estimates),
            "limitations": "Eight tasks per tier/model; variation across tasks/seeds may exceed the planning margin. Recheck allocation immediately before submission.",
            "cells": estimates}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot-root", type=Path, required=True)
    p.add_argument("--accounting", type=Path, required=True)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--available-gpu-hours", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-reviewed-failures", action="store_true",
                    help="Include complete episodes flagged for explicit review in cost rates")
    a = p.parse_args()
    result = estimate(a.pilot_root, a.accounting, a.matrix, a.available_gpu_hours,
                      allow_reviewed_failures=a.allow_reviewed_failures)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "cells"}, indent=2))
    raise SystemExit(0 if result["fits_allocation"] else 1)
