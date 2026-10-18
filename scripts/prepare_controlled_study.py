#!/usr/bin/env python3
"""Freeze the approved seven-model, two-tier, three-seed paired study.

Creates configs only. Pilot results must establish resource feasibility before
the full matrix may be submitted. Exclusions retain explicit audit reasons.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from tt_bench.tools.interventions import ARMS


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def pilot_roster(rows, count=8):
    """Sample by tier, board size and distractor status; never model outcomes."""
    groups = defaultdict(list)
    for row in rows:
        task = json.loads(Path(row["path"]).read_text())
        groups[(task["board"]["width"], task["board"]["height"],
                task.get("_meta", {}).get("variant_type") or "ordinary")].append(row)
    rng = random.Random(20260920)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < min(count, len(rows)):
        for key in sorted(groups):
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
    return selected


def prepare(audit_path, models_path, output, shard_size=32):
    if output.exists():
        raise ValueError("Refusing to replace an existing frozen study")
    audit = json.loads(audit_path.read_text())
    for path, expected in audit["source_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Audit source is stale: {path}; rebuild manifest")
    models = json.loads(models_path.read_text())
    if len(models) != 7 or len({m["model"] for m in models}) != 7:
        raise ValueError("The approved roster contains seven unique models")
    eligible, exclusions = [], []
    for row in audit["files"]:
        path = Path("data/tasks") / row["path"]
        if not (path.is_relative_to("data/tasks/challenges_1comp")
                and path.match("tt-official-ch*.json")):
            continue
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Task changed since audit: {path}")
        task = json.loads(raw)
        if task.get("tier") not in (1, 2):
            continue
        entry = {**row, "path": str(path), "tier": task["tier"]}
        if task.get("_meta", {}).get("variant_type") == "unsolvable" or "unsolvable" in path.parts:
            exclusions.append({**entry, "exclusion": "marked_unsolvable"})
        elif row["status"] != "positive_candidate":
            exclusions.append({**entry, "exclusion": "mechanical_audit_failed"})
        else:
            eligible.append(entry)
    if len({r["task_id"] for r in eligible}) != len(eligible):
        raise ValueError("Duplicate eligible task IDs")
    by_tier = {tier: sorted((r for r in eligible if r["tier"] == tier), key=lambda r: r["path"])
               for tier in (1, 2)}
    if not all(by_tier.values()):
        raise ValueError("Both tiers must contain eligible tasks")
    matrix = {"pilot": [], "study": []}
    all_pilot = {tier: pilot_roster(rows) for tier, rows in by_tier.items()}
    for model_index, model in enumerate(models):
        serving = ["--tensor-parallel-size", str(model["tensor_parallel_size"]),
                   "--max-model-len", "131072", "--dtype", "auto", "--enable-auto-tool-choice",
                   "--tool-call-parser", model["tool_parser"], "--gpu-memory-utilization", "0.90"]
        if model["native_context"] < 131072:
            serving += ["--hf-overrides", json.dumps({"rope_scaling": {
                "rope_type": "yarn", "factor": 131072 / model["native_context"],
                "original_max_position_embeddings": model["native_context"]}})]
        common = {"model": model["model"], "model_revision": model["revision"],
                  "temperature": 0.7, "max_turns": 25, "max_tokens": 32768,
                  "timeout": 600, "workers": 8, "experiment_date": "2026-09-20",
                  "hardware_class": "JURECA dc-hwai, four allocated H100 GPUs",
                  "quantization": model["quantization"] or "none", "serving_config": serving}
        for tier, rows in by_tier.items():
            for phase in ("pilot", "study"):
                phase_rows = all_pilot[tier] if phase == "pilot" else rows
                shards = [phase_rows] if phase == "pilot" else [
                    phase_rows[i:i + shard_size] for i in range(0, len(phase_rows), shard_size)]
                for seed in ([1001] if phase == "pilot" else [1001, 1002, 1003]):
                    for shard_index, shard in enumerate(shards):
                        shard_id = f"shard_{shard_index:03}"
                        config = {**common, "tier": tier, "seed": seed, "shard_id": shard_id,
                                  "tasks": [r["path"] for r in shard]}
                        filename = output / "configs" / phase / f"m{model_index}_t{tier}_s{seed}_{shard_id}.json"
                        save(filename, config)
                        # Rotate arm order independently of observed outcomes.
                        arms = list(ARMS[:5])
                        random.Random(20260920 + model_index * 10000 + tier * 1000 + seed + shard_index).shuffle(arms)
                        matrix[phase].append({"config": str(filename), "arms": arms,
                            "output_root": f"benchmark_results/controlled_ladder_2026-09-20_{phase}"})
    for phase, entries in matrix.items():
        save(output / f"{phase}_matrix.json", entries)
    save(output / "eligible_tasks.json", eligible)
    save(output / "excluded_tasks.json", exclusions)
    save(output / "pilot_tasks.json", all_pilot)
    save(output / "design.json", {
        "models": models, "tiers": [1, 2], "seeds": [1001, 1002, 1003],
        "temperature": 0.7, "arms": list(ARMS[:5]), "max_concurrent_jobs": 2,
        "max_turns": 25, "max_tokens_per_turn": 32768, "episode_workers": 8,
        "task_counts": {tier: len(rows) for tier, rows in by_tier.items()},
        "exclusions": dict(Counter(r["exclusion"] for r in exclusions)),
        "episodes": len(models) * len(eligible) * 3 * 5,
        "candidate_attempts": len(models) * len(eligible) * 3 * 8,
        "array_elements": {phase: len(entries) for phase, entries in matrix.items()},
        "pilot_maximum_gpu_hours": len(matrix["pilot"]) * 2 * 4,
        "pilot_time_limit": "02:00:00", "shard_size": shard_size,
        "full_submission_gate": "Completed pilot, inspected logs, measured resource/storage estimate fits remaining allocation",
        "audit": str(audit_path), "audit_scope": audit["scope"],
        "task_glob": "data/tasks/challenges_1comp/**/tt-official-ch*.json",
        "selection_score": ["observable_validity", "caught/released", "caught", "negative_violation_count"],
        "selection_ties": "earliest candidate", "bootstrap_unit": "puzzle ID; retain paired seeds",
        "known_limitations": ["Mechanical eligibility is not semantic certification",
                              "Generated variants may share parent puzzles; report this dependence",
                              "Fixed one-component tasks cannot establish general multi-component solving gains",
                              "A5 is deferred until A0-A4 costs and findings are available"],
    })
    return {phase: len(entries) for phase, entries in matrix.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.audit, args.models, args.output)))
