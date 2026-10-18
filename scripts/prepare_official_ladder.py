#!/usr/bin/env python3
"""Prepare paired A0--A4 pilot and full matrices for the 58 official boards."""
import argparse, json, random
from pathlib import Path

ARMS = ["A0_baseline", "A1_step_sim", "A2_structured_feedback", "A3_legality_masking", "A4_best_of_4"]

def load_tasks(root):
    rows = []
    for path in sorted((root / "data/tasks/official/challenges/json").glob("*.json")):
        task = json.loads(path.read_text())
        if task.get("tier") not in (1, 2, 3, 4):
            raise ValueError(f"official task has invalid tier: {path}")
        board = task.get("board", {})
        dims = board.get("dimensions", {})
        rows.append({"path": str(path.relative_to(root)), "tier": task["tier"],
                     "area": int(dims.get("width", 0)) * int(dims.get("height", 0)),
                     "task_id": task.get("task_id", path.stem)})
    if len(rows) != 58:
        raise ValueError(f"expected 58 official boards, found {len(rows)}")
    return rows

def evenly(rows, n):
    rows = sorted(rows, key=lambda r: (r["area"], r["task_id"]))
    if n > len(rows):
        raise ValueError("pilot larger than tier")
    indices = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
    return [rows[i] for i in indices]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); root = a.root.resolve(); out = a.output.resolve()
    rows = load_tasks(root)
    tiers = sorted({r["tier"] for r in rows})
    pilot = {tier: evenly([r for r in rows if r["tier"] == tier], 4) for tier in tiers}
    by_tier = {tier: [r for r in rows if r["tier"] == tier] for tier in tiers}
    source = root / "review/controlled_ladder_2026-09-20/frozen_study_v2"
    source_configs = list((source / "configs/study").glob("*.json"))
    templates = {}
    for path in source_configs:
        cfg = json.loads(path.read_text()); templates.setdefault(cfg["model"], cfg)
    design = json.loads((source / "design.json").read_text())
    models = [m["model"] for m in design["models"]]
    matrices = {"pilot": [], "full": []}
    for mi, model in enumerate(models):
        if model not in templates:
            raise ValueError(f"missing template for {model}")
        for tier in tiers:
            for phase, seeds, groups in (("pilot", [1001], [pilot[tier]]),
                                         ("full", [1001, 1002, 1003], [by_tier[tier][i:i+32] for i in range(0, len(by_tier[tier]), 32)])):
                for seed in seeds:
                    for si, group in enumerate(groups):
                        shard = f"official_{phase}_{si:02d}"
                        cfg = dict(templates[model]); cfg.update({"tier": tier, "seed": seed, "shard_id": shard,
                            "tasks": [r["path"] for r in group], "temperature": 0.7, "max_turns": 25,
                            "max_tokens": 32768, "workers": 8})
                        cp = out / "configs" / phase / f"m{mi}_t{tier}_s{seed}_{si:02d}.json"
                        cp.parent.mkdir(parents=True, exist_ok=True); cp.write_text(json.dumps(cfg, indent=2))
                        rng = random.Random(20260921 + mi * 10000 + tier * 1000 + seed + si)
                        arms = ARMS[:]; rng.shuffle(arms)
                        matrices[phase].append({"config": str(cp.relative_to(root)), "arms": arms,
                            "output_root": f"benchmark_results/controlled_ladder_2026-09-20_official_{phase}"})
    out.mkdir(parents=True, exist_ok=True)
    for phase, matrix in matrices.items():
        (out / f"{phase}_matrix.json").write_text(json.dumps(matrix, indent=2))
    (out / "design.json").write_text(json.dumps({"boards": 58, "pilot_boards": 16,
        "tier_counts": {str(t): len(by_tier[t]) for t in tiers}, "pilot_cells": len(matrices["pilot"]),
        "full_cells": len(matrices["full"]), "seeds": [1001,1002,1003], "arms": ARMS,
        "temperature": 0.7, "selection": "observable simulator score; earliest ties"}, indent=2))
    print(json.dumps({"official_boards": len(rows), "tier_counts": {str(t): len(by_tier[t]) for t in tiers},
        "pilot_cells": len(matrices["pilot"]), "full_cells": len(matrices["full"])}, indent=2))

if __name__ == "__main__": main()
