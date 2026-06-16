#!/usr/bin/env python3
"""Batch complexity metrics report generator.

Loads all challenge JSONs, computes complexity metrics for each, and
outputs CSV + JSON reports with tier metadata from INDEX.json.

Usage:
    PYTHONPATH=simulator uv run python scorer/complexity_report.py \
        --challenges-dir tasks/official/challenges/json \
        --index tasks/official/INDEX.json \
        --output complexity_metrics.csv \
        --output-json complexity_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "simulator"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import tt_sim
from scorer.complexity_metrics import compute_all_metrics

METRIC_ORDER = [
    "scr", "ctd", "dependency_depth", "gcc", "rpcc", "ibr", "hic",
    "bici", "sac", "sac_norm", "synthesis_load", "psde", "oss", "k_approx",
]


def load_index(index_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load INDEX.json, return {task_id: {tier, tags, ...}}."""
    if not index_path.exists():
        print(f"Warning: INDEX.json not found at {index_path}; skipping tier metadata")
        return {}
    with open(index_path) as f:
        data = json.load(f)
    idx: Dict[str, Dict[str, Any]] = {}
    for entry in data.get("tasks", []):
        idx[entry["task_id"]] = entry
    return idx


def main():
    parser = argparse.ArgumentParser(
        description="Compute complexity metrics for all Turing Tumble challenges"
    )
    parser.add_argument(
        "--challenges-dir",
        type=Path,
        default=Path("tasks/official/challenges/json"),
        help="Directory containing challenge JSON files",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("tasks/official/INDEX.json"),
        help="Path to INDEX.json with tier metadata",
    )
    parser.add_argument(
        "--pattern",
        default="tt-official-ch*.json",
        help="Glob pattern for challenge files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("complexity_metrics.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("complexity_metrics.json"),
        help="Output JSON path",
    )
    args = parser.parse_args()

    index = load_index(args.index)
    challenge_files = sorted(args.challenges_dir.glob(args.pattern))

    if not challenge_files:
        print(f"No challenge files found matching {args.challenges_dir / args.pattern}")
        return 1

    rows: List[Dict[str, Any]] = []

    for ch_path in challenge_files:
        task_id = ch_path.stem
        meta = index.get(task_id, {})
        tier = meta.get("tier")

        try:
            board = tt_sim.Board.from_task_json(str(ch_path))
            with open(ch_path) as f:
                task_info = json.load(f)
            # Coerce task_id to match filename stem for practice variants
            if ch_path.stem.startswith("tt-official-") and task_info.get("task_id") != ch_path.stem:
                task_info["task_id"] = ch_path.stem

            metrics = compute_all_metrics(board, task_info)
        except Exception as e:
            print(f"Error processing {task_id}: {e}")
            continue

        row: Dict[str, Any] = {
            "task_id": task_id,
            "tier": tier if tier is not None else "",
        }
        for m in METRIC_ORDER:
            val = metrics.get(m)
            row[m] = val

        rows.append(row)
        print(f"  {task_id:30s}  tier={tier or '?'}  BICI={metrics.get('bici', 0):.4f}")

    # Write CSV
    fieldnames = ["task_id", "tier"] + METRIC_ORDER
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Format floats to 4 decimal places for readability
            out = dict(row)
            for k in METRIC_ORDER:
                v = out.get(k)
                if isinstance(v, float):
                    out[k] = f"{v:.4f}"
                elif v is None:
                    out[k] = ""
            writer.writerow(out)

    # Write JSON
    json_rows = []
    for row in rows:
        r = dict(row)
        for k in METRIC_ORDER:
            v = r.get(k)
            if isinstance(v, float):
                r[k] = round(v, 6)
        json_rows.append(r)

    with open(args.output_json, "w") as f:
        json.dump(json_rows, f, indent=2)

    print(f"\nWrote {len(rows)} tasks to {args.output} and {args.output_json}")

    # Summary by tier
    by_tier: Dict[int, List[float]] = {}
    for row in rows:
        t = row.get("tier")
        if isinstance(t, (int, float)):
            t = int(t)
            by_tier.setdefault(t, []).append(row.get("bici", 0))

    if by_tier:
        print("\nTier summary (mean BICI):")
        for t in sorted(by_tier):
            vals = by_tier[t]
            mean = sum(vals) / len(vals)
            print(f"  Tier {t}: mean BICI = {mean:.4f}  (n={len(vals)})")

        # Also SCR
        by_tier_scr: Dict[int, List[float]] = {}
        for row in rows:
            t = row.get("tier")
            if isinstance(t, (int, float)):
                t = int(t)
                by_tier_scr.setdefault(t, []).append(row.get("scr", 0))
        print("\nTier summary (mean SCR):")
        for t in sorted(by_tier_scr):
            vals = by_tier_scr[t]
            mean = sum(vals) / len(vals)
            print(f"  Tier {t}: mean SCR = {mean:.4f}  (n={len(vals)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
