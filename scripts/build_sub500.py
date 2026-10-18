#!/usr/bin/env python3
"""Draw the frozen `scaled_sub500` subsample (HARNESS_ABLATION_PLAN.md 6.9).

WHY A SUBSAMPLE. qwen3.6-35B-A3B runs at 335-436 s per task
(docs/results_analysis.md section 6). At 8 workers the full 1 013-task `scaled`
set needs ~13.5 h, past the 12 h Slurm walltime, and seven arms of that is not
affordable. 500 tasks fit inside one job, and because `scaled` sits at a low base
rate the PAIRED interval is actually tighter there than on the primary set
(plan section 4.3).

WHY PHASE 2 NEEDS ITS OWN BASELINE. Commit a365ceb rebuilt the corpus AFTER the
stored `scaled` run was made. Measured against that run's frame (a365ceb^):

    stored `scaled` tier-1 tasks                        1013
      deleted or renamed by a365ceb                      292
      surviving filename, tier-1 in both                 721
      of those, unchanged on board/solution/targets      108

Only 108 of 1 013 are still the same task, so the stored `scaled` numbers in
docs/results_analysis.md cannot be paired against anything drawn today. Phase 2
therefore runs its OWN baseline arm on this subsample, and the default frame is
the working tree. `--frame-commit a365ceb` reproduces the intersection check
above; it yields 721 tasks and is kept for auditing, not for sampling.

The primary set is unaffected: `scaled_1comp` is 432 tier-1 tasks at a365ceb^ and
432 today, all byte-identical on board, solution, available_parts,
input_sequence, required_output and expected_output. Phase 1 pairs exactly
against the five stored greedy repetitions. `scaled_2comp` (402) is likewise
unchanged.

The drawn ids are written to a text file that belongs in version control; the
symlink directory the benchmark reads is generated from it and is disposable.

Usage:
    python scripts/build_sub500.py --write-ids
    python scripts/build_sub500.py --materialise data/tasks/scaled_sub500
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCALED_DIR = Path("data/tasks/scaled")
PATTERN = "tt-official-ch*.json"
TIER = 1
SIZE = 500
SEED = 20260918
# Default: sample the working tree, because the stored `scaled` run is not
# pairable (see the module docstring). Passing a commit here restricts the frame
# to files that existed at <commit>^, which is the audit that established it.
FRAME_COMMIT = "none"
IDS_PATH = Path("review/harness_ablation/sub500_task_ids.txt")


def frame_filenames(commit: str) -> set[str] | None:
    """Filenames present under SCALED_DIR at <commit>^, or None if unavailable."""
    try:
        out = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", f"{commit}^", "--", str(SCALED_DIR)],
            capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  ! cannot read the frame commit ({exc}); falling back to the "
              "working tree. Run a fresh baseline on the drawn ids.", file=sys.stderr)
        return None
    return {Path(line).name for line in out.splitlines() if line.strip()}


def stratum(task: dict) -> tuple:
    """Board size and reference-solution size — the two difficulty axes here."""
    board = task.get("board", {})
    n_components = len(task.get("solution", {}).get("placed_components", []))
    return (board.get("width"), board.get("height"), min(n_components, 3))


def collect(frame: set[str] | None) -> dict[tuple, list[tuple[str, str]]]:
    """stratum -> [(task_id, filename)], for tier-TIER tasks inside the frame."""
    by_stratum: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
    for path in sorted(SCALED_DIR.glob(PATTERN)):
        if frame is not None and path.name not in frame:
            continue
        try:
            task = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"  ! unreadable, skipped: {path.name}", file=sys.stderr)
            continue
        if task.get("tier") != TIER:
            continue
        by_stratum[stratum(task)].append((task["task_id"], path.name))
    return by_stratum


def draw(by_stratum: dict[tuple, list], size: int, seed: int) -> list[tuple[str, str]]:
    """Proportional allocation, largest-remainder, deterministic under `seed`."""
    rng = random.Random(seed)
    total = sum(len(v) for v in by_stratum.values())
    if total <= size:
        sys.exit(f"Frame holds {total} tasks, which is not more than {size}: "
                 "use the whole set rather than a subsample.")

    exact = {k: len(v) * size / total for k in by_stratum for v in [by_stratum[k]]}
    quota = {k: int(v) for k, v in exact.items()}
    # Largest remainder, ties broken by stratum key so the draw is reproducible.
    short = size - sum(quota.values())
    for key in sorted(exact, key=lambda k: (-(exact[k] - quota[k]), k))[:short]:
        quota[key] += 1

    drawn: list[tuple[str, str]] = []
    for key in sorted(by_stratum):
        pool = sorted(by_stratum[key])
        drawn.extend(rng.sample(pool, quota[key]))
    return sorted(drawn)


def report(by_stratum: dict[tuple, list], drawn: list[tuple[str, str]]) -> None:
    total = sum(len(v) for v in by_stratum.values())
    drawn_ids = {task_id for task_id, _ in drawn}
    drawn_strata: Counter = Counter()
    for key, items in by_stratum.items():
        drawn_strata[key] = sum(1 for task_id, _ in items if task_id in drawn_ids)

    print(f"\nframe {total} tasks -> drawn {len(drawn)}")
    print(f"{'w x h':>9} {'ncomp':>6} {'frame':>7} {'drawn':>7} {'frame %':>8} {'drawn %':>8} {'delta':>7}")
    worst = 0.0
    for key in sorted(by_stratum):
        w, h, n = key
        f_pct = 100.0 * len(by_stratum[key]) / total
        d_pct = 100.0 * drawn_strata[key] / len(drawn)
        worst = max(worst, abs(d_pct - f_pct))
        print(f"{f'{w}x{h}':>9} {n:>6} {len(by_stratum[key]):>7} {drawn_strata[key]:>7} "
              f"{f_pct:>7.2f}% {d_pct:>7.2f}% {d_pct - f_pct:>+6.2f}")
    print(f"\nlargest stratum drift: {worst:.2f} pp (plan 6.9 allows 2 pp)")
    if worst > 2.0:
        sys.exit("Stratification drifted past the 2 pp the plan allows.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-ids", action="store_true",
                    help=f"write the drawn ids to {IDS_PATH}")
    ap.add_argument("--materialise", type=Path, metavar="DIR",
                    help="create a directory of symlinks to the drawn tasks, "
                         "readable with --challenges-dir DIR --pattern '*.json'")
    ap.add_argument("--frame-commit", default=FRAME_COMMIT,
                    help="restrict the frame to files present at <commit>^ "
                         f"(default: {FRAME_COMMIT!r} = the working tree). Use "
                         "a365ceb to reproduce the corpus-drift audit.")
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if not SCALED_DIR.is_dir():
        sys.exit(f"No {SCALED_DIR} — run from the repository root.")

    frame = None if args.frame_commit == "none" else frame_filenames(args.frame_commit)
    by_stratum = collect(frame)
    if not by_stratum:
        sys.exit("Frame is empty; check --frame-commit and the tier filter.")

    drawn = draw(by_stratum, args.size, args.seed)
    report(by_stratum, drawn)

    if args.write_ids:
        IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "# scaled_sub500 — frozen subsample for the harness ablation.",
            "# Generated by scripts/build_sub500.py; regenerate, do not hand-edit.",
            f"# frame: {SCALED_DIR}/{PATTERN}, tier {TIER}, as of "
            f"{args.frame_commit}^" if frame is not None else "# frame: working tree",
            f"# size: {len(drawn)}   seed: {args.seed}",
            "# columns: task_id<TAB>filename",
        ]
        IDS_PATH.write_text(
            "\n".join(header + [f"{tid}\t{name}" for tid, name in drawn]) + "\n"
        )
        print(f"\nwrote {IDS_PATH} ({len(drawn)} ids)")

    if args.materialise:
        target = args.materialise
        target.mkdir(parents=True, exist_ok=True)
        for link in target.glob("*.json"):
            link.unlink()
        for _task_id, name in drawn:
            (target / name).symlink_to(Path("..") / SCALED_DIR.name / name)
        print(f"materialised {len(drawn)} symlinks in {target}")
        print(f"  uv run tt-benchmark --challenges-dir {target} --pattern '*.json' --tiers 1")

    return 0


if __name__ == "__main__":
    sys.exit(main())
