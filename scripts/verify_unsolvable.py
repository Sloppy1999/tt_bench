#!/usr/bin/env python3
"""Prove that the `unsolvable` variants really admit no solution.

`generate_insight_unsolvable.py` writes every variant with `solution.verified:
false` — it ASSUMES that swapping a part type or opening an extra gap makes a
board unsolvable, and never checks. That assumption is load-bearing: if even one
variant is quietly still solvable, a model that solves it looks like a scoring
bug, and a model that fails it looks like appropriate refusal. Both readings
would be wrong.

This brute-forces the placement space through `validate_synthesis` — the very
function the benchmark scores with — so "unsolvable" means unsolvable by the
harness's own definition rather than by the generator's intent.

Exhaustive only when the inventory holds ONE part: the search is
(cells)^(parts) and blows up past that. Variants with a larger inventory are
reported as UNPROVEN rather than silently assumed, since the 1comp sets are the
ones that hand out a single part.

Usage:
    uv run python scripts/verify_unsolvable.py
    uv run python scripts/verify_unsolvable.py --dir data/tasks/challenges_2comp
    uv run python scripts/verify_unsolvable.py --pattern 'unsolvable/*ch01*.json'
"""

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tt_bench.benchmark.runner import TuringTumbleBenchmark  # noqa: E402
from tt_bench.llm import client as llm_client_  # noqa: E402


def _make_runner(challenges_dir: Path) -> TuringTumbleBenchmark:
    """A benchmark instance used only for its validator — no LLM is ever called.

    output_dir goes to a temp directory because the constructor mkdir -p's it,
    and this script has no results to write.
    """
    return TuringTumbleBenchmark(
        llm_client=llm_client_.create_llm_client(
            llm_client_.LLMConfig(provider="mock", model="mock")
        ),
        challenges_dir=challenges_dir,
        output_dir=Path(tempfile.mkdtemp(prefix="verify_unsolvable_")),
    )


def exhaustive_check(
    runner: TuringTumbleBenchmark, task_info: Dict[str, Any]
) -> Tuple[Optional[List[Tuple[str, int, int]]], str]:
    """Try the single available part in every cell.

    Returns (winning_placements, note). winning_placements is None when the
    inventory is too large to enumerate.
    """
    parts = {t: n for t, n in task_info.get("available_parts", {}).items() if n > 0}
    total = sum(parts.values())
    if total != 1:
        return None, f"inventory holds {total} parts ({parts}) — not enumerated"

    ptype = next(iter(parts))
    board = task_info.get("board", {})
    rows, cols = board.get("rows", 11), board.get("cols", 11)

    wins = []
    for y in range(rows):
        for x in range(cols):
            ok, _ = runner.validate_synthesis(
                task_info, [{"component_type": ptype, "x": x, "y": y}]
            )
            if ok:
                wins.append((ptype, x, y))
    return wins, f"{rows * cols} placements of 1x {ptype}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="data/tasks/challenges_1comp")
    ap.add_argument("--pattern", default="unsolvable/tt-official-ch*.json")
    ap.add_argument(
        "--control",
        default="tt-official-ch*.json",
        help="Glob for the well-posed parents, run as a positive control. "
        "If these do not come out SOLVABLE the validator is broken, and a "
        "clean sweep of unsolvables proves nothing.",
    )
    args = ap.parse_args()

    challenges_dir = (REPO_ROOT / args.dir).resolve()
    if not challenges_dir.is_dir():
        print(f"No such directory: {challenges_dir}", file=sys.stderr)
        return 2

    runner = _make_runner(challenges_dir)
    exit_code = 0

    for heading, pattern, expect_solvable in (
        ("Positive control (well-posed parents)", args.control, True),
        ("Unsolvable variants", args.pattern, False),
    ):
        files = sorted(challenges_dir.glob(pattern))
        print(f"\n=== {heading} — {len(files)} file(s) matching '{pattern}' ===")
        if not files:
            print("  (nothing matched)")
            continue

        for f in files:
            task_info, _ = runner.load_task(f)
            wins, note = exhaustive_check(runner, task_info)

            if wins is None:
                verdict, bad = "UNPROVEN", not expect_solvable
            elif wins:
                verdict, bad = f"SOLVABLE ({len(wins)})", not expect_solvable
            else:
                verdict, bad = "unsolvable", expect_solvable

            if bad:
                exit_code = 1
            flag = "  <-- UNEXPECTED" if bad else ""
            detail = f" {wins[:3]}" if wins else ""
            print(f"  {f.name:52s} {verdict:16s} [{note}]{detail}{flag}")

    print()
    if exit_code:
        print("FAILED: at least one file did not match its expected verdict.")
    else:
        print("OK: controls are solvable, unsolvable variants admit no solution.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
