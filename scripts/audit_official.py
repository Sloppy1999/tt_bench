#!/usr/bin/env python3
"""Audit the official challenge corpus, board by board.

Checks three layers that have to agree before a task can score anything:

1. the board runs (no free-fall, no lost marbles)
2. it has a complete target output to be scored against
3. its ``ball_path`` answers obey the simulator's routing rules

Intended as the gate while the official challenges are re-authored: a task is
trustworthy only once its row reads OK in every column.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tt_bench.simulator import Board, verify_task

REPO = Path(__file__).resolve().parent.parent
CHALLENGE_DIR = REPO / "data" / "tasks" / "official" / "challenges" / "json"
QUESTION_DIR = REPO / "data" / "tasks" / "official" / "questions"

RAMP_STEP = {"ramp_right": 1, "ramp_left": -1}
COORD = re.compile(
    r"(ramp_right|ramp_left|crossover|bit|gear_bit|gear|interceptor)"
    r"\s*(?:at)?\s*\((\d+)\s*,\s*(\d+)\)"
)


def board_status(challenge: dict) -> str:
    """Why the board does or does not run."""
    try:
        board = Board.from_task_dict(challenge)
    except Exception as exc:
        return f"build-error({type(exc).__name__})"

    sequence = challenge.get("input_sequence") or ["blue"]
    if isinstance(sequence, str):
        sequence = [s.strip() for s in sequence.split(",") if s.strip()]
    try:
        results = board.run(sequence)
    except Exception as exc:
        return f"run-error({type(exc).__name__})"

    for result in results:
        path = result.path or []
        for i, curr in enumerate(path[1:], start=1):
            prev = path[i - 1]
            x, y = curr
            if prev[1] < 0 and y >= 0:
                continue
            nxt = path[i + 1] if i + 1 < len(path) else None
            if (
                y == board.rows - 1
                and nxt is not None
                and nxt[1] >= board.rows
                and x in (board.left_catcher_x, board.right_catcher_x)
            ):
                continue
            if 0 <= x < board.cols and 0 <= y < board.rows and curr not in board.components:
                return "free-fall"

    if any(
        r.caught_by is None
        and r.termination_reason not in ("no_blue_balls", "no_red_balls")
        for r in results
    ):
        return "marbles-lost"

    try:
        return "OK" if verify_task(challenge) else "output-mismatch"
    except Exception:
        return "output-mismatch"


def target_status(challenge: dict) -> str:
    """Whether there is a complete output to score against."""
    fms = challenge.get("solution", {}).get("final_marble_state")
    if not fms:
        expected = challenge.get("expected_output") or {}
        if any(k in expected for k in ("left_catcher", "right_catcher", "intercepted")):
            return "counts-only"
        return "missing"
    if not all(str(c) in ("blue", "red", "intercepted") for c in fms):
        return "placeholder"

    sequence = challenge.get("input_sequence") or []
    if sequence and len(fms) < len(sequence):
        return f"partial({len(fms)}/{len(sequence)})"
    return "OK"


def path_answer_status(stem: str) -> str:
    """Whether documented ball paths follow the simulator's routing."""
    qfile = QUESTION_DIR / f"{stem}_questions.json"
    if not qfile.exists():
        return "no-questions"
    checked = bad = vague = 0
    for question in json.loads(qfile.read_text()).get("questions", []):
        if question.get("type") != "ball_path":
            continue
        answer = str(question.get("answer", question.get("expected_answer", "")))
        steps = [(t, int(x), int(y)) for t, x, y in COORD.findall(answer)]
        if len(steps) < 2:
            vague += 1
            continue
        checked += 1
        for (t0, x0, y0), (t1, x1, y1) in zip(steps, steps[1:]):
            if t0 not in RAMP_STEP:
                if y1 != y0 + 1:
                    bad += 1
                    break
                continue
            if (x1, y1) != (x0 + RAMP_STEP[t0], y0 + 1):
                bad += 1
                break
    if not checked:
        return "unverifiable" if vague else "n/a"
    return "OK" if not bad else f"{bad}/{checked} impossible"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="list every board, not just broken ones")
    args = parser.parse_args()

    files = sorted(p for p in CHALLENGE_DIR.rglob("*.json") if "_questions" not in p.stem)
    rows = []
    for path in files:
        challenge = json.loads(path.read_text())
        rows.append((
            path.stem,
            board_status(challenge),
            target_status(challenge),
            path_answer_status(path.stem),
        ))

    healthy = [r for r in rows if r[1] == "OK" and r[2] == "OK" and r[3] in ("OK", "n/a", "no-questions")]

    print(f"{'challenge':28s} {'board':18s} {'target':16s} paths")
    print("-" * 78)
    for stem, board, target, paths in rows:
        if not args.all and (stem, board, target, paths) in [tuple(h) for h in healthy]:
            continue
        print(f"{stem:28s} {board:18s} {target:16s} {paths}")

    print("-" * 78)
    print(f"{len(rows)} boards — {len(healthy)} fully trustworthy")
    for label, idx in (("board", 1), ("target", 2), ("paths", 3)):
        counts: dict[str, int] = {}
        for row in rows:
            counts[row[idx]] = counts.get(row[idx], 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
        print(f"  {label:7s} {summary}")


if __name__ == "__main__":
    main()
