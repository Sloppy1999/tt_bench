#!/usr/bin/env python3
"""Recompute the mechanically checkable answers on transcribed challenges.

The question files were written against boards that do not behave as described —
most printed ball paths cannot happen under the simulator's routing rules. Once a
board has been transcribed from the practice guide and verified, the answers that
follow from the board itself can simply be computed instead of asserted.

Only answers that are derivable from a simulation are touched. Conceptual
questions are left exactly as they were and reported.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tt_bench.simulator import Board, verify_task

REPO = Path(__file__).resolve().parent.parent
CHALLENGE_DIR = REPO / "data" / "tasks" / "official" / "challenges" / "json"
QUESTION_DIR = REPO / "data" / "tasks" / "official" / "questions"
DERIVABLE = {"ball_path", "output_sequence", "component_count", "parts_count"}


def simulate(task: dict) -> list:
    sequence = task.get("input_sequence") or ["blue"]
    if isinstance(sequence, str):
        sequence = [s.strip() for s in sequence.split(",") if s.strip()]
    return Board.from_task_dict(task).run(sequence)


def describe_path(task: dict, results: list) -> str | None:
    """The components the first ball actually touches, in order."""
    if not results:
        return None
    board = Board.from_task_dict(task)
    first = results[0]
    steps = []
    for x, y in (first.path or []):
        comp = board.components.get((x, y))
        if comp is not None:
            steps.append(f"{comp.component_type.value} at ({x}, {y})")
    if not steps:
        return None

    colour = first.colour or "blue"
    if first.caught_by == "left_catcher":
        ending = "and is caught at the left exit"
    elif first.caught_by == "right_catcher":
        ending = "and is caught at the right exit"
    elif first.caught_by:
        ending = "and is intercepted"
    else:
        ending = "and leaves the board"
    start = (first.path or [(0, 0)])[0]
    return (f"The first {colour} ball drops from ({start[0]}, {start[1]}), "
            f"rolls onto {', then '.join(steps)}, {ending}.")


def describe_output(results: list) -> str:
    caught = [r for r in results if r.caught_by]
    left = sum(1 for r in caught if r.caught_by == "left_catcher")
    right = sum(1 for r in caught if r.caught_by == "right_catcher")
    stopped = sum(1 for r in caught if "interceptor" in str(r.caught_by))
    order = []
    for r in caught:
        if "interceptor" in str(r.caught_by):
            order.append("intercepted")
        else:
            order.append(f"{r.colour} exits {'left' if r.caught_by == 'left_catcher' else 'right'}")
    parts = [f"{len(caught)} balls reach an exit in the order: {', '.join(order)}."]
    parts.append(f"{left} exit left, {right} exit right, "
                 f"{stopped} {'is' if stopped == 1 else 'are'} intercepted.")
    return " ".join(parts)


def describe_counts(task: dict, question: str, placed_only: bool) -> str:
    fixed = task["board"].get("fixed_components", [])
    placed = task.get("solution", {}).get("placed_components", [])
    chosen = placed if placed_only else fixed + placed
    tally = Counter(c["type"] for c in chosen)
    wanted = [t for t in ("ramp_right", "ramp_left", "crossover", "bit",
                          "gear_bit", "gear", "interceptor")
              if t.replace("_", " ") in question.lower() or t in question.lower()]
    if "ramp" in question.lower() and not wanted:
        wanted = ["ramp_right", "ramp_left"]
    if wanted:
        total = sum(tally[t] for t in wanted)
        breakdown = ", ".join(f"{tally[t]} {t}" for t in wanted if tally[t])
        scope = "placed by the solution" if placed_only else "on the board"
        return f"There are {total} ({breakdown}) {scope}." if breakdown else f"There are {total} {scope}."
    if placed_only:
        return (f"The solution places {len(placed)} parts: "
                + ", ".join(f"{n} {t}" for t, n in sorted(tally.items())) + ".")
    return (f"There are {len(chosen)} components in total: {len(fixed)} in the starting setup "
            f"and {len(placed)} placed by the solution.")


def refresh(stem: str) -> tuple[int, int, str]:
    task_path = CHALLENGE_DIR / f"{stem}.json"
    qpath = QUESTION_DIR / f"{stem}_questions.json"
    if not qpath.exists():
        return 0, 0, "no questions"
    task = json.loads(task_path.read_text())
    if task.get("provenance", {}).get("board", "").startswith("transcribed") is False:
        return 0, 0, "board not transcribed"
    try:
        if not verify_task(task):
            return 0, 0, "board does not verify"
    except Exception:
        return 0, 0, "board does not verify"

    results = simulate(task)
    data = json.loads(qpath.read_text())
    updated = untouched = 0
    for question in data.get("questions", []):
        kind = question.get("type")
        text = question.get("question", "")
        if kind not in DERIVABLE:
            untouched += 1
            continue
        if kind == "ball_path":
            answer = describe_path(task, results)
        elif kind == "output_sequence":
            answer = describe_output(results)
        else:
            answer = describe_counts(task, text, placed_only=(kind == "parts_count"))
        if not answer:
            untouched += 1
            continue
        question["answer"] = answer
        question["answer_source"] = "recomputed from the transcribed board"
        updated += 1
    qpath.write_text(json.dumps(data, indent=2) + "\n")
    return updated, untouched, "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    total_updated = total_left = 0
    for path in sorted(CHALLENGE_DIR.glob("*.json")):
        task = json.loads(path.read_text())
        if not str(task.get("provenance", {}).get("board", "")).startswith("transcribed"):
            continue
        if not args.write:
            print(f"  would refresh {path.stem}")
            continue
        updated, left, note = refresh(path.stem)
        total_updated += updated
        total_left += left
        print(f"  {path.stem:24s} recomputed={updated} left_alone={left} {note}")
    if args.write:
        print(f"\nrecomputed {total_updated} answers; {total_left} left for a human")
    else:
        print("\n(dry run — pass --write to apply)")


if __name__ == "__main__":
    main()
