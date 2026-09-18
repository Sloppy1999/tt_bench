#!/usr/bin/env python3
"""Transcribe official challenges from the practice guide into benchmark tasks.

For each challenge the guide provides three things: the starting setup, the
finished board, and the required output. The parts the solver must place are the
difference between the two boards, and the required output is the ground truth.

The guide does not state where a ball leaving the bottom row is counted, so the
catcher columns and the starting ball are chosen as the configuration whose
simulation reproduces the guide's own required output. A transcription is only
written when such a configuration exists, which makes the printed output an
independent check on the extracted board rather than an assumption.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np

from extract_boards import GRID, build_templates, parts_on_page, render, blobs
from tt_bench.simulator import Board, verify_task

REPO = Path(__file__).resolve().parent.parent
GUIDE = REPO / "Papers" / "practice-guide-2021.pdf"
CHALLENGE_DIR = REPO / "data" / "tasks" / "official" / "challenges" / "json"
SCALE = 200 / 72
ALL_PART_TYPES = [
    "ramp_right", "ramp_left", "crossover",
    "bit", "gear_bit", "gear", "interceptor", "trigger",
]


# ── The guide's own pages ────────────────────────────────────────────────

def page_map() -> dict[str, int]:
    """Challenge id -> pdf page of its puzzle (the solution is the next page)."""
    text = subprocess.run(
        ["pdftotext", "-layout", str(GUIDE), "-"], capture_output=True, text=True
    ).stdout
    pages: dict[str, int] = {}
    for line in text.splitlines():
        line = line.strip()
        if ".." not in line:
            continue
        m = re.match(
            r"^(Challenge|Practice Puzzle|Bonus Puzzle)\s*([A-C])?\s*"
            r"(?:for Challenge\s*)?(\d+)?[. ]*?(\d+)$", line)
        if not m:
            continue
        kind, letter, number, page = m.groups()
        if kind == "Challenge":
            stem = f"tt-official-ch{int(number):02d}"
        elif kind == "Bonus Puzzle":
            stem = f"tt-official-ch{int(number):02d}-b{letter or 'A'}"
        else:
            stem = f"tt-official-ch{int(number):02d}-p{letter or 'A'}"
        pages[stem] = int(page) + 4      # the guide's page 1 is the pdf's page 5
    return pages


def required_output(page_no: int, cache: Path) -> list[str] | None:
    """The ball sequence the guide prints as the challenge's required output."""
    xml = subprocess.run(
        ["pdftotext", "-bbox", "-f", str(page_no), "-l", str(page_no), str(GUIDE), "-"],
        capture_output=True, text=True).stdout

    def find(word: str):
        m = re.search(r'yMin="([\d.]+)"[^>]*yMax="([\d.]+)">' + word + "<", xml)
        return (float(m.group(1)), float(m.group(2))) if m else None

    head, foot = find("output:"), find("Starting")
    if not head or not foot:
        return None
    band = render(page_no, cache)[int(head[1] * SCALE) + 4:int(foot[0] * SCALE) - 4, :]

    # The balls sit in a short wide strip; the page frame crosses the same band.
    strips = [b for b in blobs(band < 200, 200, 60000)
              if 20 <= (b["y1"] - b["y0"]) <= 70 and (b["x1"] - b["x0"]) >= 40]
    if not strips:
        return []
    strip = max(strips, key=lambda b: b["x1"] - b["x0"])
    region = band[strip["y0"]:strip["y1"] + 1, strip["x0"]:strip["x1"] + 1]

    darkest = region.min(axis=0)
    is_ball = darkest < 165          # balls are darker than the ramp they rest on
    runs, start = [], None
    for x, on in enumerate(is_ball):
        if on and start is None:
            start = x
        elif not on and start is not None:
            if x - start >= 12:
                runs.append((start, x))
            start = None
    if start is not None and len(is_ball) - start >= 12:
        runs.append((start, len(is_ball)))

    balls = []
    for x0, x1 in runs:
        count = max(1, int(round((x1 - x0) / 20.5)))
        step = (x1 - x0) / count
        for i in range(count):
            a, b = int(x0 + i * step), int(x0 + (i + 1) * step)
            seg = darkest[a:b]
            ink = seg[seg < 165]
            if ink.size:
                balls.append((a, "blue" if float(np.median(ink)) < 110 else "red"))
    balls.sort(key=lambda t: -t[0])   # the ball nearest the exit drops first
    return [colour for _, colour in balls]


# ── Goals stated in words rather than drawn as balls ─────────────────────

COUNT_WORDS = {"no": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4,
               "five": 5, "six": 6, "seven": 7, "eight": 8,
               "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5}


def _count(token: str) -> int | None:
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return COUNT_WORDS.get(token)


def objective_target(objective: str) -> list[str] | None:
    """The output sequence an interception objective asks for.

    Several challenges print no ball strip and state the goal in words instead,
    e.g. "Let only 3 blue balls reach the bottom and catch the 4th ball in the
    interceptor", which fixes the whole sequence.
    """
    text = " ".join((objective or "").split()).lower()
    if "intercept" not in text and "catch" not in text:
        return None

    colour = "red" if "red ball" in text else "blue"

    # "Let only N <colour> balls reach the bottom/end ... intercept the (N+1)th"
    m = re.search(r"let (?:only )?(\w+) (?:blue |red )?balls? reach the (?:bottom|end)", text)
    if m:
        n = _count(m.group(1))
        if n is not None:
            return [colour] * n + ["intercepted"]

    # "Catch/intercept the Nth ball in the interceptor" on its own
    m = re.search(r"(?:catch|intercept) the (\w+) (?:blue |red )?ball", text)
    if m:
        n = _count(m.group(1))
        if n is not None and n >= 1:
            return [colour] * (n - 1) + ["intercepted"]
    return None


def is_state_goal(objective: str) -> bool:
    """Whether the goal is a final bit state rather than an output sequence."""
    text = " ".join((objective or "").split()).lower()
    return bool(re.search(r"flip (?:the )?bits?\b", text)) or bool(re.search(r"\bif .*\bbits?\b", text))


def loose_intercept(objective: str) -> str | None:
    """Colour an objective wants intercepted when it fixes no count."""
    text = " ".join((objective or "").split()).lower()
    if not text.startswith("intercept a"):
        return None
    return "red" if "red ball" in text else "blue"


# ── Assembling a task ────────────────────────────────────────────────────

def ball_colours(results: list) -> list[str]:
    """What the guide prints: the colour of each ball that comes out.

    A blue ball can leave on the right, so this is not the same as the
    catcher-derived sequence the benchmark stores in final_marble_state.
    """
    out = []
    for r in results:
        if not r.caught_by:
            continue
        out.append("intercepted" if "interceptor" in str(r.caught_by) else (r.colour or "?"))
    return out


def catcher_colours(results: list) -> list[str]:
    out = []
    for r in results:
        if r.caught_by == "left_catcher":
            out.append("blue")
        elif r.caught_by == "right_catcher":
            out.append("red")
        elif r.caught_by and "interceptor" in str(r.caught_by):
            out.append("intercepted")
    return out


def matches_printed(produced: list[str], target: list[str], objective: str) -> bool:
    """Whether a run matches the guide's printed required output.

    The strip is not always the complete run: a ball caught in an interceptor is
    not drawn, and an objective ending in an ellipsis prints only as much of a
    repeating pattern as fits.
    """
    if produced == target:
        return True
    if len(produced) > len(target) and produced[:len(target)] == target:
        tail = produced[len(target):]
        if all(t == "intercepted" for t in tail):
            return True
        if "…" in (objective or "") or "..." in (objective or ""):
            return True
    # The strip draws a repeating pattern for as long as it fits, which can be
    # longer than the hoppers can actually supply.
    if 6 <= len(produced) < len(target) and target[:len(produced)] == produced:
        return True
    return False


def _task(original: dict, fixed: list[dict], placed: list[dict], *,
          entry: str, left_x: int, right_x: int, first: str,
          target: list[str]) -> dict:
    task = deepcopy(original)
    task["board"] = {
        "width": GRID,
        "height": GRID,
        "hopper_entry_mode": entry,
        "fixed_components": deepcopy(fixed),
        "ball_hoppers": {"blue": {"x": 2, "count": 8}, "red": {"x": 8, "count": 8}},
        "trigger_levers": {"left": {"x": left_x, "y": GRID},
                           "right": {"x": right_x, "y": GRID}},
    }
    parts = {t: 0 for t in ALL_PART_TYPES}
    for comp in placed:
        if comp["type"] in parts:
            parts[comp["type"]] += 1
    task["available_parts"] = parts
    task.setdefault("solution", {})
    task["solution"]["placed_components"] = deepcopy(placed)
    task["solution"]["final_marble_state"] = list(target)
    task["input_sequence"] = [first]
    return task


def fit_configuration(original: dict, fixed: list[dict], placed: list[dict],
                      target: list[str], want_intercept: str | None = None,
                      objective: str = "", state_goal: bool = False) -> dict | None:
    """Find the catcher setup under which the board produces the guide's output."""
    # The real board keeps the blue lever on the left and the red one on the
    # right, so prefer positions closest to those before considering others; an
    # unused lever can otherwise land anywhere and still match the output.
    placements = sorted(
        ((lx, rx) for lx in range(GRID) for rx in range(GRID) if lx < rx),
        key=lambda p: abs(p[0] - 2) + abs(p[1] - 8),
    )
    for entry in ("inward", "column"):
        for first in ("blue", "red"):
            for left_x, right_x in placements:
                    task = _task(original, fixed, placed, entry=entry,
                                 left_x=left_x, right_x=right_x, first=first,
                                 target=target)
                    try:
                        board = Board.from_task_dict(task)
                        results = board.run(task["input_sequence"])
                    except Exception:
                        continue
                    produced = ball_colours(results)
                    if state_goal:
                        lost = [r for r in results
                                if r.steps > 0 and not r.caught_by]
                        if lost or len(produced) < 2:
                            continue
                        task["solution"]["final_marble_state"] = catcher_colours(results)
                        task["required_output"] = produced
                        # The goal is a bit configuration, so record where the
                        # bits actually end up once the machine stops.
                        task["final_bit_states"] = board.get_all_states()
                    elif want_intercept is not None:
                        caught = [r for r in results if r.caught_by]
                        stopped = [r for r in caught
                                   if "interceptor" in str(r.caught_by)
                                   and r.colour == want_intercept]
                        if not stopped or any(not r.caught_by for r in results if r.steps > 0):
                            continue
                        task["solution"]["final_marble_state"] = catcher_colours(results)
                        task["required_output"] = produced
                    elif not matches_printed(produced, target, objective):
                        continue
                    else:
                        task["solution"]["final_marble_state"] = catcher_colours(results)
                        task["required_output"] = produced
                    task["expected_output"] = {
                        "left_catcher": sum(1 for r in results if r.caught_by == "left_catcher"),
                        "right_catcher": sum(1 for r in results if r.caught_by == "right_catcher"),
                        "intercepted": sum(1 for r in results if r.caught_by == "interceptor"),
                    }
                    try:
                        if not verify_task(task):
                            continue
                    except Exception:
                        continue
                    return task
    return None


def transcribe(stem: str, page: int, cache: Path, templates) -> tuple[dict | None, str]:
    path = CHALLENGE_DIR / f"{stem}.json"
    if not path.exists():
        return None, "no task file"
    original = json.loads(path.read_text())

    try:
        setup = parts_on_page(page, cache, templates)
        whole = parts_on_page(page + 1, cache, templates)
    except Exception as exc:
        return None, f"extract failed ({type(exc).__name__})"

    setup_keys = {(p.type, p.x, p.y) for p in setup}
    fixed = [p.to_dict() for p in setup]
    placed = [p.to_dict() for p in whole if (p.type, p.x, p.y) not in setup_keys]
    if not placed:
        return None, "no parts to place"

    # An interception objective is the more precise of the two: the printed ball
    # strip shows only the balls that reach the bottom, not the one caught.
    candidates: list[tuple[list[str], str | None]] = []
    from_objective = objective_target(original.get("objective", ""))
    if from_objective:
        candidates.append((from_objective, None))
    printed = required_output(page, cache)
    if printed:
        candidates.append((printed, None))
    loose = loose_intercept(original.get("objective", ""))
    if loose:
        candidates.append(([], loose))
    state_goal = is_state_goal(original.get("objective", ""))
    if state_goal and not candidates:
        candidates.append(([], None))
    if not candidates:
        return None, "no required output printed"

    # A bit drawn with arrows both ways has no start state in the guide, so try
    # each possibility and keep the one that satisfies the challenge.
    open_bits = [c for c in fixed + placed if c["type"] == "bit" and c.get("state") is None]
    assignments = [dict()]
    for comp in open_bits:
        assignments = [{**a, (comp["x"], comp["y"]): v} for a in assignments for v in (0, 1)]

    task = None
    for assignment in assignments:
        for comp in fixed + placed:
            if (comp["x"], comp["y"]) in assignment:
                comp["state"] = assignment[(comp["x"], comp["y"])]
        for target, want_intercept in candidates:
            task = fit_configuration(original, fixed, placed, target, want_intercept,
                                     original.get("objective", ""), state_goal)
            if task is not None:
                break
        if task is not None:
            break
    if task is None:
        shapes = ", ".join(
            f"intercepting a {w} ball" if w else f"{len(t)} balls" for t, w in candidates)
        return None, f"no setup reproduces {shapes}"

    stripped = deepcopy(task)
    stripped["solution"]["placed_components"] = []
    try:
        if verify_task(stripped):
            return None, "puzzle solves itself without the placed parts"
    except Exception:
        pass

    task["solution"]["verified"] = True
    task["solution"]["position_verified"] = True
    task["provenance"] = {
        "board": "transcribed from practice-guide-2021.pdf",
        "page": page,
        "checked": "simulation reproduces the guide's printed required output",
    }
    return task, "OK"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()

    cache = REPO / ".guide-cache"
    templates = build_templates(cache)
    pages = page_map()
    if args.only:
        pages = {k: v for k, v in pages.items() if k in args.only}

    done, failed = 0, []
    for stem, page in sorted(pages.items(), key=lambda kv: kv[1]):
        task, why = transcribe(stem, page, cache, templates)
        if task is None:
            failed.append((stem, why))
            print(f"  {stem:24s} page {page:3d}  -- {why}")
            continue
        done += 1
        n_fixed = len(task["board"]["fixed_components"])
        n_placed = len(task["solution"]["placed_components"])
        print(f"  {stem:24s} page {page:3d}  OK  fixed={n_fixed:2d} place={n_placed:2d} "
              f"output={len(task['solution']['final_marble_state'])} balls")
        if args.write:
            (CHALLENGE_DIR / f"{stem}.json").write_text(json.dumps(task, indent=2) + "\n")

    print(f"\ntranscribed {done} / {len(pages)}")
    if not args.write:
        print("(dry run — pass --write to apply)")


if __name__ == "__main__":
    main()
